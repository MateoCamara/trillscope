"""
Segment PRESEEA files for MFA alignment.

This script:
1. Parses PRESEEA XML transcripts to get utterances with timestamps
2. Groups utterances into ~15 second segments
3. Extracts corresponding audio segments
4. Creates matching txt files for MFA
5. Runs MFA on the segmented corpus

Supports parallel processing for both preparation and MFA alignment.
"""

import argparse
import logging
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from pydub import AudioSegment
from src.ingestion.preseea.xml_parser import PreseeaXMLParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A segment of audio/transcript for MFA."""
    file_id: str
    segment_idx: int
    start_ms: float
    end_ms: float
    text: str
    speaker_code: str


def parse_timestamp(ts: str) -> Optional[float]:
    """Parse timestamp string to milliseconds."""
    if not ts:
        return None

    # Format: "MM:SS" or "HH:MM:SS"
    parts = ts.split(':')
    try:
        if len(parts) == 2:
            minutes, seconds = int(parts[0]), int(parts[1])
            return (minutes * 60 + seconds) * 1000
        elif len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            return (hours * 3600 + minutes * 60 + seconds) * 1000
    except ValueError:
        pass
    return None


def clean_text_for_mfa(text: str) -> str:
    """Clean transcript text for MFA processing."""
    # Remove XML tags if present
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove special markers
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)

    # Keep only letters, spaces, and basic punctuation
    text = re.sub(r'[^\w\sáéíóúñüÁÉÍÓÚÑÜ.,;:!?¿¡-]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def segment_file(mp3_path: Path, parser: PreseeaXMLParser,
                 segment_duration_ms: int = 15000) -> List[Segment]:
    """
    Segment a PRESEEA file into fixed-length chunks for MFA.

    Args:
        mp3_path: Path to MP3 file
        parser: XML parser instance
        segment_duration_ms: Target segment duration in ms (default 15s)

    Returns:
        List of Segment objects
    """
    txt_path = mp3_path.with_suffix('.txt')
    if not txt_path.exists():
        logger.warning(f"No transcript for {mp3_path.name}")
        return []

    # Parse transcript
    result = parser.parse(txt_path)
    if not result:
        logger.warning(f"Failed to parse {txt_path.name}")
        return []

    metadata, utterances = result
    if not utterances:
        return []

    file_id = mp3_path.stem

    # Get audio duration
    try:
        audio = AudioSegment.from_mp3(str(mp3_path))
        audio_duration_ms = len(audio)
    except Exception as e:
        logger.warning(f"Failed to load audio {mp3_path.name}: {e}")
        return []

    # Combine all informant text
    all_text = []
    for utt in utterances:
        if utt.speaker_code == 'I':  # Only informant speech
            text = clean_text_for_mfa(utt.text)
            if text:
                all_text.append(text)

    full_text = ' '.join(all_text)
    words = full_text.split()
    if not words:
        logger.warning(f"No words found for {mp3_path.name}")
        return []

    # Calculate number of segments
    num_segments = max(1, int(audio_duration_ms / segment_duration_ms))

    # Distribute words across segments proportionally
    words_per_segment = len(words) / num_segments

    segments = []
    for i in range(num_segments):
        start_ms = i * segment_duration_ms
        end_ms = min((i + 1) * segment_duration_ms, audio_duration_ms)

        # Get words for this segment
        word_start = int(i * words_per_segment)
        word_end = int((i + 1) * words_per_segment)
        segment_words = words[word_start:word_end]

        if segment_words and (end_ms - start_ms) >= 5000:  # At least 5s
            segments.append(Segment(
                file_id=file_id,
                segment_idx=i,
                start_ms=start_ms,
                end_ms=end_ms,
                text=' '.join(segment_words),
                speaker_code='I'
            ))

    return segments


def process_single_file(args: Tuple[Path, Path, int]) -> int:
    """Process a single PRESEEA file (for parallel execution)."""
    mp3_path, output_dir, segment_duration_ms = args
    parser = PreseeaXMLParser()

    segments = segment_file(mp3_path, parser, segment_duration_ms)
    if not segments:
        return 0

    # Load audio once
    try:
        audio = AudioSegment.from_mp3(str(mp3_path))
        audio = audio.set_frame_rate(16000).set_channels(1)
    except Exception as e:
        return 0

    count = 0
    for seg in segments:
        seg_id = f"{seg.file_id}_seg{seg.segment_idx:03d}"

        # Extract audio segment
        audio_seg = audio[seg.start_ms:seg.end_ms]
        if len(audio_seg) < 1000:  # Skip very short segments
            continue

        wav_path = output_dir / f"{seg_id}.wav"
        txt_path = output_dir / f"{seg_id}.txt"

        # Export audio
        audio_seg.export(str(wav_path), format='wav')

        # Export text
        txt_path.write_text(seg.text, encoding='utf-8')

        count += 1

    return count


def prepare_mfa_corpus(preseea_dir: Path, output_dir: Path,
                       max_files: int = None,
                       segment_duration_ms: int = 15000,
                       num_workers: int = None) -> int:
    """
    Prepare segmented MFA corpus from PRESEEA files using parallel processing.

    Args:
        preseea_dir: PRESEEA corpus directory
        output_dir: Output directory for MFA input
        max_files: Maximum files to process (for testing)
        segment_duration_ms: Target segment duration
        num_workers: Number of parallel workers (default: CPU count)

    Returns:
        Number of segments created
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    mp3_files = sorted(preseea_dir.glob('*.mp3'))
    if max_files:
        mp3_files = mp3_files[:max_files]

    if num_workers is None:
        num_workers = os.cpu_count() or 4

    logger.info(f"Processing {len(mp3_files)} files with {num_workers} workers...")

    # Prepare args for parallel processing
    args_list = [(mp3, output_dir, segment_duration_ms) for mp3 in mp3_files]

    total_segments = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, args): args[0].name
                   for args in args_list}

        for future in as_completed(futures):
            filename = futures[future]
            try:
                count = future.result()
                total_segments += count
                completed += 1
                if completed % 50 == 0:
                    logger.info(f"Completed {completed}/{len(mp3_files)} files, {total_segments} segments so far")
            except Exception as e:
                logger.warning(f"Failed to process {filename}: {e}")

    logger.info(f"Created {total_segments} segments from {completed} files")
    return total_segments


def run_mfa_alignment(input_dir: Path, output_dir: Path,
                      use_g2p: bool = True, num_jobs: int = None) -> bool:
    """Run MFA alignment on prepared corpus with parallel jobs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_jobs is None:
        num_jobs = os.cpu_count() or 4

    cmd = [
        'mfa', 'align',
        str(input_dir),
        'spanish_mfa',
        'spanish_mfa',
        str(output_dir),
        '-j', str(num_jobs),
        '--clean',
        '--beam', '100',
        '--retry_beam', '400'
    ]

    if use_g2p:
        cmd.extend(['--use_g2p', '--g2p_model_path', 'spanish_spain_mfa'])

    logger.info(f"Running MFA: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"MFA failed: {result.stderr}")
        return False

    # Count TextGrids created
    textgrids = list(output_dir.glob('*.TextGrid'))
    logger.info(f"Created {len(textgrids)} TextGrid files")

    return len(textgrids) > 0


def main():
    parser = argparse.ArgumentParser(description='Segment PRESEEA for MFA')
    parser.add_argument('--preseea-dir', type=Path, default=Path('dataset/preseea'),
                        help='PRESEEA corpus directory')
    parser.add_argument('--output-dir', type=Path, default=Path('mfa_preseea'),
                        help='Output directory for MFA files')
    parser.add_argument('--segment-duration', type=int, default=15,
                        help='Target segment duration in seconds (default: 15)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum files to process (for testing)')
    parser.add_argument('-j', '--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Only prepare corpus, do not run MFA')
    parser.add_argument('--run-mfa-only', action='store_true',
                        help='Only run MFA (corpus already prepared)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    num_workers = args.num_workers or os.cpu_count() or 4
    logger.info(f"Using {num_workers} parallel workers")

    input_subdir = args.output_dir / 'input'
    output_subdir = args.output_dir / 'output'

    if not args.run_mfa_only:
        logger.info("Preparing segmented corpus...")
        num_segments = prepare_mfa_corpus(
            args.preseea_dir,
            input_subdir,
            max_files=args.max_files,
            segment_duration_ms=args.segment_duration * 1000,
            num_workers=num_workers
        )
        logger.info(f"Created {num_segments} segments total")

    if not args.prepare_only:
        logger.info("Running MFA alignment...")
        success = run_mfa_alignment(input_subdir, output_subdir, num_jobs=num_workers)
        if success:
            logger.info("MFA alignment complete!")
        else:
            logger.error("MFA alignment failed")
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
