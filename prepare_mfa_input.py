"""Prepare datasets for MFA alignment."""

import shutil
import subprocess
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_dataset(dataset: str, metadata_path: Path, output_dir: Path,
                    max_files: int = None, convert_audio: bool = True):
    """
    Prepare a dataset for MFA alignment.

    Args:
        dataset: Dataset name (mailabs, tedx, heroico, commonvoice)
        metadata_path: Path to metadata parquet
        output_dir: MFA input directory
        max_files: Maximum files to process (for testing)
        convert_audio: Whether to convert audio to 16kHz
    """
    logger.info(f"Preparing {dataset} for MFA alignment...")

    df = pd.read_parquet(metadata_path)
    logger.info(f"Loaded {len(df)} records")

    if max_files:
        df = df.head(max_files)
        logger.info(f"Limited to {max_files} files for testing")

    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Preparing {dataset}"):
        try:
            audio_path = Path(row['audio_path'])
            if not audio_path.exists():
                error_count += 1
                continue

            # Use utterance ID as filename
            utt_id = row['utt_id']

            # Output paths
            wav_out = output_dir / f"{utt_id}.wav"
            txt_out = output_dir / f"{utt_id}.txt"

            # Skip if already processed
            if wav_out.exists() and txt_out.exists():
                success_count += 1
                continue

            # Get transcript
            transcript = row.get('transcript_text', '')
            if not transcript or pd.isna(transcript):
                error_count += 1
                continue

            # Clean transcript for MFA
            import re
            transcript = re.sub(r'<[^>]+>', ' ', str(transcript))  # Remove XML tags
            transcript = re.sub(r'\[.*?\]', '', transcript)  # Remove brackets
            transcript = re.sub(r'\(.*?\)', '', transcript)  # Remove parentheses
            transcript = re.sub(r'[^\w\sáéíóúñüÁÉÍÓÚÑÜ.,;:!?¿¡-]', ' ', transcript)
            transcript = re.sub(r'\s+', ' ', transcript).strip()

            if not transcript:
                error_count += 1
                continue

            # Convert/copy audio
            if convert_audio:
                # Convert to 16kHz mono WAV using ffmpeg
                result = subprocess.run([
                    'ffmpeg', '-y', '-i', str(audio_path),
                    '-ar', '16000', '-ac', '1',
                    '-loglevel', 'error',
                    str(wav_out)
                ], capture_output=True)

                if result.returncode != 0:
                    error_count += 1
                    continue
            else:
                # Just copy if already 16kHz WAV
                shutil.copy(audio_path, wav_out)

            # Write transcript
            txt_out.write_text(transcript, encoding='utf-8')

            success_count += 1

        except Exception as e:
            logger.error(f"Error processing {row.get('utt_id', idx)}: {e}")
            error_count += 1

    logger.info(f"Prepared {success_count} files, {error_count} errors")
    return success_count, error_count


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare datasets for MFA')
    parser.add_argument('dataset', choices=['mailabs', 'tedx', 'heroico', 'commonvoice', 'all'])
    parser.add_argument('--max-files', type=int, help='Max files to process')
    parser.add_argument('--no-convert', action='store_true', help='Skip audio conversion')
    args = parser.parse_args()

    datasets = ['mailabs', 'tedx', 'heroico', 'commonvoice'] if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        metadata_path = Path(f'metadata/{ds}_raw.parquet')
        output_dir = Path(f'mfa_work/{ds}/input')

        if metadata_path.exists():
            # TEDx is already 16kHz
            convert = not args.no_convert and ds != 'tedx'
            prepare_dataset(ds, metadata_path, output_dir, args.max_files, convert)
        else:
            logger.warning(f"Metadata not found: {metadata_path}")
