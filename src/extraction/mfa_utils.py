"""Montreal Forced Aligner utilities for PRESEEA corpus."""

import subprocess
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MFAPhoneme:
    """Phoneme interval from MFA TextGrid output."""
    start_sec: float
    end_sec: float
    label: str

    @property
    def start_ms(self) -> float:
        return self.start_sec * 1000

    @property
    def end_ms(self) -> float:
        return self.end_sec * 1000

    @property
    def duration_ms(self) -> float:
        return (self.end_sec - self.start_sec) * 1000


@dataclass
class MFAWord:
    """Word interval from MFA TextGrid output."""
    start_sec: float
    end_sec: float
    word: str


class MFARunner:
    """
    Run Montreal Forced Aligner and parse outputs.

    MFA requires:
    - Audio files in WAV format (16kHz recommended)
    - Text files with word-level transcripts
    - Acoustic model and pronunciation dictionary for Spanish
    """

    # Spanish trill phonemes in MFA's Spanish model
    TRILL_PHONEMES = {'r', 'rr', 'ɾ'}  # MFA may use different symbols

    def __init__(self, work_dir: Optional[Path] = None):
        """
        Initialize MFA runner.

        Args:
            work_dir: Working directory for MFA files (default: temp dir)
        """
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix='mfa_'))
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.input_dir = self.work_dir / 'input'
        self.output_dir = self.work_dir / 'output'
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    def check_mfa_installed(self) -> bool:
        """Check if MFA is installed and accessible."""
        try:
            result = subprocess.run(
                ['mfa', 'version'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def download_spanish_model(self) -> bool:
        """Download Spanish acoustic model and dictionary."""
        try:
            # Download acoustic model
            subprocess.run(
                ['mfa', 'model', 'download', 'acoustic', 'spanish_mfa'],
                check=True,
                capture_output=True
            )

            # Download dictionary
            subprocess.run(
                ['mfa', 'model', 'download', 'dictionary', 'spanish_mfa'],
                check=True,
                capture_output=True
            )

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download MFA Spanish models: {e}")
            return False
        except FileNotFoundError:
            logger.error("MFA not installed")
            return False

    def prepare_file(self, audio_path: Path, transcript: str,
                     file_id: str) -> Tuple[Path, Path]:
        """
        Prepare audio and transcript files for MFA.

        Args:
            audio_path: Path to source audio file
            transcript: Plain text transcript
            file_id: Unique identifier for the file pair

        Returns:
            Tuple of (wav_path, txt_path) in input directory
        """
        # Convert audio to 16kHz WAV if needed
        wav_path = self.input_dir / f"{file_id}.wav"
        self._convert_audio(audio_path, wav_path)

        # Write transcript
        txt_path = self.input_dir / f"{file_id}.txt"
        # Clean transcript for MFA
        clean_text = self._clean_transcript(transcript)
        txt_path.write_text(clean_text, encoding='utf-8')

        return wav_path, txt_path

    def _convert_audio(self, src_path: Path, dst_path: Path) -> None:
        """Convert audio to 16kHz mono WAV."""
        try:
            # Try using ffmpeg
            subprocess.run([
                'ffmpeg', '-y', '-i', str(src_path),
                '-ar', '16000', '-ac', '1',
                str(dst_path)
            ], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: try pydub
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(str(src_path))
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(str(dst_path), format='wav')
            except ImportError:
                logger.warning(f"Could not convert {src_path}, copying as-is")
                shutil.copy(src_path, dst_path)

    def _clean_transcript(self, text: str) -> str:
        """Clean transcript for MFA processing."""
        import re

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

    def run_alignment(self, num_jobs: int = 4) -> bool:
        """
        Run MFA alignment on prepared files.

        Args:
            num_jobs: Number of parallel jobs

        Returns:
            True if alignment succeeded
        """
        try:
            cmd = [
                'mfa', 'align',
                str(self.input_dir),
                'spanish_mfa',  # Dictionary
                'spanish_mfa',  # Acoustic model
                str(self.output_dir),
                '-j', str(num_jobs),
                '--clean',
            ]

            logger.info(f"Running MFA: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"MFA failed: {result.stderr}")
                return False

            return True

        except FileNotFoundError:
            logger.error("MFA not installed")
            return False
        except Exception as e:
            logger.error(f"MFA alignment failed: {e}")
            return False

    def parse_textgrid(self, textgrid_path: Path) -> Tuple[List[MFAWord], List[MFAPhoneme]]:
        """
        Parse MFA output TextGrid file.

        Args:
            textgrid_path: Path to .TextGrid file

        Returns:
            Tuple of (words, phonemes) lists
        """
        words = []
        phonemes = []

        try:
            from praatio import textgrid

            tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)

            # Parse word tier
            for tier_name in tg.tierNames:
                tier = tg.getTier(tier_name)
                tier_lower = tier_name.lower()

                if 'word' in tier_lower:
                    for entry in tier.entries:
                        if entry.label.strip():
                            words.append(MFAWord(
                                start_sec=entry.start,
                                end_sec=entry.end,
                                word=entry.label.strip()
                            ))

                elif 'phone' in tier_lower:
                    for entry in tier.entries:
                        if entry.label.strip():
                            phonemes.append(MFAPhoneme(
                                start_sec=entry.start,
                                end_sec=entry.end,
                                label=entry.label.strip()
                            ))

        except ImportError:
            logger.error("praatio not installed, cannot parse TextGrid")
        except Exception as e:
            logger.error(f"Failed to parse TextGrid {textgrid_path}: {e}")

        return words, phonemes

    def find_trills(self, phonemes: List[MFAPhoneme]) -> List[Tuple[int, MFAPhoneme]]:
        """
        Find trill phonemes in MFA output.

        Args:
            phonemes: List of phonemes from TextGrid

        Returns:
            List of (index, phoneme) tuples for trills
        """
        trills = []

        # Spanish MFA model uses these symbols for trill
        trill_labels = {'r', 'rr', 'ɾ', 'ʀ'}

        for i, phoneme in enumerate(phonemes):
            # Check if this is a trill
            # Note: MFA Spanish model might use 'r' for both tap and trill
            # We'll flag all 'r' sounds and classify later
            if phoneme.label.lower() in trill_labels:
                trills.append((i, phoneme))

        return trills

    def find_word_for_phoneme(self, words: List[MFAWord],
                              phoneme: MFAPhoneme) -> Optional[str]:
        """Find the word containing a phoneme based on timing."""
        phoneme_mid = (phoneme.start_sec + phoneme.end_sec) / 2

        for word in words:
            if word.start_sec <= phoneme_mid <= word.end_sec:
                return word.word

        return None

    def get_output_textgrid(self, file_id: str) -> Optional[Path]:
        """Get path to MFA output TextGrid for a file."""
        textgrid_path = self.output_dir / f"{file_id}.TextGrid"
        if textgrid_path.exists():
            return textgrid_path
        return None

    def cleanup(self) -> None:
        """Remove temporary working directory."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
