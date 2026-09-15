"""SAM/SES binary audio parser for ALBAYZIN corpus."""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import struct


@dataclass
class SAMHeader:
    """Parsed SAM file header information."""
    sample_rate: int = 16000  # Fixed for ALBAYZIN
    bit_depth: int = 16
    channels: int = 1
    num_samples: int = 0
    duration_seconds: float = 0.0


class SAMParser:
    """
    Parse ALBAYZIN SAM format audio files (.SES).

    SAM format specification:
    - Raw 16-bit signed PCM
    - Little-endian byte order
    - Mono channel
    - 16000 Hz sample rate
    - No header (raw samples)

    File size calculation: file_bytes / 2 = num_samples
    Duration: num_samples / 16000 = seconds
    """

    DEFAULT_SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2

    def get_audio_metadata(self, ses_path: Path) -> SAMHeader:
        """
        Get metadata from .SES file (inferred from file size).

        Args:
            ses_path: Path to .SES audio file

        Returns:
            SAMHeader with computed values
        """
        file_size = ses_path.stat().st_size
        num_samples = file_size // self.BYTES_PER_SAMPLE
        duration = num_samples / self.DEFAULT_SAMPLE_RATE

        return SAMHeader(
            sample_rate=self.DEFAULT_SAMPLE_RATE,
            bit_depth=16,
            channels=1,
            num_samples=num_samples,
            duration_seconds=duration
        )

    def parse_header_from_seo(self, seo_path: Path) -> SAMHeader:
        """
        Extract audio parameters from corresponding .SEO file.

        SEO fields used:
        - SAM: sample rate (always 16000)
        - BEG: start sample
        - END: end sample
        - SNB: bytes per sample (2)
        - SSB: bits per sample (16)
        - NCH: number of channels (1)

        Args:
            seo_path: Path to .SEO label file

        Returns:
            SAMHeader with values from SEO file
        """
        values = {}

        with open(seo_path, 'r', encoding='latin-1', errors='replace') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, _, value = line.partition(':')
                    key = key.strip()
                    value = value.strip()

                    if key in ['SAM', 'BEG', 'END', 'SNB', 'SSB', 'NCH']:
                        try:
                            values[key] = int(value)
                        except ValueError:
                            pass

        sample_rate = values.get('SAM', self.DEFAULT_SAMPLE_RATE)
        begin_sample = values.get('BEG', 0)
        end_sample = values.get('END', 0)
        num_samples = end_sample - begin_sample if end_sample > begin_sample else 0
        duration = num_samples / sample_rate if sample_rate > 0 else 0.0

        return SAMHeader(
            sample_rate=sample_rate,
            bit_depth=values.get('SSB', 16),
            channels=values.get('NCH', 1),
            num_samples=num_samples,
            duration_seconds=duration
        )

    def validate_format(self, ses_path: Path, seo_path: Optional[Path] = None) -> list:
        """
        Validate SAM audio file format.

        Args:
            ses_path: Path to .SES audio file
            seo_path: Optional path to corresponding .SEO file

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        if not ses_path.exists():
            issues.append(f"SES file not found: {ses_path}")
            return issues

        file_size = ses_path.stat().st_size

        # Check file size is even (16-bit samples)
        if file_size % 2 != 0:
            issues.append(f"SES file size is odd ({file_size} bytes), expected even for 16-bit PCM")

        # Check minimum size (at least 1 second of audio)
        min_size = self.DEFAULT_SAMPLE_RATE * self.BYTES_PER_SAMPLE
        if file_size < min_size:
            issues.append(f"SES file too small ({file_size} bytes), less than 1 second of audio")

        # Cross-validate with SEO if available
        if seo_path and seo_path.exists():
            seo_header = self.parse_header_from_seo(seo_path)
            expected_samples = seo_header.num_samples
            actual_samples = file_size // self.BYTES_PER_SAMPLE

            if expected_samples > 0 and actual_samples != expected_samples:
                issues.append(
                    f"Sample count mismatch: SEO says {expected_samples}, "
                    f"SES file has {actual_samples}"
                )

        return issues

    def read_samples(self, ses_path: Path, start_sample: int = 0,
                    num_samples: Optional[int] = None) -> bytes:
        """
        Read raw PCM samples from SES file.

        Args:
            ses_path: Path to .SES audio file
            start_sample: Starting sample index
            num_samples: Number of samples to read (None = all)

        Returns:
            Raw bytes of PCM audio data
        """
        with open(ses_path, 'rb') as f:
            if start_sample > 0:
                f.seek(start_sample * self.BYTES_PER_SAMPLE)

            if num_samples is not None:
                return f.read(num_samples * self.BYTES_PER_SAMPLE)
            else:
                return f.read()
