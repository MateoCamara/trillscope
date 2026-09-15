"""Phoneme file parser for DIMEx100 corpus."""

from pathlib import Path
from dataclasses import dataclass
from typing import List
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhonemeSegment:
    """Single phoneme segment from .phn file."""
    start_ms: float
    end_ms: float
    phoneme: str
    is_trill: bool = False  # True if phoneme is 'r('
    is_silence: bool = False  # True if phoneme is '.sil'

    @property
    def duration_ms(self) -> float:
        """Duration of segment in milliseconds."""
        return self.end_ms - self.start_ms


class DIMExPHNParser:
    """
    Parse DIMEx100 .phn phoneme/word alignment files.

    Format (from s00101.phn):
    MillisecondsPerFrame: 1.0
    END OF HEADER
    0.000000 56.149734 .sil
    56.149734 117.647057 k
    ...
    662.164063 761.060730 r(    <- trill marker!

    The 'r(' notation indicates the trill /r/ phoneme.
    Regular 'r' indicates the tap /r/.
    '.sil' indicates silence.
    """

    # Phoneme markers
    TRILL_MARKER = 'r('
    SILENCE_MARKER = '.sil'

    def parse_phoneme_file(self, phn_path: Path) -> List[PhonemeSegment]:
        """
        Parse T22/T44/T54 style phoneme files.

        Args:
            phn_path: Path to .phn file

        Returns:
            List of PhonemeSegment objects
        """
        segments = []

        try:
            # Try UTF-8 first, then Latin-1
            content = self._read_file(phn_path)

            in_header = True
            for line in content.split('\n'):
                line = line.strip()

                if not line:
                    continue

                if 'END OF HEADER' in line:
                    in_header = False
                    continue

                if in_header:
                    continue

                # Parse phoneme line: start_ms end_ms phoneme
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                        phoneme = parts[2]

                        segment = PhonemeSegment(
                            start_ms=start,
                            end_ms=end,
                            phoneme=phoneme,
                            is_trill=(phoneme == self.TRILL_MARKER),
                            is_silence=(phoneme == self.SILENCE_MARKER)
                        )
                        segments.append(segment)

                    except ValueError as e:
                        logger.warning(f"Failed to parse line in {phn_path}: {line} - {e}")

        except Exception as e:
            logger.error(f"Failed to parse {phn_path}: {e}")

        return segments

    def _read_file(self, path: Path) -> str:
        """Read file with encoding fallback."""
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # Last resort
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    def find_trill_segments(self, phn_path: Path) -> List[PhonemeSegment]:
        """
        Extract only trill phoneme segments.

        Args:
            phn_path: Path to .phn file

        Returns:
            List of PhonemeSegment objects where is_trill=True
        """
        all_segments = self.parse_phoneme_file(phn_path)
        return [s for s in all_segments if s.is_trill]

    def get_total_duration(self, phn_path: Path) -> float:
        """
        Get total duration of the phoneme file in milliseconds.

        Args:
            phn_path: Path to .phn file

        Returns:
            Duration in milliseconds
        """
        segments = self.parse_phoneme_file(phn_path)
        if segments:
            return segments[-1].end_ms
        return 0.0

    def get_phoneme_context(self, segments: List[PhonemeSegment],
                           target_idx: int, window: int = 2) -> List[str]:
        """
        Get phoneme context around a target segment.

        Args:
            segments: List of all segments
            target_idx: Index of target phoneme
            window: Number of phonemes on each side

        Returns:
            List of phonemes in context (2*window + 1)
        """
        start = max(0, target_idx - window)
        end = min(len(segments), target_idx + window + 1)

        return [s.phoneme for s in segments[start:end]]

    def count_phonemes(self, phn_path: Path) -> dict:
        """
        Count occurrences of each phoneme in file.

        Args:
            phn_path: Path to .phn file

        Returns:
            Dict mapping phoneme to count
        """
        segments = self.parse_phoneme_file(phn_path)
        counts = {}
        for s in segments:
            counts[s.phoneme] = counts.get(s.phoneme, 0) + 1
        return counts
