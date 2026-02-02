"""Parser for Praat TextGrid files used in Glissando corpus."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class Interval:
    """A single interval in a TextGrid tier."""
    xmin: float  # Start time in seconds
    xmax: float  # End time in seconds
    text: str    # Label/content


@dataclass
class IntervalTier:
    """An interval tier from a TextGrid file."""
    name: str
    xmin: float
    xmax: float
    intervals: List[Interval] = field(default_factory=list)


@dataclass
class TextGridData:
    """Parsed TextGrid data."""
    xmin: float
    xmax: float
    tiers: Dict[str, IntervalTier] = field(default_factory=dict)

    def get_tier(self, name: str) -> Optional[IntervalTier]:
        """Get a tier by name."""
        return self.tiers.get(name)

    def get_phonemes(self) -> Optional[IntervalTier]:
        """Get the phonetic transcription tier."""
        return self.tiers.get('PhoneticTranscription')

    def get_words(self) -> Optional[IntervalTier]:
        """Get the orthographic transcription tier."""
        return self.tiers.get('OrthographicTranscription')


class TextGridParser:
    """
    Parser for Praat TextGrid files.

    Supports the standard Praat TextGrid format used by Glissando corpus.
    TextGrid files contain time-aligned tiers with phonetic/orthographic annotations.
    """

    # Tier names in Glissando TextGrids
    TIER_ORTHOGRAPHIC = 'OrthographicTranscription'
    TIER_PHONETIC = 'PhoneticTranscription'
    TIER_SYLLABLES = 'Syllables'
    TIER_INTONATION_PHRASE = 'IntonationalPhrase'
    TIER_INTONATION_GROUPS = 'IntonationGroups'

    def __init__(self):
        self._number_pattern = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

    def parse(self, filepath: Path) -> TextGridData:
        """
        Parse a TextGrid file.

        Args:
            filepath: Path to the TextGrid file

        Returns:
            TextGridData with parsed tiers
        """
        filepath = Path(filepath)

        # Try different encodings
        content = None
        for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
            try:
                content = filepath.read_text(encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            raise ValueError(f"Could not decode TextGrid file: {filepath}")

        return self._parse_content(content, filepath)

    def _parse_content(self, content: str, filepath: Path) -> TextGridData:
        """Parse TextGrid content string."""
        lines = content.split('\n')
        lines = [line.strip() for line in lines]

        # Parse header
        if not lines or 'TextGrid' not in lines[0] and 'TextGrid' not in lines[1]:
            raise ValueError(f"Not a valid TextGrid file: {filepath}")

        # Find global xmin/xmax
        xmin = 0.0
        xmax = 0.0

        for i, line in enumerate(lines):
            if line.startswith('xmin ='):
                xmin = self._extract_number(line)
            elif line.startswith('xmax =') and xmax == 0.0:
                xmax = self._extract_number(line)
                break

        # Parse tiers
        tiers = {}
        tier_start_indices = []

        for i, line in enumerate(lines):
            if 'item [' in line and ']:' in line:
                tier_start_indices.append(i)

        for idx, start_idx in enumerate(tier_start_indices):
            end_idx = tier_start_indices[idx + 1] if idx + 1 < len(tier_start_indices) else len(lines)
            tier_lines = lines[start_idx:end_idx]

            tier = self._parse_tier(tier_lines)
            if tier:
                tiers[tier.name] = tier

        return TextGridData(xmin=xmin, xmax=xmax, tiers=tiers)

    def _parse_tier(self, lines: List[str]) -> Optional[IntervalTier]:
        """Parse a single tier from lines."""
        tier_name = None
        tier_xmin = 0.0
        tier_xmax = 0.0
        intervals = []

        i = 0
        while i < len(lines):
            line = lines[i]

            if 'class = "IntervalTier"' in line:
                pass  # We only support IntervalTier
            elif line.startswith('name ='):
                tier_name = self._extract_string(line)
            elif line.startswith('xmin =') and tier_xmin == 0.0:
                tier_xmin = self._extract_number(line)
            elif line.startswith('xmax =') and tier_xmax == 0.0:
                tier_xmax = self._extract_number(line)
            elif 'intervals [' in line and ']:' in line:
                # Parse interval block
                interval_xmin = None
                interval_xmax = None
                interval_text = None

                j = i + 1
                while j < len(lines) and not ('intervals [' in lines[j] and ']:' in lines[j]):
                    interval_line = lines[j]
                    if interval_line.startswith('xmin ='):
                        interval_xmin = self._extract_number(interval_line)
                    elif interval_line.startswith('xmax ='):
                        interval_xmax = self._extract_number(interval_line)
                    elif interval_line.startswith('text ='):
                        interval_text = self._extract_string(interval_line)
                    j += 1

                if interval_xmin is not None and interval_xmax is not None:
                    intervals.append(Interval(
                        xmin=interval_xmin,
                        xmax=interval_xmax,
                        text=interval_text or ''
                    ))

                i = j - 1  # Will be incremented at end of loop

            i += 1

        if tier_name:
            return IntervalTier(
                name=tier_name,
                xmin=tier_xmin,
                xmax=tier_xmax,
                intervals=intervals
            )
        return None

    def _extract_number(self, line: str) -> float:
        """Extract a number from a line like 'xmin = 1.234'."""
        match = self._number_pattern.search(line.split('=', 1)[-1])
        if match:
            return float(match.group())
        return 0.0

    def _extract_string(self, line: str) -> str:
        """Extract a string from a line like 'name = "foo"'."""
        # Find content between quotes
        parts = line.split('=', 1)
        if len(parts) < 2:
            return ''

        value = parts[1].strip()
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        return value


def find_trill_phonemes(textgrid: TextGridData) -> List[Dict[str, Any]]:
    """
    Find trill /r/ phonemes in a TextGrid.

    In Spanish SAMPA used by Glissando:
    - 'rr' = trill /r/ (vibrante multiple)
    - 'r' = tap /ɾ/ (vibrante simple)

    For trill /r/ analysis, we include:
    1. 'rr' - always a trill (intervocalic rr, word-initial r)
    2. Word-initial 'r' in certain contexts

    Returns:
        List of dicts with phoneme info including timing
    """
    phoneme_tier = textgrid.get_phonemes()
    word_tier = textgrid.get_words()

    if not phoneme_tier:
        return []

    trills = []

    for i, interval in enumerate(phoneme_tier.intervals):
        phoneme = interval.text.strip().lower()

        # Skip empty or silence
        if not phoneme or phoneme in ('...', 'sil', 'sp', ''):
            continue

        # Check for trill 'rr'
        if phoneme == 'rr':
            # Get context phonemes
            prev_phoneme = phoneme_tier.intervals[i-1].text if i > 0 else ''
            next_phoneme = phoneme_tier.intervals[i+1].text if i < len(phoneme_tier.intervals)-1 else ''

            # Find associated word
            word = _find_word_at_time(word_tier, interval.xmin) if word_tier else None

            trills.append({
                'phoneme_label': 'rr',
                'start_sec': interval.xmin,
                'end_sec': interval.xmax,
                'duration_ms': (interval.xmax - interval.xmin) * 1000,
                'prev_phoneme': prev_phoneme.strip(),
                'next_phoneme': next_phoneme.strip(),
                'word': word,
                'is_trill': True,
            })

    return trills


def _find_word_at_time(word_tier: IntervalTier, time: float) -> Optional[str]:
    """Find the word that contains a given time point."""
    if not word_tier:
        return None

    for interval in word_tier.intervals:
        if interval.xmin <= time < interval.xmax:
            word = interval.text.strip()
            # Skip silence/pause markers
            if word and not word.startswith('CP'):
                return word
    return None
