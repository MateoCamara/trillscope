"""Pronunciation dictionary parser for DIMEx100 corpus."""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class DictionaryEntry:
    """Single dictionary entry."""
    word: str
    pronunciation: List[str]  # List of phonemes
    variant_num: int = 1      # For entries like WORD(2)
    has_trill: bool = False


class DIMExDicParser:
    """
    Parse DIMEx100 pronunciation dictionaries.

    Format (from T22.full.dic):
    A	a
    ABANDONAR	a b a n d o n a r(
    ABIERTA	a b i e r( t a

    Key phoneme symbols:
    - r( = trill /r/
    - r = tap /r/ (typically in clusters like 'br', 'tr')
    - tS = affricate
    - x = velar fricative (j sound)
    - .sil = silence

    Word variants are marked with (N) suffix, e.g., A(2), ABSOLUTA(2)
    """

    TRILL_SYMBOL = 'r('
    VARIANT_PATTERN = re.compile(r'^(.+)\((\d+)\)$')

    def parse(self, dic_path: Path) -> Dict[str, List[DictionaryEntry]]:
        """
        Parse dictionary file into word -> entries mapping.

        Args:
            dic_path: Path to .dic file

        Returns:
            Dict mapping word to list of DictionaryEntry objects
        """
        entries: Dict[str, List[DictionaryEntry]] = {}

        try:
            content = self._read_file(dic_path)

            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Split on tab
                parts = line.split('\t')
                if len(parts) < 2:
                    continue

                word_raw = parts[0].strip()
                phonemes_str = parts[1].strip()

                # Parse phonemes
                phonemes = phonemes_str.split()

                # Handle variants like WORD(2)
                word, variant = self._parse_variant(word_raw)

                entry = DictionaryEntry(
                    word=word,
                    pronunciation=phonemes,
                    variant_num=variant,
                    has_trill=self.TRILL_SYMBOL in phonemes
                )

                if word not in entries:
                    entries[word] = []
                entries[word].append(entry)

        except Exception as e:
            logger.error(f"Failed to parse dictionary {dic_path}: {e}")

        return entries

    def _read_file(self, path: Path) -> str:
        """Read file with encoding fallback."""
        for encoding in ['latin-1', 'utf-8', 'cp1252']:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    def _parse_variant(self, word_raw: str) -> tuple:
        """
        Parse word with possible variant suffix.

        Args:
            word_raw: Word string like "WORD" or "WORD(2)"

        Returns:
            Tuple of (word, variant_num)
        """
        match = self.VARIANT_PATTERN.match(word_raw)
        if match:
            return match.group(1), int(match.group(2))
        return word_raw, 1

    def find_trill_words(self, dic_path: Path) -> Set[str]:
        """
        Find all words containing trill phoneme.

        Args:
            dic_path: Path to .dic file

        Returns:
            Set of words that have at least one pronunciation with trill
        """
        entries = self.parse(dic_path)
        trill_words = set()

        for word, entry_list in entries.items():
            if any(e.has_trill for e in entry_list):
                trill_words.add(word)

        return trill_words

    def get_trill_context_patterns(self, dic_path: Path) -> Dict[str, int]:
        """
        Analyze phonetic contexts where trill appears.

        Returns:
            Dict mapping context pattern to count
        """
        entries = self.parse(dic_path)
        patterns: Dict[str, int] = {}

        for word, entry_list in entries.items():
            for entry in entry_list:
                if not entry.has_trill:
                    continue

                phonemes = entry.pronunciation
                for i, p in enumerate(phonemes):
                    if p == self.TRILL_SYMBOL:
                        # Get context (prev_phoneme_trill_next_phoneme)
                        prev = phonemes[i-1] if i > 0 else '_'
                        next_p = phonemes[i+1] if i < len(phonemes) - 1 else '_'
                        pattern = f"{prev}_{self.TRILL_SYMBOL}_{next_p}"

                        patterns[pattern] = patterns.get(pattern, 0) + 1

        return patterns

    def get_word_pronunciation(self, entries: Dict[str, List[DictionaryEntry]],
                               word: str) -> Optional[List[str]]:
        """
        Get primary pronunciation for a word.

        Args:
            entries: Parsed dictionary entries
            word: Word to look up (case-insensitive)

        Returns:
            List of phonemes or None if not found
        """
        word_upper = word.upper()
        if word_upper in entries:
            # Return first variant
            return entries[word_upper][0].pronunciation
        return None

    def count_trill_occurrences(self, dic_path: Path) -> dict:
        """
        Count trill occurrences by word position.

        Returns:
            Dict with counts: {'word_initial': N, 'intervocalic': N, 'other': N}
        """
        entries = self.parse(dic_path)
        counts = {'word_initial': 0, 'intervocalic': 0, 'cluster': 0, 'other': 0}

        vowels = {'a', 'e', 'i', 'o', 'u'}

        for word, entry_list in entries.items():
            for entry in entry_list:
                if not entry.has_trill:
                    continue

                phonemes = entry.pronunciation
                for i, p in enumerate(phonemes):
                    if p == self.TRILL_SYMBOL:
                        if i == 0:
                            counts['word_initial'] += 1
                        elif i > 0:
                            prev = phonemes[i-1]
                            if prev in vowels:
                                # Check if next is also vowel (intervocalic)
                                if i < len(phonemes) - 1 and phonemes[i+1] in vowels:
                                    counts['intervocalic'] += 1
                                else:
                                    counts['other'] += 1
                            elif prev in {'n', 'l', 's'}:
                                counts['other'] += 1  # After consonant
                            else:
                                counts['cluster'] += 1

        return counts
