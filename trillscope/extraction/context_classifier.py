"""Context classifier for trill /r/ occurrences."""

import re
from typing import Optional, Tuple


# Spanish vowels (including accented)
VOWELS = set('aeiouáéíóúAEIOUÁÉÍÓÚ')

# Consonants after which /r/ is typically trill at syllable onset
TRILL_TRIGGERING_CONSONANTS = {'n', 'l', 's', 'N', 'L', 'S'}

# Clusters where /r/ is typically tap (not trill)
TAP_CLUSTERS = {'br', 'cr', 'dr', 'fr', 'gr', 'kr', 'pr', 'tr',
                'Br', 'Cr', 'Dr', 'Fr', 'Gr', 'Kr', 'Pr', 'Tr',
                'BR', 'CR', 'DR', 'FR', 'GR', 'KR', 'PR', 'TR'}


def classify_r_context(word: str, r_position: int) -> str:
    """
    Classify /r/ context based on orthographic rules.

    Args:
        word: The word containing /r/
        r_position: Character position of 'r' in word (0-indexed)

    Returns:
        Context label: 'intervocalic_rr', 'word_initial', 'after_nls',
                       'tap_cluster', 'coda', 'unknown'
    """
    word_lower = word.lower()
    n = len(word_lower)

    if r_position < 0 or r_position >= n:
        return 'unknown'

    # Check for 'rr' (intervocalic double-r)
    if r_position > 0 and r_position < n - 1:
        # Check if this is part of 'rr'
        if (word_lower[r_position] == 'r' and
            r_position + 1 < n and word_lower[r_position + 1] == 'r'):
            # First r of 'rr' - check if intervocalic
            prev_char = word_lower[r_position - 1] if r_position > 0 else ''
            next_char = word_lower[r_position + 2] if r_position + 2 < n else ''
            if prev_char in VOWELS and next_char in VOWELS:
                return 'intervocalic_rr'
            return 'intervocalic_rr'  # Still 'rr' even if not perfectly intervocalic

        if (word_lower[r_position] == 'r' and
            r_position > 0 and word_lower[r_position - 1] == 'r'):
            # Second r of 'rr' - same classification
            return 'intervocalic_rr'

    # Word-initial 'r' (always trill)
    if r_position == 0:
        return 'word_initial'

    # 'r' after n, l, s (at syllable boundary - typically trill)
    prev_char = word_lower[r_position - 1]
    if prev_char in TRILL_TRIGGERING_CONSONANTS:
        return 'after_nls'

    # Check for tap clusters (br, tr, dr, etc.)
    if r_position > 0:
        cluster = word_lower[r_position - 1:r_position + 1]
        if cluster in TAP_CLUSTERS:
            return 'tap_cluster'

    # Intervocalic single 'r' (tap, not trill)
    if r_position > 0 and r_position < n - 1:
        prev_char = word_lower[r_position - 1]
        next_char = word_lower[r_position + 1]
        if prev_char in VOWELS and next_char in VOWELS:
            return 'intervocalic_tap'

    # Final/coda 'r'
    if r_position == n - 1:
        return 'coda'

    # Before consonant (usually coda)
    if r_position < n - 1:
        next_char = word_lower[r_position + 1]
        if next_char not in VOWELS and next_char != 'r':
            return 'coda'

    return 'unknown'


def is_trill_context(context_label: str) -> bool:
    """
    Check if context typically produces trill /r/.

    Args:
        context_label: From classify_r_context()

    Returns:
        True if trill is expected, False otherwise
    """
    trill_contexts = {'intervocalic_rr', 'word_initial', 'after_nls'}
    return context_label in trill_contexts


def find_r_positions(word: str) -> list:
    """
    Find all positions of 'r' or 'R' in a word.

    Args:
        word: The word to search

    Returns:
        List of (position, is_double_r) tuples
    """
    positions = []
    word_lower = word.lower()
    i = 0
    while i < len(word_lower):
        if word_lower[i] == 'r':
            # Check if this is 'rr'
            if i + 1 < len(word_lower) and word_lower[i + 1] == 'r':
                positions.append((i, True))  # First r of 'rr'
                i += 2  # Skip both r's
            else:
                positions.append((i, False))
                i += 1
        else:
            i += 1
    return positions


def classify_word_r_occurrences(word: str) -> list:
    """
    Classify all /r/ occurrences in a word.

    Args:
        word: The word to analyze

    Returns:
        List of dicts with position, is_trill, context_label
    """
    results = []
    r_positions = find_r_positions(word)

    for idx, (pos, is_double) in enumerate(r_positions):
        context = classify_r_context(word, pos)
        results.append({
            'position': pos,
            'r_idx': idx,
            'is_double_r': is_double,
            'context_label': context,
            'is_trill': is_trill_context(context),
        })

    return results


def classify_phoneme_context(prev_phoneme: Optional[str],
                             next_phoneme: Optional[str],
                             phoneme_label: str) -> str:
    """
    Classify /r/ context based on phoneme-level alignment.

    Args:
        prev_phoneme: Previous phoneme (or None)
        next_phoneme: Next phoneme (or None)
        phoneme_label: The /r/ phoneme label (e.g., 'r(', 'rr', 'r')

    Returns:
        Context label
    """
    # Phoneme-level vowel markers (includes IPA vowels from MFA)
    vowel_phonemes = {
        'a', 'e', 'i', 'o', 'u',  # Basic vowels
        'A', 'E', 'I', 'O', 'U',  # Uppercase
        'ä', 'ë', 'ï', 'ö', 'ü',  # Umlauts
        'á', 'é', 'í', 'ó', 'ú',  # Accented
    }

    # Trill-triggering consonant phonemes (includes IPA variants)
    trill_triggers = {
        'n', 'l', 's',
        'N', 'L', 'S',
        'ɲ', 'ʎ',  # IPA palatals
        'n̪', 'ɲ̟',  # IPA dentals
    }

    # Silence/boundary markers
    silence_markers = {'.sil', '#', '', 'sil', 'sp', 'spn', None}

    # If phoneme is explicitly marked as trill (various notations)
    # Includes: r( (DIMEx100), rr (ALBAYZIN), r (MFA trill)
    if phoneme_label in ('r(', 'rr', 'R', 'r'):
        # Already known to be trill, classify by position
        if prev_phoneme is None or prev_phoneme in silence_markers:
            return 'word_initial'

        # Clean prev/next phonemes for comparison
        prev_clean = prev_phoneme.lower() if prev_phoneme else ''
        next_clean = next_phoneme.lower() if next_phoneme else ''

        # Check if previous phoneme is a vowel
        is_prev_vowel = prev_clean in vowel_phonemes or (
            len(prev_clean) == 1 and prev_clean in 'aeiouáéíóú'
        )

        # Check if next phoneme is a vowel
        is_next_vowel = next_clean in vowel_phonemes or (
            len(next_clean) == 1 and next_clean in 'aeiouáéíóú'
        )

        if is_prev_vowel:
            if is_next_vowel:
                return 'intervocalic_rr'
            return 'post_vocalic'

        if prev_phoneme in trill_triggers or prev_clean in {'n', 'l', 's'}:
            return 'after_nls'

        # Word-initial when preceded by silence or unknown
        return 'word_initial'

    return 'unknown'
