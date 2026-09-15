"""Glissando-sp trill /r/ extractor using existing TextGrid alignment files."""

from pathlib import Path
from typing import List, Optional
import logging

from .base import TrillCandidate, ExtractionResult
from ..ingestion.glissando.textgrid_parser import (
    TextGridParser, Interval
)

logger = logging.getLogger(__name__)


class GlissandoExtractor:
    """
    Extract trill /r/ candidates from Glissando-sp corpus.

    Uses existing TextGrid phoneme alignment files where 'rr' marks trill
    in SAMPA notation.
    """

    TRILL_MARKER = 'rr'
    DATASET = 'glissando'

    # SAMPA vowel phonemes
    VOWEL_PHONEMES = {
        'a', 'e', 'i', 'o', 'u',
        'a_&quot', 'e_&quot', 'i_&quot', 'o_&quot', 'u_&quot',  # Escaped quotes in TextGrid
    }

    # Trill-triggering consonants in SAMPA
    TRILL_TRIGGERS = {'n', 'l', 's'}

    def __init__(self, corpus_root: Path):
        """
        Initialize extractor.

        Args:
            corpus_root: Path to Glissando-sp directory (03 glissando-sp/)
        """
        self.corpus_root = corpus_root
        self.textgrid_parser = TextGridParser()

    def extract_all(self) -> ExtractionResult:
        """
        Extract all trill candidates from corpus.

        Returns:
            ExtractionResult with all candidates
        """
        result = ExtractionResult(dataset=self.DATASET)

        # Process News subcorpus
        news_path = self.corpus_root / 'News'
        if news_path.exists():
            self._extract_from_news(news_path, result)

        # Process Task_dialogues
        task_path = self.corpus_root / 'Task_dialogues'
        if task_path.exists():
            self._extract_from_dialogues(task_path, 'Task_dialogues', result)

        # Process Free_dialogues
        free_path = self.corpus_root / 'Free_dialogues'
        if free_path.exists():
            self._extract_from_dialogues(free_path, 'Free_dialogues', result)

        result.compute_statistics()
        logger.info(f"Extracted {len(result.candidates)} trill candidates")

        return result

    def _extract_from_news(self, news_path: Path, result: ExtractionResult) -> None:
        """Extract from News subcorpus (read speech)."""
        logger.info("Processing News subcorpus...")

        for speaker_dir in news_path.iterdir():
            if not speaker_dir.is_dir() or not speaker_dir.name.startswith('sp_'):
                continue

            speaker_id = speaker_dir.name[3:]  # Remove 'sp_' prefix

            # Process Prosodic and Phonetic subdirectories
            for subcorpus_type in ['Prosodic', 'Phonetic']:
                subcorpus_dir = speaker_dir / subcorpus_type
                if not subcorpus_dir.exists():
                    continue

                for tg_file in subcorpus_dir.glob('*.TextGrid'):
                    self._extract_from_textgrid(
                        tg_file, speaker_id, 'News', 'read', result
                    )

    def _extract_from_dialogues(self, dialogue_path: Path, subcorpus: str,
                                result: ExtractionResult) -> None:
        """Extract from dialogue subcorpora."""
        logger.info(f"Processing {subcorpus}...")

        speech_style = 'spontaneous' if subcorpus == 'Free_dialogues' else 'task_oriented'

        for pair_dir in dialogue_path.iterdir():
            if not pair_dir.is_dir() or not pair_dir.name.startswith('sp_'):
                continue

            # For Task_dialogues, iterate over task types
            if subcorpus == 'Task_dialogues':
                for task_type in ['Transport', 'University', 'Tourist']:
                    task_dir = pair_dir / task_type
                    if task_dir.exists():
                        # Process Turns directory
                        turns_dir = task_dir / 'Turns'
                        if turns_dir.exists():
                            self._extract_from_turns(
                                turns_dir, subcorpus, speech_style, result
                            )
            else:
                # Free_dialogues - process Turns directly
                turns_dir = pair_dir / 'Turns'
                if turns_dir.exists():
                    self._extract_from_turns(
                        turns_dir, subcorpus, speech_style, result
                    )

    def _extract_from_turns(self, turns_dir: Path, subcorpus: str,
                            speech_style: str, result: ExtractionResult) -> None:
        """Extract from a Turns directory."""
        for tg_file in turns_dir.glob('*.TextGrid'):
            # Extract speaker ID from filename
            speaker_id = self._extract_speaker_from_filename(tg_file.stem)
            if not speaker_id:
                result.add_issue(f"Could not extract speaker from {tg_file}")
                continue

            self._extract_from_textgrid(
                tg_file, speaker_id, subcorpus, speech_style, result
            )

    def _extract_speaker_from_filename(self, filename: str) -> Optional[str]:
        """Extract speaker ID from filename like sp_f11r_fcd001."""
        import re
        pattern = r'(?:^|_)((?:f|m)\d{2}[ras])(?:_|$)'
        match = re.search(pattern, filename.lower())
        if match:
            return match.group(1)
        return None

    def _extract_from_textgrid(self, tg_path: Path, speaker_id: str,
                               subcorpus: str, speech_style: str,
                               result: ExtractionResult) -> None:
        """Extract trill candidates from a single TextGrid file."""
        try:
            tg = self.textgrid_parser.parse(tg_path)

            phoneme_tier = tg.get_phonemes()
            word_tier = tg.get_words()

            if not phoneme_tier:
                result.add_issue(f"No phoneme tier in {tg_path}")
                return

            # Generate utterance ID
            utt_id = f"GLI_{speaker_id}_{tg_path.stem}"

            # Find corresponding audio file
            audio_path = self._find_audio_file(tg_path)

            # Track word index
            word_idx = 0

            for i, interval in enumerate(phoneme_tier.intervals):
                phoneme = interval.text.strip().lower()

                # Check for trill 'rr'
                if phoneme == self.TRILL_MARKER:
                    # Get context phonemes
                    prev_phoneme = self._get_phoneme(phoneme_tier.intervals, i - 1)
                    next_phoneme = self._get_phoneme(phoneme_tier.intervals, i + 1)

                    # Classify context
                    context_label = self._classify_context(prev_phoneme, next_phoneme)

                    # Find associated word
                    word = self._find_word_at_time(word_tier, interval.xmin)

                    # Convert to milliseconds
                    start_ms = interval.xmin * 1000
                    end_ms = interval.xmax * 1000

                    candidate = TrillCandidate(
                        utt_id=utt_id,
                        speaker_id=speaker_id,
                        dataset=self.DATASET,
                        word=word or '[unknown]',
                        word_idx=word_idx,
                        r_idx_in_word=0,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        phoneme_label='rr',
                        prev_phoneme=prev_phoneme,
                        next_phoneme=next_phoneme,
                        context_label=context_label,
                        alignment_source='textgrid',
                        audio_path=str(audio_path) if audio_path else None,
                    )

                    result.add_candidate(candidate)
                    word_idx += 1

        except Exception as e:
            result.add_issue(f"Failed to extract from {tg_path}: {e}")
            logger.warning(f"Failed to extract from {tg_path}: {e}")

    def _get_phoneme(self, intervals: List[Interval], idx: int) -> Optional[str]:
        """Get phoneme at index, handling bounds."""
        if idx < 0 or idx >= len(intervals):
            return None
        phoneme = intervals[idx].text.strip()
        # Skip silence markers
        if phoneme in ('...', 'sil', 'sp', '', '#'):
            return None
        return phoneme

    def _classify_context(self, prev_phoneme: Optional[str],
                          next_phoneme: Optional[str]) -> str:
        """Classify trill context based on surrounding phonemes."""
        # Clean phonemes (remove SAMPA escape sequences)
        prev_clean = prev_phoneme.replace('_&quot', '') if prev_phoneme else None
        next_clean = next_phoneme.replace('_&quot', '') if next_phoneme else None

        # Word initial (no previous phoneme or silence)
        if not prev_clean:
            return 'word_initial'

        # Check for vowel context (intervocalic)
        prev_is_vowel = prev_clean in self.VOWEL_PHONEMES or prev_clean in 'aeiou'
        next_is_vowel = next_clean in self.VOWEL_PHONEMES if next_clean else False
        if not next_is_vowel and next_clean:
            next_is_vowel = next_clean in 'aeiou'

        if prev_is_vowel and next_is_vowel:
            return 'intervocalic_rr'

        # After n, l, s
        if prev_clean in self.TRILL_TRIGGERS:
            return 'after_nls'

        # Post-vocalic (after vowel but not intervocalic)
        if prev_is_vowel:
            return 'post_vocalic'

        return 'unknown'

    def _find_audio_file(self, tg_path: Path) -> Optional[Path]:
        """Find audio file corresponding to TextGrid."""
        stem = tg_path.stem
        parent = tg_path.parent

        # Try different audio file patterns
        patterns = [
            f"{stem}.wav",
            f"{stem}.fix.wav",
            f"{stem}.wir.wav",
        ]

        for pattern in patterns:
            audio_file = parent / pattern
            if audio_file.exists():
                return audio_file

        return None

    def _find_word_at_time(self, word_tier, time: float) -> Optional[str]:
        """Find word containing a given time point."""
        if not word_tier:
            return None

        for interval in word_tier.intervals:
            if interval.xmin <= time < interval.xmax:
                word = interval.text.strip()
                # Skip silence/pause markers
                if word and not word.startswith('CP'):
                    return word
        return None

    def extract_utterance(self, tg_path: Path, speaker_id: str) -> List[TrillCandidate]:
        """
        Extract candidates from a single utterance.

        Args:
            tg_path: Path to TextGrid file
            speaker_id: Speaker ID

        Returns:
            List of TrillCandidate objects
        """
        result = ExtractionResult(dataset=self.DATASET)
        self._extract_from_textgrid(tg_path, speaker_id, 'unknown', 'unknown', result)
        return result.candidates
