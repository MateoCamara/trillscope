"""ALBAYZIN trill /r/ extractor using SEO and TextGrid alignment files."""

from pathlib import Path
from typing import List, Optional, Tuple
import logging

from .base import TrillCandidate, ExtractionResult
from .context_classifier import classify_phoneme_context, is_trill_context
from ..ingestion.albayzin.seo_parser import SEOParser, SEORecord, PhonemeLabel

logger = logging.getLogger(__name__)


class AlbayzinExtractor:
    """
    Extract trill /r/ candidates from ALBAYZIN corpus.

    Uses SEO files with phoneme labels where 'rr' marks trill.
    Sample positions are converted to milliseconds.
    """

    TRILL_MARKER = 'rr'
    DATASET = 'albayzin'
    DEFAULT_SAMPLE_RATE = 16000

    def __init__(self, corpus_root: Path):
        """
        Initialize extractor.

        Args:
            corpus_root: Path to ALBAYZIN corpus directory
        """
        self.corpus_root = corpus_root
        self.seo_parser = SEOParser()

    def extract_all(self) -> ExtractionResult:
        """
        Extract all trill candidates from corpus.

        Returns:
            ExtractionResult with all candidates
        """
        result = ExtractionResult(dataset=self.DATASET)

        # Discover subcorpora
        subcorpora = ['Albayzin1', 'Albayzin2', 'Albayzin3', 'Albayzin4', 'Albayzin5']

        for subcorpus in subcorpora:
            subcorpus_path = self.corpus_root / subcorpus
            if subcorpus_path.exists():
                logger.info(f"Processing {subcorpus}...")
                self._extract_subcorpus(subcorpus_path, subcorpus, result)
            else:
                logger.warning(f"Subcorpus not found: {subcorpus_path}")

        result.compute_statistics()
        logger.info(f"Extracted {len(result.candidates)} trill candidates")

        return result

    def _extract_subcorpus(self, subcorpus_path: Path, subcorpus_name: str,
                           result: ExtractionResult) -> None:
        """Extract candidates from a single subcorpus."""

        # Find all SEO files
        seo_files = list(subcorpus_path.rglob('*.SEO')) + list(subcorpus_path.rglob('*.seo'))
        logger.info(f"Found {len(seo_files)} SEO files in {subcorpus_name}")

        for seo_path in seo_files:
            self._extract_from_seo(seo_path, subcorpus_name, result)

    def _extract_from_seo(self, seo_path: Path, subcorpus: str,
                          result: ExtractionResult) -> None:
        """Extract candidates from a single SEO file."""

        try:
            record = self.seo_parser.parse(seo_path)
            if not record.phoneme_labels:
                return

            # Find trill phonemes
            trills = self.seo_parser.find_trill_phonemes(record)
            if not trills:
                return

            # Generate IDs
            speaker_id = record.speaker_code or seo_path.stem[:2]
            utt_id = f"ALB_{speaker_id}_{seo_path.stem}"

            # Find audio file
            audio_path = self._find_audio_file(seo_path, record)

            # Sample rate for conversion
            sample_rate = record.sample_rate or self.DEFAULT_SAMPLE_RATE

            # Process each trill
            for i, trill_label in enumerate(trills):
                # Get timing
                start_ms, end_ms = self._get_phoneme_timing(
                    record, trill_label, sample_rate
                )

                # Get context phonemes
                prev_phoneme, next_phoneme = self._get_context_phonemes(
                    record.phoneme_labels, trill_label
                )

                # Try to find the word
                word = self._find_word_for_phoneme(
                    record.orthographic_text, trill_label, record
                )

                # Classify context based on ORTHOGRAPHIC rules (not just phoneme labels)
                # ALBAYZIN 'rr' phoneme label includes non-trill contexts like word-final
                context_label = 'unknown'
                if word:
                    # Find where 'r' appears in the word to classify orthographically
                    ortho_context = self._classify_orthographic_context(word)
                    if ortho_context:
                        context_label = ortho_context

                # Fallback to phoneme-based classification
                if context_label == 'unknown':
                    context_label = classify_phoneme_context(
                        prev_phoneme, next_phoneme, trill_label.phoneme
                    )

                # Skip non-trill contexts (coda, tap_cluster, post_vocalic without rr)
                if not is_trill_context(context_label):
                    continue

                candidate = TrillCandidate(
                    utt_id=utt_id,
                    speaker_id=speaker_id,
                    dataset=self.DATASET,
                    word=word or '[unknown]',
                    word_idx=i,
                    r_idx_in_word=0,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    phoneme_label=trill_label.phoneme,
                    prev_phoneme=prev_phoneme,
                    next_phoneme=next_phoneme,
                    context_label=context_label,
                    alignment_source='seo',
                    audio_path=str(audio_path) if audio_path else None,
                )

                result.add_candidate(candidate)

        except Exception as e:
            result.add_issue(f"Failed to extract from {seo_path}: {e}")
            logger.warning(f"Failed to extract from {seo_path}: {e}")

    def _get_phoneme_timing(self, record: SEORecord, label: PhonemeLabel,
                            sample_rate: int) -> Tuple[float, float]:
        """
        Calculate start and end time for a phoneme in milliseconds.

        Args:
            record: SEO record with all labels
            label: Target phoneme label
            sample_rate: Sample rate in Hz

        Returns:
            (start_ms, end_ms) tuple
        """
        labels = record.phoneme_labels
        start_sample = label.sample_position

        # Find index of this label
        label_idx = None
        for i, lbl in enumerate(labels):
            if lbl.sample_position == label.sample_position:
                label_idx = i
                break

        if label_idx is None:
            # Fallback: use label position as start, estimate end
            start_ms = start_sample / sample_rate * 1000
            end_ms = start_ms + 50  # Assume 50ms default duration
            return start_ms, end_ms

        # End is the start of next phoneme
        if label_idx + 1 < len(labels):
            end_sample = labels[label_idx + 1].sample_position
        else:
            # Last phoneme - use end of recording
            end_sample = record.end_sample

        start_ms = start_sample / sample_rate * 1000
        end_ms = end_sample / sample_rate * 1000

        return start_ms, end_ms

    def _get_context_phonemes(self, labels: List[PhonemeLabel],
                              target: PhonemeLabel) -> Tuple[Optional[str], Optional[str]]:
        """Get previous and next phonemes for context."""
        prev_phoneme = None
        next_phoneme = None

        for i, label in enumerate(labels):
            if label.sample_position == target.sample_position:
                if i > 0:
                    prev_phoneme = labels[i - 1].phoneme
                if i + 1 < len(labels):
                    next_phoneme = labels[i + 1].phoneme
                break

        return prev_phoneme, next_phoneme

    def _find_audio_file(self, seo_path: Path, record: SEORecord) -> Optional[Path]:
        """Find the audio file corresponding to an SEO file."""
        # Audio files can be .wav or .SES (SAM format)
        parent = seo_path.parent
        stem = seo_path.stem

        # Check for .wav first
        for ext in ['.wav', '.WAV', '.SES', '.ses']:
            audio_path = parent / f"{stem}{ext}"
            if audio_path.exists():
                return audio_path

        # Try using SRC field
        if record.source_file:
            src_path = parent / record.source_file
            if src_path.exists():
                return src_path

        return None

    def _classify_orthographic_context(self, word: str) -> Optional[str]:
        """
        Classify /r/ context based on orthographic rules.

        Returns the trill context if the word contains a trill /r/, otherwise None.
        """
        if not word:
            return None

        word_lower = word.lower()

        # Check for orthographic 'rr' (intervocalic double-r) - TRILL
        if 'rr' in word_lower:
            return 'intervocalic_rr'

        # Check for word-initial 'r' - TRILL
        if word_lower.startswith('r'):
            return 'word_initial'

        # Check for 'r' after n, l, s (at syllable boundary) - TRILL
        # Patterns like: enr, alr, isr (e.g., enredo, alrededor, Israel)
        import re
        if re.search(r'[nls]r', word_lower):
            return 'after_nls'

        # Other positions (coda, clusters) are NOT trill contexts
        return None

    def _find_word_for_phoneme(self, text: str, label: PhonemeLabel,
                               record: SEORecord) -> Optional[str]:
        """
        Find the word containing a phoneme.

        This is a heuristic - proper alignment would need word-level timing.
        """
        if not text:
            return None

        # Find words with 'r' or 'rr'
        import re
        words = re.findall(r'\b\w+\b', text, re.UNICODE)

        # Prioritize words with 'rr' (intervocalic trill)
        for word in words:
            if 'rr' in word.lower():
                return word

        # Then word-initial 'r'
        for word in words:
            if word.lower().startswith('r'):
                return word

        # Then 'r' after n,l,s
        for word in words:
            if re.search(r'[nls]r', word.lower()):
                return word

        # Fallback to any word with 'r'
        for word in words:
            if 'r' in word.lower():
                return word

        return None

    def extract_from_textgrid(self, textgrid_path: Path,
                              result: ExtractionResult) -> None:
        """
        Extract candidates from a TextGrid file.

        TextGrid provides more precise alignment than SEO in some cases.
        """
        try:
            # Use praatio for TextGrid parsing
            from praatio import textgrid

            tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)

            # Look for phone tier
            phone_tier = None
            for tier_name in tg.tierNames:
                if 'phone' in tier_name.lower() or 'phon' in tier_name.lower():
                    phone_tier = tg.getTier(tier_name)
                    break

            if not phone_tier:
                return

            # Generate IDs
            speaker_id = textgrid_path.stem[:2]
            utt_id = f"ALB_{speaker_id}_{textgrid_path.stem}"

            # Find trills
            intervals = list(phone_tier.entries)
            for i, interval in enumerate(intervals):
                label = interval.label.strip()
                if label.lower() == 'rr':
                    start_ms = interval.start * 1000
                    end_ms = interval.end * 1000

                    # Context
                    prev_phoneme = intervals[i - 1].label if i > 0 else None
                    next_phoneme = intervals[i + 1].label if i + 1 < len(intervals) else None

                    context_label = classify_phoneme_context(
                        prev_phoneme, next_phoneme, label
                    )

                    candidate = TrillCandidate(
                        utt_id=utt_id,
                        speaker_id=speaker_id,
                        dataset=self.DATASET,
                        word='[from_textgrid]',
                        word_idx=i,
                        r_idx_in_word=0,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        phoneme_label=label,
                        prev_phoneme=prev_phoneme,
                        next_phoneme=next_phoneme,
                        context_label=context_label,
                        alignment_source='textgrid',
                    )

                    result.add_candidate(candidate)

        except ImportError:
            logger.warning("praatio not installed, skipping TextGrid extraction")
        except Exception as e:
            result.add_issue(f"Failed to parse TextGrid {textgrid_path}: {e}")
