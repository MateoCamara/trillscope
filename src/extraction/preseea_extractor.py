"""PRESEEA trill /r/ extractor using Montreal Forced Aligner."""

from pathlib import Path
from typing import List, Optional
import logging
import re

from .base import TrillCandidate, ExtractionResult
from .context_classifier import classify_phoneme_context, classify_r_context, is_trill_context
from .mfa_utils import MFARunner, MFAPhoneme, MFAWord
from ..ingestion.preseea.xml_parser import PreseeaXMLParser

logger = logging.getLogger(__name__)


class PreseeaExtractor:
    """
    Extract trill /r/ candidates from PRESEEA corpus.

    PRESEEA has no phoneme-level alignment, so we use:
    1. Montreal Forced Aligner (MFA) to generate alignments
    2. Parse MFA output TextGrids for trill phonemes
    """

    DATASET = 'preseea'

    def __init__(self, corpus_root: Path, work_dir: Optional[Path] = None,
                 use_mfa: bool = True):
        """
        Initialize extractor.

        Args:
            corpus_root: Path to PRESEEA corpus directory
            work_dir: Working directory for MFA (default: temp)
            use_mfa: Whether to run MFA alignment (False = orthographic only)
        """
        self.corpus_root = corpus_root
        self.use_mfa = use_mfa
        self.xml_parser = PreseeaXMLParser()

        if use_mfa:
            self.mfa_runner = MFARunner(work_dir)
        else:
            self.mfa_runner = None

    def extract_all(self, run_mfa: bool = True) -> ExtractionResult:
        """
        Extract all trill candidates from corpus.

        Args:
            run_mfa: If True, run MFA alignment first. If False, use
                    existing alignments or orthographic-only extraction.

        Returns:
            ExtractionResult with all candidates
        """
        result = ExtractionResult(dataset=self.DATASET)

        # Find all audio/transcript pairs
        mp3_files = sorted(self.corpus_root.glob('*.mp3'))
        logger.info(f"Found {len(mp3_files)} MP3 files")

        if self.use_mfa and run_mfa:
            # Prepare files for MFA
            logger.info("Preparing files for MFA alignment...")
            self._prepare_mfa_files(mp3_files, result)

            # Run MFA
            if self.mfa_runner.check_mfa_installed():
                logger.info("Running MFA alignment (this may take a while)...")
                success = self.mfa_runner.run_alignment()
                if not success:
                    logger.warning("MFA alignment failed, falling back to orthographic extraction")
                    self._extract_orthographic_only(mp3_files, result)
                    return result
            else:
                logger.warning("MFA not installed, using orthographic extraction only")
                self._extract_orthographic_only(mp3_files, result)
                return result

        # Extract from MFA outputs
        if self.use_mfa:
            self._extract_from_mfa_outputs(mp3_files, result)
        else:
            self._extract_orthographic_only(mp3_files, result)

        result.compute_statistics()
        logger.info(f"Extracted {len(result.candidates)} trill candidates")

        return result

    def _prepare_mfa_files(self, mp3_files: List[Path],
                           result: ExtractionResult) -> None:
        """Prepare audio and transcript files for MFA."""
        for mp3_path in mp3_files:
            try:
                # Find transcript
                txt_path = mp3_path.with_suffix('.txt')
                if not txt_path.exists():
                    continue

                # Parse XML transcript
                parsed = self.xml_parser.parse(txt_path)
                if not parsed:
                    continue

                # Parser returns (metadata, utterances) tuple
                metadata, utterances = parsed
                if not utterances:
                    continue

                # Combine all utterance texts
                transcript_text = ' '.join(u.text for u in utterances if u.text)
                if not transcript_text:
                    continue

                # Prepare for MFA
                file_id = mp3_path.stem
                self.mfa_runner.prepare_file(mp3_path, transcript_text, file_id)

            except Exception as e:
                result.add_issue(f"Failed to prepare {mp3_path.stem}: {e}")

    def _extract_from_mfa_outputs(self, mp3_files: List[Path],
                                   result: ExtractionResult) -> None:
        """Extract candidates from MFA output TextGrids."""
        for mp3_path in mp3_files:
            file_id = mp3_path.stem

            # Get TextGrid output
            textgrid_path = self.mfa_runner.get_output_textgrid(file_id)
            if not textgrid_path:
                # No alignment, try orthographic extraction
                self._extract_orthographic_file(mp3_path, result)
                continue

            # Parse TextGrid
            words, phonemes = self.mfa_runner.parse_textgrid(textgrid_path)
            if not phonemes:
                continue

            # Get speaker info from transcript
            txt_path = mp3_path.with_suffix('.txt')
            speaker_id = file_id.split('_')[0] if '_' in file_id else file_id[:4]

            utt_id = f"PRE_{file_id}"

            # Find trills
            trill_indices = self.mfa_runner.find_trills(phonemes)

            for idx, phoneme in trill_indices:
                # Get context
                prev_phoneme = phonemes[idx - 1].label if idx > 0 else None
                next_phoneme = phonemes[idx + 1].label if idx + 1 < len(phonemes) else None

                # Find word
                word = self.mfa_runner.find_word_for_phoneme(words, phoneme)

                # Classify context
                context_label = classify_phoneme_context(
                    prev_phoneme, next_phoneme, phoneme.label
                )

                # Additional check: is this actually a trill context?
                if word:
                    word_context = self._check_word_trill_context(word, phoneme)
                    if not word_context:
                        continue  # Skip if word analysis suggests tap

                candidate = TrillCandidate(
                    utt_id=utt_id,
                    speaker_id=speaker_id,
                    dataset=self.DATASET,
                    word=word or '[unknown]',
                    word_idx=idx,
                    r_idx_in_word=0,
                    start_ms=phoneme.start_ms,
                    end_ms=phoneme.end_ms,
                    phoneme_label=phoneme.label,
                    prev_phoneme=prev_phoneme,
                    next_phoneme=next_phoneme,
                    context_label=context_label,
                    alignment_source='mfa',
                    audio_path=str(mp3_path),
                )

                result.add_candidate(candidate)

    def _check_word_trill_context(self, word: str, phoneme: MFAPhoneme) -> bool:
        """
        Check if word context suggests this /r/ is a trill.

        Args:
            word: The word containing the /r/
            phoneme: The phoneme from MFA

        Returns:
            True if likely trill, False if likely tap
        """
        word_lower = word.lower()

        # Check for 'rr' - always trill
        if 'rr' in word_lower:
            return True

        # Check for word-initial 'r' - always trill
        if word_lower.startswith('r'):
            return True

        # Check for 'r' after n, l, s - trill
        for trigger in ['nr', 'lr', 'sr']:
            if trigger in word_lower:
                return True

        # Check for tap clusters - not trill
        for cluster in ['br', 'cr', 'dr', 'fr', 'gr', 'pr', 'tr']:
            if cluster in word_lower:
                return False

        # Default: could be either, include for analysis
        return True

    def _extract_orthographic_only(self, mp3_files: List[Path],
                                   result: ExtractionResult) -> None:
        """Extract candidates based on orthographic rules only (no timing)."""
        for mp3_path in mp3_files:
            self._extract_orthographic_file(mp3_path, result)

    def _extract_orthographic_file(self, mp3_path: Path,
                                   result: ExtractionResult) -> None:
        """Extract candidates from a single file using orthographic rules."""
        try:
            # Find transcript
            txt_path = mp3_path.with_suffix('.txt')
            if not txt_path.exists():
                return

            # Parse XML transcript
            parsed = self.xml_parser.parse(txt_path)
            if not parsed:
                return

            # Parser returns (metadata, utterances) tuple
            metadata, utterances = parsed
            if not utterances:
                return

            # Combine all utterance texts
            transcript_text = ' '.join(u.text for u in utterances if u.text)
            if not transcript_text:
                return

            file_id = mp3_path.stem
            speaker_id = file_id.split('_')[0] if '_' in file_id else file_id[:4]
            utt_id = f"PRE_{file_id}"

            # Find words with trill contexts
            words = re.findall(r'\b\w+\b', transcript_text, re.UNICODE)

            for word_idx, word in enumerate(words):
                word_clean = ''.join(c for c in word if c.isalpha())
                if not word_clean:
                    continue

                # Analyze /r/ occurrences in word
                word_lower = word_clean.lower()

                # Check for 'rr' (intervocalic double-r)
                if 'rr' in word_lower:
                    candidate = TrillCandidate(
                        utt_id=utt_id,
                        speaker_id=speaker_id,
                        dataset=self.DATASET,
                        word=word_clean,
                        word_idx=word_idx,
                        r_idx_in_word=0,
                        start_ms=0.0,  # Unknown without alignment
                        end_ms=0.0,
                        phoneme_label='rr',
                        prev_phoneme=None,
                        next_phoneme=None,
                        context_label='intervocalic_rr',
                        alignment_source='orthographic',
                        audio_path=str(mp3_path),
                    )
                    result.add_candidate(candidate)

                # Check for word-initial 'r'
                elif word_lower.startswith('r'):
                    candidate = TrillCandidate(
                        utt_id=utt_id,
                        speaker_id=speaker_id,
                        dataset=self.DATASET,
                        word=word_clean,
                        word_idx=word_idx,
                        r_idx_in_word=0,
                        start_ms=0.0,
                        end_ms=0.0,
                        phoneme_label='r',
                        prev_phoneme=None,
                        next_phoneme=None,
                        context_label='word_initial',
                        alignment_source='orthographic',
                        audio_path=str(mp3_path),
                    )
                    result.add_candidate(candidate)

                # Check for 'r' after n, l, s
                else:
                    for trigger in ['nr', 'lr', 'sr']:
                        if trigger in word_lower:
                            candidate = TrillCandidate(
                                utt_id=utt_id,
                                speaker_id=speaker_id,
                                dataset=self.DATASET,
                                word=word_clean,
                                word_idx=word_idx,
                                r_idx_in_word=0,
                                start_ms=0.0,
                                end_ms=0.0,
                                phoneme_label='r',
                                prev_phoneme=trigger[0],
                                next_phoneme=None,
                                context_label='after_nls',
                                alignment_source='orthographic',
                                audio_path=str(mp3_path),
                            )
                            result.add_candidate(candidate)
                            break

        except Exception as e:
            result.add_issue(f"Failed orthographic extraction for {mp3_path.stem}: {e}")

    def extract_without_mfa(self) -> ExtractionResult:
        """
        Extract candidates without running MFA (orthographic only).

        Returns:
            ExtractionResult with candidates (no timing info)
        """
        result = ExtractionResult(dataset=self.DATASET)

        mp3_files = sorted(self.corpus_root.glob('*.mp3'))
        self._extract_orthographic_only(mp3_files, result)

        result.compute_statistics()
        return result
