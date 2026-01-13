"""DIMEx100 trill /r/ extractor using existing .phn alignment files."""

from pathlib import Path
from typing import List, Optional
import logging

from .base import TrillCandidate, ExtractionResult
from .context_classifier import classify_phoneme_context
from ..ingestion.dimex100.phn_parser import DIMExPHNParser, PhonemeSegment

logger = logging.getLogger(__name__)


class DIMEx100Extractor:
    """
    Extract trill /r/ candidates from DIMEx100 corpus.

    Uses existing .phn phoneme alignment files where 'r(' marks trill.
    """

    TRILL_MARKER = 'r('
    DATASET = 'dimex100'

    def __init__(self, corpus_root: Path):
        """
        Initialize extractor.

        Args:
            corpus_root: Path to CorpusDimex100 directory
        """
        self.corpus_root = corpus_root
        self.phn_parser = DIMExPHNParser()

    def extract_all(self) -> ExtractionResult:
        """
        Extract all trill candidates from corpus.

        Returns:
            ExtractionResult with all candidates
        """
        result = ExtractionResult(dataset=self.DATASET)

        # Discover speaker directories
        speaker_dirs = sorted(self.corpus_root.glob('s[0-9][0-9][0-9]'))
        logger.info(f"Found {len(speaker_dirs)} speaker directories")

        for speaker_dir in speaker_dirs:
            speaker_id = speaker_dir.name
            self._extract_speaker(speaker_dir, speaker_id, result)

        result.compute_statistics()
        logger.info(f"Extracted {len(result.candidates)} trill candidates")

        return result

    def _extract_speaker(self, speaker_dir: Path, speaker_id: str,
                         result: ExtractionResult) -> None:
        """Extract candidates from a single speaker directory."""

        # Find all .phn files in T22, T44, T54 subdirectories
        for task_dir in ['T22', 'T44', 'T54']:
            task_path = speaker_dir / task_dir
            if not task_path.exists():
                continue

            phn_files = sorted(task_path.glob('*.phn'))
            for phn_path in phn_files:
                self._extract_from_phn(phn_path, speaker_id, task_dir, result)

    def _extract_from_phn(self, phn_path: Path, speaker_id: str,
                          task: str, result: ExtractionResult) -> None:
        """Extract candidates from a single .phn file."""

        try:
            segments = self.phn_parser.parse_phoneme_file(phn_path)
            if not segments:
                return

            # Generate utterance ID
            utt_id = f"DIM_{speaker_id}_{phn_path.stem}"

            # Find corresponding audio file
            audio_path = self._find_audio_file(phn_path, speaker_id)

            # Find corresponding transcript for word context
            transcript = self._load_transcript(phn_path, speaker_id)

            # Track word position heuristically
            word_idx = 0
            current_word = ""

            for i, segment in enumerate(segments):
                if segment.is_trill:
                    # Get context phonemes
                    prev_phoneme = segments[i - 1].phoneme if i > 0 else None
                    next_phoneme = segments[i + 1].phoneme if i < len(segments) - 1 else None

                    # Classify context
                    context_label = classify_phoneme_context(
                        prev_phoneme, next_phoneme, segment.phoneme
                    )

                    # Try to find the word containing this trill
                    word = self._find_word_at_time(
                        transcript, segment.start_ms, segment.end_ms
                    )

                    candidate = TrillCandidate(
                        utt_id=utt_id,
                        speaker_id=speaker_id,
                        dataset=self.DATASET,
                        word=word or '[unknown]',
                        word_idx=word_idx,
                        r_idx_in_word=0,  # Would need more analysis for multiple /r/ in word
                        start_ms=segment.start_ms,
                        end_ms=segment.end_ms,
                        phoneme_label=segment.phoneme,
                        prev_phoneme=prev_phoneme,
                        next_phoneme=next_phoneme,
                        context_label=context_label,
                        alignment_source='phn',
                        audio_path=str(audio_path) if audio_path else None,
                    )

                    result.add_candidate(candidate)
                    word_idx += 1

        except Exception as e:
            result.add_issue(f"Failed to extract from {phn_path}: {e}")
            logger.warning(f"Failed to extract from {phn_path}: {e}")

    def _find_audio_file(self, phn_path: Path, speaker_id: str) -> Optional[Path]:
        """Find the audio file corresponding to a .phn file."""
        # audio_editado contains the edited audio files
        audio_dir = self.corpus_root / speaker_id / 'audio_editado'
        if audio_dir.exists():
            # Match by stem (e.g., s00101.phn -> s00101.wav)
            audio_file = audio_dir / f"{phn_path.stem}.wav"
            if audio_file.exists():
                return audio_file

        return None

    def _load_transcript(self, phn_path: Path, speaker_id: str) -> Optional[str]:
        """Load transcript text for a .phn file."""
        # texto/ contains transcripts
        texto_dir = self.corpus_root / speaker_id / 'texto'
        if texto_dir.exists():
            txt_file = texto_dir / f"{phn_path.stem}.txt"
            if txt_file.exists():
                try:
                    # Try different encodings
                    for enc in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            return txt_file.read_text(encoding=enc).strip()
                        except UnicodeDecodeError:
                            continue
                except Exception:
                    pass
        return None

    def _find_word_at_time(self, transcript: Optional[str],
                           start_ms: float, end_ms: float) -> Optional[str]:
        """
        Find the word containing a given time span.

        This is a heuristic approach - proper word alignment would be better.
        For now, we just return the transcript words that might contain 'r'.
        """
        if not transcript:
            return None

        # Simple heuristic: find words containing 'r' or 'rr'
        words = transcript.split()
        for word in words:
            word_clean = ''.join(c for c in word if c.isalpha())
            if 'r' in word_clean.lower():
                return word_clean

        return None

    def extract_utterance(self, phn_path: Path, speaker_id: str) -> List[TrillCandidate]:
        """
        Extract candidates from a single utterance.

        Args:
            phn_path: Path to .phn file
            speaker_id: Speaker ID

        Returns:
            List of TrillCandidate objects
        """
        result = ExtractionResult(dataset=self.DATASET)
        self._extract_from_phn(phn_path, speaker_id, 'T22', result)
        return result.candidates
