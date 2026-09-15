"""Generic MFA-based trill /r/ extractor for datasets with MFA alignment."""

import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
from tqdm import tqdm

from .base import TrillCandidate, ExtractionResult
from .context_classifier import classify_phoneme_context
from .mfa_utils import MFARunner, MFAPhoneme, MFAWord

logger = logging.getLogger(__name__)


class MFAExtractor:
    """
    Generic extractor for datasets aligned with Montreal Forced Aligner.

    Works with any dataset that has:
    1. MFA TextGrid outputs in mfa_work/{dataset}/output/
    2. Metadata parquet with utt_id matching TextGrid filenames
    """

    # MFA trill phonemes (excludes tap ɾ)
    TRILL_LABELS = {'r', 'rr', 'ʀ'}

    def __init__(
        self,
        dataset: str,
        mfa_output_dir: Path,
        metadata_path: Path,
        audio_base_dir: Optional[Path] = None
    ):
        """
        Initialize extractor.

        Args:
            dataset: Dataset identifier (mailabs, tedx, heroico, commonvoice)
            mfa_output_dir: Directory with MFA TextGrid outputs
            metadata_path: Path to {dataset}_raw.parquet
            audio_base_dir: Base directory for audio files (optional)
        """
        self.dataset = dataset
        self.mfa_output_dir = Path(mfa_output_dir)
        self.metadata_path = Path(metadata_path)
        self.audio_base_dir = Path(audio_base_dir) if audio_base_dir else None

        # Create MFA runner for TextGrid parsing
        self.mfa_runner = MFARunner()

    def extract_all(self) -> ExtractionResult:
        """
        Extract all trill candidates from MFA alignments.

        Returns:
            ExtractionResult with candidates
        """
        result = ExtractionResult(dataset=self.dataset)

        # Load metadata
        if not self.metadata_path.exists():
            logger.error(f"Metadata not found: {self.metadata_path}")
            result.add_issue(f"Metadata not found: {self.metadata_path}")
            return result

        df = pd.read_parquet(self.metadata_path)
        logger.info(f"Loaded {len(df)} records from metadata")

        # Create lookup from utt_id to metadata row
        metadata_lookup = {row['utt_id']: row for _, row in df.iterrows()}

        # Find all TextGrid files
        textgrid_files = list(self.mfa_output_dir.glob('*.TextGrid'))
        if not textgrid_files:
            logger.error(f"No TextGrid files found in {self.mfa_output_dir}")
            result.add_issue(f"No TextGrid files in {self.mfa_output_dir}")
            return result

        logger.info(f"Found {len(textgrid_files)} TextGrid files")

        # Process each TextGrid
        for tg_path in tqdm(textgrid_files, desc=f"Extracting {self.dataset}"):
            utt_id = tg_path.stem  # TextGrid name = utt_id

            # Get metadata for this utterance
            if utt_id not in metadata_lookup:
                result.add_issue(f"No metadata for {utt_id}")
                continue

            metadata = metadata_lookup[utt_id]

            # Extract speaker ID and audio path from metadata
            speaker_id = metadata['speaker_id'] if 'speaker_id' in metadata else 'unknown'
            audio_path = metadata['audio_path'] if 'audio_path' in metadata else ''

            # Parse TextGrid
            try:
                words, phonemes = self.mfa_runner.parse_textgrid(tg_path)
            except Exception as e:
                result.add_issue(f"Failed to parse {tg_path}: {e}")
                continue

            if not phonemes:
                continue

            # Find trills
            trill_candidates = self._find_trills_with_context(
                phonemes, words, utt_id, speaker_id, audio_path
            )

            for candidate in trill_candidates:
                result.add_candidate(candidate)

        result.compute_statistics()
        logger.info(f"Extracted {len(result.candidates)} trill candidates from {self.dataset}")

        return result

    def _find_trills_with_context(
        self,
        phonemes: List[MFAPhoneme],
        words: List[MFAWord],
        utt_id: str,
        speaker_id: str,
        audio_path: str
    ) -> List[TrillCandidate]:
        """
        Find trill phonemes and create candidates with context filtering.

        Args:
            phonemes: Phonemes from TextGrid
            words: Words from TextGrid
            utt_id: Utterance ID
            speaker_id: Speaker ID
            audio_path: Path to audio file

        Returns:
            List of TrillCandidate objects
        """
        candidates = []

        for idx, phoneme in enumerate(phonemes):
            # Only process trill labels
            if phoneme.label not in self.TRILL_LABELS:
                continue

            # Get context phonemes
            prev_phoneme = phonemes[idx - 1].label if idx > 0 else None
            next_phoneme = phonemes[idx + 1].label if idx + 1 < len(phonemes) else None

            # Find word containing this phoneme
            word = self.mfa_runner.find_word_for_phoneme(words, phoneme)

            # Apply word-context filtering
            if word:
                if not self._is_trill_context(word):
                    continue  # Skip non-trill contexts
            else:
                word = '[unknown]'

            # Classify phonetic context
            context_label = classify_phoneme_context(
                prev_phoneme, next_phoneme, phoneme.label
            )

            candidate = TrillCandidate(
                utt_id=utt_id,
                speaker_id=speaker_id,
                dataset=self.dataset,
                word=word,
                word_idx=idx,
                r_idx_in_word=0,
                start_ms=phoneme.start_ms,
                end_ms=phoneme.end_ms,
                phoneme_label=phoneme.label,
                prev_phoneme=prev_phoneme,
                next_phoneme=next_phoneme,
                context_label=context_label,
                alignment_source='mfa',
                audio_path=audio_path,
                has_overlap=False,
            )
            candidates.append(candidate)

        return candidates

    def _is_trill_context(self, word: str) -> bool:
        """
        Check if word context indicates a trill (not tap).

        Spanish trill contexts:
        1. 'rr' - intervocalic double-r (perro)
        2. Word-initial 'r' (rojo)
        3. 'r' after n, l, s (enredo, alrededor, Israel)

        Non-trill (tap) contexts:
        1. Intervocalic single 'r' (pero)
        2. Consonant clusters: br, cr, dr, fr, gr, pr, tr (brazo, tres)
        3. Coda/final 'r' (árbol, mar)

        Args:
            word: The word containing /r/

        Returns:
            True if trill context, False if tap context
        """
        word_lower = word.lower()

        # No 'r' at all - shouldn't happen but skip
        if 'r' not in word_lower:
            return False

        # 'rr' is always trill
        if 'rr' in word_lower:
            return True

        # Word-initial 'r' is always trill
        if word_lower.startswith('r'):
            return True

        # 'r' after n, l, s at syllable boundary is trill
        for trigger in ['nr', 'lr', 'sr']:
            if trigger in word_lower:
                return True

        # Consonant clusters with 'r' are taps
        for cluster in ['br', 'cr', 'dr', 'fr', 'gr', 'pr', 'tr', 'kr']:
            if cluster in word_lower:
                return False

        # Intervocalic single 'r' is a tap
        import re
        vowels = 'aeiouáéíóú'
        for match in re.finditer(r'r', word_lower):
            pos = match.start()
            prev_char = word_lower[pos - 1] if pos > 0 else ''
            next_char = word_lower[pos + 1] if pos + 1 < len(word_lower) else ''

            # Check if it's part of 'rr' (skip)
            if pos > 0 and word_lower[pos - 1] == 'r':
                continue
            if pos + 1 < len(word_lower) and word_lower[pos + 1] == 'r':
                continue

            # Intervocalic single r is tap
            if prev_char in vowels and next_char in vowels:
                return False

            # Coda r is variable, exclude
            if prev_char in vowels and (next_char not in vowels or not next_char):
                return False

        # Default: exclude uncertain cases
        return False


def extract_mfa_dataset(
    dataset: str,
    mfa_output_dir: Path = None,
    metadata_path: Path = None,
    output_path: Path = None
) -> ExtractionResult:
    """
    Extract trill candidates from an MFA-aligned dataset.

    Args:
        dataset: Dataset name (mailabs, tedx, heroico, commonvoice)
        mfa_output_dir: Directory with MFA outputs (default: mfa_work/{dataset}/output)
        metadata_path: Path to metadata (default: metadata/{dataset}_raw.parquet)
        output_path: Output parquet path (default: outputs/tables/r_candidates_{dataset}.parquet)

    Returns:
        ExtractionResult
    """
    # Set defaults
    if mfa_output_dir is None:
        mfa_output_dir = Path(f'mfa_work/{dataset}/output')
    if metadata_path is None:
        metadata_path = Path(f'metadata/{dataset}_raw.parquet')
    if output_path is None:
        output_path = Path(f'outputs/tables/r_candidates_{dataset}.parquet')

    # Create extractor and run
    extractor = MFAExtractor(
        dataset=dataset,
        mfa_output_dir=mfa_output_dir,
        metadata_path=metadata_path
    )

    result = extractor.extract_all()

    # Save to parquet
    if result.candidates:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([c.to_dict() for c in result.candidates])
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(result.candidates)} candidates to {output_path}")
    else:
        logger.warning(f"No candidates extracted for {dataset}")

    return result


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Extract trills from MFA-aligned datasets')
    parser.add_argument(
        'dataset',
        choices=['mailabs', 'tedx', 'heroico', 'commonvoice', 'all'],
        help='Dataset to extract from'
    )
    args = parser.parse_args()

    datasets = ['mailabs', 'tedx', 'heroico', 'commonvoice'] if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Extracting {ds.upper()}")
        print(f"{'='*60}")

        result = extract_mfa_dataset(ds)

        print(f"Total candidates: {len(result.candidates)}")
        print(f"Issues: {len(result.issues)}")
        if result.statistics:
            print(f"Unique speakers: {result.statistics.get('unique_speakers', 0)}")
            print(f"By context: {result.statistics.get('by_context', {})}")
