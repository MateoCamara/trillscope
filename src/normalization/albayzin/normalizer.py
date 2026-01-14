"""ALBAYZIN corpus normalizer."""

from pathlib import Path
from typing import List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import pandas as pd

from ..base import (
    DatasetNormalizer,
    NormalizedUtterance,
    NormalizationResult,
)
from ..utils.audio_converter import AudioConverter
from ..utils.transcript_normalizer import TranscriptNormalizer

logger = logging.getLogger(__name__)


class AlbayzinNormalizer(DatasetNormalizer):
    """
    Normalize ALBAYZIN corpus.

    Input: metadata/albayzin_raw.parquet (from Block A)
    Output:
    - data_norm/albayzin/audio_16k/*.wav
    - data_norm/albayzin/transcripts/*.json (with phoneme timing)
    - outputs/tables/utt_index_albayzin.csv
    - reports/normalization/albayzin.md
    """

    DATASET_NAME = 'albayzin'

    def __init__(self, parquet_path: Path, output_root: Path):
        super().__init__(parquet_path, output_root)

        # Output directories
        self.audio_dir = output_root / 'data_norm' / 'albayzin' / 'audio_16k'
        self.transcript_dir = output_root / 'data_norm' / 'albayzin' / 'transcripts'

        # Utilities
        self.audio_converter = AudioConverter()
        self.transcript_normalizer = TranscriptNormalizer()

    def load_ingestion_data(self) -> pd.DataFrame:
        """Load data from Block A parquet file."""
        logger.info(f"Loading ingestion data from {self.parquet_path}")
        df = pd.read_parquet(self.parquet_path)
        logger.info(f"Loaded {len(df):,} utterances")
        return df

    def normalize_audio(self, row: pd.Series, output_dir: Path,
                        skip_existing: bool = False) -> Tuple[str, float, List[str]]:
        """
        Convert ALBAYZIN audio (SAM/SES) to WAV.

        Args:
            row: Row from ingestion parquet
            output_dir: Directory to save normalized audio
            skip_existing: Skip if output exists

        Returns:
            Tuple of (output_path, duration_ms, issues_list)
        """
        utt_id = row['utt_id']
        audio_path = Path(row['audio_path'])
        output_path = output_dir / f"{utt_id}.wav"

        result = self.audio_converter.convert(
            audio_path, output_path, skip_existing=skip_existing
        )

        if result.success:
            # Return relative path from data_norm root
            rel_path = str(output_path.relative_to(self.output_root / 'data_norm'))
            return rel_path, result.duration_ms, result.issues
        else:
            return '', 0, result.issues

    def normalize_transcript(self, row: pd.Series, output_dir: Path,
                             skip_existing: bool = False) -> Tuple[Optional[str], bool, Optional[str], List[str]]:
        """
        Normalize ALBAYZIN SEO transcript to JSON.

        Args:
            row: Row from ingestion parquet
            output_dir: Directory to save normalized transcript
            skip_existing: Skip if output exists

        Returns:
            Tuple of (output_path, has_timing, transcript_text, issues_list)
        """
        issues = []
        utt_id = row['utt_id']
        transcript_path = row.get('transcript_path')

        if not transcript_path or pd.isna(transcript_path):
            issues.append('no_transcript_path')
            return None, False, None, issues

        transcript_path = Path(transcript_path)
        if not transcript_path.exists():
            issues.append(f'transcript_not_found: {transcript_path}')
            return None, False, None, issues

        output_path = output_dir / f"{utt_id}.json"

        if skip_existing and output_path.exists():
            # Read existing to get text
            import json
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                rel_path = str(output_path.relative_to(self.output_root / 'data_norm'))
                return rel_path, data.get('has_timing', False), data.get('text', ''), ['skipped_existing']
            except Exception:
                pass

        transcript = self.transcript_normalizer.normalize_seo_to_json(
            transcript_path, output_path, utt_id
        )

        if transcript.text:
            rel_path = str(output_path.relative_to(self.output_root / 'data_norm'))
            return rel_path, transcript.has_timing, transcript.text, issues
        else:
            issues.append('transcript_empty')
            return None, False, None, issues

    def _normalize_row(self, row: pd.Series, skip_existing: bool) -> NormalizedUtterance:
        """Normalize a single row (audio + transcript)."""
        utt_id = row['utt_id']
        speaker_id = row['speaker_id']
        issues = []

        # Normalize audio
        audio_path, duration_ms, audio_issues = self.normalize_audio(
            row, self.audio_dir, skip_existing
        )
        issues.extend(audio_issues)

        # Normalize transcript
        transcript_path, has_timing, text, transcript_issues = self.normalize_transcript(
            row, self.transcript_dir, skip_existing
        )
        issues.extend(transcript_issues)

        # Determine validity
        is_valid = bool(audio_path) and bool(transcript_path)

        # Filter out non-error issues for validation
        validation_issues = [i for i in issues if not i.startswith('skipped_') and not i.startswith('already_')]

        return NormalizedUtterance(
            utt_id=utt_id,
            speaker_id=speaker_id,
            dataset=self.DATASET_NAME,
            original_audio_path=row['audio_path'],
            original_transcript_path=row.get('transcript_path'),
            normalized_audio_path=audio_path,
            normalized_transcript_path=transcript_path,
            duration_ms=duration_ms if duration_ms > 0 else row.get('duration_ms', 0),
            sample_rate=16000,
            transcript_text=text,
            has_timing=has_timing,
            is_valid=is_valid,
            validation_issues=validation_issues
        )

    def normalize(self, skip_existing: bool = False,
                  num_workers: int = 1) -> NormalizationResult:
        """
        Run full normalization pipeline.

        Args:
            skip_existing: Skip files that already exist
            num_workers: Number of parallel workers (not used for ALBAYZIN - fast enough)

        Returns:
            NormalizationResult with all normalized utterances
        """
        logger.info(f"Starting ALBAYZIN normalization")
        logger.info(f"  Output root: {self.output_root}")
        logger.info(f"  Audio dir: {self.audio_dir}")
        logger.info(f"  Transcript dir: {self.transcript_dir}")

        # Create output directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        df = self.load_ingestion_data()

        # Process each utterance
        utterances = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Normalizing ALBAYZIN"):
            try:
                utt = self._normalize_row(row, skip_existing)
                utterances.append(utt)

                # Log issues
                for issue in utt.validation_issues:
                    self.log_issue('warning', issue, f"{utt.utt_id}: {issue}")

            except Exception as e:
                logger.error(f"Error processing {row['utt_id']}: {e}")
                self.log_issue('error', 'processing_error', f"{row['utt_id']}: {str(e)}")

        # Calculate statistics
        total = len(utterances)
        valid = sum(1 for u in utterances if u.is_valid)
        with_timing = sum(1 for u in utterances if u.has_timing)
        total_duration_hrs = sum(u.duration_ms for u in utterances) / 3600000

        statistics = {
            'total_utterances': total,
            'valid_utterances': valid,
            'invalid_utterances': total - valid,
            'with_timing': with_timing,
            'total_duration_hours': total_duration_hrs,
        }

        logger.info(f"ALBAYZIN normalization complete:")
        logger.info(f"  Total: {total:,} utterances")
        logger.info(f"  Valid: {valid:,} ({100*valid/total:.1f}%)")
        logger.info(f"  With timing: {with_timing:,}")
        logger.info(f"  Duration: {total_duration_hrs:.2f} hours")

        return NormalizationResult(
            dataset_name=self.DATASET_NAME,
            utterances=utterances,
            issues=self.issues,
            statistics=statistics
        )
