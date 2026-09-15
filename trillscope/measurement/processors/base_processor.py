"""Base processor for acoustic measurement."""

import logging
from pathlib import Path
from typing import Optional
from abc import ABC

import pandas as pd

from ..base import (
    AcousticMetrics, MeasurementConfig, MeasurementResult, MeasurementStatus
)
from ..utils.audio_loader import load_audio_segment
from ..utils.cycle_detector import detect_trill_cycles, compute_cycle_rate
from ..utils.voicing_analyzer import analyze_voicing
from ..utils.intensity_analyzer import analyze_intensity

logger = logging.getLogger(__name__)


class BaseMeasurementProcessor(ABC):
    """Base class for dataset-specific measurement processors."""

    DATASET: str = "base"

    def __init__(
        self,
        candidates_path: Path,
        config: Optional[MeasurementConfig] = None
    ):
        """
        Initialize processor.

        Args:
            candidates_path: Path to r_candidates parquet file
            config: Measurement configuration
        """
        self.candidates_path = candidates_path
        self.config = config or MeasurementConfig()

        # Load candidates
        self.candidates_df = pd.read_parquet(candidates_path)
        logger.info(f"Loaded {len(self.candidates_df)} candidates from {candidates_path}")

    def process_all(
        self,
        skip_invalid_timing: bool = True,
        progress_callback=None
    ) -> MeasurementResult:
        """
        Process all candidates and compute acoustic measurements.

        Args:
            skip_invalid_timing: Skip tokens without valid timing
            progress_callback: Optional callback(current, total) for progress

        Returns:
            MeasurementResult with all measurements
        """
        result = MeasurementResult(dataset=self.DATASET, config=self.config)

        # Filter candidates
        df = self.candidates_df.copy()

        if skip_invalid_timing:
            original_count = len(df)
            df = df[df['duration_ms'] > 0]
            skipped = original_count - len(df)
            if skipped > 0:
                logger.info(f"Skipped {skipped} tokens without timing")
                result.add_issue(f"Skipped {skipped} tokens without valid timing")

        total = len(df)
        logger.info(f"Processing {total} tokens")

        # Group by audio file for efficient loading
        audio_groups = df.groupby('audio_path')

        processed = 0
        for audio_path, group in audio_groups:
            # Process each token in this audio file
            for idx, row in group.iterrows():
                try:
                    metric = self._measure_token(row)
                    result.add_metric(metric)
                except Exception as e:
                    logger.warning(f"Error measuring token {row.get('utt_id', idx)}: {e}")
                    result.add_issue(f"Error: {row.get('utt_id', idx)}: {e}")

                    # Add failed metric
                    metric = self._create_failed_metric(row, str(e))
                    result.add_metric(metric)

                processed += 1
                if progress_callback and processed % 100 == 0:
                    progress_callback(processed, total)

        result.compute_statistics()
        logger.info(f"Measurement complete: {result.statistics.get('successful', 0)}/{total} successful")

        return result

    def _measure_token(self, row: pd.Series) -> AcousticMetrics:
        """
        Measure acoustic properties of a single token.

        Args:
            row: DataFrame row with token info

        Returns:
            AcousticMetrics for this token
        """
        # Create base metric
        metric = AcousticMetrics(
            utt_id=row['utt_id'],
            speaker_id=row['speaker_id'],
            dataset=row['dataset'],
            word=row['word'],
            start_ms=row['start_ms'],
            end_ms=row['end_ms'],
            duration_ms=row['duration_ms'],
            context_label=row.get('context_label', ''),
            phoneme_label=row.get('phoneme_label', ''),
            audio_path=row.get('audio_path', ''),
        )

        # Check duration
        if row['duration_ms'] < self.config.min_duration_ms:
            metric.status = MeasurementStatus.AUDIO_TOO_SHORT
            metric.warnings.append(f"Duration {row['duration_ms']:.1f}ms < {self.config.min_duration_ms}ms")
            return metric

        if row['duration_ms'] > self.config.max_duration_ms:
            metric.warnings.append(f"Duration {row['duration_ms']:.1f}ms > {self.config.max_duration_ms}ms (unusual)")

        # Load audio segment
        audio, sr = load_audio_segment(
            row['audio_path'],
            row['start_ms'],
            row['end_ms'],
            target_sr=self.config.target_sr,
            context_ms=self.config.context_ms
        )

        if audio is None:
            metric.status = MeasurementStatus.AUDIO_NOT_FOUND
            return metric

        if len(audio) < 100:
            metric.status = MeasurementStatus.AUDIO_TOO_SHORT
            metric.warnings.append("Audio segment too short after loading")
            return metric

        # Measure cycles
        try:
            num_cycles, intervals, regularity, confidence = detect_trill_cycles(
                audio, sr,
                min_cycle_duration_ms=self.config.min_cycle_duration_ms,
                max_cycle_duration_ms=self.config.max_cycle_duration_ms,
                envelope_smoothing_ms=self.config.envelope_smoothing_ms,
                min_prominence=self.config.min_cycle_prominence,
                method=self.config.cycle_method,
                bp_low=self.config.bp_freq_low_hz,
                bp_high=self.config.bp_freq_high_hz,
                bp_filter_order=self.config.bp_filter_order,
                spec_win_ms=self.config.spectrogram_win_ms,
                spec_hop_ms=self.config.spectrogram_hop_ms,
            )

            metric.num_cycles = num_cycles
            metric.inter_cycle_intervals_ms = intervals
            metric.cycle_regularity = regularity
            metric.cycle_confidence = confidence
            metric.cycle_rate_hz = compute_cycle_rate(num_cycles, row['duration_ms'])

        except Exception as e:
            metric.warnings.append(f"Cycle detection error: {e}")

        # Measure voicing
        try:
            voicing_pct, mean_f0, f0_range, mean_hnr = analyze_voicing(
                audio, sr,
                pitch_floor=self.config.pitch_floor_hz,
                pitch_ceiling=self.config.pitch_ceiling_hz,
                voicing_threshold=self.config.voicing_threshold
            )

            metric.voicing_pct = voicing_pct
            metric.mean_f0_hz = mean_f0
            metric.f0_range_hz = f0_range
            metric.mean_hnr_db = mean_hnr

        except Exception as e:
            metric.warnings.append(f"Voicing analysis error: {e}")

        # Measure intensity
        try:
            mean_db, max_db, min_db, range_db = analyze_intensity(audio, sr)

            metric.mean_intensity_db = mean_db
            metric.max_intensity_db = max_db
            metric.min_intensity_db = min_db
            metric.intensity_range_db = range_db

        except Exception as e:
            metric.warnings.append(f"Intensity analysis error: {e}")

        return metric

    def _create_failed_metric(
        self,
        row: pd.Series,
        error_msg: str
    ) -> AcousticMetrics:
        """Create a metric for a failed measurement."""
        return AcousticMetrics(
            utt_id=row.get('utt_id', 'unknown'),
            speaker_id=row.get('speaker_id', 'unknown'),
            dataset=row.get('dataset', self.DATASET),
            word=row.get('word', ''),
            start_ms=row.get('start_ms', 0),
            end_ms=row.get('end_ms', 0),
            duration_ms=row.get('duration_ms', 0),
            status=MeasurementStatus.PROCESSING_ERROR,
            warnings=[error_msg],
            context_label=row.get('context_label', ''),
            phoneme_label=row.get('phoneme_label', ''),
            audio_path=row.get('audio_path', ''),
        )

    def to_dataframe(self, result: MeasurementResult) -> pd.DataFrame:
        """Convert measurement result to DataFrame."""
        if not result.metrics:
            return pd.DataFrame()

        rows = [m.to_dict() for m in result.metrics]
        return pd.DataFrame(rows)

    def save_result(
        self,
        result: MeasurementResult,
        output_path: Path
    ) -> None:
        """Save measurement result to parquet file."""
        df = self.to_dataframe(result)

        if len(df) == 0:
            logger.warning(f"No measurements to save for {self.DATASET}")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} measurements to {output_path}")
