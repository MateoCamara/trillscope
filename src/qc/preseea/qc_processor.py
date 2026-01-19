"""PRESEEA audio QC processor."""

from pathlib import Path
from typing import Optional
import logging

import pandas as pd
from tqdm import tqdm
import numpy as np

from ..base import QCMetrics, QCResult, QCThresholds
from ..utils.metrics import (
    load_wav_samples,
    calculate_clipping,
    calculate_rms_db,
    calculate_frame_energies,
    calculate_silence_percentage,
)
from ..utils.vad import calculate_snr_vad

logger = logging.getLogger(__name__)


class PreseeaQCProcessor:
    """QC processor for PRESEEA dataset."""

    DATASET_NAME = 'preseea'

    def __init__(
        self,
        index_path: Path,
        data_norm_dir: Path,
        thresholds: Optional[QCThresholds] = None
    ):
        """
        Initialize processor.

        Args:
            index_path: Path to utt_index_preseea.csv from Block B
            data_norm_dir: Root path to data_norm/ directory
            thresholds: QC thresholds (uses defaults if None)
        """
        self.index_path = Path(index_path)
        self.data_norm_dir = Path(data_norm_dir)
        self.thresholds = thresholds or QCThresholds()

    def process_all(
        self,
        skip_invalid: bool = True
    ) -> QCResult:
        """
        Process all utterances and compute QC metrics.

        Note: PRESEEA audio is kept as MP3 by default in Block B.
        For QC, WAV conversion is required (run with --convert-audio).

        Args:
            skip_invalid: Skip utterances marked invalid in Block B

        Returns:
            QCResult with all metrics and statistics
        """
        result = QCResult(
            dataset=self.DATASET_NAME,
            thresholds=self.thresholds
        )

        # Load index from Block B
        logger.info(f"Loading index from {self.index_path}")
        df = pd.read_csv(self.index_path)
        logger.info(f"Found {len(df)} utterances in index")

        # Filter to valid only if requested
        if skip_invalid and 'is_valid' in df.columns:
            df = df[df['is_valid'] == True]
            logger.info(f"Processing {len(df)} valid utterances")

        # Check for WAV files
        wav_count = 0
        mp3_count = 0
        for _, row in df.iterrows():
            audio_path = row.get('normalized_audio_path', '')
            if audio_path.endswith('.wav'):
                wav_count += 1
            elif audio_path.endswith('.mp3'):
                mp3_count += 1

        if mp3_count > 0 and wav_count == 0:
            logger.warning(
                f"PRESEEA audio is in MP3 format ({mp3_count} files). "
                "Run 'python -m src.normalization.cli normalize preseea --convert-audio' "
                "to convert to WAV for QC analysis."
            )
            result.add_issue(
                f"PRESEEA requires WAV conversion for QC. Found {mp3_count} MP3 files."
            )
            return result

        # Process each utterance
        for _, row in tqdm(df.iterrows(), total=len(df), desc="QC PRESEEA"):
            try:
                audio_path = row['normalized_audio_path']

                # Skip MP3 files
                if audio_path.endswith('.mp3'):
                    result.add_issue(f"Skipped MP3 file: {row['utt_id']}")
                    continue

                metric = self.process_utterance(
                    utt_id=row['utt_id'],
                    speaker_id=row['speaker_id'],
                    audio_path=audio_path
                )
                if metric:
                    # Apply thresholds
                    self.apply_thresholds(metric)
                    result.add_metric(metric)
                else:
                    result.add_issue(f"Failed to process {row['utt_id']}")
            except Exception as e:
                logger.error(f"Error processing {row['utt_id']}: {e}")
                result.add_issue(f"Error processing {row['utt_id']}: {str(e)}")

        # Compute statistics
        result.compute_statistics()

        return result

    def process_utterance(
        self,
        utt_id: str,
        speaker_id: str,
        audio_path: str
    ) -> Optional[QCMetrics]:
        """
        Process a single utterance and compute QC metrics.

        Args:
            utt_id: Utterance identifier
            speaker_id: Speaker identifier
            audio_path: Relative path to normalized WAV file (from index)

        Returns:
            QCMetrics or None if processing failed
        """
        # Resolve full path
        full_audio_path = self.data_norm_dir / audio_path

        if not full_audio_path.exists():
            logger.warning(f"Audio file not found: {full_audio_path}")
            return None

        try:
            # Load audio
            samples, sample_rate = load_wav_samples(full_audio_path)

            # Calculate duration
            duration_ms = len(samples) / sample_rate * 1000

            # Calculate clipping
            clipping_pct, num_clipped, num_samples = calculate_clipping(
                samples,
                near_max_fraction=self.thresholds.clipping_near_max_fraction
            )

            # Calculate RMS
            rms_db = calculate_rms_db(samples)

            # Calculate frame energies
            frame_energies = calculate_frame_energies(
                samples,
                sample_rate,
                frame_size_ms=self.thresholds.frame_size_ms,
                hop_size_ms=self.thresholds.hop_size_ms
            )

            # Calculate silence percentage
            silence_pct = calculate_silence_percentage(
                frame_energies,
                threshold_db=self.thresholds.silence_threshold_db
            )

            # Calculate SNR using VAD
            snr_db, speech_energy_db, noise_energy_db = calculate_snr_vad(
                samples,
                sample_rate,
                frame_size_ms=self.thresholds.frame_size_ms,
                hop_size_ms=self.thresholds.hop_size_ms,
                speech_threshold_db=self.thresholds.vad_speech_threshold_db,
                noise_threshold_db=self.thresholds.vad_noise_threshold_db
            )

            return QCMetrics(
                utt_id=utt_id,
                speaker_id=speaker_id,
                dataset=self.DATASET_NAME,
                audio_path=str(audio_path),
                duration_ms=duration_ms,
                clipping_pct=clipping_pct,
                rms_db=rms_db,
                snr_db=snr_db,
                silence_pct=silence_pct,
                num_samples=num_samples,
                num_clipped_samples=num_clipped,
                speech_energy_db=speech_energy_db,
                noise_energy_db=noise_energy_db,
            )

        except Exception as e:
            logger.error(f"Failed to process audio {full_audio_path}: {e}")
            return None

    def apply_thresholds(self, metrics: QCMetrics) -> QCMetrics:
        """
        Apply QC thresholds and set pass/fail status.

        Modifies metrics.pass_qc and metrics.failed_checks in place.

        Args:
            metrics: QCMetrics to evaluate

        Returns:
            Same metrics object with pass_qc and failed_checks updated
        """
        failed_checks = []

        # Check clipping
        if metrics.clipping_pct > self.thresholds.clipping_max_pct:
            failed_checks.append('clipping')

        # Check duration
        if metrics.duration_ms < self.thresholds.duration_min_ms:
            failed_checks.append('duration')

        # Check SNR (handle inf values)
        if np.isfinite(metrics.snr_db):
            if metrics.snr_db < self.thresholds.snr_min_db:
                failed_checks.append('snr')
        elif metrics.snr_db == float('-inf'):
            failed_checks.append('snr')

        # Check silence
        if metrics.silence_pct > self.thresholds.silence_max_pct:
            failed_checks.append('silence')

        metrics.failed_checks = failed_checks
        metrics.pass_qc = len(failed_checks) == 0

        return metrics
