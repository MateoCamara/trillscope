"""Base classes and data types for audio quality control (Block E)."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QCThresholds:
    """Configurable thresholds for QC filtering."""

    # Clipping: discard if > threshold (percentage of samples near max)
    clipping_max_pct: float = 0.1

    # Duration: discard if < threshold (milliseconds)
    duration_min_ms: float = 500.0

    # SNR: discard if < threshold (dB)
    snr_min_db: float = 10.0

    # Silence: discard if > threshold (percentage of frames)
    silence_max_pct: float = 60.0

    # Near-max sample threshold for clipping detection (fraction of max value)
    clipping_near_max_fraction: float = 0.99

    # Energy threshold for silence detection (dB below peak)
    silence_threshold_db: float = -40.0

    # Frame size for energy analysis (ms)
    frame_size_ms: float = 25.0

    # Hop size for energy analysis (ms)
    hop_size_ms: float = 10.0

    # VAD thresholds for SNR estimation
    vad_speech_threshold_db: float = -35.0
    vad_noise_threshold_db: float = -50.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QCThresholds':
        """Create thresholds from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'clipping_max_pct': self.clipping_max_pct,
            'duration_min_ms': self.duration_min_ms,
            'snr_min_db': self.snr_min_db,
            'silence_max_pct': self.silence_max_pct,
            'clipping_near_max_fraction': self.clipping_near_max_fraction,
            'silence_threshold_db': self.silence_threshold_db,
            'frame_size_ms': self.frame_size_ms,
            'hop_size_ms': self.hop_size_ms,
            'vad_speech_threshold_db': self.vad_speech_threshold_db,
            'vad_noise_threshold_db': self.vad_noise_threshold_db,
        }


@dataclass
class QCMetrics:
    """Per-utterance quality metrics."""

    # Identity
    utt_id: str
    speaker_id: str
    dataset: str
    audio_path: str

    # Measured metrics
    duration_ms: float
    clipping_pct: float  # Percentage of samples at/near max value
    rms_db: float  # RMS level in dB (ref: 1.0 for normalized float)
    snr_db: float  # Estimated SNR using VAD
    silence_pct: float  # Percentage of frames classified as silence

    # QC result (fields with defaults must come after required fields)
    pass_qc: bool = True
    failed_checks: List[str] = field(default_factory=list)

    # Additional info
    num_samples: int = 0
    num_clipped_samples: int = 0
    speech_energy_db: float = 0.0
    noise_energy_db: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'utt_id': self.utt_id,
            'speaker_id': self.speaker_id,
            'dataset': self.dataset,
            'audio_path': self.audio_path,
            'duration_ms': self.duration_ms,
            'clipping_pct': self.clipping_pct,
            'rms_db': self.rms_db,
            'snr_db': self.snr_db,
            'silence_pct': self.silence_pct,
            'pass_qc': self.pass_qc,
            'failed_checks': ';'.join(self.failed_checks),
            'num_samples': self.num_samples,
            'num_clipped_samples': self.num_clipped_samples,
            'speech_energy_db': self.speech_energy_db,
            'noise_energy_db': self.noise_energy_db,
        }


@dataclass
class QCStatistics:
    """Aggregate statistics for a QC run."""

    total_utterances: int = 0
    passed_utterances: int = 0
    failed_utterances: int = 0

    # Failure counts by cause
    failed_by_clipping: int = 0
    failed_by_duration: int = 0
    failed_by_snr: int = 0
    failed_by_silence: int = 0

    # Distribution statistics
    duration_percentiles: Dict[str, float] = field(default_factory=dict)
    clipping_percentiles: Dict[str, float] = field(default_factory=dict)
    snr_percentiles: Dict[str, float] = field(default_factory=dict)
    silence_percentiles: Dict[str, float] = field(default_factory=dict)
    rms_percentiles: Dict[str, float] = field(default_factory=dict)

    # Total audio duration
    total_duration_hours: float = 0.0
    passed_duration_hours: float = 0.0


@dataclass
class QCResult:
    """Result of QC processing for a dataset."""

    dataset: str
    metrics: List[QCMetrics] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    statistics: QCStatistics = field(default_factory=QCStatistics)
    thresholds: QCThresholds = field(default_factory=QCThresholds)

    def add_metric(self, metric: QCMetrics) -> None:
        """Add a QC metric result."""
        self.metrics.append(metric)

    def add_issue(self, issue: str) -> None:
        """Log an issue."""
        self.issues.append(issue)

    def compute_statistics(self) -> None:
        """Compute aggregate statistics from metrics."""
        if not self.metrics:
            return

        self.statistics.total_utterances = len(self.metrics)
        self.statistics.passed_utterances = sum(1 for m in self.metrics if m.pass_qc)
        self.statistics.failed_utterances = (
            self.statistics.total_utterances - self.statistics.passed_utterances
        )

        # Count failures by cause
        for m in self.metrics:
            if 'clipping' in m.failed_checks:
                self.statistics.failed_by_clipping += 1
            if 'duration' in m.failed_checks:
                self.statistics.failed_by_duration += 1
            if 'snr' in m.failed_checks:
                self.statistics.failed_by_snr += 1
            if 'silence' in m.failed_checks:
                self.statistics.failed_by_silence += 1

        # Calculate percentiles
        percentile_keys = ['p5', 'p25', 'p50', 'p75', 'p95']
        percentile_values = [5, 25, 50, 75, 95]

        durations = [m.duration_ms for m in self.metrics]
        clippings = [m.clipping_pct for m in self.metrics]
        # Filter out inf values for SNR percentiles
        snrs = [m.snr_db for m in self.metrics if np.isfinite(m.snr_db)]
        silences = [m.silence_pct for m in self.metrics]
        rms_values = [m.rms_db for m in self.metrics if np.isfinite(m.rms_db)]

        for key, pct in zip(percentile_keys, percentile_values):
            self.statistics.duration_percentiles[key] = float(np.percentile(durations, pct))
            self.statistics.clipping_percentiles[key] = float(np.percentile(clippings, pct))
            if snrs:
                self.statistics.snr_percentiles[key] = float(np.percentile(snrs, pct))
            self.statistics.silence_percentiles[key] = float(np.percentile(silences, pct))
            if rms_values:
                self.statistics.rms_percentiles[key] = float(np.percentile(rms_values, pct))

        # Total durations
        self.statistics.total_duration_hours = sum(durations) / 3600000
        self.statistics.passed_duration_hours = sum(
            m.duration_ms for m in self.metrics if m.pass_qc
        ) / 3600000

    def to_csv(self, output_path: Path) -> None:
        """Save QC flags to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([m.to_dict() for m in self.metrics])
        df.to_csv(output_path, index=False)

        logger.info(f"Saved {len(df)} QC records to {output_path}")

    def to_parquet(self, output_path: Path) -> None:
        """Save QC flags to parquet."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([m.to_dict() for m in self.metrics])
        df.to_parquet(output_path, index=False)

        logger.info(f"Saved {len(df)} QC records to {output_path}")
