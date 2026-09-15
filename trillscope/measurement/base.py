"""Base classes and data types for acoustic measurement."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class MeasurementStatus(Enum):
    """Status of acoustic measurement for a token."""
    SUCCESS = "success"
    INVALID_TIMING = "invalid_timing"
    AUDIO_NOT_FOUND = "audio_not_found"
    AUDIO_TOO_SHORT = "audio_too_short"
    PROCESSING_ERROR = "processing_error"


@dataclass
class MeasurementConfig:
    """Configuration for acoustic measurement."""

    # Audio loading
    target_sr: int = 16000
    context_ms: float = 50.0  # Extra context before/after segment

    # Cycle detection
    min_cycle_duration_ms: float = 10.0   # Min time between peaks
    max_cycle_duration_ms: float = 60.0   # Max time between peaks
    envelope_smoothing_ms: float = 5.0    # Envelope smoothing window
    min_cycle_prominence: float = 0.1     # Min peak prominence (normalized)

    # Cycle detection - multi-method
    cycle_method: str = 'multi'           # 'multi', 'envelope', 'legacy'
    bp_freq_low_hz: float = 500.0         # Band-pass low cutoff
    bp_freq_high_hz: float = 4000.0       # Band-pass high cutoff
    bp_filter_order: int = 4              # Butterworth filter order
    spectrogram_win_ms: float = 4.0       # STFT window size
    spectrogram_hop_ms: float = 1.0       # STFT hop size

    # Voicing detection (Praat parameters)
    pitch_floor_hz: float = 75.0
    pitch_ceiling_hz: float = 500.0
    voicing_threshold: float = 0.45

    # Quality thresholds
    min_duration_ms: float = 5.0          # Skip very short segments
    max_duration_ms: float = 500.0        # Flag unusually long segments

    # Processing
    n_jobs: int = 1


@dataclass
class AcousticMetrics:
    """Acoustic measurements for a single trill /r/ token."""

    # Identity (from r_candidates)
    utt_id: str
    speaker_id: str
    dataset: str
    word: str

    # Timing (from r_candidates)
    start_ms: float
    end_ms: float
    duration_ms: float

    # Cycle detection
    num_cycles: int = 0
    cycle_rate_hz: float = 0.0
    inter_cycle_intervals_ms: List[float] = field(default_factory=list)
    cycle_regularity: float = 0.0  # CV of intervals (0=perfect)
    cycle_confidence: float = 0.0  # 0-1 consensus confidence

    # Voicing
    voicing_pct: float = 0.0       # 0-100%
    mean_f0_hz: Optional[float] = None
    f0_range_hz: Optional[float] = None
    mean_hnr_db: float = 0.0       # Harmonics-to-noise ratio

    # Intensity
    mean_intensity_db: float = 0.0
    max_intensity_db: float = 0.0
    min_intensity_db: float = 0.0
    intensity_range_db: float = 0.0

    # Quality flags
    status: MeasurementStatus = MeasurementStatus.SUCCESS
    warnings: List[str] = field(default_factory=list)

    # Metadata (from r_candidates)
    context_label: str = ""
    phoneme_label: str = ""
    audio_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'utt_id': self.utt_id,
            'speaker_id': self.speaker_id,
            'dataset': self.dataset,
            'word': self.word,
            'start_ms': self.start_ms,
            'end_ms': self.end_ms,
            'duration_ms': self.duration_ms,
            'num_cycles': self.num_cycles,
            'cycle_rate_hz': self.cycle_rate_hz,
            'cycle_regularity': self.cycle_regularity,
            'cycle_confidence': self.cycle_confidence,
            'voicing_pct': self.voicing_pct,
            'mean_f0_hz': self.mean_f0_hz,
            'f0_range_hz': self.f0_range_hz,
            'mean_hnr_db': self.mean_hnr_db,
            'mean_intensity_db': self.mean_intensity_db,
            'max_intensity_db': self.max_intensity_db,
            'min_intensity_db': self.min_intensity_db,
            'intensity_range_db': self.intensity_range_db,
            'status': self.status.value,
            'warnings': ';'.join(self.warnings) if self.warnings else '',
            'context_label': self.context_label,
            'phoneme_label': self.phoneme_label,
            'audio_path': self.audio_path,
        }


@dataclass
class MeasurementResult:
    """Result of measurement for a dataset."""

    dataset: str
    metrics: List[AcousticMetrics] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    config: MeasurementConfig = field(default_factory=MeasurementConfig)

    def add_metric(self, metric: AcousticMetrics) -> None:
        """Add a measurement result."""
        self.metrics.append(metric)

    def add_issue(self, issue: str) -> None:
        """Log an issue."""
        self.issues.append(issue)

    def compute_statistics(self) -> None:
        """Compute summary statistics."""
        successful = [m for m in self.metrics if m.status == MeasurementStatus.SUCCESS]

        self.statistics = {
            'total_tokens': len(self.metrics),
            'successful': len(successful),
            'failed': len(self.metrics) - len(successful),
            'total_issues': len(self.issues),
        }

        # Status breakdown
        status_counts = {}
        for m in self.metrics:
            status = m.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        self.statistics['by_status'] = status_counts

        if successful:
            # Cycle count distribution
            cycle_counts = {}
            for m in successful:
                c = m.num_cycles
                cycle_counts[c] = cycle_counts.get(c, 0) + 1
            self.statistics['cycle_distribution'] = cycle_counts

            # Duration stats
            durations = [m.duration_ms for m in successful]
            self.statistics['mean_duration_ms'] = sum(durations) / len(durations)

            # Voicing stats
            voicing = [m.voicing_pct for m in successful]
            self.statistics['mean_voicing_pct'] = sum(voicing) / len(voicing)

            # Confidence distribution
            confidences = [m.cycle_confidence for m in successful]
            high_conf = sum(1 for c in confidences if c > 0.7)
            med_conf = sum(1 for c in confidences if 0.3 <= c <= 0.7)
            low_conf = sum(1 for c in confidences if c < 0.3)
            self.statistics['mean_confidence'] = sum(confidences) / len(confidences)
            self.statistics['confidence_distribution'] = {
                'high': high_conf,
                'medium': med_conf,
                'low': low_conf,
            }

            # F0 stats (only for tokens with F0)
            f0_values = [m.mean_f0_hz for m in successful if m.mean_f0_hz is not None]
            if f0_values:
                self.statistics['mean_f0_hz'] = sum(f0_values) / len(f0_values)
