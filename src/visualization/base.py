"""Base classes for visualization module."""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class VisualizationConfig:
    """Configuration for trill /r/ visualization."""

    # Audio processing
    target_sr: int = 16000  # Target sample rate for consistent spectrograms
    context_ms: float = 250.0  # Context window before/after /r/ region (increased for better audibility)

    # Spectrogram settings
    fft_size: int = 512
    hop_length: int = 128
    freq_max: int = 8000  # Max frequency to display

    # Figure settings
    figsize: Tuple[float, float] = (12, 6)
    dpi: int = 150

    # Colors
    highlight_color: str = '#FF6B6B'  # Red for /r/ region
    highlight_alpha: float = 0.3
    waveform_color: str = '#2E86AB'  # Blue for waveform

    # Text
    title_fontsize: int = 12
    label_fontsize: int = 10


@dataclass
class PlotResult:
    """Result of plotting a single trill example."""

    utt_id: str
    dataset: str
    output_path: str
    success: bool
    error: Optional[str] = None
    word: str = ''
    context_label: str = ''
    duration_ms: float = 0.0
