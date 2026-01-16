"""Core plotting functions for trill /r/ visualization."""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import soundfile as sf
from scipy.signal import spectrogram, resample

from .base import VisualizationConfig, PlotResult

logger = logging.getLogger(__name__)


def load_audio_segment(
    audio_path: str,
    start_ms: float,
    end_ms: float,
    context_ms: float = 100.0,
    target_sr: int = 16000
) -> Tuple[np.ndarray, int, float, float]:
    """
    Load audio segment around the /r/ region.

    Returns:
        audio: Audio samples (resampled if needed)
        sr: Sample rate
        r_start_in_segment: Start of /r/ within segment (seconds)
        r_end_in_segment: End of /r/ within segment (seconds)
    """
    # Load full audio
    audio, sr = sf.read(audio_path)

    # Handle stereo -> mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Calculate segment boundaries in samples
    segment_start_ms = max(0, start_ms - context_ms)
    segment_end_ms = end_ms + context_ms

    start_sample = int(segment_start_ms / 1000 * sr)
    end_sample = int(segment_end_ms / 1000 * sr)
    end_sample = min(end_sample, len(audio))

    # Extract segment
    segment = audio[start_sample:end_sample]

    # Calculate /r/ position within segment
    r_start_in_segment = (start_ms - segment_start_ms) / 1000  # seconds
    r_end_in_segment = (end_ms - segment_start_ms) / 1000  # seconds

    # Resample if needed
    if sr != target_sr:
        num_samples = int(len(segment) * target_sr / sr)
        segment = resample(segment, num_samples)
        sr = target_sr

    return segment, sr, r_start_in_segment, r_end_in_segment


def compute_spectrogram_data(
    audio: np.ndarray,
    sr: int,
    config: VisualizationConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectrogram for visualization."""
    f, t, Sxx = spectrogram(
        audio,
        fs=sr,
        nperseg=config.fft_size,
        noverlap=config.fft_size - config.hop_length,
        nfft=config.fft_size
    )

    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # Limit frequency range
    freq_mask = f <= config.freq_max
    f = f[freq_mask]
    Sxx_db = Sxx_db[freq_mask, :]

    return f, t, Sxx_db


def plot_trill(
    audio_path: str,
    start_ms: float,
    end_ms: float,
    output_path: str,
    word: str,
    context_label: str,
    dataset: str,
    speaker_id: str,
    config: Optional[VisualizationConfig] = None
) -> PlotResult:
    """
    Generate two-panel plot (waveform + spectrogram) for a trill /r/.

    Returns PlotResult with success/failure status.
    """
    # Import matplotlib here to avoid issues if not installed
    import matplotlib.pyplot as plt

    if config is None:
        config = VisualizationConfig()

    utt_id = Path(audio_path).stem
    duration_ms = end_ms - start_ms

    try:
        # Load audio
        audio, sr, r_start, r_end = load_audio_segment(
            audio_path, start_ms, end_ms,
            context_ms=config.context_ms,
            target_sr=config.target_sr
        )

        # Create time axis for waveform
        time_axis = np.arange(len(audio)) / sr

        # Compute spectrogram
        f, t, Sxx_db = compute_spectrogram_data(audio, sr, config)

        # Create figure
        fig, (ax_wave, ax_spec) = plt.subplots(
            2, 1,
            figsize=config.figsize,
            sharex=True,
            gridspec_kw={'height_ratios': [1, 2]}
        )

        # Plot waveform
        ax_wave.plot(time_axis, audio, color=config.waveform_color, linewidth=0.5)
        ax_wave.axvspan(r_start, r_end,
                       color=config.highlight_color,
                       alpha=config.highlight_alpha,
                       label='/r/ region')
        ax_wave.axvline(r_start, color=config.highlight_color, linewidth=1, linestyle='--')
        ax_wave.axvline(r_end, color=config.highlight_color, linewidth=1, linestyle='--')
        ax_wave.set_ylabel('Amplitude', fontsize=config.label_fontsize)
        ax_wave.set_xlim(0, time_axis[-1])
        ax_wave.legend(loc='upper right', fontsize=8)
        ax_wave.grid(True, alpha=0.3)

        # Plot spectrogram
        ax_spec.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        ax_spec.axvline(r_start, color='white', linewidth=1.5, linestyle='--')
        ax_spec.axvline(r_end, color='white', linewidth=1.5, linestyle='--')
        ax_spec.set_ylabel('Frequency (Hz)', fontsize=config.label_fontsize)
        ax_spec.set_xlabel('Time (s)', fontsize=config.label_fontsize)
        ax_spec.set_ylim(0, config.freq_max)

        # Title
        title = f"{dataset.upper()}: '{word}' [{context_label}] - {duration_ms:.1f}ms"
        subtitle = f"Speaker: {speaker_id} | Audio: {Path(audio_path).name}"
        fig.suptitle(title, fontsize=config.title_fontsize, fontweight='bold')
        ax_wave.set_title(subtitle, fontsize=9, color='gray')

        plt.tight_layout()

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved plot: {output_path}")

        return PlotResult(
            utt_id=utt_id,
            dataset=dataset,
            output_path=output_path,
            success=True,
            word=word,
            context_label=context_label,
            duration_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Failed to plot {audio_path}: {e}")
        return PlotResult(
            utt_id=utt_id,
            dataset=dataset,
            output_path=output_path,
            success=False,
            error=str(e)
        )
