"""Intensity analysis for acoustic measurement."""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def analyze_intensity(
    audio: np.ndarray,
    sr: int
) -> Tuple[float, float, float, float]:
    """
    Analyze intensity characteristics.

    Args:
        audio: Audio samples (normalized to [-1, 1])
        sr: Sample rate

    Returns:
        Tuple of:
        - mean_db: Mean intensity in dB
        - max_db: Maximum intensity in dB
        - min_db: Minimum intensity in dB
        - range_db: Intensity range (max - min)
    """
    if len(audio) < 10:
        return 0.0, 0.0, 0.0, 0.0

    try:
        import parselmouth

        # Create Praat Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)

        # Get intensity contour
        intensity = sound.to_intensity(
            minimum_pitch=100.0,  # Affects window size
            time_step=0.005      # 5ms frames
        )

        # Get intensity values
        intensity_values = intensity.values[0]

        # Filter out undefined values
        valid_intensity = intensity_values[~np.isnan(intensity_values)]
        valid_intensity = valid_intensity[valid_intensity > -200]  # Filter out silence markers

        if len(valid_intensity) == 0:
            return _intensity_fallback(audio)

        mean_db = float(np.mean(valid_intensity))
        max_db = float(np.max(valid_intensity))
        min_db = float(np.min(valid_intensity))
        range_db = max_db - min_db

        return mean_db, max_db, min_db, range_db

    except ImportError:
        logger.warning("parselmouth not installed, using fallback intensity")
        return _intensity_fallback(audio)
    except Exception as e:
        logger.warning(f"Intensity analysis error: {e}")
        return _intensity_fallback(audio)


def _intensity_fallback(audio: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fallback intensity calculation using RMS.

    Converts RMS amplitude to approximate dB SPL.
    """
    if len(audio) < 10:
        return 0.0, 0.0, 0.0, 0.0

    try:
        # Calculate RMS in frames
        frame_size = 160  # ~10ms at 16kHz
        hop_size = 80

        num_frames = (len(audio) - frame_size) // hop_size + 1
        if num_frames < 1:
            # Single RMS value
            rms = np.sqrt(np.mean(audio ** 2))
            db = _rms_to_db(rms)
            return db, db, db, 0.0

        rms_values = []
        for i in range(num_frames):
            start = i * hop_size
            frame = audio[start:start + frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            rms_values.append(rms)

        rms_values = np.array(rms_values)

        # Filter out very low values (silence)
        valid_rms = rms_values[rms_values > 1e-6]
        if len(valid_rms) == 0:
            valid_rms = rms_values

        db_values = np.array([_rms_to_db(r) for r in valid_rms])

        mean_db = float(np.mean(db_values))
        max_db = float(np.max(db_values))
        min_db = float(np.min(db_values))
        range_db = max_db - min_db

        return mean_db, max_db, min_db, range_db

    except Exception as e:
        logger.warning(f"Intensity fallback error: {e}")
        return 0.0, 0.0, 0.0, 0.0


def _rms_to_db(rms: float, ref: float = 1.0) -> float:
    """
    Convert RMS amplitude to decibels.

    Args:
        rms: RMS amplitude value
        ref: Reference amplitude (default 1.0 for normalized audio)

    Returns:
        dB value (approximately)
    """
    if rms <= 0:
        return -100.0  # Floor value for silence

    # dB = 20 * log10(rms / ref)
    # For normalized audio, this gives values typically in range -60 to 0
    # We add an offset to make it more like typical intensity values (40-80 dB range)
    db = 20 * np.log10(rms / ref)

    # Add offset to convert to approximate dB SPL-like scale
    # This is a rough approximation; actual SPL depends on recording conditions
    db_adjusted = db + 70  # Shift to typical speech range

    return float(db_adjusted)


def compute_intensity_contour(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 10.0
) -> np.ndarray:
    """
    Compute intensity contour for visualization.

    Args:
        audio: Audio samples
        sr: Sample rate
        frame_ms: Frame size in milliseconds

    Returns:
        Array of intensity values in dB
    """
    frame_size = int(frame_ms * sr / 1000)
    hop_size = frame_size // 2

    num_frames = (len(audio) - frame_size) // hop_size + 1
    if num_frames < 1:
        return np.array([])

    intensities = []
    for i in range(num_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        db = _rms_to_db(rms)
        intensities.append(db)

    return np.array(intensities)
