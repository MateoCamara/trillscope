"""Voicing analysis using Praat via parselmouth."""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def analyze_voicing(
    audio: np.ndarray,
    sr: int,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 500.0,
    voicing_threshold: float = 0.45
) -> Tuple[float, Optional[float], Optional[float], float]:
    """
    Analyze voicing characteristics using Praat algorithms.

    Args:
        audio: Audio samples (normalized to [-1, 1])
        sr: Sample rate
        pitch_floor: Minimum pitch in Hz
        pitch_ceiling: Maximum pitch in Hz
        voicing_threshold: Voicing threshold (Praat default: 0.45)

    Returns:
        Tuple of:
        - voicing_pct: Percentage of frames with voicing (0-100)
        - mean_f0: Mean F0 where voiced (None if fully unvoiced)
        - f0_range: F0 range (max - min, None if < 2 voiced frames)
        - mean_hnr: Mean harmonics-to-noise ratio in dB
    """
    if len(audio) < 100:
        return 0.0, None, None, 0.0

    try:
        import parselmouth
        from parselmouth.praat import call

        # Create Praat Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)

        # Pitch analysis
        pitch = sound.to_pitch(
            time_step=0.005,  # 5ms frames
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling
        )

        # Get pitch values
        pitch_values = pitch.selected_array['frequency']

        # Voiced frames have F0 > 0
        voiced_mask = pitch_values > 0
        total_frames = len(pitch_values)

        if total_frames == 0:
            voicing_pct = 0.0
        else:
            voicing_pct = (np.sum(voiced_mask) / total_frames) * 100.0

        # Calculate F0 statistics
        voiced_f0 = pitch_values[voiced_mask]

        if len(voiced_f0) >= 1:
            mean_f0 = float(np.mean(voiced_f0))
        else:
            mean_f0 = None

        if len(voiced_f0) >= 2:
            f0_range = float(np.max(voiced_f0) - np.min(voiced_f0))
        else:
            f0_range = None

        # Harmonicity (HNR) analysis
        try:
            harmonicity = sound.to_harmonicity()
            hnr_values = harmonicity.values[0]
            # Filter out undefined values (typically -200 dB)
            valid_hnr = hnr_values[hnr_values > -100]
            if len(valid_hnr) > 0:
                mean_hnr = float(np.mean(valid_hnr))
            else:
                mean_hnr = 0.0
        except Exception:
            mean_hnr = 0.0

        return voicing_pct, mean_f0, f0_range, mean_hnr

    except ImportError:
        logger.warning("parselmouth not installed, using fallback voicing detection")
        return _voicing_fallback(audio, sr)
    except Exception as e:
        logger.warning(f"Voicing analysis error: {e}")
        return 0.0, None, None, 0.0


def _voicing_fallback(
    audio: np.ndarray,
    sr: int
) -> Tuple[float, Optional[float], Optional[float], float]:
    """
    Fallback voicing detection using zero-crossing rate.

    This is a simple heuristic: voiced speech typically has
    lower zero-crossing rates than unvoiced speech.
    """
    try:
        # Calculate zero-crossing rate in frames
        frame_size = int(0.01 * sr)  # 10ms frames
        hop_size = frame_size // 2

        num_frames = (len(audio) - frame_size) // hop_size + 1
        if num_frames < 1:
            return 0.0, None, None, 0.0

        voiced_count = 0

        for i in range(num_frames):
            start = i * hop_size
            frame = audio[start:start + frame_size]

            # Zero-crossing rate
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))

            # Frame energy
            energy = np.sqrt(np.mean(frame ** 2))

            # Heuristic: voiced if low ZCR and sufficient energy
            if zcr < 0.15 and energy > 0.01:
                voiced_count += 1

        voicing_pct = (voiced_count / num_frames) * 100.0

        return voicing_pct, None, None, 0.0

    except Exception:
        return 0.0, None, None, 0.0


def analyze_voicing_librosa(
    audio: np.ndarray,
    sr: int,
    fmin: float = 75.0,
    fmax: float = 500.0
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Alternative voicing analysis using librosa's pyin.

    Args:
        audio: Audio samples
        sr: Sample rate
        fmin: Minimum F0
        fmax: Maximum F0

    Returns:
        Tuple of (voicing_pct, mean_f0, f0_range)
    """
    try:
        import librosa

        # Use pyin for probabilistic F0 estimation with voicing
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=2048,
            hop_length=512
        )

        if f0 is None or len(f0) == 0:
            return 0.0, None, None

        # Voicing percentage based on voiced flags
        voicing_pct = (np.sum(voiced_flag) / len(voiced_flag)) * 100.0

        # F0 statistics for voiced frames
        voiced_f0 = f0[voiced_flag]
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

        if len(voiced_f0) >= 1:
            mean_f0 = float(np.nanmean(voiced_f0))
        else:
            mean_f0 = None

        if len(voiced_f0) >= 2:
            f0_range = float(np.nanmax(voiced_f0) - np.nanmin(voiced_f0))
        else:
            f0_range = None

        return voicing_pct, mean_f0, f0_range

    except ImportError:
        logger.warning("librosa not installed for pyin")
        return 0.0, None, None
    except Exception as e:
        logger.warning(f"librosa voicing error: {e}")
        return 0.0, None, None
