"""Voice Activity Detection for SNR estimation."""

from typing import Tuple
import numpy as np

from .metrics import calculate_frame_energies


def energy_based_vad(
    frame_energies_db: np.ndarray,
    speech_threshold_db: float = -35.0,
    noise_threshold_db: float = -50.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple energy-based VAD to classify frames as speech or noise.

    Uses a two-threshold approach:
    - Frames above speech_threshold_db are speech
    - Frames below noise_threshold_db are noise
    - Frames in between are ambiguous (excluded from SNR calc)

    Args:
        frame_energies_db: Frame energies in dB
        speech_threshold_db: Threshold above which frames are speech
        noise_threshold_db: Threshold below which frames are noise

    Returns:
        Tuple of (speech_mask, noise_mask) as boolean arrays
    """
    speech_mask = frame_energies_db >= speech_threshold_db
    noise_mask = frame_energies_db <= noise_threshold_db

    return speech_mask, noise_mask


def calculate_snr_vad(
    samples: np.ndarray,
    sample_rate: int,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0,
    speech_threshold_db: float = -35.0,
    noise_threshold_db: float = -50.0,
    min_speech_frames: int = 10,
    min_noise_frames: int = 5
) -> Tuple[float, float, float]:
    """
    Estimate SNR using VAD-based separation of speech and noise.

    Algorithm:
    1. Compute frame-wise energies
    2. Use energy-based VAD to classify frames
    3. Average energy of speech frames vs noise frames
    4. SNR = 10 * log10(speech_energy / noise_energy)

    Args:
        samples: Audio samples normalized to [-1, 1]
        sample_rate: Sample rate
        frame_size_ms: Frame length for analysis
        hop_size_ms: Hop between frames
        speech_threshold_db: VAD speech threshold
        noise_threshold_db: VAD noise threshold
        min_speech_frames: Minimum speech frames required for valid estimate
        min_noise_frames: Minimum noise frames required for valid estimate

    Returns:
        Tuple of (snr_db, speech_energy_db, noise_energy_db)
        Returns (float('inf'), speech_energy, -100.0) if no noise detected
        Returns (float('-inf'), -100.0, noise_energy) if no speech detected
    """
    # Calculate frame energies
    frame_energies_db = calculate_frame_energies(
        samples, sample_rate, frame_size_ms, hop_size_ms
    )

    # Classify frames
    speech_mask, noise_mask = energy_based_vad(
        frame_energies_db, speech_threshold_db, noise_threshold_db
    )

    # Count frames
    num_speech = np.sum(speech_mask)
    num_noise = np.sum(noise_mask)

    # Calculate energies (convert from dB back to linear for averaging)
    speech_energies_linear = 10 ** (frame_energies_db[speech_mask] / 10) if num_speech > 0 else []
    noise_energies_linear = 10 ** (frame_energies_db[noise_mask] / 10) if num_noise > 0 else []

    # Handle edge cases
    if num_speech < min_speech_frames:
        # Not enough speech detected
        noise_energy_db = (
            10 * np.log10(np.mean(noise_energies_linear))
            if num_noise >= min_noise_frames
            else -100.0
        )
        return float('-inf'), -100.0, noise_energy_db

    if num_noise < min_noise_frames:
        # No noise detected - very clean signal
        speech_energy_db = 10 * np.log10(np.mean(speech_energies_linear))
        return float('inf'), speech_energy_db, -100.0

    # Calculate average energies
    speech_energy_linear = np.mean(speech_energies_linear)
    noise_energy_linear = np.mean(noise_energies_linear)

    speech_energy_db = 10 * np.log10(speech_energy_linear)
    noise_energy_db = 10 * np.log10(noise_energy_linear)

    # Calculate SNR
    snr_db = speech_energy_db - noise_energy_db

    return snr_db, speech_energy_db, noise_energy_db
