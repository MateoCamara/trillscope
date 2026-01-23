"""Trill cycle detection algorithm."""

import numpy as np
from typing import Tuple, List
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import uniform_filter1d
import logging

logger = logging.getLogger(__name__)


def detect_trill_cycles(
    audio: np.ndarray,
    sr: int,
    min_cycle_duration_ms: float = 10.0,
    max_cycle_duration_ms: float = 60.0,
    envelope_smoothing_ms: float = 5.0,
    min_prominence: float = 0.1
) -> Tuple[int, List[float], float]:
    """
    Detect occlusion-release cycles in trill audio.

    Algorithm:
    1. Compute amplitude envelope using Hilbert transform
    2. Smooth envelope to reduce noise
    3. Find local minima (occlusions) between maxima (releases)
    4. Filter by timing constraints and prominence
    5. Count valid cycles

    Args:
        audio: Audio samples (normalized to [-1, 1])
        sr: Sample rate
        min_cycle_duration_ms: Minimum time between cycle peaks (ms)
        max_cycle_duration_ms: Maximum time between cycle peaks (ms)
        envelope_smoothing_ms: Smoothing window size (ms)
        min_prominence: Minimum peak prominence (fraction of max envelope)

    Returns:
        Tuple of:
        - num_cycles: Number of detected cycles
        - intervals_ms: Inter-cycle intervals in milliseconds
        - regularity: Coefficient of variation of intervals (0=perfect regularity)
    """
    if len(audio) < 10:
        return 0, [], 0.0

    try:
        # Step 1: Compute amplitude envelope using Hilbert transform
        analytic_signal = hilbert(audio)
        envelope = np.abs(analytic_signal)

        # Step 2: Smooth the envelope
        smoothing_samples = max(1, int(envelope_smoothing_ms * sr / 1000))
        envelope_smooth = uniform_filter1d(envelope, size=smoothing_samples)

        # Normalize envelope
        env_max = envelope_smooth.max()
        if env_max == 0:
            return 0, [], 0.0
        envelope_norm = envelope_smooth / env_max

        # Step 3: Calculate distance constraints in samples
        min_distance = int(min_cycle_duration_ms * sr / 1000)
        max_distance = int(max_cycle_duration_ms * sr / 1000)

        # Step 4: Find peaks (releases) in the envelope
        # Prominence threshold based on envelope amplitude
        prominence = min_prominence * env_max

        peaks, peak_props = find_peaks(
            envelope_smooth,
            distance=min_distance,
            prominence=prominence
        )

        if len(peaks) < 2:
            # Less than 2 peaks means 0 or 1 cycle
            return max(0, len(peaks)), [], 0.0

        # Step 5: Calculate inter-peak intervals
        intervals_samples = np.diff(peaks)
        intervals_ms = (intervals_samples / sr * 1000).tolist()

        # Filter intervals that are too long (suggesting missing cycles)
        valid_intervals = [i for i in intervals_ms if i <= max_cycle_duration_ms]

        # Number of cycles = number of valid peaks - 1 (or number of valid intervals)
        # If we have peaks, each peak-to-peak is one cycle
        num_cycles = len(valid_intervals)

        # Calculate regularity (coefficient of variation)
        if len(valid_intervals) >= 2:
            mean_interval = np.mean(valid_intervals)
            std_interval = np.std(valid_intervals)
            regularity = std_interval / mean_interval if mean_interval > 0 else 0.0
        else:
            regularity = 0.0

        return num_cycles, valid_intervals, regularity

    except Exception as e:
        logger.warning(f"Cycle detection error: {e}")
        return 0, [], 0.0


def compute_cycle_rate(num_cycles: int, duration_ms: float) -> float:
    """
    Compute cycle rate in Hz.

    Args:
        num_cycles: Number of detected cycles
        duration_ms: Duration of the segment in milliseconds

    Returns:
        Cycle rate in Hz (cycles per second)
    """
    if duration_ms <= 0 or num_cycles <= 0:
        return 0.0

    duration_sec = duration_ms / 1000.0
    return num_cycles / duration_sec


def detect_cycles_energy_based(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 5.0,
    min_cycle_ms: float = 15.0,
    max_cycle_ms: float = 50.0
) -> Tuple[int, List[float]]:
    """
    Alternative cycle detection using frame-based energy.

    This method computes short-time energy and looks for
    periodic dips (occlusions) in the energy contour.

    Args:
        audio: Audio samples
        sr: Sample rate
        frame_ms: Frame size in ms
        min_cycle_ms: Minimum cycle duration
        max_cycle_ms: Maximum cycle duration

    Returns:
        Tuple of (num_cycles, intervals_ms)
    """
    frame_samples = int(frame_ms * sr / 1000)
    hop_samples = frame_samples // 2

    # Compute frame energies
    num_frames = (len(audio) - frame_samples) // hop_samples + 1
    if num_frames < 3:
        return 0, []

    energies = []
    for i in range(num_frames):
        start = i * hop_samples
        frame = audio[start:start + frame_samples]
        energy = np.sqrt(np.mean(frame ** 2))
        energies.append(energy)

    energies = np.array(energies)

    # Normalize
    if energies.max() == 0:
        return 0, []
    energies_norm = energies / energies.max()

    # Find minima (occlusions) - invert and find peaks
    min_distance_frames = int(min_cycle_ms / (hop_samples / sr * 1000))

    # Find valleys by finding peaks in inverted signal
    valleys, _ = find_peaks(
        1 - energies_norm,
        distance=max(1, min_distance_frames),
        prominence=0.1
    )

    if len(valleys) < 2:
        return 0, []

    # Calculate intervals
    intervals_frames = np.diff(valleys)
    intervals_ms = (intervals_frames * hop_samples / sr * 1000).tolist()

    # Filter by max cycle duration
    valid_intervals = [i for i in intervals_ms if i <= max_cycle_ms]
    num_cycles = len(valid_intervals)

    return num_cycles, valid_intervals
