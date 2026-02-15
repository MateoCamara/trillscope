"""Trill cycle detection algorithm.

Provides three complementary sub-detectors and a consensus mechanism:
  (A) Band-pass filtered envelope: isolates 500-4000 Hz to remove voicing energy
  (B) Spectrogram intensity valley detection: finds periodic energy dips
  (C) Envelope autocorrelation: estimates dominant periodicity

The consensus mechanism combines results for robust cycle counting.
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.signal import hilbert, find_peaks, butter, sosfilt, stft
from scipy.ndimage import uniform_filter1d
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-detector A: Band-pass filtered envelope
# ---------------------------------------------------------------------------

def _detect_bandpass_envelope(
    audio: np.ndarray,
    sr: int,
    bp_low: float = 500.0,
    bp_high: float = 4000.0,
    filter_order: int = 4,
    min_cycle_ms: float = 10.0,
    max_cycle_ms: float = 60.0,
    smoothing_ms: float = 5.0,
    min_prominence: float = 0.1,
) -> Tuple[int, List[float], Optional[np.ndarray]]:
    """
    Detect cycles using Hilbert envelope on band-pass filtered audio.

    Band-pass filtering to 500-4000 Hz removes low-frequency voicing energy
    that masks occlusion dips, improving contrast between occlusion and release.

    Returns:
        (num_cycles, intervals_ms, peak_positions_samples)
    """
    if len(audio) < 10:
        return 0, [], None

    try:
        # Clamp bp_high to Nyquist
        nyquist = sr / 2.0
        bp_high_clamped = min(bp_high, nyquist * 0.95)
        if bp_low >= bp_high_clamped:
            # Fallback: skip filtering if impossible
            filtered = audio
        else:
            sos = butter(filter_order, [bp_low, bp_high_clamped], btype='band',
                         fs=sr, output='sos')
            filtered = sosfilt(sos, audio)

        # Hilbert envelope
        analytic = hilbert(filtered)
        envelope = np.abs(analytic)

        # Smooth
        smooth_samples = max(1, int(smoothing_ms * sr / 1000))
        envelope_smooth = uniform_filter1d(envelope, size=smooth_samples)

        env_max = envelope_smooth.max()
        if env_max == 0:
            return 0, [], None

        # Peak detection
        min_distance = max(1, int(min_cycle_ms * sr / 1000))
        prominence = min_prominence * env_max

        peaks, _ = find_peaks(
            envelope_smooth,
            distance=min_distance,
            prominence=prominence,
        )

        if len(peaks) < 2:
            return max(0, len(peaks)), [], peaks if len(peaks) else None

        # Intervals
        intervals_samples = np.diff(peaks)
        intervals_ms = (intervals_samples / sr * 1000).tolist()

        # Filter intervals exceeding max_cycle_ms
        valid_intervals = [i for i in intervals_ms if i <= max_cycle_ms]
        num_cycles = len(valid_intervals)

        return num_cycles, valid_intervals, peaks

    except Exception as e:
        logger.debug(f"Bandpass envelope detection error: {e}")
        return 0, [], None


# ---------------------------------------------------------------------------
# Sub-detector B: Spectrogram intensity valley detection
# ---------------------------------------------------------------------------

def _detect_spectrogram_valleys(
    audio: np.ndarray,
    sr: int,
    freq_low: float = 500.0,
    freq_high: float = 4000.0,
    win_ms: float = 4.0,
    hop_ms: float = 1.0,
    min_cycle_ms: float = 10.0,
    max_cycle_ms: float = 60.0,
    min_prominence: float = 0.1,
) -> Tuple[int, List[float], Optional[np.ndarray]]:
    """
    Detect cycles by finding periodic intensity valleys in the spectrogram.

    Occlusions appear as vertical intensity reductions across formant bands.
    Summing energy in 500-4000 Hz produces a contour where valleys = occlusions.

    Returns:
        (num_cycles, intervals_ms, valley_positions_frames)
    """
    if len(audio) < 10:
        return 0, [], None

    try:
        nperseg = max(4, int(win_ms * sr / 1000))
        # Ensure nperseg is even for FFT efficiency
        nperseg = nperseg + (nperseg % 2)
        hop_samples = max(1, int(hop_ms * sr / 1000))
        noverlap = nperseg - hop_samples

        # Guard: noverlap must be < nperseg and >= 0
        noverlap = max(0, min(noverlap, nperseg - 1))

        # Need enough audio for at least one window
        if len(audio) < nperseg:
            return 0, [], None

        freqs, times, Zxx = stft(audio, fs=sr, nperseg=nperseg,
                                 noverlap=noverlap, window='hann')

        # Power spectrogram
        power = np.abs(Zxx) ** 2

        # Select frequency band
        freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
        if not np.any(freq_mask):
            return 0, [], None

        # Band energy contour (sum across selected frequencies)
        band_energy = power[freq_mask, :].sum(axis=0)

        if len(band_energy) < 3:
            return 0, [], None

        # Smooth contour
        smooth_frames = max(1, int(3.0 / (hop_ms if hop_ms > 0 else 1.0)))
        band_smooth = uniform_filter1d(band_energy, size=smooth_frames)

        be_max = band_smooth.max()
        if be_max == 0:
            return 0, [], None

        # Find valleys (occlusions) by inverting and finding peaks
        min_distance_frames = max(1, int(min_cycle_ms / hop_ms))
        prominence = min_prominence * be_max

        valleys, _ = find_peaks(
            -band_smooth,
            distance=min_distance_frames,
            prominence=prominence,
        )

        if len(valleys) < 2:
            return 0, [], valleys if len(valleys) else None

        # Convert valley positions to time intervals
        intervals_frames = np.diff(valleys)
        intervals_ms = (intervals_frames * hop_ms).tolist()

        valid_intervals = [i for i in intervals_ms if i <= max_cycle_ms]
        num_cycles = len(valid_intervals)

        return num_cycles, valid_intervals, valleys

    except Exception as e:
        logger.debug(f"Spectrogram valley detection error: {e}")
        return 0, [], None


# ---------------------------------------------------------------------------
# Sub-detector C: Envelope autocorrelation
# ---------------------------------------------------------------------------

def _detect_autocorrelation(
    audio: np.ndarray,
    sr: int,
    min_cycle_ms: float = 10.0,
    max_cycle_ms: float = 60.0,
    smoothing_ms: float = 3.0,
) -> Tuple[int, float, float]:
    """
    Estimate cycle count via autocorrelation of the amplitude envelope.

    Rather than finding individual peaks, this measures the dominant
    periodicity. The autocorrelation peak height indicates regularity.

    Returns:
        (estimated_cycles, estimated_period_ms, acf_peak_height)
        acf_peak_height is 0-1, where higher = more periodic/regular.
    """
    if len(audio) < 10:
        return 0, 0.0, 0.0

    try:
        # Envelope
        analytic = hilbert(audio)
        envelope = np.abs(analytic)

        # Smooth
        smooth_samples = max(1, int(smoothing_ms * sr / 1000))
        envelope_smooth = uniform_filter1d(envelope, size=smooth_samples)

        # Mean-subtract for autocorrelation
        env_centered = envelope_smooth - np.mean(envelope_smooth)
        norm = np.dot(env_centered, env_centered)
        if norm == 0:
            return 0, 0.0, 0.0

        # Normalized autocorrelation via correlate
        acf_full = np.correlate(env_centered, env_centered, mode='full')
        acf = acf_full[len(env_centered) - 1:]  # positive lags only
        acf = acf / norm  # normalize so acf[0] = 1.0

        # Search range: lags corresponding to min_cycle_ms .. max_cycle_ms
        min_lag = max(1, int(min_cycle_ms * sr / 1000))
        max_lag = min(len(acf) - 1, int(max_cycle_ms * sr / 1000))

        if min_lag >= max_lag:
            return 0, 0.0, 0.0

        acf_search = acf[min_lag:max_lag + 1]

        if len(acf_search) == 0:
            return 0, 0.0, 0.0

        # Find first significant peak in the search range
        peaks, props = find_peaks(acf_search, prominence=0.05)

        if len(peaks) == 0:
            return 0, 0.0, 0.0

        # Take the first (lowest-lag) peak = fundamental period
        first_peak_idx = peaks[0]
        best_lag = min_lag + first_peak_idx
        acf_peak_height = float(acf[best_lag])

        # Estimated period
        period_ms = best_lag / sr * 1000.0

        # Estimated cycle count = total duration / period
        total_duration_ms = len(audio) / sr * 1000.0
        estimated_cycles = int(round(total_duration_ms / period_ms)) - 1
        estimated_cycles = max(0, estimated_cycles)

        return estimated_cycles, period_ms, max(0.0, acf_peak_height)

    except Exception as e:
        logger.debug(f"Autocorrelation detection error: {e}")
        return 0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Consensus logic
# ---------------------------------------------------------------------------

def _consensus(
    count_a: int,
    intervals_a: List[float],
    count_b: int,
    intervals_b: List[float],
    count_c: int,
    acf_height: float,
) -> Tuple[int, List[float], float, float]:
    """
    Combine results from three sub-detectors into a consensus.

    Rules:
    1. All 3 agree within +/-1: median count, confidence 1.0
    2. 2 of 3 agree within +/-1: agreed pair mean (rounded), confidence 0.7
    3. No pair agrees: use method A as primary, confidence 0.3

    For intervals and regularity, prefer the method whose count matches
    the consensus (A > B > C).

    Returns:
        (num_cycles, intervals_ms, regularity, confidence)
    """
    counts = [count_a, count_b, count_c]

    def _agree(x, y):
        return abs(x - y) <= 1

    all_agree = _agree(count_a, count_b) and _agree(count_b, count_c) and _agree(count_a, count_c)
    ab_agree = _agree(count_a, count_b)
    ac_agree = _agree(count_a, count_c)
    bc_agree = _agree(count_b, count_c)

    if all_agree:
        # All three agree: use median
        final_count = int(np.median(counts))
        confidence = 1.0
    elif ab_agree:
        final_count = round((count_a + count_b) / 2.0)
        confidence = 0.7
    elif ac_agree:
        final_count = round((count_a + count_c) / 2.0)
        confidence = 0.7
    elif bc_agree:
        final_count = round((count_b + count_c) / 2.0)
        confidence = 0.7
    else:
        # No agreement: use method A
        final_count = count_a
        confidence = 0.3

    # Select intervals from matching method (prefer A, then B)
    if final_count == count_a or _agree(final_count, count_a):
        intervals = intervals_a
    elif final_count == count_b or _agree(final_count, count_b):
        intervals = intervals_b
    else:
        intervals = intervals_a  # fallback

    # Regularity from intervals
    if len(intervals) >= 2:
        mean_iv = np.mean(intervals)
        std_iv = np.std(intervals)
        regularity = float(std_iv / mean_iv) if mean_iv > 0 else 0.0
    else:
        regularity = 0.0

    return final_count, intervals, regularity, confidence


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_trill_cycles(
    audio: np.ndarray,
    sr: int,
    min_cycle_duration_ms: float = 10.0,
    max_cycle_duration_ms: float = 60.0,
    envelope_smoothing_ms: float = 5.0,
    min_prominence: float = 0.1,
    method: str = 'multi',
    bp_low: float = 500.0,
    bp_high: float = 4000.0,
    bp_filter_order: int = 4,
    spec_win_ms: float = 4.0,
    spec_hop_ms: float = 1.0,
) -> Tuple[int, List[float], float, float]:
    """
    Detect occlusion-release cycles in trill audio.

    Args:
        audio: Audio samples (normalized to [-1, 1])
        sr: Sample rate
        min_cycle_duration_ms: Minimum time between cycle peaks (ms)
        max_cycle_duration_ms: Maximum time between cycle peaks (ms)
        envelope_smoothing_ms: Smoothing window size (ms)
        min_prominence: Minimum peak prominence (fraction of max envelope)
        method: Detection method - 'multi' (3 detectors + consensus),
                'envelope' (bandpass envelope only), 'legacy' (original Hilbert)
        bp_low: Band-pass low cutoff Hz (for 'multi' and 'envelope')
        bp_high: Band-pass high cutoff Hz (for 'multi' and 'envelope')
        bp_filter_order: Butterworth filter order
        spec_win_ms: STFT window size ms (for spectrogram detector)
        spec_hop_ms: STFT hop size ms (for spectrogram detector)

    Returns:
        Tuple of:
        - num_cycles: Number of detected cycles
        - intervals_ms: Inter-cycle intervals in milliseconds
        - regularity: Coefficient of variation of intervals (0=perfect)
        - confidence: Consensus confidence score (0.0-1.0)
    """
    if len(audio) < 10:
        return 0, [], 0.0, 1.0

    if method == 'legacy':
        return _detect_legacy(
            audio, sr,
            min_cycle_duration_ms, max_cycle_duration_ms,
            envelope_smoothing_ms, min_prominence,
        )

    if method == 'envelope':
        count, intervals, _ = _detect_bandpass_envelope(
            audio, sr,
            bp_low=bp_low, bp_high=bp_high, filter_order=bp_filter_order,
            min_cycle_ms=min_cycle_duration_ms,
            max_cycle_ms=max_cycle_duration_ms,
            smoothing_ms=envelope_smoothing_ms,
            min_prominence=min_prominence,
        )
        regularity = _compute_regularity(intervals)
        return count, intervals, regularity, 0.5  # single-method confidence

    # method == 'multi': run all three detectors + consensus
    total_duration_ms = len(audio) / sr * 1000.0

    # (A) Band-pass envelope
    count_a, intervals_a, _ = _detect_bandpass_envelope(
        audio, sr,
        bp_low=bp_low, bp_high=bp_high, filter_order=bp_filter_order,
        min_cycle_ms=min_cycle_duration_ms,
        max_cycle_ms=max_cycle_duration_ms,
        smoothing_ms=envelope_smoothing_ms,
        min_prominence=min_prominence,
    )

    # (B) Spectrogram valleys
    count_b, intervals_b, _ = _detect_spectrogram_valleys(
        audio, sr,
        freq_low=bp_low, freq_high=bp_high,
        win_ms=spec_win_ms, hop_ms=spec_hop_ms,
        min_cycle_ms=min_cycle_duration_ms,
        max_cycle_ms=max_cycle_duration_ms,
        min_prominence=min_prominence,
    )

    # (C) Autocorrelation - skip for very short segments (<20ms)
    if total_duration_ms >= 20.0:
        count_c, period_ms, acf_height = _detect_autocorrelation(
            audio, sr,
            min_cycle_ms=min_cycle_duration_ms,
            max_cycle_ms=max_cycle_duration_ms,
            smoothing_ms=3.0,
        )
    else:
        count_c, period_ms, acf_height = 0, 0.0, 0.0

    # Consensus
    num_cycles, intervals, regularity, confidence = _consensus(
        count_a, intervals_a,
        count_b, intervals_b,
        count_c, acf_height,
    )

    return num_cycles, intervals, regularity, confidence


# ---------------------------------------------------------------------------
# Legacy method (original Hilbert-only, preserved for comparison)
# ---------------------------------------------------------------------------

def _detect_legacy(
    audio: np.ndarray,
    sr: int,
    min_cycle_duration_ms: float = 10.0,
    max_cycle_duration_ms: float = 60.0,
    envelope_smoothing_ms: float = 5.0,
    min_prominence: float = 0.1,
) -> Tuple[int, List[float], float, float]:
    """Original Hilbert envelope method (no band-pass filtering)."""
    try:
        analytic_signal = hilbert(audio)
        envelope = np.abs(analytic_signal)

        smoothing_samples = max(1, int(envelope_smoothing_ms * sr / 1000))
        envelope_smooth = uniform_filter1d(envelope, size=smoothing_samples)

        env_max = envelope_smooth.max()
        if env_max == 0:
            return 0, [], 0.0, 1.0

        min_distance = int(min_cycle_duration_ms * sr / 1000)
        prominence = min_prominence * env_max

        peaks, _ = find_peaks(
            envelope_smooth,
            distance=min_distance,
            prominence=prominence,
        )

        if len(peaks) < 2:
            return max(0, len(peaks)), [], 0.0, 0.5

        intervals_samples = np.diff(peaks)
        intervals_ms = (intervals_samples / sr * 1000).tolist()
        valid_intervals = [i for i in intervals_ms if i <= max_cycle_duration_ms]
        num_cycles = len(valid_intervals)

        regularity = _compute_regularity(valid_intervals)

        return num_cycles, valid_intervals, regularity, 0.5

    except Exception as e:
        logger.warning(f"Legacy cycle detection error: {e}")
        return 0, [], 0.0, 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_regularity(intervals: List[float]) -> float:
    """Compute coefficient of variation of intervals."""
    if len(intervals) < 2:
        return 0.0
    mean_iv = np.mean(intervals)
    std_iv = np.std(intervals)
    return float(std_iv / mean_iv) if mean_iv > 0 else 0.0


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
