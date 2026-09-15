"""Closure-based trill detector.

Counts occlusion–release events rather than envelope peaks. A closure is a
local energy minimum of the band-limited envelope (by default the 500–3500 Hz
band, which carries turbulence and release bursts) that is followed shortly
by a mid-band release burst. This anchors the count to the same object that
the phonetic literature counts manually on spectrograms (Quilis 1993, Blecua
2001, Henriksen & Willis 2010, Bradley & Willis 2012, Henriksen et al. 2023).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class DetectorConfig:
    """Detector parameters. Physical-scale constants come first.

    The defaults are conservative. The parameters used for the published
    analysis live in ``trillscope/config/detector.yaml``; load them with
    :func:`trillscope.detector.load_config`.
    """

    # Frame analysis
    frame_ms: float = 2.5

    # Band-pass design (Butterworth, order 4)
    low_band: tuple[float, float] = (60.0, 500.0)
    mid_band: tuple[float, float] = (500.0, 3500.0)

    # Closure candidate detection
    min_inter_closure_ms: float = 15.0   # taps are ~18–30 ms; closures separated by ≥ 15 ms
    max_inter_closure_ms: float = 50.0   # full periods rarely exceed 50 ms in trills
    closure_window_ms: float = 50.0      # window for local-max normalisation
    closure_threshold_pct: float = 0.25  # min must drop below 25 % of local max (~ -12 dB)
    closure_prominence_db: float = 8.0   # min prominence vs. neighbours

    # Release verification
    release_min_lag_ms: float = 3.0
    release_max_lag_ms: float = 25.0
    release_threshold_pct: float = 0.40  # mid-band peak ≥ 40 % of local mid max

    # Periodicity refinement
    enable_periodicity_refinement: bool = True
    period_z_threshold: float = 2.0      # reject intervals > z SD from inferred period

    # Token-level boundary expansion (to absorb MFA alignment error)
    boundary_expand_ms: float = 20.0

    # Which envelope to inspect for closure candidates.
    # - "mid":      use only the 500-3500 Hz turbulence band. Default.
    #               Detects voiced-through trills (lab-quality ALBAYZIN style)
    #               where voicing continues during occlusion.
    # - "combined": geometric mean of low + mid envelopes.
    #               Requires both bands to drop; only triggers when voicing
    #               actually stops during the closure (casual-speech pattern).
    closure_envelope: str = "mid"


@dataclass(frozen=True)
class Closure:
    """A single occlusion–release event."""

    closure_t_ms: float          # time of the closure (energy minimum)
    release_t_ms: float          # time of the release burst
    depth_db: float              # how much energy drops at the closure
    interval_to_next_ms: float = float("nan")  # filled later


@dataclass(frozen=True)
class DetectionResult:
    """Detection result for a single token."""

    closures: list[Closure]
    confidence: float            # in [0, 1]
    notes: list[str] = field(default_factory=list)

    @property
    def n_closures(self) -> int:
        return len(self.closures)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _rms_envelope(x: np.ndarray, sr: int, frame_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Frame-wise RMS. Returns (envelope, time_axis_ms)."""
    hop = max(1, int(round(sr * frame_ms / 1000.0)))
    win = hop  # non-overlapping
    n_frames = len(x) // hop
    if n_frames == 0:
        return np.array([]), np.array([])
    # vectorised
    trim = n_frames * hop
    frames = x[:trim].reshape(n_frames, win)
    env = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    t = np.arange(n_frames) * frame_ms + frame_ms / 2.0
    return env, t


def _to_db(x: np.ndarray, ref: float | None = None) -> np.ndarray:
    ref = float(np.max(x)) if ref is None else ref
    ref = max(ref, 1e-12)
    return 20.0 * np.log10(np.maximum(x, 1e-12) / ref)


def _bandpass(x: np.ndarray, sr: int, band: tuple[float, float]) -> np.ndarray:
    low, high = band
    nyq = 0.5 * sr
    # clamp to nyquist
    high = min(high, nyq * 0.99)
    if low <= 0:
        # low <= 0: low-pass filter only
        b, a = signal.butter(4, high / nyq, btype="low")
    else:
        b, a = signal.butter(4, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def _find_closure_candidates(
    env_db: np.ndarray,
    t_ms: np.ndarray,
    cfg: DetectorConfig,
) -> list[int]:
    """Return frame indices of closure candidates (local minima in the envelope)."""
    if len(env_db) < 5:
        return []
    # Closure = local minimum of -env_db = local maximum of (-env_db)
    inverted = -env_db
    if len(t_ms) < 2:
        return []
    frame_ms = float(t_ms[1] - t_ms[0])
    distance = max(1, int(round(cfg.min_inter_closure_ms / frame_ms)))
    # NOTE: do not pass prominence= here. signal.find_peaks measures prominence
    # relative to adjacent saddle points, which collapses to a small value on
    # uniformly periodic envelopes (every dip's neighbour is another equally
    # deep dip, so each one looks unprominent). For trills, that erroneously
    # killed the candidate detection. Instead we gate with the
    # closure_threshold_pct test below, which compares each candidate to the
    # local max within a window — a more robust criterion for periodic signals.
    # Use a very small prominence (1 dB) only to filter out noise-floor wiggles.
    # The real prominence gate is the manual drop-from-local-max check below.
    peaks, _props = signal.find_peaks(inverted, distance=distance, prominence=1.0)
    win_frames = max(2, int(round(cfg.closure_window_ms / frame_ms)))
    kept: list[int] = []
    for p in peaks:
        lo = max(0, p - win_frames // 2)
        hi = min(len(env_db), p + win_frames // 2 + 1)
        local_max_lin = float(np.max(10.0 ** (env_db[lo:hi] / 20.0)))
        local_val_lin = float(10.0 ** (env_db[p] / 20.0))
        if local_max_lin <= 0:
            continue
        ratio = local_val_lin / local_max_lin
        if ratio > cfg.closure_threshold_pct:
            continue
        # Re-check prominence manually: drop in dB from the local max to the
        # candidate must be at least closure_prominence_db. This is the same
        # idea as scipy's prominence but referenced to the window's max instead
        # of adjacent saddles, so periodic dips are not penalised.
        drop_db = float(env_db[lo:hi].max()) - float(env_db[p])
        if drop_db < cfg.closure_prominence_db:
            continue
        kept.append(int(p))
    return kept


def _verify_releases(
    closure_frames: list[int],
    env_mid: np.ndarray,
    t_ms: np.ndarray,
    cfg: DetectorConfig,
) -> list[tuple[int, int]]:
    """For each closure, find a verified release peak in the mid band."""
    if not closure_frames:
        return []
    frame_ms = float(t_ms[1] - t_ms[0])
    lag_lo = int(round(cfg.release_min_lag_ms / frame_ms))
    lag_hi = int(round(cfg.release_max_lag_ms / frame_ms))
    verified: list[tuple[int, int]] = []
    for ci in closure_frames:
        lo = ci + lag_lo
        hi = min(len(env_mid), ci + lag_hi + 1)
        if lo >= len(env_mid) or hi <= lo:
            continue
        window = env_mid[lo:hi]
        # local context for normalisation: ± release_max_lag*2 around closure
        ctx_lo = max(0, ci - lag_hi * 2)
        ctx_hi = min(len(env_mid), ci + lag_hi * 2 + 1)
        local_max = float(np.max(env_mid[ctx_lo:ctx_hi]))
        if local_max <= 0:
            continue
        peak_idx = int(np.argmax(window))
        peak_val = float(window[peak_idx])
        if peak_val / local_max >= cfg.release_threshold_pct:
            verified.append((ci, lo + peak_idx))
    return verified


def _refine_by_periodicity(
    pairs: list[tuple[int, int]],
    t_ms: np.ndarray,
    cfg: DetectorConfig,
) -> tuple[list[tuple[int, int]], float]:
    """Reject closures whose interval to neighbours deviates > z SD from median period.

    Returns kept pairs and a regularity score in [0, 1].
    """
    if len(pairs) < 3 or not cfg.enable_periodicity_refinement:
        # regularity undefined; return everything with a neutral score
        return pairs, 0.5
    closure_times = np.array([t_ms[p[0]] for p in pairs])
    intervals = np.diff(closure_times)
    median_iv = float(np.median(intervals))
    std_iv = float(np.std(intervals)) + 1e-9
    # tag intervals as outliers
    z = np.abs(intervals - median_iv) / std_iv
    # mark closures to keep: closure i is dropped if BOTH adjacent intervals are outliers
    keep_mask = np.ones(len(pairs), dtype=bool)
    for i in range(len(pairs)):
        left_outlier = i > 0 and z[i - 1] > cfg.period_z_threshold
        right_outlier = i < len(pairs) - 1 and z[i] > cfg.period_z_threshold
        # if both adjacent intervals are outliers, this closure is likely spurious
        if left_outlier and right_outlier:
            keep_mask[i] = False
    kept = [pairs[i] for i in range(len(pairs)) if keep_mask[i]]
    # regularity score = 1 - clipped CV of intervals
    cv = std_iv / max(median_iv, 1e-6)
    regularity = float(np.clip(1.0 - cv, 0.0, 1.0))
    return kept, regularity


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_closures(
    audio: np.ndarray,
    sr: int,
    cfg: DetectorConfig | None = None,
    roi_ms: tuple[float, float] | None = None,
) -> DetectionResult:
    """Detect closures in a (mono, float) trill token.

    Parameters
    ----------
    audio : np.ndarray
        Mono float audio of the trill token. In production, callers should pass
        an excerpt that covers ±20 ms around the MFA/native boundary so the
        detector can see release energy that crosses the boundary, then pass
        the *core* boundary in ``roi_ms``.
    sr : int
        Sample rate. The detector does not resample; the caller must normalise.
    cfg : DetectorConfig, optional
    roi_ms : (start_ms, end_ms), optional
        Region within ``audio`` where closures are counted as part of the trill.
        Closures detected outside this window are discarded as boundary noise.
        If None, the full audio is considered ROI.

    Returns
    -------
    DetectionResult
    """
    cfg = cfg or DetectorConfig()
    notes: list[str] = []
    if audio.ndim != 1:
        audio = audio.mean(axis=-1)
    if len(audio) < int(sr * 0.030):
        notes.append("audio shorter than 30 ms; skipping")
        return DetectionResult(closures=[], confidence=0.0, notes=notes)

    # Two-band envelopes
    audio_low = _bandpass(audio, sr, cfg.low_band)
    audio_mid = _bandpass(audio, sr, cfg.mid_band)
    env_low, t_ms = _rms_envelope(audio_low, sr, cfg.frame_ms)
    env_mid, _ = _rms_envelope(audio_mid, sr, cfg.frame_ms)
    if cfg.closure_envelope == "combined":
        # Both bands must drop: only fires when voicing stops during the closure.
        env_for_closures = np.sqrt(env_low * env_mid + 1e-12)
    else:
        # Mid band only: detects closures even when voicing continues
        # through the constriction (lab-quality voiced-through trills).
        env_for_closures = env_mid
    env_db = _to_db(env_for_closures)
    env_combined_db = _to_db(np.sqrt(env_low * env_mid + 1e-12))

    # Closure candidates
    candidate_frames = _find_closure_candidates(env_db, t_ms, cfg)
    if cfg.closure_envelope == "mid" and candidate_frames:
        # Sanity check on the combined envelope: a real closure should also
        # show at least a small drop (>=3 dB) in the combined envelope. This
        # rejects boundary artifacts where the mid band drops because the
        # following segment is a voiced vowel (mid is naturally lower in
        # voicing-only audio), while the combined envelope barely changes.
        win_frames = max(2, int(round(cfg.closure_window_ms / (t_ms[1] - t_ms[0]))))
        filtered = []
        for p in candidate_frames:
            lo = max(0, p - win_frames // 2)
            hi = min(len(env_combined_db), p + win_frames // 2 + 1)
            local_max = float(env_combined_db[lo:hi].max())
            drop = local_max - float(env_combined_db[p])
            if drop >= 3.0:
                filtered.append(p)
        candidate_frames = filtered
    if not candidate_frames:
        notes.append("no closure candidates found")
        return DetectionResult(closures=[], confidence=0.0, notes=notes)

    # Release verification
    verified = _verify_releases(candidate_frames, env_mid, t_ms, cfg)
    if not verified:
        notes.append(f"{len(candidate_frames)} closure candidates but none had a verified release")
        return DetectionResult(closures=[], confidence=0.0, notes=notes)
    n_dropped_no_release = len(candidate_frames) - len(verified)
    if n_dropped_no_release:
        notes.append(f"dropped {n_dropped_no_release} candidates without valid release")

    # Periodicity refinement
    refined, regularity = _refine_by_periodicity(verified, t_ms, cfg)
    n_dropped_irregular = len(verified) - len(refined)
    if n_dropped_irregular:
        notes.append(f"dropped {n_dropped_irregular} candidates as periodicity outliers")

    # ROI filter: drop closures outside the requested region of interest
    if roi_ms is not None:
        roi_lo, roi_hi = roi_ms
        before = len(refined)
        refined = [(ci, ri) for ci, ri in refined if roi_lo <= float(t_ms[ci]) <= roi_hi]
        dropped_outside = before - len(refined)
        if dropped_outside:
            notes.append(f"dropped {dropped_outside} closures outside ROI [{roi_lo:.0f}, {roi_hi:.0f}] ms")

    # Build Closure objects
    closures: list[Closure] = []
    # local max for depth computation
    env_lin = 10.0 ** (env_db / 20.0)
    global_max = float(np.max(env_lin)) if len(env_lin) else 1e-12
    for i, (ci, ri) in enumerate(refined):
        # depth_db = drop below global max
        depth_db = float(-_to_db(np.array([env_lin[ci]]), ref=global_max)[0])
        iv_next = float("nan")
        if i + 1 < len(refined):
            iv_next = float(t_ms[refined[i + 1][0]] - t_ms[ci])
        closures.append(
            Closure(
                closure_t_ms=float(t_ms[ci]),
                release_t_ms=float(t_ms[ri]),
                depth_db=depth_db,
                interval_to_next_ms=iv_next,
            )
        )

    # Filter by max interval (no two valid closures should be > max_inter_closure_ms apart;
    # if they are, the segment is probably non-trill noise gluing two trills)
    filtered = [closures[0]] if closures else []
    for c in closures[1:]:
        if c.closure_t_ms - filtered[-1].closure_t_ms <= cfg.max_inter_closure_ms * 1.5:
            filtered.append(c)
        else:
            notes.append(
                f"gap {c.closure_t_ms - filtered[-1].closure_t_ms:.0f} ms > "
                f"{cfg.max_inter_closure_ms * 1.5:.0f} ms; truncating trill"
            )
            break
    closures = filtered

    # Confidence
    if not closures:
        confidence = 0.0
    else:
        depth_uniformity = 1.0 - float(np.std([c.depth_db for c in closures])) / (
            float(np.mean([c.depth_db for c in closures])) + 1e-6
        )
        depth_uniformity = float(np.clip(depth_uniformity, 0.0, 1.0))
        confidence = float(np.clip(0.5 * regularity + 0.5 * depth_uniformity, 0.0, 1.0))

    return DetectionResult(closures=closures, confidence=confidence, notes=notes)
