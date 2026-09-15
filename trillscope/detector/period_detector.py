"""Independent autocorrelation-based n_closures estimator.

Used to cross-validate `detect_closures` (the v2 closure detector). The two
algorithms share only the bandpass / envelope front-end; the back-end is
deliberately different:

  - `detect_closures` finds individual closure events (energy minima) and
    counts them.
  - `detect_n_closures_by_period` finds the dominant cycle period via
    autocorrelation of the envelope and infers n_closures from the ROI
    duration divided by that period.

If both produce the same count (within +/-1) on the same token, that token's
closure count is unlikely to be a detector artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cycle_detector import _bandpass, _rms_envelope


@dataclass(frozen=True)
class PeriodDetectorConfig:
    """Parameters of the autocorrelation-based detector."""

    band: tuple[float, float] = (500.0, 3500.0)
    frame_ms: float = 2.5
    lag_min_ms: float = 20.0
    lag_max_ms: float = 60.0
    min_regularity: float = 0.20  # below this, treat as non-periodic (n=0)
    min_closure_spacing_ms: float = 15.0  # physical lower bound, mirrors v2


@dataclass(frozen=True)
class PeriodResult:
    n_closures: int
    period_ms: float
    regularity: float
    notes: list[str] = field(default_factory=list)


def detect_n_closures_by_period(
    audio: np.ndarray,
    sr: int,
    roi_ms: tuple[float, float] | None = None,
    cfg: PeriodDetectorConfig | None = None,
) -> PeriodResult:
    """Estimate the number of closures in a trill token by autocorrelation."""
    cfg = cfg or PeriodDetectorConfig()
    notes: list[str] = []
    if audio.ndim != 1:
        audio = audio.mean(axis=-1)

    if roi_ms is not None:
        i0 = max(0, int(round(roi_ms[0] / 1000.0 * sr)))
        i1 = int(round(roi_ms[1] / 1000.0 * sr))
        chunk = audio[i0:i1]
        roi_duration_ms = (roi_ms[1] - roi_ms[0])
    else:
        chunk = audio
        roi_duration_ms = len(audio) * 1000.0 / sr

    if len(chunk) < int(sr * 0.030) or roi_duration_ms < 30.0:
        notes.append("roi shorter than 30 ms")
        return PeriodResult(n_closures=0, period_ms=float("nan"), regularity=0.0, notes=notes)

    mid = _bandpass(chunk, sr, cfg.band)
    env, _t = _rms_envelope(mid, sr, cfg.frame_ms)
    if len(env) < 4:
        notes.append("envelope too short")
        return PeriodResult(n_closures=0, period_ms=float("nan"), regularity=0.0, notes=notes)

    env = env - env.mean()
    if float(np.dot(env, env)) <= 0:
        notes.append("envelope has zero variance")
        return PeriodResult(n_closures=0, period_ms=float("nan"), regularity=0.0, notes=notes)

    lag_lo = max(1, int(round(cfg.lag_min_ms / cfg.frame_ms)))
    lag_hi = min(len(env) - 1, int(round(cfg.lag_max_ms / cfg.frame_ms)))
    if lag_hi <= lag_lo:
        notes.append("lag window collapsed")
        return PeriodResult(n_closures=0, period_ms=float("nan"), regularity=0.0, notes=notes)

    acf = np.empty(lag_hi - lag_lo + 1, dtype=np.float64)
    n = len(env)
    for k, lag in enumerate(range(lag_lo, lag_hi + 1)):
        a = env[: n - lag]
        b = env[lag:]
        denom_a = float(np.dot(a, a))
        denom_b = float(np.dot(b, b))
        if denom_a <= 0 or denom_b <= 0:
            acf[k] = 0.0
        else:
            # Pearson-style normalised cross-correlation: length-invariant.
            acf[k] = float(np.dot(a, b)) / np.sqrt(denom_a * denom_b)

    peak_idx = int(np.argmax(acf))
    regularity = float(np.clip(acf[peak_idx], 0.0, 1.0))
    period_ms = (lag_lo + peak_idx) * cfg.frame_ms

    if regularity < cfg.min_regularity:
        notes.append(f"acf peak {regularity:.2f} below {cfg.min_regularity:.2f}")
        return PeriodResult(n_closures=0, period_ms=period_ms, regularity=regularity, notes=notes)

    raw = roi_duration_ms / max(period_ms, 1e-6)
    n_closures = int(round(raw))
    max_allowed = int(np.floor(roi_duration_ms / cfg.min_closure_spacing_ms))
    if n_closures > max_allowed:
        notes.append(f"clamped n={n_closures} -> {max_allowed} by spacing")
        n_closures = max_allowed
    n_closures = max(0, n_closures)

    return PeriodResult(
        n_closures=n_closures,
        period_ms=period_ms,
        regularity=regularity,
        notes=notes,
    )
