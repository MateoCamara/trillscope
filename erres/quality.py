"""Detector-agnostic quality gate for trill /r/ tokens.

The gate is a single fixed criterion. It only consumes acoustic features that
do not depend on which cycle detector is run downstream, so the population of
tokens that survives is reproducible from the raw audio + boundaries alone.

Criterion:
- duration_ms in [50, 200]       : excludes sub-cycle taps and outliers
- voicing_pct >= 80              : excludes devoiced / mis-extracted tokens
- periodicity_score >= 0.40      : confirms periodic energy modulation in the
                                   18-40 Hz cycle-rate band (canonical Spanish
                                   trill range, Quilis 1993, Henriksen 2010)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .detector.cycle_detector import _bandpass, _rms_envelope


QUALITY_FILTER = {
    "duration_min_ms": 50.0,
    "duration_max_ms": 200.0,
    "voicing_min_pct": 80.0,
    "periodicity_min": 0.40,
}

PERIOD_BAND_HZ = (500.0, 3500.0)
ENV_FRAME_MS = 2.5
PERIOD_LAG_MIN_MS = 25.0
PERIOD_LAG_MAX_MS = 55.0


def compute_periodicity_score(
    audio: np.ndarray,
    sr: int,
    t0_ms: float,
    t1_ms: float,
) -> float:
    """Autocorrelation peak ratio of the mid-band envelope inside the ROI.

    The score lives in [0, 1]:
    - 0 means the envelope has no detectable cycle in the trill-period range
    - 1 means the envelope repeats itself perfectly at some lag in [25, 55] ms

    The score is independent of the closure detector. It only inspects whether
    the signal has the kind of periodic energy modulation that defines a trill.
    """
    i0 = max(0, int(round(t0_ms / 1000.0 * sr)))
    i1 = int(round(t1_ms / 1000.0 * sr))
    chunk = audio[i0:i1]
    if len(chunk) < int(sr * 0.060):
        return 0.0
    mid = _bandpass(chunk, sr, PERIOD_BAND_HZ)
    env, _ = _rms_envelope(mid, sr, ENV_FRAME_MS)
    if len(env) < 4:
        return 0.0
    env = env - env.mean()
    denom = float(np.dot(env, env))
    if denom <= 0:
        return 0.0
    n = len(env)
    lag_lo = max(1, int(round(PERIOD_LAG_MIN_MS / ENV_FRAME_MS)))
    lag_hi = min(n - 1, int(round(PERIOD_LAG_MAX_MS / ENV_FRAME_MS)))
    if lag_hi <= lag_lo:
        return 0.0
    peak = 0.0
    for lag in range(lag_lo, lag_hi + 1):
        num = float(np.dot(env[: n - lag], env[lag:]))
        score = num / denom
        if score > peak:
            peak = score
    return float(np.clip(peak, 0.0, 1.0))


def apply_quality_filter(df: pd.DataFrame, criterion: dict = QUALITY_FILTER) -> pd.DataFrame:
    """Drop tokens that fail any of the three quality invariants.

    Required columns: duration_ms, voicing_pct, periodicity_score.
    """
    mask = (
        df["duration_ms"].between(criterion["duration_min_ms"], criterion["duration_max_ms"])
        & (df["voicing_pct"] >= criterion["voicing_min_pct"])
        & (df["periodicity_score"] >= criterion["periodicity_min"])
    )
    return df[mask].reset_index(drop=True)
