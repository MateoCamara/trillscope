"""Synthetic-trill tests for the independent autocorrelation detector."""

from __future__ import annotations

import numpy as np
import pytest

from trillscope.detector import detect_n_closures_by_period
from trillscope.synthetic import TrillSpec, synthesize_trill


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_period_count_clean_canonical(n: int) -> None:
    spec = TrillSpec(n_closures=n, period_ms=30.0, period_jitter_ms=0.0, noise_snr_db=30.0)
    audio, _truth_times, sr, roi = synthesize_trill(spec)
    result = detect_n_closures_by_period(audio, sr, roi_ms=roi)
    assert abs(result.n_closures - n) <= 1
    assert 25.0 <= result.period_ms <= 40.0
    assert result.regularity >= 0.4


def test_period_low_regularity_on_white_noise() -> None:
    """White noise has no periodic envelope; regularity must stay low."""
    sr = 16000
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 1, sr // 2).astype(np.float32)  # 500 ms
    result = detect_n_closures_by_period(audio, sr, roi_ms=(50.0, 450.0))
    assert result.regularity < 0.30


def test_period_robust_to_jitter() -> None:
    spec = TrillSpec(n_closures=4, period_ms=32.0, period_jitter_ms=4.0, noise_snr_db=20.0)
    audio, _truth_times, sr, roi = synthesize_trill(spec)
    result = detect_n_closures_by_period(audio, sr, roi_ms=roi)
    assert abs(result.n_closures - 4) <= 1
