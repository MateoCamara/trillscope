"""Debug helper: visualise envelope, candidates, and verification stages."""
import numpy as np

from erres.detector.cycle_detector import (
    DetectorConfig,
    _bandpass,
    _rms_envelope,
    _to_db,
    _find_closure_candidates,
    _verify_releases,
)
from tests.synthetic_trills import TrillSpec, synthesize_trill


def debug(spec: TrillSpec):
    audio, gt_ms, sr = synthesize_trill(spec)
    cfg = DetectorConfig()

    audio_low = _bandpass(audio, sr, cfg.low_band)
    audio_mid = _bandpass(audio, sr, cfg.mid_band)
    env_low, t_ms = _rms_envelope(audio_low, sr, cfg.frame_ms)
    env_mid, _ = _rms_envelope(audio_mid, sr, cfg.frame_ms)
    env_combined = np.sqrt(env_low * env_mid + 1e-12)
    env_db = _to_db(env_combined)

    print(f"n_frames={len(env_db)}, frame_ms={cfg.frame_ms}, total_t={t_ms[-1]:.0f} ms")
    print(f"env_db range: {env_db.min():.1f} .. {env_db.max():.1f}")
    print(f"env_mid range: {env_mid.min():.4f} .. {env_mid.max():.4f}")
    print(f"env_low range: {env_low.min():.4f} .. {env_low.max():.4f}")
    print(f"ground truth closures (ms): {gt_ms.tolist()}")
    print(f"per-frame: idx t_ms env_low env_mid env_db")
    for i in range(len(env_db)):
        marker = ""
        for g in gt_ms:
            if abs(t_ms[i] - g) <= 5.0:
                marker = " <- GT"
        print(f"  {i:2d} {t_ms[i]:6.1f}  {env_low[i]:.4f}  {env_mid[i]:.4f}  {env_db[i]:+6.2f}{marker}")

    # candidate detection
    cands = _find_closure_candidates(env_db, t_ms, cfg)
    print(f"candidates (frame_idx, time_ms, env_db):")
    for c in cands:
        print(f"  frame={c} t={t_ms[c]:.1f}ms env={env_db[c]:.1f}dB")

    # verification
    verified = _verify_releases(cands, env_mid, t_ms, cfg)
    print(f"verified pairs (closure_t_ms, release_t_ms):")
    for ci, ri in verified:
        print(f"  closure_t={t_ms[ci]:.1f}ms release_t={t_ms[ri]:.1f}ms "
              f"mid_at_release={env_mid[ri]:.4f}")

    # For each candidate that failed verification, show why
    for ci in cands:
        if any(v[0] == ci for v in verified):
            continue
        frame_ms = float(t_ms[1] - t_ms[0])
        lag_lo = int(round(cfg.release_min_lag_ms / frame_ms))
        lag_hi = int(round(cfg.release_max_lag_ms / frame_ms))
        lo = ci + lag_lo
        hi = min(len(env_mid), ci + lag_hi + 1)
        if lo >= len(env_mid):
            print(f"  candidate frame={ci} t={t_ms[ci]:.1f} no room for release window")
            continue
        window = env_mid[lo:hi]
        ctx_lo = max(0, ci - lag_hi * 2)
        ctx_hi = min(len(env_mid), ci + lag_hi * 2 + 1)
        local_max = float(np.max(env_mid[ctx_lo:ctx_hi]))
        peak_val = float(np.max(window)) if len(window) else 0.0
        print(f"  candidate t={t_ms[ci]:.1f}: window_peak={peak_val:.4f} "
              f"local_max={local_max:.4f} ratio={peak_val/max(local_max,1e-9):.2f} "
              f"(needed >={cfg.release_threshold_pct})")


if __name__ == "__main__":
    spec = TrillSpec(n_closures=2, period_ms=30.0, period_jitter_ms=0.5,
                     noise_snr_db=35.0, rng_seed=42)
    debug(spec)
