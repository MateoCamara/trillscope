"""Synthetic trill generator for detector validation.

Produces audio that mimics the acoustic structure of a Spanish trill:
- A voiced base at a chosen F0 (50–250 Hz).
- An amplitude envelope with N occlusion–release events: each is a deep V-shaped
  minimum followed by a release burst spike.
- Optional jitter in inter-closure interval, optional noise, optional partial
  devoicing during closures.

The generator returns audio + ground-truth closure times so detectors can be
validated quantitatively.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrillSpec:
    n_closures: int
    duration_ms: float = 100.0
    f0_hz: float = 130.0
    period_ms: float = 30.0          # inter-closure interval (~33 Hz cycle rate)
    period_jitter_ms: float = 2.0
    closure_depth_db: float = 25.0   # how deep the closure goes below peak
    closure_width_ms: float = 8.0    # half-width of the V at base
    release_overshoot: float = 1.2   # release burst is 1.2× the steady-state amplitude
    voicing_dropout: bool = True     # devoice during closures
    noise_snr_db: float = 25.0
    sr: int = 16000
    rng_seed: int = 0


def _envelope(spec: TrillSpec, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return amplitude envelope + ground-truth closure times in ms."""
    rng = np.random.default_rng(spec.rng_seed)
    env = np.ones_like(t)
    # place closures evenly with jitter, centred in the segment
    total_span = (spec.n_closures - 1) * spec.period_ms if spec.n_closures > 1 else 0.0
    start = (spec.duration_ms - total_span) / 2.0
    closure_times_ms = []
    for i in range(spec.n_closures):
        ct = start + i * spec.period_ms + rng.normal(0, spec.period_jitter_ms)
        closure_times_ms.append(ct)
    closure_times_ms.sort()

    depth_lin = 10.0 ** (-spec.closure_depth_db / 20.0)
    width_s = spec.closure_width_ms / 1000.0
    for ct in closure_times_ms:
        ct_s = ct / 1000.0
        # V-shape: deep at the closure, recovering linearly within `width`
        dist = np.abs(t - ct_s)
        within = dist < width_s
        # at dist=0, env = depth_lin; at dist=width, env = 1
        v = depth_lin + (1.0 - depth_lin) * (dist / width_s)
        env = np.where(within, np.minimum(env, v), env)
        # release: brief burst right after (5–10 ms after closure)
        release_t_s = ct_s + 0.006
        release_w_s = 0.002
        dist_r = np.abs(t - release_t_s)
        burst_mask = dist_r < release_w_s
        burst = 1.0 + (spec.release_overshoot - 1.0) * (1.0 - dist_r / release_w_s)
        env = np.where(burst_mask, np.maximum(env, burst), env)
    return env, np.array(closure_times_ms)


def synthesize_trill(spec: TrillSpec) -> tuple[np.ndarray, np.ndarray, int, tuple[float, float]]:
    """Generate a synthetic trill.

    Returns
    -------
    audio : np.ndarray (float32, mono)
    closure_times_ms : np.ndarray
        Ground-truth closure times (absolute time in ``audio``).
    sr : int
    roi_ms : (start_ms, end_ms)
        Boundaries of the trill region in ``audio`` — the simulated MFA segment.
        Pad regions (vowel context) lie outside this window.
    """
    rng = np.random.default_rng(spec.rng_seed)
    # Adapt duration so n closures fit comfortably with the requested period
    span_ms = max(0.0, (spec.n_closures - 1) * spec.period_ms)
    duration_ms = max(spec.duration_ms, span_ms + 30.0)
    n_samples = int(round(spec.sr * duration_ms / 1000.0))
    t = np.arange(n_samples) / spec.sr

    # Voiced carrier — sum of first few harmonics for richer spectrum
    carrier = np.zeros_like(t)
    for k in range(1, 6):
        carrier += (1.0 / k) * np.sin(2 * np.pi * spec.f0_hz * k * t + rng.uniform(0, 2 * np.pi))
    carrier = carrier / np.max(np.abs(carrier) + 1e-9)

    # use the actual duration for envelope placement
    effective_spec = TrillSpec(
        **{**spec.__dict__, "duration_ms": duration_ms}
    )
    env, closure_times_ms = _envelope(effective_spec, t)
    sig = carrier * env

    # Optional voicing dropout: zero out the voicing during deep closures
    if spec.voicing_dropout:
        depth_lin = 10.0 ** (-spec.closure_depth_db / 20.0)
        # below 1.5× depth, fully devoiced (just noise floor)
        mask = env < depth_lin * 1.5
        sig = np.where(mask, sig * 0.1, sig)

    # Add Gaussian noise at requested SNR
    sig_rms = float(np.sqrt(np.mean(sig ** 2)) + 1e-12)
    noise_rms = sig_rms / (10.0 ** (spec.noise_snr_db / 20.0))
    sig = sig + rng.normal(0, noise_rms, size=sig.shape)

    # Pre-pad with 20 ms of full-amplitude voicing to mimic the adjacent vowel.
    # Use steady-state amplitude (no fade) so the pad does not create a sham
    # amplitude transition the detector might confuse with a closure.
    pad_samples = int(0.020 * spec.sr)
    t_pre = np.arange(pad_samples) / spec.sr
    t_post = np.arange(pad_samples) / spec.sr
    pre_carrier = np.zeros_like(t_pre)
    post_carrier = np.zeros_like(t_post)
    for k in range(1, 6):
        pre_carrier += (1.0 / k) * np.sin(2 * np.pi * spec.f0_hz * k * t_pre + rng.uniform(0, 2 * np.pi))
        post_carrier += (1.0 / k) * np.sin(2 * np.pi * spec.f0_hz * k * t_post + rng.uniform(0, 2 * np.pi))
    pre_carrier = pre_carrier / (np.max(np.abs(pre_carrier)) + 1e-9)
    post_carrier = post_carrier / (np.max(np.abs(post_carrier)) + 1e-9)
    pre = pre_carrier + rng.normal(0, noise_rms, pad_samples)
    post = post_carrier + rng.normal(0, noise_rms, pad_samples)
    sig = np.concatenate([pre, sig, post]).astype(np.float32)
    # Shift ground-truth times by pad
    pad_ms = 1000.0 * pad_samples / spec.sr
    closure_times_ms = closure_times_ms + pad_ms
    roi_ms = (pad_ms, pad_ms + duration_ms)
    return sig, closure_times_ms, spec.sr, roi_ms
