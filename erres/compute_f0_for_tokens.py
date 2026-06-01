"""Compute a per-token fundamental frequency (F0) for every quality-filtered trill.

The v1 measurement pipeline left `mean_f0_hz` empty in every row of
outputs/tables/acoustic_measurements_*.parquet (parselmouth was unavailable and the
fallback returned no F0). We need a per-token F0 to test the paper's central claim:
that the envelope-peak over-count -- the source of the spurious sex effect -- grows
as F0 falls. F0 is estimated with librosa.pyin
over each token's region of interest, padded with real surrounding audio so the pitch
tracker has enough context.

Read-only over existing tables/audio. Writes only outputs/tables/per_token_f0.parquet.
Run from repo root:  python -m erres.compute_f0_for_tokens
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from erres.audio_io import load_audio_full

CORPORA = ["dimex100", "albayzin", "glissando", "preseea", "tedx", "heroico"]
TABLES_DIR = Path("outputs/tables")
OUT = TABLES_DIR / "per_token_f0.parquet"

SR = 16_000
FMIN = 65.0          # below modal Spanish male F0, with margin
FMAX = 400.0         # above modal Spanish female F0
FRAME_LENGTH = 2048  # librosa.pyin default; >= 2 periods of FMIN
HOP = 256            # 16 ms resolution
PAD = 1024           # real-audio context (64 ms) on each side of the ROI


def _token_f0(audio: np.ndarray, sr: int, start_ms: float, end_ms: float):
    """Median F0 of the voiced frames whose centre falls inside [start_ms, end_ms]."""
    import librosa

    s = int(round(start_ms / 1000.0 * sr))
    e = int(round(end_ms / 1000.0 * sr))
    lo = max(0, s - PAD)
    hi = min(len(audio), e + PAD)
    seg = audio[lo:hi]
    if seg.size < FRAME_LENGTH // 2:
        return np.nan, 0

    f0, _, _ = librosa.pyin(
        seg, fmin=FMIN, fmax=FMAX, sr=sr,
        frame_length=FRAME_LENGTH, hop_length=HOP, center=True,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=HOP)
    abs_ms = (lo / sr * 1000.0) + times * 1000.0
    in_roi = (abs_ms >= start_ms) & (abs_ms <= end_ms)
    voiced = in_roi & np.isfinite(f0)
    n = int(voiced.sum())
    if n == 0:
        return np.nan, 0
    return float(np.nanmedian(f0[voiced])), n


def main() -> None:
    frames = []
    for ds in CORPORA:
        p = TABLES_DIR / f"closures_v2_{ds}.parquet"
        if not p.exists():
            print(f"  [skip] missing {p}")
            continue
        df = pd.read_parquet(p)[["dataset", "utt_id", "speaker_id",
                                  "start_ms", "end_ms", "audio_path"]].copy()
        frames.append(df)
    tokens = pd.concat(frames, ignore_index=True)
    # Process one audio file at a time (many tokens share a file).
    tokens = tokens.sort_values(["audio_path", "start_ms"]).reset_index(drop=True)
    print(f"Computing F0 for {len(tokens)} tokens "
          f"({tokens['audio_path'].nunique()} unique audio files)...")

    cache_path = None
    cache_audio = None
    out_rows = []
    last_pct = -1
    for i, row in enumerate(tokens.itertuples(index=False)):
        f0_hz, n_voiced, status = np.nan, 0, "ok"
        try:
            if row.audio_path != cache_path:
                cache_audio, _ = load_audio_full(row.audio_path, target_sr=SR)
                cache_path = row.audio_path
            f0_hz, n_voiced = _token_f0(cache_audio, SR, row.start_ms, row.end_ms)
            if not np.isfinite(f0_hz):
                status = "no_voiced_frame"
        except Exception as exc:  # noqa: BLE001 - record and continue
            status = f"error: {type(exc).__name__}"
        out_rows.append({
            "dataset": row.dataset, "utt_id": row.utt_id, "speaker_id": row.speaker_id,
            "start_ms": row.start_ms, "end_ms": row.end_ms,
            "mean_f0_hz": f0_hz, "n_voiced_roi": n_voiced, "f0_status": status,
        })
        pct = (i + 1) * 100 // len(tokens)
        if pct != last_pct and pct % 10 == 0:
            print(f"  {pct}% ({i + 1}/{len(tokens)})")
            last_pct = pct

    out = pd.DataFrame(out_rows)
    out.to_parquet(OUT, index=False)
    ok = out["mean_f0_hz"].notna()
    print(f"\nWrote {OUT}")
    print(f"  F0 recovered: {ok.sum()}/{len(out)} tokens "
          f"({100 * ok.mean():.1f}%)")
    print(f"  median F0 overall: {out.loc[ok, 'mean_f0_hz'].median():.1f} Hz "
          f"(IQR {out.loc[ok, 'mean_f0_hz'].quantile(0.25):.0f}-"
          f"{out.loc[ok, 'mean_f0_hz'].quantile(0.75):.0f})")
    print(out.loc[ok].groupby("dataset")["mean_f0_hz"]
          .agg(["count", "median"]).round(1).to_string())


if __name__ == "__main__":
    main()
