"""Per-token spectrogram + contact sheets for validation reports.

`render_token_spectrogram` writes one PNG per token (audio waveform + 0–5 kHz
spectrogram, with the trill ROI highlighted). `make_contact_sheet` tiles the
PNGs of a corpus into a single grid, colouring each cell's border green if
the v2 detector and the period detector agree (|Δn|≤1), red otherwise.

This module is the visual companion of `trillscope.validate` — it produces no
filtering decisions, only documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import stft

from .audio_io import load_audio_full

log = logging.getLogger(__name__)


def render_token_spectrogram(
    audio: np.ndarray,
    sr: int,
    trill_t0_s: float,
    trill_t1_s: float,
    out_path: Path,
    *,
    title: str,
) -> None:
    """Waveform + 0–5 kHz spectrogram with the trill region highlighted."""
    n_fft = 512
    hop = 64  # 4 ms at 16 kHz

    fig, (ax_wave, ax_spec) = plt.subplots(
        2, 1, figsize=(8, 4), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
    )
    t = np.arange(len(audio)) / sr
    ax_wave.plot(t, audio, color="black", linewidth=0.5)
    ax_wave.set_ylabel("amp")
    ax_wave.set_xlim(0, t[-1] if len(t) else 1.0)
    ax_wave.axvspan(trill_t0_s, trill_t1_s, color="orange", alpha=0.2)

    win = np.hanning(n_fft)
    f, t_spec, Z = stft(
        audio, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop,
        boundary=None, padded=False,
    )
    Sxx = np.abs(Z) ** 2
    fmask = f <= 5000
    Sdb = 10.0 * np.log10(np.maximum(Sxx[fmask], 1e-12))
    ax_spec.pcolormesh(t_spec, f[fmask], Sdb, shading="auto", cmap="magma")
    ax_spec.axvline(trill_t0_s, color="white", linewidth=1)
    ax_spec.axvline(trill_t1_s, color="white", linewidth=1)
    ax_spec.set_ylabel("Hz")
    ax_spec.set_xlabel("time (s)")
    ax_spec.set_ylim(0, 5000)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def render_corpus_specs(
    closures_df: pd.DataFrame,
    xval_df: pd.DataFrame,
    specs_dir: Path,
    target_sr: int = 16_000,
    max_per_corpus: int = 144,
) -> pd.DataFrame:
    """Render up to `max_per_corpus` token PNGs from a corpus. Returns the
    selection actually rendered, with the agreement column attached so the
    contact sheet can colour cells correctly."""
    ok = closures_df[closures_df["status_v2"] == "ok"].copy()
    if len(ok) > max_per_corpus:
        ok = ok.sample(n=max_per_corpus, random_state=0).reset_index(drop=True)

    # Join agreement info if available
    if not xval_df.empty:
        key_cols = ["utt_id", "start_ms", "end_ms"]
        ok = ok.merge(
            xval_df[key_cols + ["n_closures_period", "agreement"]],
            on=key_cols, how="left",
        )
    else:
        ok["n_closures_period"] = pd.NA
        ok["agreement"] = pd.NA

    specs_dir.mkdir(parents=True, exist_ok=True)
    rendered_rows: list[dict] = []

    for audio_path, group in ok.groupby("audio_path", sort=False):
        try:
            audio, sr = load_audio_full(audio_path, target_sr=target_sr)
        except Exception as exc:
            log.warning("audio load failed: %s: %s", audio_path, exc)
            continue
        for _, row in group.iterrows():
            pad_ms = 100.0
            t0 = float(row["start_ms"])
            t1 = float(row["end_ms"])
            chunk_start_ms = max(0.0, t0 - pad_ms)
            chunk_end_ms = t1 + pad_ms
            i0 = int(round(chunk_start_ms / 1000.0 * sr))
            i1 = int(round(chunk_end_ms / 1000.0 * sr))
            chunk = audio[i0:i1]
            if len(chunk) < int(sr * 0.05):
                continue
            trill_t0_s = (t0 - chunk_start_ms) / 1000.0
            trill_t1_s = (t1 - chunk_start_ms) / 1000.0
            n_v2 = int(row["n_closures_v2"])
            n_p = row["n_closures_period"]
            n_p_str = "?" if pd.isna(n_p) else f"{int(n_p)}"
            title = (
                f"{row['utt_id']}  |  dur={t1 - t0:.0f} ms  "
                f"v2={n_v2}  period={n_p_str}"
            )
            token_id = f"{row['dataset']}-{row['utt_id']}-{int(t0)}"
            out_png = specs_dir / f"{token_id}.png"
            render_token_spectrogram(chunk, sr, trill_t0_s, trill_t1_s, out_png, title=title)
            rendered_rows.append({
                "token_id": token_id,
                "png_path": str(out_png),
                "n_closures_v2": n_v2,
                "n_closures_period": n_p_str,
                "agreement": int(row["agreement"]) if not pd.isna(row["agreement"]) else 0,
            })

    return pd.DataFrame(rendered_rows)


def make_contact_sheet(
    rendered: pd.DataFrame,
    dataset: str,
    out_path: Path,
    n_cols: int = 12,
    cell_size_px: int = 240,
) -> None:
    """Tile the rendered PNGs into one grid with green/red borders."""
    if rendered.empty:
        log.warning("no rendered PNGs for %s; skipping contact sheet", dataset)
        return

    n = len(rendered)
    n_rows = (n + n_cols - 1) // n_cols
    border = 6
    cell_w = cell_size_px
    cell_h = int(cell_size_px * 0.55)  # PNGs are ~ 880x440 -> ratio 0.5
    sheet_w = n_cols * cell_w
    sheet_h = n_rows * cell_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")

    for i, row in rendered.reset_index(drop=True).iterrows():
        x = (i % n_cols) * cell_w
        y = (i // n_cols) * cell_h
        try:
            img = Image.open(row["png_path"]).convert("RGB")
        except Exception:
            continue
        img.thumbnail((cell_w - 2 * border, cell_h - 2 * border))
        sheet.paste(img, (x + border, y + border))
        # Coloured border
        color = (0, 170, 0) if row["agreement"] == 1 else (200, 30, 30)
        # draw 4 thin rectangles around the cell
        from PIL import ImageDraw
        draw = ImageDraw.Draw(sheet)
        draw.rectangle(
            [x + 1, y + 1, x + cell_w - 2, y + cell_h - 2],
            outline=color, width=border // 2,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    log.info("wrote contact sheet: %s (%d cells)", out_path, n)
