"""Regenerate the v2 IberSpeech figures into paper/iberspeech2026/figures/.

Produces:
  fig1_spectrogram.png  - example 2-closure trill with detector closure marks
  fig2_context.png      - closure count & cycle rate by context (median + 95% CI)
  fig3_effect_sizes.png - effect-size heatmap (measures x predictors)
  fig4_robustness.png   - filter sweep (p_sex, eps_ctx) + detector sweep pass grid

Run from repo root:
    python -m src.paper.figures_v2
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from src.statistics.data_loader import load_analysis_data, _aggregate_by_speaker
from src.statistics.inferential import bootstrap_median_ci

logging.getLogger().setLevel(logging.WARNING)

FIG = Path("paper/iberspeech2026/figures")
CORPORA = ["dimex100", "albayzin", "glissando", "preseea", "tedx", "heroico"]
CTX_ORDER = ["intervocalic_rr", "after_nls", "word_initial"]
CTX_LABEL = {"intervocalic_rr": "Interv.\nrr", "after_nls": "Post-\n/nls/", "word_initial": "Word-\ninit."}

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _tokens() -> pd.DataFrame:
    df, _ = load_analysis_data(
        Path("outputs/tables_v2"), Path("metadata/metadata_unified.parquet"),
        datasets=CORPORA, aggregate_by_speaker=False,
    )
    return df


def fig1_spectrogram() -> None:
    """Spectrogram of a representative 2-closure Glissando trill with marks."""
    from erres.audio_io import load_audio_full
    cl = pd.read_parquet("outputs/tables/closures_v2_glissando.parquet")
    ok = cl[(cl.status_v2 == "ok") & (cl.n_closures_v2 == 2)].copy()
    # prefer an intervocalic token with a regular period
    ok = ok.sort_values("period_cv_v2")
    row = ok.iloc[0]
    audio, sr = load_audio_full(row["audio_path"], target_sr=16000)
    pad = 20.0
    cs = max(0.0, row["start_ms"] - pad)
    ce = row["end_ms"] + pad
    i0, i1 = int(cs / 1000 * sr), int(ce / 1000 * sr)
    seg = audio[i0:i1]
    t = np.linspace(0, len(seg) / sr * 1000, len(seg))
    closures = list(row["closure_times_ms_v2"])  # chunk-relative ms

    fig, axes = plt.subplots(2, 1, figsize=(5.2, 3.6), height_ratios=[1, 2])
    axes[0].plot(t, seg, color="#333", lw=0.5)
    axes[0].set_ylabel("Amp.")
    axes[0].set_xlim(0, t[-1])
    axes[0].set_xticks([])
    f, tt, Sxx = signal.spectrogram(seg, sr, nperseg=min(256, len(seg) // 4),
                                    noverlap=min(256, len(seg) // 4) // 2)
    axes[1].pcolormesh(tt * 1000, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="inferno")
    axes[1].set_ylim(0, 5000)
    axes[1].set_ylabel("Freq. (Hz)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_xlim(0, t[-1])
    for c in closures:
        for ax in axes:
            ax.axvline(c, color="cyan", ls="--", lw=1.0, alpha=0.9)
    axes[1].axvspan(pad, pad + row["duration_ms"], color="white", alpha=0.10)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_spectrogram.png")
    plt.close(fig)


def fig2_context(df: pd.DataFrame) -> None:
    sc = _aggregate_by_speaker(df, include_context=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, col, ylab, tag in [
        (axes[0], "num_cycles", "Closure count", "(a)"),
        (axes[1], "cycle_rate_hz", "Cycle rate (Hz)", "(b)"),
    ]:
        meds, los, his = [], [], []
        for c in CTX_ORDER:
            v = sc.loc[sc.context_label == c, col].dropna()
            m = float(v.median())
            lo, hi = bootstrap_median_ci(v)
            meds.append(m); los.append(m - lo); his.append(hi - m)
        ax.errorbar(range(3), meds, yerr=[los, his], fmt="o", capsize=4,
                    color="#3C5488", ms=6, lw=1.3)
        ax.set_xticks(range(3))
        ax.set_xticklabels([CTX_LABEL[c] for c in CTX_ORDER])
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(tag, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_context.png")
    plt.close(fig)


def fig3_effect_sizes(df: pd.DataFrame) -> None:
    """Uniform Kruskal-Wallis epsilon-squared for every predictor.

    Using one effect-size metric (epsilon-squared) on a single colour scale
    avoids the misleading impression created by mixing rank-biserial r (sex)
    with epsilon-squared (the multi-group factors): a rank-biserial r of 0.14
    is negligible, whereas epsilon-squared 0.14 is large.
    """
    from matplotlib.patches import Rectangle
    from scipy import stats as ss
    spk = _aggregate_by_speaker(df, include_context=False)
    sc = _aggregate_by_speaker(df, include_context=True)
    measures = [("num_cycles", "Closures"), ("cycle_rate_hz", "Cycle rate"), ("duration_ms", "Duration")]
    preds = [("context_label", "Context", sc), ("speech_style", "Style", spk),
             ("age_bin", "Age", spk), ("education_bin", "Educ.", spk), ("sex", "Sex", spk)]
    nrow, ncol = len(measures), len(preds)
    M = np.full((nrow, ncol), np.nan)
    pval = np.full((nrow, ncol), np.nan)
    for i, (mc, _ml) in enumerate(measures):
        for j, (pc, _pl, frame) in enumerate(preds):
            if mc not in frame.columns or pc not in frame.columns:
                continue
            sub = frame[~frame[pc].isin(["unknown", "ambiguous", "", None])][[mc, pc]].dropna()
            groups = [g[mc].values for _lvl, g in sub.groupby(pc) if len(g) >= 5]
            if len(groups) < 2:
                continue
            H, p = ss.kruskal(*groups)
            k = len(groups); n = int(sum(len(g) for g in groups))
            M[i, j] = max(0.0, (H - k + 1) / (n - k))
            pval[i, j] = p
    sig = np.zeros((nrow, ncol), dtype=bool)
    for j in range(ncol):  # Bonferroni within each predictor (across its outcomes)
        col = pval[:, j]
        nj = int(np.isfinite(col).sum())
        if nj:
            sig[:, j] = col * nj < 0.05

    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=0.14, aspect="auto")
    ax.set_xticks(range(ncol)); ax.set_xticklabels([p[1] for p in preds])
    ax.set_yticks(range(nrow)); ax.set_yticklabels([m[1] for m in measures])
    for i in range(nrow):
        for j in range(ncol):
            if np.isnan(M[i, j]):
                continue
            star = "*" if sig[i, j] else ""
            ax.text(j, i, f"{M[i, j]:.2f}{star}", ha="center", va="center",
                    color="white" if M[i, j] < 0.09 else "black", fontsize=9)
            if not sig[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       hatch="////", edgecolor="white", lw=0, alpha=0.35))
    fig.colorbar(im, ax=ax, label="$\\varepsilon^2$")
    ax.set_title("Kruskal--Wallis $\\varepsilon^2$ ($^*$sig.\\ Bonferroni; n.s.\\ hatched)")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_effect_sizes.png")
    plt.close(fig)


def _heat(ax, piv, title, cmap, vmin, vmax, fmt):
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{r:g}" for r in piv.index])
    ax.set_title(title, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if v == v:
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7,
                        color="white" if cmap != "RdYlGn" else "black")
    return im


def fig4_robustness() -> None:
    sw = pd.read_csv("outputs/v2/filter_sensitivity/sweep.csv")
    gd = pd.read_csv("outputs/v2/detector_sensitivity/grid_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8))
    ps = sw.pivot(index="voicing_min", columns="periodicity_min", values="p_sex")
    _heat(axes[0], ps, "(a) Sex $p$ (filter grid)", "viridis", 0, 0.7, lambda v: f"{v:.2f}")
    axes[0].set_xlabel("periodicity$_{min}$"); axes[0].set_ylabel("voicing$_{min}$")
    ec = sw.pivot(index="voicing_min", columns="periodicity_min", values="eps_ctx")
    _heat(axes[1], ec, "(b) Context $\\varepsilon^2$ (filter grid)", "viridis", 0, 0.14, lambda v: f"{v:.2f}")
    axes[1].set_xlabel("periodicity$_{min}$"); axes[1].set_ylabel("voicing$_{min}$")
    pg = gd.pivot(index="prominence_db", columns="threshold_pct", values="sum")
    _heat(axes[2], pg, "(c) Corpora passing (detector grid)", "RdYlGn", 0, 6, lambda v: f"{int(v)}")
    axes[2].set_xlabel("threshold$_{pct}$"); axes[2].set_ylabel("prominence$_{dB}$")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_robustness.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = _tokens()
    fig1_spectrogram()
    fig2_context(df)
    fig3_effect_sizes(df)
    fig4_robustness()
    print("Wrote fig1_spectrogram, fig2_context, fig3_effect_sizes, fig4_robustness to", FIG)


if __name__ == "__main__":
    main()
