"""Sensitivity of the closure-count conclusions to the quality-filter thresholds.

Re-applies ``apply_quality_filter`` over a grid of (periodicity_min,
voicing_min) to the full token pool with unfiltered closure counts (produced by
``trillscope.detect_unfiltered``), and for each cell recomputes the sex
(Mann-Whitney, speaker-level) and context (Kruskal-Wallis, speaker x context)
tests on num_cycles. Writes a tidy CSV plus two heatmaps showing whether the
qualitative conclusions (sex not robustly significant, context a modest medium
effect) hold across a reasonable filter range.

Duration is fixed at [50, 200] ms (literature-justified). The production point
is periodicity_min=0.40, voicing_min=80; p-values here are uncorrected (the
sweep checks per-cell stability, not the corrected family from `analyze all`).

Run:
    python -m trillscope.sweep_quality
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .quality import apply_quality_filter

log = logging.getLogger(__name__)

CORPORA = ["dimex100", "albayzin", "glissando", "preseea", "tedx", "heroico"]
UNFILT = Path("outputs/tables_unfiltered")
TOKENS = Path("outputs/tables/tokens.parquet")
OUT_DIR = Path("outputs/v2/filter_sensitivity")

PERIODICITY_GRID = [0.30, 0.35, 0.40, 0.45, 0.50]
VOICING_GRID = [70.0, 80.0, 90.0]
MIN_GROUP = 5


def _speaker_sex_map() -> dict:
    """speaker_id -> sex using the SAME path as the main analysis (trillscope.statistics
    data_loader), so the sweep's sex test runs on the full sex-labelled sample
    (incl. DIMEx100 F0-inferred sex) rather than the sparse 'sex' column that
    build_tokens leaves in tokens.parquet. Non-circular here: the outcome is
    num_cycles, not F0.
    """
    from trillscope.statistics.data_loader import load_analysis_data
    df, _ = load_analysis_data(
        Path("outputs/tables_v2"), Path("metadata/metadata_unified.parquet"),
        datasets=CORPORA, aggregate_by_speaker=False,
    )
    return df.drop_duplicates("speaker_id").set_index("speaker_id")["sex"].to_dict()


def load_pool() -> pd.DataFrame:
    """tokens.parquet (filter inputs + grouping) joined to unfiltered n_closures."""
    tok = pd.read_parquet(TOKENS)
    cl = pd.concat(
        [pd.read_parquet(UNFILT / f"closures_v2_{d}.parquet") for d in CORPORA],
        ignore_index=True,
    )
    tok["_k"] = list(zip(tok["audio_path"], tok["t0_ms"].round(3), tok["t1_ms"].round(3)))
    cl["_k"] = list(zip(cl["audio_path"], cl["start_ms"].round(3), cl["end_ms"].round(3)))
    cl_n = cl[["_k", "n_closures_v2", "status_v2"]].drop_duplicates("_k")
    pool = tok.merge(cl_n, on="_k", how="inner").drop(columns="_k")
    # Use the analysis-grade sex labelling (the same 365-speaker sample as the
    # main sex analysis).
    sex_map = _speaker_sex_map()
    pool["sex"] = pool["speaker_id"].map(sex_map).fillna("unknown")
    log.info(
        "pool: %d/%d tokens.parquet rows matched; %d speakers with F/M sex",
        len(pool), len(tok),
        pool.loc[pool.sex.isin(["F", "M"]), "speaker_id"].nunique(),
    )
    return pool


def _sex_test(df: pd.DataFrame):
    spk = (
        df.groupby(["speaker_id", "corpus"])
        .agg(n=("n_closures_v2", "mean"), sex=("sex", "first"))
        .reset_index()
    )
    F = spk.loc[spk.sex == "F", "n"]
    M = spk.loc[spk.sex == "M", "n"]
    if len(F) < MIN_GROUP or len(M) < MIN_GROUP:
        return np.nan, np.nan, len(F), len(M)
    u, p = scipy_stats.mannwhitneyu(F, M, alternative="two-sided")
    eff = 1 - 2 * u / (len(F) * len(M))
    return p, eff, len(F), len(M)


def _context_test(df: pd.DataFrame):
    sc = (
        df.groupby(["speaker_id", "corpus", "context"])
        .agg(n=("n_closures_v2", "mean"))
        .reset_index()
    )
    groups = [
        g["n"].values
        for lvl, g in sc.groupby("context")
        if lvl not in ("unknown", "", None) and len(g) >= MIN_GROUP
    ]
    if len(groups) < 2:
        return np.nan, np.nan
    h, p = scipy_stats.kruskal(*groups)
    k = len(groups)
    nt = sum(len(g) for g in groups)
    eps = max(0.0, (h - k + 1) / (nt - k))
    return p, eps


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pool = load_pool()
    rows = []
    for pmin in PERIODICITY_GRID:
        for vmin in VOICING_GRID:
            crit = {
                "duration_min_ms": 50.0,
                "duration_max_ms": 200.0,
                "voicing_min_pct": vmin,
                "periodicity_min": pmin,
            }
            f = apply_quality_filter(pool, crit)
            p_sex, eff_sex, nF, nM = _sex_test(f)
            p_ctx, eps_ctx = _context_test(f)
            rows.append({
                "periodicity_min": pmin,
                "voicing_min": vmin,
                "n_tokens": len(f),
                "median_n": float(f["n_closures_v2"].median()) if len(f) else np.nan,
                "n_F": nF,
                "n_M": nM,
                "p_sex": p_sex,
                "eff_sex": eff_sex,
                "p_ctx": p_ctx,
                "eps_ctx": eps_ctx,
            })
            log.info(
                "per=%.2f voi=%2.0f -> n=%5d  p_sex=%.3f  p_ctx=%.1e  eps_ctx=%.3f",
                pmin, vmin, len(f), p_sex, p_ctx, eps_ctx,
            )
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "sweep.csv", index=False)
    _heatmaps(out)
    print(out.to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'sweep.csv'} + heatmaps")


def _heatmaps(out: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for col, title, fname in [
        ("p_sex", "Sex Mann-Whitney p (num_cycles, uncorrected)", "heatmap_p_sex.png"),
        ("eps_ctx", "Context epsilon-squared (num_cycles)", "heatmap_eps_ctx.png"),
    ]:
        piv = out.pivot(index="voicing_min", columns="periodicity_min", values=col)
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([int(v) for v in piv.index])
        ax.set_xlabel("periodicity_min")
        ax.set_ylabel("voicing_min")
        ax.set_title(title)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if v == v:  # not NaN
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
