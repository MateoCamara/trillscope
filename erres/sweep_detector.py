"""C: sensitivity of detector v2 to its two main thresholds.

Re-runs detect + cross-validate + literature validation over a grid of
(closure_prominence_db, closure_threshold_pct), holding everything else at the
production config and applying the production quality filter (same token set as
production). Reports, per (config, corpus): median n_closures, median period,
cross-detector agreement, and whether the corpus passes LITERATURE_TARGETS.
Writes grid.csv + grid_summary.csv + a stability figure.

The production config (config/detector.yaml) and the production tables in
outputs/tables/ are NOT modified — all work happens in a temp dir that is
removed at the end.

Run:
    python -m erres.sweep_detector
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .build_tokens import CORPUS_CANONICAL
from .cli import load_config
from .cross_validate import run_dataset as run_xval
from .measure_v2 import run_dataset as run_detect
from .quality import apply_quality_filter
from .validate import validate

log = logging.getLogger(__name__)

CORPORA = ["dimex100", "albayzin", "glissando", "preseea", "tedx", "heroico"]
CAND_DIR = Path("outputs/tables")
OUT_DIR = Path("outputs/v2/detector_sensitivity")
WORK = OUT_DIR / "_work"

PROM_GRID = [3.0, 5.0, 8.0]
THR_GRID = [0.30, 0.40, 0.50]


def filtered_keys() -> dict[str, set]:
    """Production-filtered (utt_id, start_ms, end_ms) keys per dataset.

    Mirrors erres.cli._load_filter_keys: apply the production quality filter to
    tokens.parquet, then reconstruct keys so detect runs on exactly the
    production token set for every config in the sweep.
    """
    tokens = pd.read_parquet(CAND_DIR / "tokens.parquet")
    filt = apply_quality_filter(tokens)
    c2k = {v: k for k, v in CORPUS_CANONICAL.items()}
    out: dict[str, set] = {}
    for corpus, g in filt.groupby("corpus"):
        ds = c2k.get(corpus)
        if ds is None:
            continue
        utt = [t[len(ds) + 1:].rsplit("-", 2)[0] for t in g["token_id"]]
        out[ds] = set(zip(utt, g["t0_ms"].round(4), g["t1_ms"].round(4)))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = load_config(Path("config/detector.yaml"))
    fkeys = filtered_keys()
    WORK.mkdir(parents=True, exist_ok=True)

    rows = []
    for prom in PROM_GRID:
        for thr in THR_GRID:
            cfg = replace(base, closure_prominence_db=prom, closure_threshold_pct=thr)
            for ds in CORPORA:
                cand = CAND_DIR / f"r_candidates_{ds}.parquet"
                if not cand.exists():
                    continue
                clp = WORK / f"closures_{ds}.parquet"
                xvp = WORK / f"xval_{ds}.parquet"
                run_detect(cand, clp, cfg=cfg, keep_keys=fkeys.get(ds), progress_every=0)
                run_xval(clp, xvp)
                r = validate(ds, pd.read_parquet(clp), pd.read_parquet(xvp))
                rows.append({
                    "prominence_db": prom,
                    "threshold_pct": thr,
                    "corpus": ds,
                    "n_tokens": r.n_tokens,
                    "median_n": r.median_n_closures,
                    "median_period_ms": r.median_period_ms,
                    "agreement_pct": r.pct_cross_detector_agreement,
                    "passes": bool(r.passes),
                })
            cell = [x for x in rows if x["prominence_db"] == prom and x["threshold_pct"] == thr]
            log.info("prom=%.0f thr=%.2f -> %d/%d corpora PASS",
                     prom, thr, sum(x["passes"] for x in cell), len(cell))

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "grid.csv", index=False)
    summ = out.groupby(["prominence_db", "threshold_pct"])["passes"].agg(["sum", "count"]).reset_index()
    summ.to_csv(OUT_DIR / "grid_summary.csv", index=False)
    _figure(out)
    shutil.rmtree(WORK, ignore_errors=True)

    print(summ.to_string(index=False))
    print("\nProduction config (prom=5, thr=0.40):")
    prod = out[(out.prominence_db == 5.0) & (out.threshold_pct == 0.40)]
    print(prod[["corpus", "median_n", "median_period_ms", "agreement_pct", "passes"]].to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'grid.csv'} + stability.png")


def _figure(out: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summ = out.groupby(["prominence_db", "threshold_pct"])["passes"].sum().reset_index()
    piv = summ.pivot(index="prominence_db", columns="threshold_pct", values="passes")
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=6)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{r:.0f}" for r in piv.index])
    ax.set_xlabel("closure_threshold_pct")
    ax.set_ylabel("closure_prominence_db")
    ax.set_title("# corpora passing LITERATURE_TARGETS (of 6)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{int(piv.values[i, j])}", ha="center", va="center", fontsize=12)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stability.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
