"""Bootstrap 95% CIs for the medians reported in the v2 paper tables.

Reuses the v2 statistics loader (``src.statistics.data_loader``) so the unit of
analysis matches the inferential tests: speaker-level for sex / corpus, and
speaker x context for the phonotactic-context effect. ``num_cycles`` and
``duration_ms`` come from the bridged v2 measurements; ``period_ms`` comes from
the independent cross-detector table (``period_ms_v2``).

Run:
    python -m src.statistics.bootstrap_cis
Output:
    outputs/v2/tables/bootstrap_cis.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loader import load_analysis_data, _aggregate_by_speaker
from .inferential import bootstrap_median_ci

CORPORA = ["albayzin", "dimex100", "preseea", "glissando", "tedx", "heroico"]
MEAS_DIR = Path("outputs/tables_v2")
XVAL_DIR = Path("outputs/tables")
META = Path("metadata/metadata_unified.parquet")
OUT = Path("outputs/v2/tables/bootstrap_cis.csv")

_SKIP = {"unknown", "ambiguous", "", None}


def _emit(rows: list, metric: str, grouping: str, frame: pd.DataFrame) -> None:
    """Append one bootstrap-CI row per level of ``grouping`` for ``metric``."""
    if metric not in frame.columns or grouping not in frame.columns:
        return
    for level, sub in frame.groupby(grouping):
        if level in _SKIP:
            continue
        vals = sub[metric].dropna()
        if len(vals) == 0:
            continue
        lo, hi = bootstrap_median_ci(vals)
        rows.append({
            "metric": metric,
            "grouping": grouping,
            "level": level,
            "n": int(len(vals)),
            "median": float(vals.median()),
            "ci_lo": lo,
            "ci_hi": hi,
        })


def main() -> None:
    # Token-level merged frame (num_cycles, duration_ms, sex, context_label, ...)
    df, _ = load_analysis_data(MEAS_DIR, META, datasets=CORPORA, aggregate_by_speaker=False)
    spk = _aggregate_by_speaker(df, include_context=False)
    spk_ctx = _aggregate_by_speaker(df, include_context=True)

    rows: list[dict] = []
    for metric in ("num_cycles", "duration_ms"):
        _emit(rows, metric, "sex", spk)
        _emit(rows, metric, "dataset", spk)
        _emit(rows, metric, "context_label", spk_ctx)

    # period_ms from the independent cross-detector table, aggregated to speaker
    cv = pd.concat(
        [pd.read_parquet(XVAL_DIR / f"cross_validation_{d}.parquet") for d in CORPORA],
        ignore_index=True,
    ).rename(columns={"period_ms_v2": "period_ms"})
    sex_map = spk.drop_duplicates("speaker_id").set_index("speaker_id")["sex"]
    cv["sex"] = cv["speaker_id"].map(sex_map)
    cv_spk = cv.groupby(["speaker_id", "dataset"], as_index=False).agg(
        period_ms=("period_ms", "mean"), sex=("sex", "first")
    )
    cv_spk_ctx = cv.groupby(["speaker_id", "dataset", "context_label"], as_index=False).agg(
        period_ms=("period_ms", "mean")
    )
    _emit(rows, "period_ms", "sex", cv_spk)
    _emit(rows, "period_ms", "dataset", cv_spk)
    _emit(rows, "period_ms", "context_label", cv_spk_ctx)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {len(out)} rows to {OUT}")


if __name__ == "__main__":
    main()
