"""Export closure counts in the acoustic_measurements schema.

`trillscope.statistics` reads `acoustic_measurements_<dataset>.parquet` and uses
`num_cycles`, `voicing_pct`, `mean_f0_hz`, `mean_hnr_db`, etc. To run the
statistical analysis on closure counts, this module writes a parallel set of
parquets under `outputs/tables_v2/` with the same schema, where:

  num_cycles        <- n_closures_v2 (number of closures)
  cycle_rate_hz     <- num_cycles / duration
  cycle_regularity  <- period_cv_v2 (CV of inter-closure intervals; 0 = regular)
  cycle_confidence  <- confidence_v2

All other acoustic features (voicing_pct, mean_f0_hz, mean_hnr_db, intensity)
are taken from the envelope-peak measurement tables
(`outputs/tables/acoustic_measurements_<dataset>.parquet`); only the cycle
count is replaced.

Run:
    python -m trillscope.bridge_to_v1_stats
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CORPORA = ["dimex100", "albayzin", "glissando", "preseea", "tedx", "heroico"]


def bridge_dataset(
    closures_v2_path: Path,
    measurements_v1_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    if not closures_v2_path.exists():
        log.warning("missing %s", closures_v2_path)
        return pd.DataFrame()
    if not measurements_v1_path.exists():
        log.warning("missing %s", measurements_v1_path)
        return pd.DataFrame()

    v2 = pd.read_parquet(closures_v2_path)
    v1 = pd.read_parquet(measurements_v1_path)
    log.info("%s: %d closure rows, %d envelope-peak measurement rows",
             closures_v2_path.stem, len(v2), len(v1))

    v1 = v1.drop_duplicates(subset=["utt_id", "start_ms", "end_ms"])
    v2 = v2.drop_duplicates(subset=["utt_id", "start_ms", "end_ms"])
    v1["start_ms_r"] = v1["start_ms"].round(4)
    v1["end_ms_r"] = v1["end_ms"].round(4)
    v2["start_ms_r"] = v2["start_ms"].round(4)
    v2["end_ms_r"] = v2["end_ms"].round(4)

    feature_cols = [
        "voicing_pct", "mean_f0_hz", "f0_range_hz", "mean_hnr_db",
        "mean_intensity_db", "max_intensity_db", "min_intensity_db",
        "intensity_range_db", "warnings",
    ]
    feature_cols = [c for c in feature_cols if c in v1.columns]

    merged = v2.merge(
        v1[["utt_id", "start_ms_r", "end_ms_r", *feature_cols]],
        on=["utt_id", "start_ms_r", "end_ms_r"],
        how="left",
    )

    duration_ms = merged["duration_ms"].astype(float)
    n_cycles = merged["n_closures_v2"].astype(int)
    out = pd.DataFrame({
        "utt_id": merged["utt_id"],
        "speaker_id": merged["speaker_id"],
        "dataset": merged["dataset"],
        "word": merged["word"] if "word" in merged.columns else "",
        "start_ms": merged["start_ms"],
        "end_ms": merged["end_ms"],
        "duration_ms": duration_ms,
        "num_cycles": n_cycles,
        "cycle_rate_hz": np.where(
            duration_ms > 0, 1000.0 * n_cycles / duration_ms, 0.0
        ),
        "cycle_regularity": merged["period_cv_v2"],
        "cycle_confidence": merged["confidence_v2"],
        "status": np.where(merged["status_v2"] == "ok", "success", "failed"),
        "context_label": merged["context_label"],
        "phoneme_label": merged["phoneme_label"],
        "audio_path": merged["audio_path"],
    })
    for c in feature_cols:
        if c == "warnings":
            continue
        out[c] = merged[c]
    out["warnings"] = merged.get("warnings", "")

    n_unmatched = out["voicing_pct"].isna().sum() if "voicing_pct" in out.columns else 0
    log.info(
        "  -> %s: %d rows (%d unmatched against envelope-peak measurement features)",
        out_path.name, len(out), n_unmatched,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="trillscope.bridge_to_v1_stats")
    ap.add_argument("--src-dir", type=Path, default=Path("outputs/tables"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/tables_v2"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    total = 0
    for ds in CORPORA:
        df = bridge_dataset(
            args.src_dir / f"closures_v2_{ds}.parquet",
            args.src_dir / f"acoustic_measurements_{ds}.parquet",
            args.out_dir / f"acoustic_measurements_{ds}.parquet",
        )
        total += len(df)
    log.info("done. %d total rows across %d corpora", total, len(CORPORA))


if __name__ == "__main__":
    main()
