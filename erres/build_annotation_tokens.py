"""Build a unified tokens.parquet for manual closure annotation (Phase A2).

Joins:
- outputs/tables/r_candidates_<dataset>.parquet  (timing, context)
- outputs/tables/acoustic_measurements_<dataset>.parquet  (auto_cycles_old)
- metadata/metadata_unified.parquet  (sex, when available)

into a single frame whose schema matches what annotations/extract_for_annotation.py
expects: token_id, audio_path, t0_ms, t1_ms, speaker_id, corpus, sex, context,
auto_cycles_old.

Only the six corpora included in the IberSpeech 2026 sample are emitted:
DIMEx100, ALBAYZIN, Glissando, PRESEEA, TEDx, Heroico. M-AILABS is excluded
(prose); CommonVoice is excluded (crowdsourced, demographic-noisy).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


CORPUS_CANONICAL = {
    "dimex100": "DIMEx100",
    "albayzin": "ALBAYZIN",
    "glissando": "Glissando",
    "preseea": "PRESEEA",
    "tedx": "TEDx",
    "heroico": "Heroico",
}

INCLUDED_CONTEXTS = {"intervocalic_rr", "word_initial", "after_nls"}


def build(
    candidates_dir: Path,
    metadata_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    meta = pd.read_parquet(metadata_path)
    speaker_sex = (
        meta.dropna(subset=["speaker_id"])
        .drop_duplicates(subset=["speaker_id"])[["speaker_id", "sex"]]
    )

    frames: list[pd.DataFrame] = []
    for ds_key, corpus_name in CORPUS_CANONICAL.items():
        cand_path = candidates_dir / f"r_candidates_{ds_key}.parquet"
        meas_path = candidates_dir / f"acoustic_measurements_{ds_key}.parquet"
        if not cand_path.exists():
            log.warning("skip %s: %s missing", ds_key, cand_path)
            continue

        cand = pd.read_parquet(cand_path)
        log.info("%s: %d candidate rows", ds_key, len(cand))

        cand = cand[cand["duration_ms"] > 0]
        if "alignment_source" in cand.columns:
            cand = cand[cand["alignment_source"] != "orthographic"]
        cand = cand[cand["context_label"].isin(INCLUDED_CONTEXTS)]
        log.info("%s: %d after timing+context filters", ds_key, len(cand))

        if meas_path.exists():
            meas = pd.read_parquet(meas_path)
            meas = (
                meas[meas["status"] == "success"][
                    [
                        "utt_id", "start_ms", "end_ms",
                        "num_cycles", "voicing_pct", "cycle_confidence",
                        "mean_intensity_db",
                    ]
                ]
                .drop_duplicates(subset=["utt_id", "start_ms", "end_ms"])
                .rename(columns={"num_cycles": "auto_cycles_old"})
            )
            cand = cand.merge(meas, on=["utt_id", "start_ms", "end_ms"], how="left")
        else:
            cand["auto_cycles_old"] = pd.NA
            cand["voicing_pct"] = pd.NA
            cand["cycle_confidence"] = pd.NA
            cand["mean_intensity_db"] = pd.NA

        cand = cand.merge(speaker_sex, on="speaker_id", how="left")
        cand["sex"] = cand["sex"].fillna("U")

        cand["token_id"] = (
            ds_key
            + "-"
            + cand["utt_id"].astype(str)
            + "-"
            + cand["word_idx"].astype(str)
            + "-"
            + cand["r_idx_in_word"].astype(str)
        )
        cand["corpus"] = corpus_name
        cand = cand.rename(columns={"start_ms": "t0_ms", "end_ms": "t1_ms", "context_label": "context"})

        keep = [
            "token_id", "audio_path", "t0_ms", "t1_ms", "duration_ms",
            "speaker_id", "corpus", "sex", "context", "auto_cycles_old",
            "voicing_pct", "cycle_confidence", "mean_intensity_db",
        ]
        frames.append(cand[keep])

    if not frames:
        raise RuntimeError("no candidates parquets found")

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["auto_cycles_old"])
    out["auto_cycles_old"] = out["auto_cycles_old"].astype(int)
    # v1's r_candidates have some duplicated rows (e.g. r_candidates_albayzin
    # is fully duplicated 2x). Collapse to a single row per token_id.
    out = out.drop_duplicates(subset=["token_id"], keep="first").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    log.info("wrote %s: %d rows across %d corpora", output_path, len(out), out["corpus"].nunique())
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="erres.build_annotation_tokens")
    ap.add_argument("--candidates-dir", type=Path, default=Path("outputs/tables"))
    ap.add_argument("--metadata", type=Path, default=Path("metadata/metadata_unified.parquet"))
    ap.add_argument("--out", type=Path, default=Path("outputs/tables/tokens.parquet"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    df = build(args.candidates_dir, args.metadata, args.out)
    print(df.groupby(["corpus", "sex"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
