"""Context robustness check: does the independent period cross-detector reproduce
the primary detector's per-context closure-count ranking?

The paper's second main finding is that phonotactic context modulates closure
count, with intervocalic /rr/ showing the FEWEST closures and onset trills more.
A caveat discussed in the paper is that intervocalic trills are fully voiced and
vowel-flanked, so their mid-band closure minima are shallower and may be
under-detected by the *primary* closure detector, which could partly produce
the effect.

This script tests that possibility on the existing cross-validation tables. The
cross-detector (trillscope/detector/period_detector.py) counts closures purely by
autocorrelation periodicity, not by minima depth, so if it reproduces the same
context ranking the effect is not a primary-detector artifact. Both counts live,
per token, in
outputs/tables/cross_validation_<corpus>.parquet (columns n_closures_v2 and
n_closures_period), tagged with context_label -- no join needed.

Read-only over existing tables. Writes only to reports/v2/ and outputs/v2/tables/.
Run from repo root:  python -m trillscope.context_cross_detector
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CORPORA = ["dimex100", "albayzin", "glissando", "preseea", "tedx", "heroico"]
# The three analysis contexts the paper reports (drop DIMEx's post_vocalic/unknown).
CONTEXTS = ["intervocalic_rr", "after_nls", "word_initial"]

TABLES_DIR = Path("outputs/tables")
OUT_CSV = Path("outputs/v2/tables/context_cross_detector.csv")
OUT_MD = Path("reports/v2/context_cross_detector.md")


def _load() -> pd.DataFrame:
    frames = []
    for ds in CORPORA:
        p = TABLES_DIR / f"cross_validation_{ds}.parquet"
        if not p.exists():
            print(f"  [skip] missing {p}")
            continue
        df = pd.read_parquet(p)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # Drop the 202 exact-duplicate rows so token-level means match the canonical
    # 3,560-token analysis set (speaker x context means are already robust to them).
    out = out.drop_duplicates(["dataset", "utt_id", "start_ms", "end_ms"]).reset_index(drop=True)
    return out


def _context_means(df: pd.DataFrame, count_col: str, by_speaker: bool) -> pd.Series:
    """Mean closure count per context, either token-level or speaker x context-level."""
    sub = df.dropna(subset=[count_col])
    if by_speaker:
        cell = sub.groupby(["speaker_id", "dataset", "context_label"])[count_col].mean().reset_index()
        return cell.groupby("context_label")[count_col].mean()
    return sub.groupby("context_label")[count_col].mean()


def _ranking(s: pd.Series) -> list[str]:
    return list(s.reindex(CONTEXTS).sort_values().index)


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    raw = _load()
    n_raw = len(raw)
    df = raw[raw["context_label"].isin(CONTEXTS)].copy()
    n_period_nan = int(df["n_closures_period"].isna().sum())
    print(f"Loaded {n_raw} cross-validated tokens; "
          f"{len(df)} in the 3 analysis contexts; "
          f"{n_period_nan} have no cross-detector count (period failed).")

    lines: list[str] = []
    lines.append("# Context robustness: primary vs. period cross-detector\n")
    lines.append(
        "Per-context **mean closure count** from the primary detector "
        "(`n_closures_v2`) and the independent period cross-detector "
        "(`n_closures_period`), on the same tokens. The cross-detector counts by "
        "autocorrelation periodicity, not by closure-minima depth, so agreement on "
        "the ranking shows the context effect is not an artifact of intervocalic "
        "closure under-detection.\n")
    lines.append(f"- Tokens (3 contexts): **{len(df)}** "
                 f"({n_period_nan} dropped from the cross-detector column for a failed period estimate)\n")

    # ---- Pooled, two aggregations -------------------------------------------
    rows = []
    for by_speaker in (False, True):
        label = "speaker x context" if by_speaker else "token-level"
        prim = _context_means(df, "n_closures_v2", by_speaker)
        cross = _context_means(df, "n_closures_period", by_speaker)
        for ctx in CONTEXTS:
            rows.append({
                "scope": "pooled",
                "aggregation": label,
                "context": ctx,
                "primary_mean_n": round(float(prim.get(ctx, np.nan)), 3),
                "cross_mean_n": round(float(cross.get(ctx, np.nan)), 3),
            })
        lines.append(f"\n## Pooled, {label}\n")
        lines.append("| context | primary mean n | cross-detector mean n |")
        lines.append("|---|---:|---:|")
        for ctx in CONTEXTS:
            lines.append(f"| {ctx} | {prim.get(ctx, np.nan):.3f} | {cross.get(ctx, np.nan):.3f} |")
        lines.append("")
        lines.append(f"- primary ranking (fewest->most): {' < '.join(_ranking(prim))}")
        lines.append(f"- cross-detector ranking (fewest->most): {' < '.join(_ranking(cross))}")
        prim_iv_lowest = _ranking(prim)[0] == "intervocalic_rr"
        cross_iv_lowest = _ranking(cross)[0] == "intervocalic_rr"
        lines.append(f"- intervocalic is fewest -- primary: **{prim_iv_lowest}**, "
                     f"cross-detector: **{cross_iv_lowest}**")

    # ---- Per corpus (token-level; speaker x context is sparse per corpus) ----
    lines.append("\n## Per corpus (token-level mean n)\n")
    lines.append("| corpus | ctx | primary | cross | primary rank | cross rank | iv fewest (prim/cross) |")
    lines.append("|---|---|---:|---:|---|---|---|")
    per_corpus_agree = 0
    per_corpus_total = 0
    for ds in CORPORA:
        d = df[df["dataset"] == ds]
        if d.empty:
            continue
        prim = _context_means(d, "n_closures_v2", by_speaker=False)
        cross = _context_means(d, "n_closures_period", by_speaker=False)
        present = [c for c in CONTEXTS if c in prim.index and c in cross.dropna().index]
        if len(present) < 2:
            lines.append(f"| {ds} | (only {len(present)} ctx present) | | | | | |")
            continue
        prim_rank = list(prim.reindex(present).sort_values().index)
        cross_rank = list(cross.reindex(present).sort_values().index)
        prim_iv = prim_rank[0] == "intervocalic_rr" if "intervocalic_rr" in present else None
        cross_iv = cross_rank[0] == "intervocalic_rr" if "intervocalic_rr" in present else None
        for ctx in present:
            rows.append({
                "scope": ds, "aggregation": "token-level", "context": ctx,
                "primary_mean_n": round(float(prim.get(ctx, np.nan)), 3),
                "cross_mean_n": round(float(cross.get(ctx, np.nan)), 3),
            })
        agree = prim_rank == cross_rank
        per_corpus_total += 1
        per_corpus_agree += int(agree)
        lines.append(
            f"| {ds} | {','.join(present)} | "
            f"{'/'.join(f'{prim.get(c, np.nan):.2f}' for c in present)} | "
            f"{'/'.join(f'{cross.get(c, np.nan):.2f}' for c in present)} | "
            f"{' < '.join(prim_rank)} | {' < '.join(cross_rank)} | "
            f"{prim_iv}/{cross_iv} |")

    lines.append(f"\n- Per-corpus full-ranking agreement: "
                 f"**{per_corpus_agree}/{per_corpus_total}** corpora.\n")

    # ---- Verdict -------------------------------------------------------------
    prim_sxc = _context_means(df, "n_closures_v2", by_speaker=True)
    cross_sxc = _context_means(df, "n_closures_period", by_speaker=True)
    same_rank = _ranking(prim_sxc) == _ranking(cross_sxc)
    iv_lowest_both = _ranking(prim_sxc)[0] == "intervocalic_rr" and _ranking(cross_sxc)[0] == "intervocalic_rr"
    lines.append("## Verdict\n")
    lines.append(
        f"At the pooled speaker x context level (the unit the paper reports), the "
        f"primary and period cross-detector rankings are "
        f"{'**identical**' if same_rank else 'different'} "
        f"({' < '.join(_ranking(prim_sxc))}), and intervocalic is the fewest under "
        f"{'**both**' if iv_lowest_both else 'only one'} detectors. The cross-detector, "
        f"which counts by periodicity rather than minima depth, gives a mean intervocalic "
        f"count of {cross_sxc['intervocalic_rr']:.2f} versus "
        f"{prim_sxc['intervocalic_rr']:.2f} for the primary detector; a cross-detector count "
        f"at or above the primary count indicates that the ranking is not explained by "
        f"intervocalic closure under-detection. The pattern is carried by the large "
        f"spontaneous corpora (glissando, tedx); small corpora have few tokens per context "
        f"cell and are individually noisy in both detectors.\n")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_CSV} and {OUT_MD}")
    # Echo the pooled tables to stdout for the run log.
    print("\n".join(lines))


if __name__ == "__main__":
    main()
