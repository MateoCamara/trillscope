"""Demonstrate that the spurious sex effect is a counting artifact tied to F0.

The paper argues that the v1 envelope-peak counter inflates "cycles" for low-F0
(male) voices because it registers a maximum for both the occlusion and its release,
so the count scales with harmonic density -- whereas the v2 closure counter, anchored
on energy minima with verified releases, does not.

Here we test it directly on the quality-filtered tokens, now that a real per-token F0
is available (trillscope/compute_f0_for_tokens.py). Predictions:

  1. The per-token over-count  excess = num_cycles(v1) - n_closures_v2  rises as F0
     falls  ->  Spearman rho(excess, F0) < 0.
  2. The v1 count correlates with F0 while the v2 count does not  ->
     rho(num_cycles, F0) clearly negative & significant; rho(n_closures_v2, F0) ~ 0.
  3. Consequence (independent, metadata-based sex; DIMEx excluded as its sex is
     F0-inferred and would be circular): men show a v1-count advantage but no
     v2-count advantage.

Read-only over existing tables. Writes reports/v2/mechanism_sex.md + CSV.
Run from repo root (after compute_f0_for_tokens):  python -m trillscope.mechanism_sex
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from trillscope.statistics.data_loader import load_analysis_data

CORPORA = ["dimex100", "albayzin", "glissando", "preseea", "tedx", "heroico"]
TABLES_DIR = Path("outputs/tables")
META = Path("metadata/metadata_unified.parquet")
OUT_CSV = Path("outputs/v2/tables/mechanism_sex.csv")
OUT_MD = Path("reports/v2/mechanism_sex.md")


def _rho(a: pd.Series, b: pd.Series):
    m = a.notna() & b.notna()
    if m.sum() < 10:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(a[m], b[m])
    return float(r), float(p), int(m.sum())


def _load_tokens() -> pd.DataFrame:
    # 1. quality-filtered tokens with the v2 closure count
    frames = []
    for ds in CORPORA:
        p = TABLES_DIR / f"closures_v2_{ds}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p)[
                ["dataset", "utt_id", "speaker_id", "start_ms", "end_ms",
                 "n_closures_v2", "context_label"]])
    tok = pd.concat(frames, ignore_index=True)
    # closures_v2 carries 202 exact-duplicate rows (ALBAYZIN n=0 tokens emitted
    # twice); the canonical analysis set is the 4-key-deduplicated 3,560 tokens.
    tok = tok.drop_duplicates(["dataset", "utt_id", "start_ms", "end_ms"]).reset_index(drop=True)

    # 2. real per-token F0 (same source rows -> exact key)
    f0 = (pd.read_parquet(TABLES_DIR / "per_token_f0.parquet")
          .drop_duplicates(["dataset", "utt_id", "start_ms", "end_ms"]))
    tok = tok.merge(f0[["dataset", "utt_id", "start_ms", "end_ms", "mean_f0_hz"]],
                    on=["dataset", "utt_id", "start_ms", "end_ms"], how="left")

    # 3. v1 envelope-peak count + independent sex, via the paper's loader
    adf, _ = load_analysis_data(TABLES_DIR, META, datasets=CORPORA,
                                aggregate_by_speaker=False)
    adf = adf.copy()
    adf["skey"] = adf["start_ms"].round().astype("Int64")
    v1 = (adf[["dataset", "utt_id", "skey", "num_cycles"]]
          .drop_duplicates(["dataset", "utt_id", "skey"]))
    tok["skey"] = tok["start_ms"].round().astype("Int64")
    tok = tok.merge(v1, on=["dataset", "utt_id", "skey"], how="left")

    sex_map = (adf.drop_duplicates(["dataset", "speaker_id"])
               [["dataset", "speaker_id", "sex"]])
    tok = tok.merge(sex_map, on=["dataset", "speaker_id"], how="left")

    tok["excess"] = tok["num_cycles"] - tok["n_closures_v2"]
    return tok


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    tok = _load_tokens()

    n = len(tok)
    n_f0 = int(tok["mean_f0_hz"].notna().sum())
    n_v1 = int(tok["num_cycles"].notna().sum())
    both = tok["mean_f0_hz"].notna() & tok["num_cycles"].notna()
    core = tok[both].copy()

    L: list[str] = ["# Sex artifact = counting method, not articulation\n"]
    L.append(f"- Quality-filtered tokens: **{n}**")
    L.append(f"- with real F0 (librosa pyin): **{n_f0}** ({100*n_f0/n:.1f}%)")
    L.append(f"- matched to v1 envelope-peak count: **{n_v1}** ({100*n_v1/n:.1f}%)")
    L.append(f"- usable for the F0 mechanism test (both): **{len(core)}**\n")
    L.append(f"- mean v1 count {core['num_cycles'].mean():.2f}, "
             f"mean v2 count {core['n_closures_v2'].mean():.2f}, "
             f"mean over-count (excess) {core['excess'].mean():.2f}\n")

    rows = []

    # ---- 1 & 2: token-level F0 mechanism ------------------------------------
    L.append("## Token-level F0 mechanism\n")
    L.append("| correlate | Spearman rho | p | n |")
    L.append("|---|---:|---:|---:|")
    for name, col in [("excess vs F0", "excess"),
                      ("v1 envelope-peak count vs F0", "num_cycles"),
                      ("v2 closure count vs F0", "n_closures_v2")]:
        r, p, k = _rho(core[col], core["mean_f0_hz"])
        rows.append({"scope": "token", "test": name, "rho": r, "p": p, "n": k})
        L.append(f"| {name} | {r:+.3f} | {p:.1e} | {k} |")
    L.append("")

    # ---- robustness: speaker-level ------------------------------------------
    spk = (core.groupby(["dataset", "speaker_id"])
           .agg(excess=("excess", "mean"), num_cycles=("num_cycles", "mean"),
                n_closures_v2=("n_closures_v2", "mean"),
                mean_f0_hz=("mean_f0_hz", "mean")).reset_index())
    L.append("## Speaker-level (robustness to pseudo-replication)\n")
    L.append("| correlate | Spearman rho | p | n speakers |")
    L.append("|---|---:|---:|---:|")
    for name, col in [("excess vs F0", "excess"),
                      ("v1 count vs F0", "num_cycles"),
                      ("v2 count vs F0", "n_closures_v2")]:
        r, p, k = _rho(spk[col], spk["mean_f0_hz"])
        rows.append({"scope": "speaker", "test": name, "rho": r, "p": p, "n": k})
        L.append(f"| {name} | {r:+.3f} | {p:.1e} | {k} |")
    L.append("")

    # ---- 3: independent-sex consequence (no DIMEx) --------------------------
    sx = core[(core["dataset"] != "dimex100") & (core["sex"].isin(["F", "M"]))]
    spk_sx = (sx.groupby(["dataset", "speaker_id", "sex"])
              .agg(num_cycles=("num_cycles", "mean"),
                   n_closures_v2=("n_closures_v2", "mean"),
                   excess=("excess", "mean"),
                   mean_f0_hz=("mean_f0_hz", "mean")).reset_index())
    nF = int((spk_sx["sex"] == "F").sum())
    nM = int((spk_sx["sex"] == "M").sum())
    L.append("## Consequence: metadata-based sex (DIMEx excluded, non-circular)\n")
    L.append(f"- speakers: **{nF} F, {nM} M** "
             f"(albayzin, glissando, preseea, tedx, heroico)\n")
    L.append("| measure | F mean | M mean | M-F | Mann-Whitney p | rank-biserial r |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for name, col in [("F0 (Hz)", "mean_f0_hz"),
                      ("v1 envelope-peak count", "num_cycles"),
                      ("v2 closure count", "n_closures_v2"),
                      ("over-count (excess)", "excess")]:
        f = spk_sx.loc[spk_sx["sex"] == "F", col]
        m = spk_sx.loc[spk_sx["sex"] == "M", col]
        u, p = mannwhitneyu(m, f, alternative="two-sided")
        r_rb = 2 * u / (len(m) * len(f)) - 1  # rank-biserial
        rows.append({"scope": "sex", "test": name, "F_mean": float(f.mean()),
                     "M_mean": float(m.mean()), "p": float(p),
                     "rank_biserial": float(r_rb)})
        L.append(f"| {name} | {f.mean():.2f} | {m.mean():.2f} | "
                 f"{m.mean()-f.mean():+.2f} | {p:.1e} | {r_rb:+.2f} |")
    L.append("")

    # ---- verdict ------------------------------------------------------------
    r_ex, p_ex, _ = _rho(core["excess"], core["mean_f0_hz"])
    r_v1, p_v1, _ = _rho(core["num_cycles"], core["mean_f0_hz"])
    r_v2, p_v2, _ = _rho(core["n_closures_v2"], core["mean_f0_hz"])
    demonstrated = (r_ex < 0 and p_ex < 0.05 and r_v1 < 0 and p_v1 < 0.05
                    and abs(r_v2) < abs(r_v1) / 2)
    L.append("## Verdict\n")
    L.append(
        f"The envelope-peak over-count rises as F0 falls (rho={r_ex:+.2f}, p={p_ex:.1e}). "
        f"The v1 count is F0-dependent (rho={r_v1:+.2f}, p={p_v1:.1e}) while the v2 "
        f"closure count is {'essentially F0-independent' if abs(r_v2) < abs(r_v1)/2 else 'also F0-linked'} "
        f"(rho={r_v2:+.2f}, p={p_v2:.1e}). "
        + ("**Mechanism demonstrated**: the apparent sex effect is located in the "
           "counting method, not in articulation.\n"
           if demonstrated else
           "Result is mixed; not a clean demonstration.\n"))

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT_CSV} and {OUT_MD}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
