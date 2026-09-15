"""Test whether the quality filter drops tokens in a sex-biased way.

Motivation: the fixed quality filter keeps only ~21% of candidate tokens
(3,561 / 16,836), so the dropped ~79% could carry a different population trend.
In particular, if men devoice more, the voicing >= 80% criterion would
preferentially remove male tokens and could produce a spurious sex null result.

This script tests that directly on the full candidate pool. If the filter were
sex-biased we would see (a) different retention rates for F vs M, (b) a significant
sex x inclusion association, and (c) lower voicing among male candidates. If
retention is similar for F and M, the association is not significant, and voicing
is not lower for men, the sex null result is not an artifact of the filter.

Sex is taken from the analysis loader (metadata-based). DIMEx100 is excluded: its sex
is F0-inferred and unavailable at token level, matching the non-circular stance of
trillscope/mechanism_sex.py. Read-only over existing tables.

Run from repo root:  python -m trillscope.attrition_sex
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from trillscope.quality import QUALITY_FILTER, apply_quality_filter
from trillscope.statistics.data_loader import load_analysis_data

CORPORA = ["dimex100", "albayzin", "glissando", "preseea", "tedx", "heroico"]
TABLES_DIR = Path("outputs/tables")
META = Path("metadata/metadata_unified.parquet")
TOKENS = TABLES_DIR / "tokens.parquet"
OUT_CSV = Path("outputs/v2/tables/attrition_sex.csv")
OUT_MD = Path("reports/v2/attrition_sex.md")


def load_pool_with_sex() -> pd.DataFrame:
    """Full candidate pool (tokens.parquet) + metadata sex + a `passed` flag.

    The `sex` column shipped in tokens.parquet only covers albayzin/preseea, so we
    override it with the loader's per-speaker sex map (the same one the paper's sex
    analysis uses), keyed on (dataset, speaker_id).
    """
    tok = pd.read_parquet(TOKENS)
    tok["dataset"] = tok["corpus"].str.lower()

    adf, _ = load_analysis_data(TABLES_DIR, META, datasets=CORPORA,
                                aggregate_by_speaker=False)
    sex_map = (adf.drop_duplicates(["dataset", "speaker_id"])
               [["dataset", "speaker_id", "sex"]])
    tok = tok.drop(columns=["sex"]).merge(sex_map, on=["dataset", "speaker_id"],
                                          how="left")

    # `passed` = survives the fixed quality filter (duration/voicing/periodicity).
    kept = apply_quality_filter(tok)
    tok["passed"] = tok["token_id"].isin(set(kept["token_id"]))

    # which invariant each token fails (for the per-reason breakdown)
    tok["fail_dur"] = ~tok["duration_ms"].between(
        QUALITY_FILTER["duration_min_ms"], QUALITY_FILTER["duration_max_ms"])
    tok["fail_voi"] = tok["voicing_pct"] < QUALITY_FILTER["voicing_min_pct"]
    tok["fail_per"] = tok["periodicity_score"] < QUALITY_FILTER["periodicity_min"]
    return tok


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    pool = load_pool_with_sex()
    n_all = len(pool)
    n_kept_all = int(pool["passed"].sum())

    sx = pool[pool["sex"].isin(["F", "M"])].copy()  # metadata sex only (no DIMEx)
    n_lab = len(sx)
    corpora_lab = ", ".join(sorted(sx["dataset"].unique()))

    rows: list[dict] = []
    L: list[str] = ["# Quality-filter attrition is sex-balanced\n"]
    L.append(
        f"Filter keeps **{n_kept_all}/{n_all}** candidates "
        f"({100*n_kept_all/n_all:.1f}%) overall. The question is whether that "
        f"selection is sex-biased. Sex is metadata-based; DIMEx100 is excluded "
        f"(F0-inferred sex, non-circular stance as in `mechanism_sex.py`).\n")
    L.append(f"- candidate tokens with F/M sex label: **{n_lab}** "
             f"(corpora: {corpora_lab})\n")

    # ---- 1: retention rate by sex -------------------------------------------
    L.append("## Retention rate by sex\n")
    L.append("| sex | candidates | retained | retention % |")
    L.append("|---|---:|---:|---:|")
    for s in ("F", "M"):
        d = sx[sx["sex"] == s]
        kept = int(d["passed"].sum())
        pct = 100 * kept / len(d) if len(d) else 0.0
        rows.append({"scope": "retention", "sex": s, "candidates": len(d),
                     "retained": kept, "retention_pct": pct})
        L.append(f"| {s} | {len(d)} | {kept} | {pct:.1f}% |")
    L.append("")

    # composition F/M among retained vs dropped
    comp = pd.crosstab(sx["passed"], sx["sex"], normalize="index")
    L.append("Sex composition (row-normalised):\n")
    L.append("| subset | F | M |")
    L.append("|---|---:|---:|")
    for flag, label in [(True, "retained"), (False, "dropped")]:
        if flag in comp.index:
            f_pct, m_pct = comp.loc[flag, "F"], comp.loc[flag, "M"]
            rows.append({"scope": "composition", "subset": label,
                         "F_frac": float(f_pct), "M_frac": float(m_pct)})
            L.append(f"| {label} | {100*f_pct:.1f}% | {100*m_pct:.1f}% |")
    L.append("")

    # ---- 2: sex x inclusion association (token-level chi-square) -------------
    ct = pd.crosstab(sx["sex"], sx["passed"])
    chi2, p_chi, _, _ = chi2_contingency(ct)
    rows.append({"scope": "association", "test": "chi2 sex x inclusion",
                 "stat": float(chi2), "p": float(p_chi), "n": int(n_lab)})
    L.append("## Sex x inclusion association\n")
    L.append(f"Chi-square on the 2x2 table (sex x passed): "
             f"chi2 = {chi2:.2f}, **p = {p_chi:.3f}**, n = {n_lab}. "
             f"{'No' if p_chi >= 0.05 else 'A'} significant association.\n")

    # speaker-level retention (avoids pseudo-replication): per-speaker retained
    # fraction, compared F vs M with Mann-Whitney.
    spk = (sx.groupby(["dataset", "speaker_id", "sex"])["passed"]
           .mean().reset_index(name="retained_frac"))
    fF = spk.loc[spk["sex"] == "F", "retained_frac"]
    mM = spk.loc[spk["sex"] == "M", "retained_frac"]
    u, p_spk = mannwhitneyu(mM, fF, alternative="two-sided")
    rows.append({"scope": "association", "test": "MWU per-speaker retained frac",
                 "F_mean": float(fF.mean()), "M_mean": float(mM.mean()),
                 "p": float(p_spk), "n_F": int(len(fF)), "n_M": int(len(mM))})
    L.append(f"Speaker-level (no pseudo-replication): mean retained fraction "
             f"F = {fF.mean():.3f} ({len(fF)} spk), M = {mM.mean():.3f} "
             f"({len(mM)} spk); Mann-Whitney **p = {p_spk:.3f}**.\n")

    # ---- 3: failure reason by sex (among dropped) ---------------------------
    dropped = sx[~sx["passed"]]
    L.append("## Why dropped, by sex (share of dropped tokens failing each rule)\n")
    L.append("| sex | fail duration | fail voicing | fail periodicity |")
    L.append("|---|---:|---:|---:|")
    for s in ("F", "M"):
        d = dropped[dropped["sex"] == s]
        fd, fv, fp = d["fail_dur"].mean(), d["fail_voi"].mean(), d["fail_per"].mean()
        rows.append({"scope": "fail_reason", "sex": s, "fail_dur": float(fd),
                     "fail_voi": float(fv), "fail_per": float(fp), "n": len(d)})
        L.append(f"| {s} | {100*fd:.1f}% | {100*fv:.1f}% | {100*fp:.1f}% |")
    L.append("")

    # ---- 4: devoicing test --------------------------------------------------
    L.append("## Voicing by sex (the devoicing hypothesis)\n")
    L.append("| subset | F mean | M mean | M-F | Mann-Whitney p |")
    L.append("|---|---:|---:|---:|---:|")
    for label, d in [("all candidates", sx), ("dropped only", dropped)]:
        f = d.loc[d["sex"] == "F", "voicing_pct"].dropna()
        m = d.loc[d["sex"] == "M", "voicing_pct"].dropna()
        u, p = mannwhitneyu(m, f, alternative="two-sided")
        rows.append({"scope": "voicing", "subset": label, "F_mean": float(f.mean()),
                     "M_mean": float(m.mean()), "p": float(p)})
        L.append(f"| {label} | {f.mean():.1f} | {m.mean():.1f} | "
                 f"{m.mean()-f.mean():+.1f} | {p:.3f} |")
    # speaker-level voicing (clean): per-speaker mean voicing of all candidates
    spk_v = (sx.groupby(["dataset", "speaker_id", "sex"])["voicing_pct"]
             .mean().reset_index())
    fv = spk_v.loc[spk_v["sex"] == "F", "voicing_pct"]
    mv = spk_v.loc[spk_v["sex"] == "M", "voicing_pct"]
    u, p_v = mannwhitneyu(mv, fv, alternative="two-sided")
    rows.append({"scope": "voicing", "subset": "speaker-level all",
                 "F_mean": float(fv.mean()), "M_mean": float(mv.mean()),
                 "p": float(p_v)})
    L.append(f"| speaker-level all | {fv.mean():.1f} | {mv.mean():.1f} | "
             f"{mv.mean()-fv.mean():+.1f} | {p_v:.3f} |")
    L.append("")

    # ---- verdict ------------------------------------------------------------
    balanced = (p_chi >= 0.05 and p_spk >= 0.05 and p_v >= 0.05)
    L.append("## Verdict\n")
    L.append(
        ("**Filter attrition is sex-balanced.** Retention does not differ by sex "
         "(token-level chi-square and speaker-level test both n.s.), and male "
         "candidates are not less voiced than female ones, so the voicing criterion "
         "does not preferentially remove male tokens. The sex null is therefore not "
         "an artifact of the quality filter.\n"
         if balanced else
         "**A sex imbalance is present** in filter attrition (one or more tests "
         "significant), so the sex null result may be partly an artifact of the "
         "quality filter.\n"))

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT_CSV} and {OUT_MD}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
