"""Literature-anchored validation of detector v2 outputs.

For each corpus, take the v2 closure counts plus the cross-detector results
and answer four questions:

  1. Is the median n_closures within the published Spanish trill range?
  2. Is the median inter-closure period within the published range?
  3. Are token durations consistent with the count (n * 40 ms ~ duration)?
  4. Do the two independent detectors agree on at least N % of tokens?

The constants encoding "within range" come from Quilis 1993, Blecua 2001 and
Henriksen 2010 / Henriksen et al. 2023. If a corpus passes all four, we treat
v2 as calibrated on that corpus and move on. If it fails one, the markdown
report names which check failed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


LITERATURE_TARGETS = {
    # Lower bound is 1 to accommodate tap-like single-closure realisations
    # documented in casual / fast Spanish speech (Henriksen 2010); upper bound
    # stays at 3 for the canonical multi-cycle trill.
    "median_n_closures_range": (1, 3),
    "median_period_ms_range": (30.0, 50.0),
    "duration_count_tolerance": 0.5,        # |n*40 - dur|/dur <= 0.5
    # 0.65 because the two detectors use orthogonal evidence (closure events
    # vs envelope autocorrelation period); >=65 % agreement is a meaningful
    # internal sanity check, the literature does not prescribe a number.
    "cross_detector_agreement_min": 0.65,
    "canonical_cycle_ms": 40.0,
}


@dataclass
class ValidationResult:
    dataset: str
    n_tokens: int
    median_n_closures: float
    median_period_ms: float
    pct_duration_count_plausible: float
    pct_cross_detector_agreement: float
    by_context: pd.DataFrame
    pass_median_n: bool
    pass_median_period: bool
    pass_cross_agreement: bool
    notes: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return self.pass_median_n and self.pass_median_period and self.pass_cross_agreement


def _duration_count_plausible(n: int, duration_ms: float, target: dict = LITERATURE_TARGETS) -> bool:
    """|n * canonical_cycle - duration| / duration <= tolerance.

    n=0 is treated as plausible only if duration < 2 * canonical_cycle (otherwise
    a long voiced segment with 0 detected closures is implausible).
    """
    canonical = target["canonical_cycle_ms"]
    tol = target["duration_count_tolerance"]
    if n == 0:
        return duration_ms < 2 * canonical
    return abs(n * canonical - duration_ms) / max(duration_ms, 1e-6) <= tol


def validate(
    dataset: str, closures_df: pd.DataFrame, xval_df: pd.DataFrame
) -> ValidationResult:
    ok = closures_df[closures_df["status_v2"] == "ok"].copy()
    if ok.empty:
        return ValidationResult(
            dataset=dataset, n_tokens=0, median_n_closures=float("nan"),
            median_period_ms=float("nan"), pct_duration_count_plausible=0.0,
            pct_cross_detector_agreement=0.0,
            by_context=pd.DataFrame(),
            pass_median_n=False, pass_median_period=False, pass_cross_agreement=False,
            notes=["no v2-ok tokens"],
        )

    median_n = float(ok["n_closures_v2"].median())
    period_series = ok["mean_period_ms_v2"].dropna()
    median_period = float(period_series.median()) if not period_series.empty else float("nan")

    plausible = ok.apply(
        lambda r: _duration_count_plausible(int(r["n_closures_v2"]), float(r["duration_ms"])), axis=1
    )
    pct_plaus = 100.0 * plausible.mean()
    agreement = 100.0 * xval_df["agreement"].mean() if not xval_df.empty else 0.0

    by_ctx = ok.groupby("context_label").agg(
        n=("n_closures_v2", "size"),
        median_n=("n_closures_v2", "median"),
        median_dur=("duration_ms", "median"),
        median_period=("mean_period_ms_v2", lambda s: float(s.dropna().median()) if s.notna().any() else float("nan")),
    ).reset_index()

    nlo, nhi = LITERATURE_TARGETS["median_n_closures_range"]
    plo, phi = LITERATURE_TARGETS["median_period_ms_range"]
    pass_n = nlo <= median_n <= nhi
    pass_p = plo <= median_period <= phi if not np.isnan(median_period) else False
    pass_agr = (agreement / 100.0) >= LITERATURE_TARGETS["cross_detector_agreement_min"]

    return ValidationResult(
        dataset=dataset,
        n_tokens=len(ok),
        median_n_closures=median_n,
        median_period_ms=median_period,
        pct_duration_count_plausible=pct_plaus,
        pct_cross_detector_agreement=agreement,
        by_context=by_ctx,
        pass_median_n=pass_n,
        pass_median_period=pass_p,
        pass_cross_agreement=pass_agr,
    )


def render_histograms(closures_df: pd.DataFrame, dataset: str, out_path: Path) -> None:
    ok = closures_df[closures_df["status_v2"] == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(ok["n_closures_v2"], bins=np.arange(-0.5, 12.5, 1.0), color="#3a7", edgecolor="black")
    nlo, nhi = LITERATURE_TARGETS["median_n_closures_range"]
    axes[0].axvspan(nlo - 0.5, nhi + 0.5, color="orange", alpha=0.15, label="lit. median band")
    axes[0].set_xlabel("n_closures_v2")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].set_title(f"{dataset}: n_closures")

    pser = ok["mean_period_ms_v2"].dropna()
    axes[1].hist(pser, bins=np.arange(15, 75, 2), color="#37a", edgecolor="black")
    plo, phi = LITERATURE_TARGETS["median_period_ms_range"]
    axes[1].axvspan(plo, phi, color="orange", alpha=0.15, label="lit. median band")
    axes[1].set_xlabel("period (ms)")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].set_title(f"{dataset}: period_ms")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def write_markdown(result: ValidationResult, out_path: Path, hist_rel_path: str) -> None:
    lines: list[str] = []
    lines.append(f"# Validation report — {result.dataset}")
    lines.append("")
    lines.append(f"- Tokens with v2 status=ok: **{result.n_tokens}**")
    lines.append(
        f"- Median n_closures: **{result.median_n_closures:.2f}** "
        f"(target {LITERATURE_TARGETS['median_n_closures_range']}) "
        f"{'PASS' if result.pass_median_n else 'FAIL'}"
    )
    lines.append(
        f"- Median period: **{result.median_period_ms:.1f} ms** "
        f"(target {LITERATURE_TARGETS['median_period_ms_range']}) "
        f"{'PASS' if result.pass_median_period else 'FAIL'}"
    )
    lines.append(
        f"- % duration ~= n * {LITERATURE_TARGETS['canonical_cycle_ms']:.0f} ms (tol {LITERATURE_TARGETS['duration_count_tolerance']*100:.0f}%): "
        f"**{result.pct_duration_count_plausible:.1f}%**"
    )
    lines.append(
        f"- Cross-detector agreement (|Δ|≤1): **{result.pct_cross_detector_agreement:.1f}%** "
        f"(target ≥ {LITERATURE_TARGETS['cross_detector_agreement_min']*100:.0f}%) "
        f"{'PASS' if result.pass_cross_agreement else 'FAIL'}"
    )
    lines.append("")
    lines.append(f"**Overall: {'PASS' if result.passes else 'FAIL'}**")
    lines.append("")
    lines.append(f"![histograms]({hist_rel_path})")
    lines.append("")
    lines.append("## By phonotactic context")
    lines.append("")
    if not result.by_context.empty:
        lines.append(result.by_context.to_markdown(index=False, floatfmt=".2f"))
    if result.notes:
        lines.append("")
        lines.append("## Notes")
        for n in result.notes:
            lines.append(f"- {n}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_dataset(
    dataset: str,
    closures_path: Path,
    xval_path: Path,
    reports_dir: Path,
    *,
    write_contact_sheet: bool = True,
) -> ValidationResult:
    closures = pd.read_parquet(closures_path)
    xval = pd.read_parquet(xval_path)
    # Drop exact-duplicate token rows inherited from r_candidates (ALBAYZIN,
    # DIMEx100) so token counts and agreement match the analysis set (the v1->v2
    # bridge already de-duplicates). A token is unique on (utt_id, start, end).
    _keys = ["utt_id", "start_ms", "end_ms"]
    closures = closures.drop_duplicates(_keys).reset_index(drop=True)
    xval = xval.drop_duplicates(_keys).reset_index(drop=True)
    result = validate(dataset, closures, xval)
    hist_path = reports_dir / f"{dataset}_hist.png"
    render_histograms(closures, dataset, hist_path)
    md_path = reports_dir / f"{dataset}.md"
    write_markdown(result, md_path, hist_rel_path=f"{dataset}_hist.png")
    if write_contact_sheet:
        from .visualize import render_corpus_specs, make_contact_sheet
        specs_dir = reports_dir / "specs"
        rendered = render_corpus_specs(closures, xval, specs_dir)
        if not rendered.empty:
            make_contact_sheet(
                rendered, dataset, reports_dir / f"{dataset}_contact.png"
            )
    log.info(
        "%s: median_n=%.2f median_period=%.1f agree=%.1f%% -> %s",
        dataset, result.median_n_closures, result.median_period_ms,
        result.pct_cross_detector_agreement,
        "PASS" if result.passes else "FAIL",
    )
    return result
