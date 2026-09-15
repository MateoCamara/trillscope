"""Report generator for statistical analysis results."""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base import AnalysisConfig, AnalysisResult, TestResult

logger = logging.getLogger(__name__)


def generate_report(
    result: AnalysisResult,
    output_path: Path,
    data_stats: Optional[dict] = None
) -> str:
    """
    Generate a comprehensive markdown report of statistical analysis.

    Args:
        result: AnalysisResult object
        output_path: Path to save the report
        data_stats: Optional dict with data loading statistics

    Returns:
        Path to saved report
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    # Header
    lines.append("# Statistical Analysis Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Table of Contents
    lines.append("## Table of Contents")
    lines.append("1. [Data Summary](#data-summary)")
    lines.append("2. [Descriptive Statistics](#descriptive-statistics)")
    lines.append("3. [Inferential Statistics](#inferential-statistics)")
    lines.append("4. [Effect Sizes](#effect-sizes)")
    lines.append("5. [Visualizations](#visualizations)")
    lines.append("6. [Conclusions](#conclusions)")
    lines.append("")

    # Data Summary
    lines.append("## Data Summary")
    lines.append("")
    if data_stats:
        lines.append(_format_data_summary(data_stats))
    else:
        lines.append("Data statistics not available.")
    lines.append("")

    # Descriptive Statistics
    lines.append("## Descriptive Statistics")
    lines.append("")
    lines.append(_format_descriptive_stats(result.descriptive_stats))
    lines.append("")

    # Inferential Statistics
    lines.append("## Inferential Statistics")
    lines.append("")
    lines.append(_format_inferential_stats(result.test_results, result.config))
    lines.append("")

    # Effect Sizes
    lines.append("## Effect Sizes")
    lines.append("")
    lines.append(_format_effect_sizes(result.test_results))
    lines.append("")

    # Visualizations
    lines.append("## Visualizations")
    lines.append("")
    if result.figures:
        for fig_path in result.figures:
            fig_name = Path(fig_path).name
            # Use relative path
            lines.append(f"### {_format_figure_title(fig_name)}")
            lines.append(f"![{fig_name}](../outputs/figures/{fig_name})")
            lines.append("")
    else:
        lines.append("No visualizations generated.")
    lines.append("")

    # Conclusions
    lines.append("## Conclusions")
    lines.append("")
    lines.append(_format_conclusions(result))
    lines.append("")

    # Issues
    if result.issues:
        lines.append("## Issues and Warnings")
        lines.append("")
        for issue in result.issues:
            lines.append(f"- {issue}")
        lines.append("")

    # Write report
    report_content = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info(f"Report saved to {output_path}")
    return str(output_path)


def _format_data_summary(stats: dict) -> str:
    """Format data loading statistics."""
    lines = []

    lines.append("### Dataset Overview")
    lines.append("")
    lines.append("| Dataset | Measurements | Metadata | Merged |")
    lines.append("|---------|-------------|----------|--------|")

    for dataset, ds_stats in stats.get('datasets', {}).items():
        meas = ds_stats.get('measurements', 0)
        meta = ds_stats.get('metadata_records', 0)
        merged = ds_stats.get('merged', 0)
        lines.append(f"| {dataset} | {meas:,} | {meta:,} | {merged:,} |")

    total = stats.get('total_measurements', 0)
    final = stats.get('final_records', 0)
    lines.append(f"| **Total** | {total:,} | - | - |")
    lines.append("")

    if 'aggregated_speakers' in stats:
        lines.append(f"**After speaker aggregation:** {stats['aggregated_speakers']:,} speakers")
    lines.append(f"**Final analysis dataset:** {final:,} records")

    return "\n".join(lines)


def _format_descriptive_stats(df: pd.DataFrame) -> str:
    """Format descriptive statistics table."""
    if df.empty:
        return "No descriptive statistics available."

    lines = []

    # Overall statistics
    overall = df[df['group_by'] == 'overall']
    if len(overall) > 0:
        lines.append("### Overall Statistics")
        lines.append("")
        lines.append("| Variable | N | Mean | SD | Median | Min | Max |")
        lines.append("|----------|---|------|-----|--------|-----|-----|")
        for _, row in overall.iterrows():
            lines.append(
                f"| {_format_var(row['variable'])} | "
                f"{row['n']:,} | "
                f"{row['mean']:.2f} | "
                f"{row['std']:.2f} | "
                f"{row['median']:.2f} | "
                f"{row['min']:.2f} | "
                f"{row['max']:.2f} |"
            )
        lines.append("")

    # Grouped statistics
    grouped = df[df['group_by'] != 'overall']
    if len(grouped) > 0:
        for group_var in grouped['group_by'].unique():
            group_df = grouped[grouped['group_by'] == group_var]

            lines.append(f"### By {_format_var(group_var)}")
            lines.append("")

            for var in group_df['variable'].unique():
                var_df = group_df[group_df['variable'] == var]
                lines.append(f"**{_format_var(var)}**")
                lines.append("")
                lines.append("| Group | N | Mean | SD | Median |")
                lines.append("|-------|---|------|-----|--------|")

                for _, row in var_df.iterrows():
                    lines.append(
                        f"| {row['group_level']} | "
                        f"{row['n']:,} | "
                        f"{row['mean']:.2f} | "
                        f"{row['std']:.2f} | "
                        f"{row['median']:.2f} |"
                    )
                lines.append("")

    return "\n".join(lines)


def _format_inferential_stats(tests: List[TestResult], config: AnalysisConfig) -> str:
    """Format inferential test results."""
    if not tests:
        return "No statistical tests were run."

    lines = []

    lines.append(f"**Significance level:** α = {config.alpha}")
    lines.append(f"**Multiple comparison correction:** {config.correction_method}")
    lines.append("")

    # Group tests by predictor
    tests_by_predictor = {}
    for test in tests:
        pred = test.predictor_variable
        if pred not in tests_by_predictor:
            tests_by_predictor[pred] = []
        tests_by_predictor[pred].append(test)

    for predictor, pred_tests in tests_by_predictor.items():
        lines.append(f"### {_format_var(predictor)}")
        lines.append("")
        lines.append("| Outcome | Test | Statistic | p-value | p (adj) | Significant |")
        lines.append("|---------|------|-----------|---------|---------|-------------|")

        for test in pred_tests:
            p_adj = f"{test.p_adjusted:.4f}" if test.p_adjusted else "-"
            sig = "**Yes**" if test.significant else "No"
            lines.append(
                f"| {_format_var(test.outcome_variable)} | "
                f"{test.test_type} | "
                f"{test.statistic:.2f} | "
                f"{test.p_value:.4f} | "
                f"{p_adj} | "
                f"{sig} |"
            )

        lines.append("")

        # Add group means for context
        if pred_tests:
            first_test = pred_tests[0]
            if first_test.group_means:
                lines.append("**Group Means:**")
                for group, mean in sorted(first_test.group_means.items()):
                    n = first_test.group_ns.get(group, '?')
                    lines.append(f"- {group}: n={n}")
                lines.append("")

        # Post-hoc results
        for test in pred_tests:
            if test.posthoc_results is not None and len(test.posthoc_results) > 0:
                lines.append(f"#### Post-hoc: {_format_var(test.outcome_variable)}")
                lines.append("")
                if isinstance(test.posthoc_results, pd.DataFrame):
                    if 'group1' in test.posthoc_results.columns:
                        # Pairwise format
                        lines.append("| Comparison | p (adjusted) | Significant |")
                        lines.append("|------------|--------------|-------------|")
                        for _, row in test.posthoc_results.iterrows():
                            sig = "**Yes**" if row.get('significant', row.get('p_adjusted', 1) < 0.05) else "No"
                            p = row.get('p_adjusted', row.get('p_value', 0))
                            lines.append(f"| {row['group1']} vs {row['group2']} | {p:.4f} | {sig} |")
                    else:
                        # Matrix format (scikit_posthocs)
                        lines.append("Pairwise comparison matrix (Bonferroni-adjusted p-values):")
                        lines.append("")
                        lines.append("```")
                        lines.append(test.posthoc_results.to_string())
                        lines.append("```")
                lines.append("")

    return "\n".join(lines)


def _format_effect_sizes(tests: List[TestResult]) -> str:
    """Format effect size summary."""
    if not tests:
        return "No effect sizes available."

    lines = []

    lines.append("### Effect Size Interpretations")
    lines.append("")
    lines.append("| Outcome | Predictor | Effect Size | Type | Interpretation |")
    lines.append("|---------|-----------|-------------|------|----------------|")

    for test in tests:
        if test.effect_size is not None:
            lines.append(
                f"| {_format_var(test.outcome_variable)} | "
                f"{_format_var(test.predictor_variable)} | "
                f"{test.effect_size:.3f} | "
                f"{test.effect_size_name} | "
                f"{test.effect_size_interpretation} |"
            )

    lines.append("")

    # Legend
    lines.append("### Effect Size Thresholds")
    lines.append("")
    lines.append("**Rank-biserial r (two-group comparisons):**")
    lines.append("- < 0.2: negligible")
    lines.append("- 0.2-0.5: small")
    lines.append("- 0.5-0.8: medium")
    lines.append("- > 0.8: large")
    lines.append("")
    lines.append("**Epsilon-squared (multi-group comparisons):**")
    lines.append("- < 0.01: negligible")
    lines.append("- 0.01-0.06: small")
    lines.append("- 0.06-0.14: medium")
    lines.append("- > 0.14: large")

    return "\n".join(lines)


def _format_conclusions(result: AnalysisResult) -> str:
    """Generate conclusions based on results."""
    lines = []

    significant_tests = result.get_significant_tests()
    total_tests = len(result.test_results)

    lines.append("### Summary")
    lines.append("")
    lines.append(f"Out of {total_tests} statistical tests, {len(significant_tests)} showed significant effects after correction for multiple comparisons.")
    lines.append("")

    if significant_tests:
        lines.append("### Significant Findings")
        lines.append("")

        # Group by predictor
        by_predictor = {}
        for test in significant_tests:
            pred = test.predictor_variable
            if pred not in by_predictor:
                by_predictor[pred] = []
            by_predictor[pred].append(test)

        for predictor, tests in by_predictor.items():
            lines.append(f"**{_format_var(predictor)}:**")
            for test in tests:
                effect_desc = f"{test.effect_size_interpretation} effect" if test.effect_size_interpretation else ""
                lines.append(
                    f"- {_format_var(test.outcome_variable)}: "
                    f"p = {test.p_adjusted or test.p_value:.4f}, "
                    f"{effect_desc}"
                )
            lines.append("")
    else:
        lines.append("No statistically significant effects were found after correction for multiple comparisons.")
        lines.append("")

    # Interpretation notes
    lines.append("### Interpretation Notes")
    lines.append("")
    lines.append("1. All tests used non-parametric methods (Mann-Whitney U, Kruskal-Wallis) appropriate for acoustic data.")
    lines.append("2. P-values were adjusted using Bonferroni correction to control family-wise error rate.")
    lines.append("3. Effect sizes are reported alongside p-values to indicate practical significance.")
    lines.append("4. Data were aggregated at the speaker level to avoid pseudo-replication.")

    return "\n".join(lines)


def _format_var(var_name: str) -> str:
    """Format variable name for display."""
    replacements = {
        'num_cycles': 'Number of Cycles',
        'duration_ms': 'Duration (ms)',
        'voicing_pct': 'Voicing (%)',
        'mean_f0_hz': 'Mean F0 (Hz)',
        'cycle_rate_hz': 'Cycle Rate (Hz)',
        'mean_hnr_db': 'Mean HNR (dB)',
        'mean_intensity_db': 'Mean Intensity (dB)',
        'sex': 'Sex',
        'age_bin': 'Age Group',
        'education_bin': 'Education Level',
        'country': 'Country',
        'context_label': 'Phonetic Context',
    }
    return replacements.get(var_name, var_name.replace('_', ' ').title())


def _format_figure_title(filename: str) -> str:
    """Format figure filename as title."""
    # stat_boxplot_sex.png -> Box Plot: Sex
    name = filename.replace('.png', '').replace('stat_', '')

    if 'boxplot' in name:
        var = name.replace('boxplot_', '')
        return f"Box Plot: {_format_var(var)}"
    elif 'violin' in name:
        var = name.replace('violin_', '')
        return f"Violin Plot: {_format_var(var)}"
    elif 'bar' in name:
        parts = name.replace('bar_', '').split('_')
        return f"Bar Plot: {' by '.join(_format_var(p) for p in parts)}"
    else:
        return name.replace('_', ' ').title()


def save_tables(result: AnalysisResult, output_dir: Path) -> List[str]:
    """
    Save statistical results as CSV tables.

    Args:
        result: AnalysisResult object
        output_dir: Directory to save tables

    Returns:
        List of saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Descriptive statistics
    if not result.descriptive_stats.empty:
        desc_path = output_dir / 'descriptive_stats.csv'
        result.descriptive_stats.to_csv(desc_path, index=False)
        saved.append(str(desc_path))
        logger.info(f"Saved descriptive stats: {desc_path}")

    # Test results
    tests_df = result.get_tests_df()
    if not tests_df.empty:
        tests_path = output_dir / 'statistical_tests.csv'
        tests_df.to_csv(tests_path, index=False)
        saved.append(str(tests_path))
        logger.info(f"Saved test results: {tests_path}")

    return saved
