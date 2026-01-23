"""Inferential statistical tests for sociolinguistic analysis."""

import logging
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from .base import TestResult, AnalysisConfig, interpret_effect_size

logger = logging.getLogger(__name__)


def run_two_group_test(
    df: pd.DataFrame,
    outcome_var: str,
    group_var: str,
    config: AnalysisConfig
) -> Optional[TestResult]:
    """
    Run appropriate two-group comparison test.

    Uses Mann-Whitney U test (non-parametric) by default.

    Args:
        df: Data DataFrame
        outcome_var: Name of outcome variable
        group_var: Name of grouping variable (should have 2 levels)
        config: Analysis configuration

    Returns:
        TestResult or None if test cannot be run
    """
    if outcome_var not in df.columns or group_var not in df.columns:
        return None

    # Get groups, excluding unknown
    valid_df = df[~df[group_var].isin(['unknown', None, ''])].copy()
    valid_df = valid_df.dropna(subset=[outcome_var, group_var])

    groups = valid_df[group_var].unique()
    if len(groups) != 2:
        logger.warning(f"{group_var} does not have exactly 2 groups: {groups}")
        return None

    group1_name, group2_name = sorted(groups)
    group1 = valid_df[valid_df[group_var] == group1_name][outcome_var]
    group2 = valid_df[valid_df[group_var] == group2_name][outcome_var]

    if len(group1) < config.min_group_size or len(group2) < config.min_group_size:
        logger.warning(f"Insufficient sample size for {group_var}: {len(group1)}, {len(group2)}")
        return None

    # Mann-Whitney U test
    statistic, p_value = scipy_stats.mannwhitneyu(
        group1, group2, alternative='two-sided'
    )

    # Rank-biserial correlation as effect size
    n1, n2 = len(group1), len(group2)
    effect_size = 1 - (2 * statistic) / (n1 * n2)  # Rank-biserial r

    result = TestResult(
        test_id=f"{outcome_var}_by_{group_var}",
        outcome_variable=outcome_var,
        predictor_variable=group_var,
        test_type='Mann-Whitney U',
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        effect_size_name='rank_biserial_r',
        effect_size_interpretation=interpret_effect_size(effect_size, 'rank_biserial'),
        group_ns={group1_name: n1, group2_name: n2},
        group_means={group1_name: group1.mean(), group2_name: group2.mean()},
        group_medians={group1_name: group1.median(), group2_name: group2.median()},
        significant=p_value < config.alpha,
    )

    return result


def run_multi_group_test(
    df: pd.DataFrame,
    outcome_var: str,
    group_var: str,
    config: AnalysisConfig
) -> Optional[TestResult]:
    """
    Run appropriate multi-group comparison test.

    Uses Kruskal-Wallis H test (non-parametric).

    Args:
        df: Data DataFrame
        outcome_var: Name of outcome variable
        group_var: Name of grouping variable (3+ levels)
        config: Analysis configuration

    Returns:
        TestResult or None if test cannot be run
    """
    if outcome_var not in df.columns or group_var not in df.columns:
        return None

    # Get groups, excluding unknown
    valid_df = df[~df[group_var].isin(['unknown', None, ''])].copy()
    valid_df = valid_df.dropna(subset=[outcome_var, group_var])

    groups = valid_df[group_var].unique()
    if len(groups) < 2:
        return None

    # Prepare group data
    group_data = []
    group_ns = {}
    group_means = {}
    group_medians = {}

    for group_name in sorted(groups):
        group_values = valid_df[valid_df[group_var] == group_name][outcome_var]
        if len(group_values) < config.min_group_size:
            continue
        group_data.append(group_values)
        group_ns[group_name] = len(group_values)
        group_means[group_name] = group_values.mean()
        group_medians[group_name] = group_values.median()

    if len(group_data) < 2:
        logger.warning(f"Insufficient groups for {group_var} after filtering")
        return None

    # Kruskal-Wallis H test
    statistic, p_value = scipy_stats.kruskal(*group_data)

    # Epsilon-squared effect size
    n_total = sum(len(g) for g in group_data)
    k = len(group_data)
    epsilon_squared = (statistic - k + 1) / (n_total - k)
    epsilon_squared = max(0, epsilon_squared)  # Can't be negative

    result = TestResult(
        test_id=f"{outcome_var}_by_{group_var}",
        outcome_variable=outcome_var,
        predictor_variable=group_var,
        test_type='Kruskal-Wallis H',
        statistic=statistic,
        p_value=p_value,
        effect_size=epsilon_squared,
        effect_size_name='epsilon_squared',
        effect_size_interpretation=interpret_effect_size(epsilon_squared, 'epsilon_squared'),
        group_ns=group_ns,
        group_means=group_means,
        group_medians=group_medians,
        significant=p_value < config.alpha,
    )

    # Run post-hoc tests if significant
    if result.significant and len(group_data) > 2:
        result.posthoc_results = _run_posthoc_dunn(valid_df, outcome_var, group_var)

    return result


def _run_posthoc_dunn(
    df: pd.DataFrame,
    outcome_var: str,
    group_var: str
) -> Optional[pd.DataFrame]:
    """
    Run Dunn's post-hoc test for pairwise comparisons.

    Args:
        df: Data DataFrame
        outcome_var: Outcome variable name
        group_var: Grouping variable name

    Returns:
        DataFrame with pairwise comparison results
    """
    try:
        import scikit_posthocs as sp
        results = sp.posthoc_dunn(
            df, val_col=outcome_var, group_col=group_var, p_adjust='bonferroni'
        )
        return results
    except ImportError:
        # Fallback: manual pairwise Mann-Whitney with Bonferroni
        return _manual_pairwise_tests(df, outcome_var, group_var)
    except Exception as e:
        logger.warning(f"Post-hoc test failed: {e}")
        return None


def _manual_pairwise_tests(
    df: pd.DataFrame,
    outcome_var: str,
    group_var: str
) -> pd.DataFrame:
    """Manual pairwise Mann-Whitney tests with Bonferroni correction."""
    groups = sorted(df[group_var].unique())
    n_comparisons = len(groups) * (len(groups) - 1) // 2

    results = []
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            vals1 = df[df[group_var] == g1][outcome_var]
            vals2 = df[df[group_var] == g2][outcome_var]

            if len(vals1) < 2 or len(vals2) < 2:
                continue

            stat, p = scipy_stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
            p_adj = min(p * n_comparisons, 1.0)  # Bonferroni

            results.append({
                'group1': g1,
                'group2': g2,
                'statistic': stat,
                'p_value': p,
                'p_adjusted': p_adj,
                'significant': p_adj < 0.05
            })

    return pd.DataFrame(results)


def run_all_tests(
    df: pd.DataFrame,
    outcome_vars: List[str],
    predictor_vars: List[str],
    config: AnalysisConfig
) -> List[TestResult]:
    """
    Run all appropriate statistical tests.

    Args:
        df: Data DataFrame
        outcome_vars: List of outcome variable names
        predictor_vars: List of predictor variable names
        config: Analysis configuration

    Returns:
        List of TestResult objects
    """
    results = []

    for outcome_var in outcome_vars:
        if outcome_var not in df.columns:
            continue

        for predictor_var in predictor_vars:
            if predictor_var not in df.columns:
                continue

            # Count unique levels
            valid_df = df[~df[predictor_var].isin(['unknown', None, ''])]
            n_levels = valid_df[predictor_var].nunique()

            if n_levels == 2:
                # Two-group test
                result = run_two_group_test(df, outcome_var, predictor_var, config)
            elif n_levels > 2:
                # Multi-group test
                result = run_multi_group_test(df, outcome_var, predictor_var, config)
            else:
                continue

            if result:
                results.append(result)

    # Apply multiple comparison correction across all tests
    if results:
        results = _apply_correction(results, config.correction_method)

    return results


def _apply_correction(
    results: List[TestResult],
    method: str = 'bonferroni'
) -> List[TestResult]:
    """Apply multiple comparison correction to p-values."""
    p_values = [r.p_value for r in results]
    n_tests = len(p_values)

    if method == 'bonferroni':
        p_adjusted = [min(p * n_tests, 1.0) for p in p_values]
    elif method == 'holm':
        # Holm-Bonferroni step-down
        sorted_indices = np.argsort(p_values)
        p_adjusted = [0.0] * n_tests
        for rank, idx in enumerate(sorted_indices):
            p_adjusted[idx] = min(p_values[idx] * (n_tests - rank), 1.0)
    else:
        p_adjusted = p_values

    for result, p_adj in zip(results, p_adjusted):
        result.p_adjusted = p_adj
        # Update significance based on adjusted p-value
        result.significant = p_adj < 0.05

    return results
