"""Descriptive statistics for acoustic measurements."""

import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def compute_summary_stats(
    df: pd.DataFrame,
    outcome_vars: List[str],
    group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute summary statistics for outcome variables.

    Args:
        df: Data DataFrame
        outcome_vars: List of outcome variable names
        group_by: Optional grouping variable

    Returns:
        DataFrame with summary statistics
    """
    # Filter to available columns
    outcome_vars = [v for v in outcome_vars if v in df.columns]

    if not outcome_vars:
        logger.warning("No outcome variables found in data")
        return pd.DataFrame()

    results = []

    if group_by and group_by in df.columns:
        # Grouped statistics
        for var in outcome_vars:
            for group_val, group_df in df.groupby(group_by):
                if group_val in ['unknown', None, ''] or pd.isna(group_val):
                    continue

                values = group_df[var].dropna()
                if len(values) < 2:
                    continue

                stats = _compute_stats(values)
                stats['variable'] = var
                stats['group_by'] = group_by
                stats['group_level'] = group_val
                results.append(stats)
    else:
        # Overall statistics
        for var in outcome_vars:
            values = df[var].dropna()
            if len(values) < 2:
                continue

            stats = _compute_stats(values)
            stats['variable'] = var
            stats['group_by'] = 'overall'
            stats['group_level'] = 'all'
            results.append(stats)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Reorder columns
    cols = ['variable', 'group_by', 'group_level', 'n', 'mean', 'std',
            'median', 'q25', 'q75', 'min', 'max', 'skewness', 'kurtosis']
    cols = [c for c in cols if c in result_df.columns]

    return result_df[cols]


def _compute_stats(values: pd.Series) -> dict:
    """Compute descriptive statistics for a series of values."""
    from scipy import stats as scipy_stats

    return {
        'n': len(values),
        'mean': values.mean(),
        'std': values.std(),
        'median': values.median(),
        'q25': values.quantile(0.25),
        'q75': values.quantile(0.75),
        'min': values.min(),
        'max': values.max(),
        'skewness': scipy_stats.skew(values),
        'kurtosis': scipy_stats.kurtosis(values),
    }


def compute_all_descriptive_stats(
    df: pd.DataFrame,
    outcome_vars: List[str],
    group_vars: List[str]
) -> pd.DataFrame:
    """
    Compute descriptive statistics for all combinations.

    Args:
        df: Data DataFrame
        outcome_vars: List of outcome variables
        group_vars: List of grouping variables

    Returns:
        Combined DataFrame of all statistics
    """
    all_stats = []

    # Overall statistics
    overall = compute_summary_stats(df, outcome_vars)
    if len(overall) > 0:
        all_stats.append(overall)

    # Grouped statistics
    for group_var in group_vars:
        if group_var in df.columns:
            grouped = compute_summary_stats(df, outcome_vars, group_by=group_var)
            if len(grouped) > 0:
                all_stats.append(grouped)

    if not all_stats:
        return pd.DataFrame()

    return pd.concat(all_stats, ignore_index=True)


def get_group_counts(
    df: pd.DataFrame,
    group_vars: List[str]
) -> pd.DataFrame:
    """
    Get sample sizes for each group level.

    Args:
        df: Data DataFrame
        group_vars: List of grouping variables

    Returns:
        DataFrame with group counts
    """
    results = []

    for var in group_vars:
        if var not in df.columns:
            continue

        counts = df[var].value_counts()
        for level, count in counts.items():
            if level in ['unknown', None, ''] or pd.isna(level):
                continue
            results.append({
                'variable': var,
                'level': level,
                'n': count,
                'percentage': count / len(df) * 100
            })

    return pd.DataFrame(results)
