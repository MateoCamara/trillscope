"""Statistical visualizations for sociolinguistic analysis."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def create_boxplots(
    df: pd.DataFrame,
    outcome_vars: List[str],
    group_var: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 150
) -> List[str]:
    """
    Create box plots for outcome variables by grouping variable.

    Args:
        df: Data DataFrame
        outcome_vars: List of outcome variable names
        group_var: Grouping variable name
        output_dir: Output directory for figures
        figsize: Figure size
        dpi: Figure resolution

    Returns:
        List of saved figure paths
    """
    try:
        import seaborn as sns
    except ImportError:
        logger.warning("seaborn not installed, using matplotlib only")
        return _create_boxplots_matplotlib(df, outcome_vars, group_var, output_dir, figsize, dpi)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # Filter to valid groups
    valid_df = df[~df[group_var].isin(['unknown', None, ''])].copy()
    valid_df = valid_df[valid_df[group_var].notna()]

    if len(valid_df) == 0:
        logger.warning(f"No valid data for {group_var}")
        return []

    # Get available outcome vars
    outcome_vars = [v for v in outcome_vars if v in valid_df.columns]
    if not outcome_vars:
        return []

    # Determine grid layout
    n_vars = len(outcome_vars)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Color palette
    groups = [str(g) for g in valid_df[group_var].unique()]
    n_groups = len(groups)
    palette = sns.color_palette("Set2", n_groups)

    for idx, var in enumerate(outcome_vars):
        ax = axes[idx]

        # Create box plot using matplotlib directly (seaborn 0.13.2 has bugs)
        group_data = [valid_df[valid_df[group_var] == g][var].dropna().values
                      for g in valid_df[group_var].unique()]
        bp = ax.boxplot(group_data, patch_artist=True, showfliers=True)
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
        ax.set_xticklabels([str(g) for g in valid_df[group_var].unique()], rotation=45, ha='right')

        ax.set_xlabel(group_var.replace('_', ' ').title())
        ax.set_ylabel(_format_var_name(var))
        ax.set_title(f'{_format_var_name(var)} by {group_var.replace("_", " ").title()}')

    # Hide empty subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f'stat_boxplot_{group_var}.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    saved_paths.append(str(output_path))
    logger.info(f"Saved box plot: {output_path}")

    return saved_paths


def _create_boxplots_matplotlib(
    df: pd.DataFrame,
    outcome_vars: List[str],
    group_var: str,
    output_dir: Path,
    figsize: Tuple[int, int],
    dpi: int
) -> List[str]:
    """Fallback box plots using matplotlib only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    valid_df = df[~df[group_var].isin(['unknown', None, ''])].copy()
    if len(valid_df) == 0:
        return []

    outcome_vars = [v for v in outcome_vars if v in valid_df.columns]
    if not outcome_vars:
        return []

    n_vars = len(outcome_vars)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    groups = sorted(valid_df[group_var].unique())

    for idx, var in enumerate(outcome_vars):
        ax = axes[idx]

        data = [valid_df[valid_df[group_var] == g][var].dropna() for g in groups]
        ax.boxplot(data, labels=groups)

        ax.set_xlabel(group_var.replace('_', ' ').title())
        ax.set_ylabel(_format_var_name(var))
        ax.set_title(f'{_format_var_name(var)}')

    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    output_path = output_dir / f'stat_boxplot_{group_var}.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    saved_paths.append(str(output_path))
    return saved_paths


def create_violin_plots(
    df: pd.DataFrame,
    outcome_vars: List[str],
    group_var: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 150
) -> List[str]:
    """Create violin plots for outcome variables by grouping variable."""
    try:
        import seaborn as sns
    except ImportError:
        logger.warning("seaborn not installed, skipping violin plots")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    valid_df = df[~df[group_var].isin(['unknown', None, ''])].copy()
    valid_df = valid_df[valid_df[group_var].notna()]
    if len(valid_df) == 0:
        return []

    outcome_vars = [v for v in outcome_vars if v in valid_df.columns]
    if not outcome_vars:
        return []

    n_vars = len(outcome_vars)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    groups = list(valid_df[group_var].unique())
    n_groups = len(groups)
    palette = sns.color_palette("Set2", n_groups)

    for idx, var in enumerate(outcome_vars):
        ax = axes[idx]

        # Use matplotlib violinplot directly to avoid seaborn 0.13.2 bugs
        group_data = [valid_df[valid_df[group_var] == g][var].dropna().values
                      for g in groups]
        # Filter out empty groups
        non_empty = [(d, g) for d, g in zip(group_data, groups) if len(d) > 0]
        if non_empty:
            data_list, label_list = zip(*non_empty)
            vp = ax.violinplot(list(data_list), showmedians=True)
            for i, body in enumerate(vp['bodies']):
                body.set_facecolor(palette[i % len(palette)])
                body.set_alpha(0.7)
            ax.set_xticks(range(1, len(label_list) + 1))
            ax.set_xticklabels([str(g) for g in label_list], rotation=45, ha='right')

        ax.set_xlabel(group_var.replace('_', ' ').title())
        ax.set_ylabel(_format_var_name(var))
        ax.set_title(f'{_format_var_name(var)}')

    for idx in range(n_vars, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    output_path = output_dir / f'stat_violin_{group_var}.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    saved_paths.append(str(output_path))
    logger.info(f"Saved violin plot: {output_path}")

    return saved_paths


def create_summary_barplot(
    df: pd.DataFrame,
    outcome_var: str,
    group_var: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150
) -> Optional[str]:
    """Create bar plot with error bars for a single outcome variable."""
    try:
        import seaborn as sns
    except ImportError:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    valid_df = df[~df[group_var].isin(['unknown', None, ''])].copy()
    if len(valid_df) == 0 or outcome_var not in valid_df.columns:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=valid_df,
        x=group_var,
        y=outcome_var,
        ax=ax,
        errorbar='se',
        palette='Set2'
    )

    ax.set_xlabel(group_var.replace('_', ' ').title())
    ax.set_ylabel(_format_var_name(outcome_var))
    ax.set_title(f'Mean {_format_var_name(outcome_var)} by {group_var.replace("_", " ").title()}')

    plt.tight_layout()

    output_path = output_dir / f'stat_bar_{outcome_var}_{group_var}.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return str(output_path)


def create_all_visualizations(
    df: pd.DataFrame,
    outcome_vars: List[str],
    group_vars: List[str],
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 150
) -> List[str]:
    """
    Create all statistical visualizations.

    Args:
        df: Data DataFrame
        outcome_vars: List of outcome variables
        group_vars: List of grouping variables
        output_dir: Output directory
        figsize: Figure size
        dpi: Resolution

    Returns:
        List of saved figure paths
    """
    all_paths = []

    for group_var in group_vars:
        if group_var not in df.columns:
            continue

        # Box plots
        paths = create_boxplots(df, outcome_vars, group_var, output_dir, figsize, dpi)
        all_paths.extend(paths)

        # Violin plots
        paths = create_violin_plots(df, outcome_vars, group_var, output_dir, figsize, dpi)
        all_paths.extend(paths)

    return all_paths


def _format_var_name(var_name: str) -> str:
    """Format variable name for display."""
    replacements = {
        'num_cycles': 'Number of Cycles',
        'duration_ms': 'Duration (ms)',
        'voicing_pct': 'Voicing (%)',
        'mean_f0_hz': 'Mean F0 (Hz)',
        'cycle_rate_hz': 'Cycle Rate (Hz)',
        'mean_hnr_db': 'Mean HNR (dB)',
        'mean_intensity_db': 'Mean Intensity (dB)',
        'cycle_regularity': 'Cycle Regularity',
    }
    return replacements.get(var_name, var_name.replace('_', ' ').title())
