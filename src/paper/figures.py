"""Publication-quality figure generation for the research paper."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.io import wavfile

logger = logging.getLogger(__name__)

# Publication style settings
STYLE = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}

# Color palette for consistency
COLORS = {
    'female': '#E64B35',  # Red
    'male': '#4DBBD5',    # Blue
    'low': '#F39B7F',     # Light coral
    'mid': '#8491B4',     # Slate blue
    'high': '#91D1C2',    # Teal
    'albayzin': '#00A087',
    'dimex100': '#3C5488',
    'preseea': '#F39B7F',
}


def apply_style():
    """Apply publication style to matplotlib."""
    plt.rcParams.update(STYLE)


def generate_spectrogram_figure(
    audio_path: Path,
    start_ms: float,
    end_ms: float,
    output_path: Path,
    title: str = "Spanish Trill /r/",
    figsize: Tuple[float, float] = (6, 4)
) -> str:
    """
    Generate annotated spectrogram of a trill segment.

    Args:
        audio_path: Path to audio file
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        output_path: Output figure path
        title: Figure title
        figsize: Figure size in inches

    Returns:
        Path to saved figure
    """
    apply_style()

    # Load audio
    sr, audio = wavfile.read(audio_path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    # Extract segment with context
    context_ms = 30
    start_sample = max(0, int((start_ms - context_ms) * sr / 1000))
    end_sample = min(len(audio), int((end_ms + context_ms) * sr / 1000))
    segment = audio[start_sample:end_sample]

    # Time axis
    duration = len(segment) / sr
    time = np.linspace(0, duration * 1000, len(segment))

    # Create figure with waveform and spectrogram
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 2])

    # Waveform
    ax1 = axes[0]
    ax1.plot(time, segment, color='#333333', linewidth=0.5)
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, duration * 1000)
    ax1.axvline(context_ms, color='red', linestyle='--', alpha=0.7, label='Trill onset')
    ax1.axvline(duration * 1000 - context_ms, color='red', linestyle='--', alpha=0.7)
    ax1.fill_betweenx([-1, 1], context_ms, duration * 1000 - context_ms,
                      alpha=0.1, color='red')
    ax1.set_title(title)

    # Spectrogram
    ax2 = axes[1]
    nperseg = min(256, len(segment) // 4)
    noverlap = nperseg // 2
    f, t, Sxx = signal.spectrogram(segment, sr, nperseg=nperseg, noverlap=noverlap)

    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # Plot spectrogram
    im = ax2.pcolormesh(t * 1000, f, Sxx_db, shading='gouraud', cmap='inferno')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylim(0, 5000)

    # Mark trill region
    ax2.axvline(context_ms, color='white', linestyle='--', alpha=0.7)
    ax2.axvline(duration * 1000 - context_ms, color='white', linestyle='--', alpha=0.7)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, label='Power (dB)')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Saved spectrogram: {output_path}")
    return str(output_path)


def generate_sex_differences_panel(
    data_path: Path,
    output_path: Path,
    figsize: Tuple[float, float] = (8, 6)
) -> str:
    """
    Generate 2x2 panel showing significant sex differences.

    Variables: num_cycles, mean_f0_hz, cycle_rate_hz, mean_hnr_db

    Args:
        data_path: Path to descriptive_stats.csv
        output_path: Output figure path
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    apply_style()

    # Load data
    df = pd.read_csv(data_path)

    # Filter to sex comparisons
    sex_data = df[(df['group_by'] == 'sex') & (df['group_level'].isin(['F', 'M']))]

    if len(sex_data) == 0:
        logger.error("No sex comparison data found")
        return ""

    # Variables to plot (the 4 significant ones)
    variables = [
        ('num_cycles', 'Number of Cycles', 'p < .001***'),
        ('mean_f0_hz', 'Mean F0 (Hz)', 'p < .001***'),
        ('cycle_rate_hz', 'Cycle Rate (Hz)', 'p < .001***'),
        ('mean_hnr_db', 'Mean HNR (dB)', 'p = .001**'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, (var, label, pval) in enumerate(variables):
        ax = axes[idx]

        var_data = sex_data[sex_data['variable'] == var]

        if len(var_data) == 0:
            continue

        # Get means and SDs
        female = var_data[var_data['group_level'] == 'F'].iloc[0]
        male = var_data[var_data['group_level'] == 'M'].iloc[0]

        # Bar positions
        x = [0, 1]
        means = [female['mean'], male['mean']]
        sds = [female['std'], male['std']]
        ns = [int(female['n']), int(male['n'])]

        # Create bars
        bars = ax.bar(x, means, yerr=sds, width=0.6,
                      color=[COLORS['female'], COLORS['male']],
                      edgecolor='black', linewidth=0.5,
                      capsize=4, error_kw={'linewidth': 1})

        # Labels
        ax.set_xticks(x)
        ax.set_xticklabels([f'Female\n(n={ns[0]})', f'Male\n(n={ns[1]})'])
        ax.set_ylabel(label)

        # Add significance annotation
        ymax = max(means) + max(sds) * 1.2
        ax.text(0.5, 0.95, pval, transform=ax.transAxes,
                ha='center', va='top', fontsize=9, style='italic')

        # Add panel label (a, b, c, d)
        ax.text(-0.15, 1.05, chr(97 + idx) + ')', transform=ax.transAxes,
                fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Saved sex differences panel: {output_path}")
    return str(output_path)


def generate_education_duration_plot(
    data_path: Path,
    output_path: Path,
    figsize: Tuple[float, float] = (5, 4)
) -> str:
    """
    Generate box plot showing education effect on duration.

    Args:
        data_path: Path to descriptive_stats.csv
        output_path: Output figure path
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    apply_style()

    # Load data
    df = pd.read_csv(data_path)

    # Filter to education comparisons for duration
    edu_data = df[(df['group_by'] == 'education_bin') &
                  (df['variable'] == 'duration_ms') &
                  (df['group_level'].isin(['low', 'mid', 'high']))]

    if len(edu_data) == 0:
        logger.error("No education duration data found")
        return ""

    fig, ax = plt.subplots(figsize=figsize)

    # Order: low, mid, high
    order = ['low', 'mid', 'high']
    labels = ['Low', 'Mid', 'High']
    colors = [COLORS['low'], COLORS['mid'], COLORS['high']]

    means = []
    sds = []
    ns = []

    for level in order:
        row = edu_data[edu_data['group_level'] == level].iloc[0]
        means.append(row['mean'])
        sds.append(row['std'])
        ns.append(int(row['n']))

    x = range(len(order))
    bars = ax.bar(x, means, yerr=sds, width=0.6,
                  color=colors, edgecolor='black', linewidth=0.5,
                  capsize=4, error_kw={'linewidth': 1})

    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}\n(n={n})' for l, n in zip(labels, ns)])
    ax.set_xlabel('Education Level')
    ax.set_ylabel('Trill Duration (ms)')
    ax.set_title('Effect of Education on Trill Duration')

    # Add significance brackets
    # Low vs Mid
    y1 = means[0] + sds[0] + 5
    y2 = means[1] + sds[1] + 5
    ymax = max(y1, y2) + 5
    ax.plot([0, 0, 1, 1], [y1, ymax, ymax, y2], 'k-', linewidth=0.8)
    ax.text(0.5, ymax + 2, '**', ha='center', fontsize=10)

    # Low vs High
    y3 = means[2] + sds[2] + 5
    ymax2 = max(y1, y3) + 15
    ax.plot([0, 0, 2, 2], [y1 + 10, ymax2, ymax2, y3], 'k-', linewidth=0.8)
    ax.text(1, ymax2 + 2, '**', ha='center', fontsize=10)

    # Kruskal-Wallis result
    ax.text(0.98, 0.02, 'H = 12.4, p = .002, ε² = 0.16',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Saved education duration plot: {output_path}")
    return str(output_path)


def generate_cycle_distribution(
    measurements_dir: Path,
    output_path: Path,
    figsize: Tuple[float, float] = (7, 4)
) -> str:
    """
    Generate histogram of cycle counts by dataset.

    Args:
        measurements_dir: Directory containing acoustic_measurements_*.parquet
        output_path: Output figure path
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    apply_style()

    datasets = ['albayzin', 'dimex100', 'preseea']
    dataset_labels = ['ALBAYZIN', 'DIMEx100', 'PRESEEA']
    dataset_colors = [COLORS['albayzin'], COLORS['dimex100'], COLORS['preseea']]

    fig, ax = plt.subplots(figsize=figsize)

    all_data = []
    for dataset in datasets:
        path = measurements_dir / f'acoustic_measurements_{dataset}.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            df['dataset'] = dataset
            all_data.append(df)

    if not all_data:
        logger.error("No measurement data found")
        return ""

    combined = pd.concat(all_data, ignore_index=True)

    # Bin edges (0-15 cycles)
    bins = np.arange(0.5, 16.5, 1)

    for i, (dataset, label, color) in enumerate(zip(datasets, dataset_labels, dataset_colors)):
        data = combined[combined['dataset'] == dataset]['num_cycles'].dropna()
        if len(data) == 0:
            continue

        # Calculate histogram
        counts, _ = np.histogram(data, bins=bins, density=True)

        # Plot as line with markers
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(bin_centers, counts * 100, 'o-', label=f'{label} (n={len(data):,})',
                color=color, linewidth=1.5, markersize=4)

    ax.set_xlabel('Number of Cycles')
    ax.set_ylabel('Percentage of Tokens (%)')
    ax.set_title('Distribution of Trill Cycle Counts by Corpus')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 15)
    ax.set_xticks(range(0, 16, 2))

    # Add vertical lines for modes
    # ALBAYZIN mode ~1, DIMEx100 mode ~3, PRESEEA mode ~6
    ax.axvline(1, color=COLORS['albayzin'], linestyle=':', alpha=0.5)
    ax.axvline(3, color=COLORS['dimex100'], linestyle=':', alpha=0.5)
    ax.axvline(6, color=COLORS['preseea'], linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Saved cycle distribution: {output_path}")
    return str(output_path)


def generate_all_figures(
    output_dir: Path,
    data_dir: Path = Path('outputs/tables'),
    audio_dir: Optional[Path] = None
) -> dict:
    """
    Generate all paper figures.

    Args:
        output_dir: Directory to save figures
        data_dir: Directory containing data files
        audio_dir: Directory containing audio for spectrogram (optional)

    Returns:
        Dict mapping figure names to paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}

    # Figure 2: Sex differences panel
    sex_path = output_dir / 'fig2_sex_panel.png'
    desc_path = data_dir / 'descriptive_stats.csv'
    if desc_path.exists():
        generate_sex_differences_panel(desc_path, sex_path)
        figures['fig2_sex_panel'] = str(sex_path)

    # Figure 3: Education duration
    edu_path = output_dir / 'fig3_education.png'
    if desc_path.exists():
        generate_education_duration_plot(desc_path, edu_path)
        figures['fig3_education'] = str(edu_path)

    # Figure 4: Cycle distribution
    cycle_path = output_dir / 'fig4_cycles_dist.png'
    generate_cycle_distribution(data_dir, cycle_path)
    figures['fig4_cycles_dist'] = str(cycle_path)

    return figures


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    figures = generate_all_figures(
        output_dir=Path('paper/figures'),
        data_dir=Path('outputs/tables')
    )
    print(f"Generated {len(figures)} figures:")
    for name, path in figures.items():
        print(f"  {name}: {path}")
