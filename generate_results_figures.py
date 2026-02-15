"""Generate individual figures for paper Results section.

Creates ~41 individual horizontal boxplots (one plot per image) plus supplementary figures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style - clean, minimal
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Boxplot style settings
BOXPLOT_PROPS = {
    'boxprops': {'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 1.2},
    'medianprops': {'color': '#E67E22', 'linewidth': 2},  # Orange median
    'whiskerprops': {'color': 'black', 'linewidth': 1.2},
    'capprops': {'color': 'black', 'linewidth': 1.2},
    'flierprops': {'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 5},
    'width': 0.5,
}

# Output directory
fig_dir = Path('paper_materials/figures')
fig_dir.mkdir(parents=True, exist_ok=True)

# Load all measurements
print("Loading data...")
datasets = ['albayzin', 'dimex100', 'preseea', 'glissando', 'mailabs', 'tedx', 'heroico', 'commonvoice']
dfs = []
for ds in datasets:
    path = Path(f'outputs/tables/acoustic_measurements_{ds}.parquet')
    if path.exists():
        df = pd.read_parquet(path)
        dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(data)} tokens from {len(dfs)} datasets")

# Infer sex from F0
def infer_sex(f0):
    if pd.isna(f0):
        return 'unknown'
    elif f0 > 165:
        return 'F'
    elif f0 < 145:
        return 'M'
    else:
        return 'ambiguous'

# Aggregate by speaker
speaker_data = data.groupby(['speaker_id', 'dataset']).agg({
    'num_cycles': 'mean',
    'duration_ms': 'mean',
    'voicing_pct': 'mean',
    'mean_f0_hz': 'mean',
    'cycle_rate_hz': 'mean',
    'mean_hnr_db': 'mean',
}).reset_index()

speaker_data['sex'] = speaker_data['mean_f0_hz'].apply(infer_sex)
speaker_data_fm = speaker_data[speaker_data['sex'].isin(['F', 'M'])].copy()

print(f"Speakers: {len(speaker_data_fm)} (F={len(speaker_data_fm[speaker_data_fm['sex']=='F'])}, M={len(speaker_data_fm[speaker_data_fm['sex']=='M'])})")

# Aggregate by context
context_data = data.groupby(['speaker_id', 'dataset', 'context_label']).agg({
    'num_cycles': 'mean',
    'duration_ms': 'mean',
    'voicing_pct': 'mean',
    'mean_f0_hz': 'mean',
    'cycle_rate_hz': 'mean',
    'mean_hnr_db': 'mean',
}).reset_index()

trill_contexts = ['intervocalic_rr', 'word_initial', 'after_nls', 'post_vocalic']
context_data = context_data[context_data['context_label'].isin(trill_contexts)]

# Load metadata to get country/region info
print("\nLoading metadata for regional data...")
metadata_dfs = []
for ds in ['commonvoice', 'heroico', 'mailabs', 'tedx', 'albayzin', 'dimex100', 'glissando', 'preseea']:
    path = Path(f'metadata/{ds}_raw.parquet')
    if path.exists():
        meta = pd.read_parquet(path)
        meta['dataset'] = ds
        if 'origin_raw' in meta.columns:
            meta['region'] = meta['origin_raw']
        elif 'country' in meta.columns:
            meta['region'] = meta['country']
        else:
            # Assign based on known dataset origin
            if ds == 'albayzin':
                meta['region'] = 'Spain'
            elif ds == 'dimex100':
                meta['region'] = 'Mexico'
            elif ds == 'glissando':
                meta['region'] = 'Spain: Valladolid'
            elif ds == 'preseea':
                meta['region'] = 'Spain: Madrid'
            else:
                meta['region'] = 'Unknown'
        metadata_dfs.append(meta[['speaker_id', 'dataset', 'region']].drop_duplicates())

if metadata_dfs:
    all_meta = pd.concat(metadata_dfs, ignore_index=True)
    # Get unique speaker-region mapping (take first if multiple)
    speaker_regions = all_meta.groupby(['speaker_id', 'dataset'])['region'].first().reset_index()
    print(f"Loaded region data for {len(speaker_regions)} speakers")

    # Merge with speaker data
    speaker_with_region = speaker_data.merge(speaker_regions, on=['speaker_id', 'dataset'], how='left')

    # Fill missing with dataset-based defaults
    dataset_defaults = {
        'albayzin': 'Spain',
        'dimex100': 'Mexico',
        'glissando': 'Spain: Valladolid',
        'preseea': 'Spain: Madrid',
    }
    for ds, region in dataset_defaults.items():
        mask = (speaker_with_region['dataset'] == ds) & (speaker_with_region['region'].isna())
        speaker_with_region.loc[mask, 'region'] = region

    # Clean up region names (fix encoding issues and consolidate)
    def clean_region(r):
        if pd.isna(r):
            return 'Unknown'
        r = str(r).strip()
        r_lower = r.lower()

        # Fix common encoding issues
        r = r.replace('Ã©', 'e').replace('Ã³', 'o').replace('Ã¡', 'a').replace('Ã±', 'n')
        r = r.replace('í', 'i').replace('ó', 'o').replace('á', 'a').replace('ñ', 'n').replace('é', 'e').replace('ú', 'u')

        # Map to standardized names (maximum granularity but clean)
        if 'norte peninsular' in r_lower:
            return 'Spain: North'
        if 'sur peninsular' in r_lower:
            return 'Spain: South'
        if 'centro-sur peninsular' in r_lower:
            return 'Spain: Center'
        if 'islas canarias' in r_lower or 'tenerife' in r_lower:
            return 'Canary Islands'
        if 'rioplatense' in r_lower:
            return 'Rioplatense'
        if 'andino' in r_lower:
            return 'Andean'
        if 'caribe' in r_lower:
            return 'Caribbean'
        if 'chileno' in r_lower or r_lower == 'chile':
            return 'Chilean'
        if 'central' in r_lower and 'america' in r_lower:
            return 'Central America'
        if r_lower in ['méxico', 'mexico']:
            return 'Mexico'
        if r_lower in ['españa', 'espana', 'spain']:
            return 'Spain'
        if r_lower == 'argentina':
            return 'Argentina'
        if r_lower == 'colombia':
            return 'Colombia'
        if r_lower == 'venezuela':
            return 'Venezuela'
        if r_lower == 'peru':
            return 'Peru'
        if r_lower == 'ecuador':
            return 'Ecuador'
        if r_lower == 'puerto rico':
            return 'Puerto Rico'
        if r_lower == 'france' or 'francia' in r_lower:
            return 'France'

        # Spanish cities -> assign to region
        spanish_north = ['valladolid', 'leon', 'burgos', 'palencia', 'segovia', 'avila',
                        'miranda de ebro', 'medina del campo', 'zaragoza', 'guadalajara']
        spanish_south = ['jaen', 'cartagena', 'caceres']
        spanish_center = ['madrid']
        spanish_other = ['valencia', 'ceuta']

        if r_lower in spanish_north or any(c in r_lower for c in spanish_north):
            return 'Spain: North'
        if r_lower in spanish_south or any(c in r_lower for c in spanish_south):
            return 'Spain: South'
        if r_lower in spanish_center or 'madrid' in r_lower:
            return 'Spain: Center'
        if r_lower in spanish_other:
            return 'Spain'

        if r_lower == 'unknown' or r_lower == '':
            return 'Unknown'

        return r  # Return as-is if no match

    speaker_with_region['region_clean'] = speaker_with_region['region'].apply(clean_region)
    print(f"Regions found: {speaker_with_region['region_clean'].nunique()}")
    print(speaker_with_region['region_clean'].value_counts())
else:
    speaker_with_region = speaker_data.copy()
    speaker_with_region['region_clean'] = 'Unknown'

# Speech style data
if 'speech_style' in data.columns:
    style_data = data.groupby(['speaker_id', 'dataset', 'speech_style']).agg({
        'num_cycles': 'mean',
        'duration_ms': 'mean',
        'voicing_pct': 'mean',
        'mean_f0_hz': 'mean',
        'cycle_rate_hz': 'mean',
        'mean_hnr_db': 'mean',
    }).reset_index()
else:
    # Infer from dataset
    dataset_to_style = {
        'albayzin': 'read',
        'preseea': 'spontaneous',
        'glissando': 'mixed',
        'dimex100': 'read',
        'mailabs': 'read',
        'tedx': 'spontaneous',
        'heroico': 'read',
        'commonvoice': 'read'
    }
    style_data = speaker_data.copy()
    style_data['speech_style'] = style_data['dataset'].map(dataset_to_style)

# Age and education data (if available)
age_data = None
edu_data = None

# =============================================================================
# FIGURE GENERATION FUNCTIONS
# =============================================================================

def create_horizontal_boxplot(data, x_var, y_var, title, xlabel, ylabel,
                               order=None, figsize=(8, 5), filename=None):
    """Create a single horizontal boxplot with thin white boxes and orange median."""
    fig, ax = plt.subplots(figsize=figsize)

    # Filter out NaN values
    plot_data = data.dropna(subset=[x_var, y_var])

    if order is not None:
        # Filter to only include values in order
        plot_data = plot_data[plot_data[y_var].isin(order)]

    # Create boxplot with custom style - NO OUTLIERS (showfliers=False)
    bp = ax.boxplot([plot_data[plot_data[y_var] == cat][x_var].values
                     for cat in (order if order else plot_data[y_var].unique())],
                    vert=False,
                    patch_artist=True,
                    widths=0.5,
                    showfliers=False)  # Hide outliers for cleaner x-axis

    # Style the boxes
    for box in bp['boxes']:
        box.set(facecolor='white', edgecolor='black', linewidth=1.2)
    for median in bp['medians']:
        median.set(color='#E67E22', linewidth=2)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.2)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.2)

    # Set y-tick labels
    labels = order if order else list(plot_data[y_var].unique())
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add light vertical grid lines
    ax.xaxis.grid(True, linestyle=':', alpha=0.6)
    ax.yaxis.grid(False)

    plt.tight_layout()

    if filename:
        plt.savefig(fig_dir / filename, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Created: {filename}")

    plt.close()

def create_scatter_plot(data, x_var, y_var, hue_var, title, xlabel, ylabel,
                        palette=None, figsize=(8, 6), filename=None, add_regression=False):
    """Create a scatter plot with optional regression line."""
    fig, ax = plt.subplots(figsize=figsize)

    if palette is None:
        palette = {'F': '#E74C3C', 'M': '#3498DB'}

    plot_data = data.dropna(subset=[x_var, y_var])

    for hue_val in plot_data[hue_var].unique():
        subset = plot_data[plot_data[hue_var] == hue_val]
        color = palette.get(hue_val, 'gray')
        ax.scatter(subset[x_var], subset[y_var], c=color, alpha=0.6,
                   s=30, label=hue_val)

    if add_regression:
        slope, intercept, r, p, se = stats.linregress(plot_data[x_var], plot_data[y_var])
        x_line = np.linspace(plot_data[x_var].min(), plot_data[x_var].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.7,
                label=f'r = {r:.2f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if filename:
        plt.savefig(fig_dir / filename, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Created: {filename}")

    plt.close()

# =============================================================================
# GENERATE ALL FIGURES
# =============================================================================

measures = {
    'num_cycles': ('Cycle Count', 'Number of Cycles'),
    'duration_ms': ('Duration (ms)', 'Duration'),
    'voicing_pct': ('Voicing (%)', 'Voicing Percentage'),
    'mean_f0_hz': ('F0 (Hz)', 'Fundamental Frequency'),
    'cycle_rate_hz': ('Cycle Rate (Hz)', 'Cycle Rate'),
    'mean_hnr_db': ('HNR (dB)', 'Harmonics-to-Noise Ratio')
}

print("\n=== Generating Sex Figures ===")
for var, (label, full_name) in measures.items():
    create_horizontal_boxplot(
        speaker_data_fm, x_var=var, y_var='sex',
        title=f'{full_name} by Sex',
        xlabel=label, ylabel='Sex',
        order=['F', 'M'],
        figsize=(8, 2.5),
        filename=f'sex_{var.replace("mean_", "").replace("_hz", "").replace("_db", "").replace("_pct", "").replace("_ms", "")}.png'
    )

print("\n=== Generating Context Figures ===")
context_order = ['intervocalic_rr', 'word_initial', 'after_nls', 'post_vocalic']
context_labels = ['Intervocalic rr', 'Word-initial', 'After n/l/s', 'Post-vocalic']
# Rename for display
context_data_display = context_data.copy()
context_data_display['context_display'] = context_data_display['context_label'].map(
    dict(zip(context_order, context_labels))
)

for var, (label, full_name) in measures.items():
    create_horizontal_boxplot(
        context_data_display, x_var=var, y_var='context_display',
        title=f'{full_name} by Phonological Context',
        xlabel=label, ylabel='Context',
        order=context_labels,
        figsize=(8, 3.5),
        filename=f'context_{var.replace("mean_", "").replace("_hz", "").replace("_db", "").replace("_pct", "").replace("_ms", "")}.png'
    )

print("\n=== Generating Speech Style Figures ===")
style_data_display = style_data.copy()
if 'speech_style' in style_data_display.columns:
    style_data_display['style_display'] = style_data_display['speech_style'].map(
        {'read': 'Read', 'spontaneous': 'Spontaneous', 'mixed': 'Mixed', 'task_oriented': 'Task-oriented'}
    )
    style_labels = [s for s in ['Read', 'Spontaneous', 'Mixed', 'Task-oriented']
                    if s in style_data_display['style_display'].values]

    for var, (label, full_name) in measures.items():
        create_horizontal_boxplot(
            style_data_display, x_var=var, y_var='style_display',
            title=f'{full_name} by Speech Style',
            xlabel=label, ylabel='Speech Style',
            order=style_labels,
            figsize=(8, 3),
            filename=f'style_{var.replace("mean_", "").replace("_hz", "").replace("_db", "").replace("_pct", "").replace("_ms", "")}.png'
        )

print("\n=== Generating Region Figures ===")
# Get regions that have enough data (at least 5 speakers)
region_counts = speaker_with_region['region_clean'].value_counts()
regions_to_use = region_counts[region_counts >= 5].index.tolist()
# Remove 'Unknown' from the list
regions_to_use = [r for r in regions_to_use if r != 'Unknown']
print(f"Regions with >= 5 speakers: {len(regions_to_use)}")

# Sort regions by mean cycle count for consistent ordering
region_order = speaker_with_region[speaker_with_region['region_clean'].isin(regions_to_use)].groupby('region_clean')['num_cycles'].mean().sort_values(ascending=True).index.tolist()

for var, (label, full_name) in measures.items():
    plot_data = speaker_with_region[speaker_with_region['region_clean'].isin(regions_to_use)]

    # Adjust figure height based on number of regions
    fig_height = max(4, len(regions_to_use) * 0.45)
    create_horizontal_boxplot(
        plot_data, x_var=var, y_var='region_clean',
        title=f'{full_name} by Region',
        xlabel=label, ylabel='Region',
        order=region_order,
        figsize=(10, fig_height),
        filename=f'region_{var.replace("mean_", "").replace("_hz", "").replace("_db", "").replace("_pct", "").replace("_ms", "")}.png'
    )

print("\n=== Generating Dataset Figures ===")
# Sort datasets by mean cycle count
dataset_order = speaker_data.groupby('dataset')['num_cycles'].mean().sort_values(ascending=True).index.tolist()

for var, (label, full_name) in measures.items():
    create_horizontal_boxplot(
        speaker_data, x_var=var, y_var='dataset',
        title=f'{full_name} by Dataset',
        xlabel=label, ylabel='Dataset',
        order=dataset_order,
        figsize=(10, 4.5),
        filename=f'dataset_{var.replace("mean_", "").replace("_hz", "").replace("_db", "").replace("_pct", "").replace("_ms", "")}.png'
    )

print("\n=== Generating Supplementary Figures ===")

# Correlation heatmap
corr_vars = ['num_cycles', 'duration_ms', 'voicing_pct', 'mean_f0_hz', 'cycle_rate_hz', 'mean_hnr_db']
corr_labels = ['Cycles', 'Duration', 'Voicing', 'F0', 'Cycle Rate', 'HNR']
corr_matrix = speaker_data_fm[corr_vars].corr()

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=corr_labels, yticklabels=corr_labels, ax=ax,
            vmin=-1, vmax=1, square=True)
ax.set_title('Correlation Matrix of Acoustic Measures')
plt.tight_layout()
plt.savefig(fig_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: correlation_heatmap.png")
plt.close()

# Cycle distribution histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data['num_cycles'], bins=range(0, 20), edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(x=data['num_cycles'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {data["num_cycles"].mean():.1f}')
ax.axvline(x=data['num_cycles'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median = {data["num_cycles"].median():.1f}')
ax.set_xlabel('Number of Cycles')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Trill Cycle Counts')
ax.legend()
plt.tight_layout()
plt.savefig(fig_dir / 'cycle_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: cycle_distribution.png")
plt.close()

# Duration vs Cycles scatter
create_scatter_plot(
    speaker_data_fm, x_var='duration_ms', y_var='num_cycles', hue_var='sex',
    title='Duration vs Cycle Count by Sex',
    xlabel='Mean Duration (ms)', ylabel='Mean Cycle Count',
    palette={'F': '#E74C3C', 'M': '#3498DB'},
    figsize=(8, 6),
    filename='duration_vs_cycles.png',
    add_regression=True
)

# F0 vs Cycle Rate scatter
create_scatter_plot(
    speaker_data_fm, x_var='mean_f0_hz', y_var='cycle_rate_hz', hue_var='sex',
    title='F0 vs Cycle Rate by Sex',
    xlabel='Mean F0 (Hz)', ylabel='Cycle Rate (Hz)',
    palette={'F': '#E74C3C', 'M': '#3498DB'},
    figsize=(8, 6),
    filename='f0_vs_cyclerate.png',
    add_regression=False
)

# Effect sizes bar chart
print("\n=== Generating Effect Size Figure ===")
try:
    tests = pd.read_csv('outputs/tables/statistical_tests.csv')

    fig, ax = plt.subplots(figsize=(12, 8))

    # Pivot for grouped bar chart
    pivot_data = tests.pivot(index='outcome_variable', columns='predictor_variable', values='effect_size')

    # Rename for display
    outcome_names = {
        'num_cycles': 'Cycles',
        'duration_ms': 'Duration',
        'voicing_pct': 'Voicing',
        'mean_f0_hz': 'F0',
        'cycle_rate_hz': 'Cycle Rate',
        'mean_hnr_db': 'HNR'
    }
    pivot_data.index = pivot_data.index.map(lambda x: outcome_names.get(x, x))

    pivot_data.plot(kind='barh', ax=ax, width=0.8)
    ax.axvline(x=0.14, color='green', linestyle='--', alpha=0.7, label='Large (>0.14)')
    ax.axvline(x=0.06, color='orange', linestyle='--', alpha=0.7, label='Medium (>0.06)')
    ax.set_xlabel('Effect Size (epsilon-squared)')
    ax.set_ylabel('Acoustic Measure')
    ax.set_title('Effect Sizes by Predictor Variable')
    ax.legend(title='Predictor', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(fig_dir / 'effect_sizes_all.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Created: effect_sizes_all.png")
    plt.close()
except Exception as e:
    print(f"Could not create effect sizes figure: {e}")

# Dataset comparison bar chart
print("\n=== Generating Dataset Comparison Figure ===")
dataset_summary = data.groupby('dataset').agg({
    'num_cycles': 'mean',
    'speaker_id': 'nunique',
    'utt_id': 'count'
}).reset_index()
dataset_summary.columns = ['Dataset', 'Mean Cycles', 'Speakers', 'Tokens']

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(dataset_summary))
bars = ax.bar(x, dataset_summary['Tokens'], color='steelblue', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(dataset_summary['Dataset'], rotation=45, ha='right')
ax.set_xlabel('Dataset')
ax.set_ylabel('Number of Trill Tokens')
ax.set_title('Trill Token Count by Dataset')

for bar, n in zip(bars, dataset_summary['Tokens']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{n:,}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(fig_dir / 'dataset_tokens.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: dataset_tokens.png")
plt.close()

# Speaker density plot (F0 vs Cycles by Sex)
print("\n=== Generating Speaker Density Plot ===")
fig, ax = plt.subplots(figsize=(10, 8))
try:
    sns.kdeplot(data=speaker_data_fm, x='mean_f0_hz', y='num_cycles', hue='sex',
                fill=True, alpha=0.5, levels=5, ax=ax,
                palette={'F': '#E74C3C', 'M': '#3498DB'})
    ax.set_xlabel('Mean F0 (Hz)')
    ax.set_ylabel('Mean Cycle Count')
    ax.set_title('Speaker Distribution: F0 vs Cycle Count by Sex')
    plt.tight_layout()
    plt.savefig(fig_dir / 'speaker_density.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Created: speaker_density.png")
except Exception as e:
    print(f"Could not create density plot: {e}")
plt.close()

print("\n=== DONE ===")
print(f"Generated figures in {fig_dir}")

# Count figures
fig_count = len(list(fig_dir.glob('*.png')))
print(f"Total figures: {fig_count}")
