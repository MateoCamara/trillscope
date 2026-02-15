"""Generate figures for paper."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Output directory
fig_dir = Path('paper_materials/figures')
fig_dir.mkdir(exist_ok=True)

# Load all measurements
datasets = ['albayzin', 'dimex100', 'preseea', 'glissando', 'mailabs', 'tedx', 'heroico', 'commonvoice']
dfs = []
for ds in datasets:
    path = Path(f'outputs/tables/acoustic_measurements_{ds}.parquet')
    if path.exists():
        df = pd.read_parquet(path)
        dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(data)} tokens")

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
speaker_data = speaker_data[speaker_data['sex'].isin(['F', 'M'])]

print(f"Speakers: {len(speaker_data)} (F={len(speaker_data[speaker_data['sex']=='F'])}, M={len(speaker_data[speaker_data['sex']=='M'])})")

# ============================================================
# FIGURE 1: Sex differences - Box plots (6 measures)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
measures = ['num_cycles', 'duration_ms', 'voicing_pct', 'mean_f0_hz', 'cycle_rate_hz', 'mean_hnr_db']
labels = ['Cycle Count', 'Duration (ms)', 'Voicing (%)', 'Mean F0 (Hz)', 'Cycle Rate (Hz)', 'Mean HNR (dB)']

for ax, measure, label in zip(axes.flat, measures, labels):
    sns.boxplot(data=speaker_data, x='sex', y=measure, ax=ax, palette={'F': '#E74C3C', 'M': '#3498DB'})
    ax.set_xlabel('Sex')
    ax.set_ylabel(label)
    ax.set_title(label)

plt.tight_layout()
plt.savefig(fig_dir / 'fig01_sex_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig01_sex_boxplots.png")

# ============================================================
# FIGURE 2: Sex differences - Violin plots
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, measure, label in zip(axes.flat, measures, labels):
    sns.violinplot(data=speaker_data, x='sex', y=measure, ax=ax, palette={'F': '#E74C3C', 'M': '#3498DB'})
    ax.set_xlabel('Sex')
    ax.set_ylabel(label)
    ax.set_title(label)

plt.tight_layout()
plt.savefig(fig_dir / 'fig02_sex_violins.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig02_sex_violins.png")

# ============================================================
# FIGURE 3: Context effects - Box plots
# ============================================================
context_data = data.groupby(['speaker_id', 'dataset', 'context_label']).agg({
    'num_cycles': 'mean',
    'duration_ms': 'mean',
    'voicing_pct': 'mean',
    'cycle_rate_hz': 'mean',
}).reset_index()

trill_contexts = ['intervocalic_rr', 'word_initial', 'after_nls', 'post_vocalic']
context_data = context_data[context_data['context_label'].isin(trill_contexts)]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
context_measures = ['num_cycles', 'duration_ms', 'voicing_pct', 'cycle_rate_hz']
context_labels = ['Cycle Count', 'Duration (ms)', 'Voicing (%)', 'Cycle Rate (Hz)']

for ax, measure, label in zip(axes.flat, context_measures, context_labels):
    sns.boxplot(data=context_data, x='context_label', y=measure, ax=ax,
                order=['intervocalic_rr', 'word_initial', 'after_nls', 'post_vocalic'])
    ax.set_xlabel('Phonological Context')
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(fig_dir / 'fig03_context_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig03_context_boxplots.png")

# ============================================================
# FIGURE 4: Dataset comparison - Bar chart
# ============================================================
dataset_summary = data.groupby('dataset').agg({
    'num_cycles': 'mean',
    'duration_ms': 'mean',
    'speaker_id': 'nunique',
    'utt_id': 'count'
}).reset_index()
dataset_summary.columns = ['dataset', 'mean_cycles', 'mean_duration', 'n_speakers', 'n_tokens']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
bars = ax.bar(dataset_summary['dataset'], dataset_summary['n_tokens'], color='steelblue')
ax.set_xlabel('Dataset')
ax.set_ylabel('Number of Trill Tokens')
ax.set_title('Token Count by Dataset')
ax.tick_params(axis='x', rotation=45)
for bar, n in zip(bars, dataset_summary['n_tokens']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, f'{n:,}',
            ha='center', va='bottom', fontsize=8)

ax = axes[1]
bars = ax.bar(dataset_summary['dataset'], dataset_summary['mean_cycles'], color='coral')
ax.set_xlabel('Dataset')
ax.set_ylabel('Mean Cycle Count')
ax.set_title('Mean Cycle Count by Dataset')
ax.tick_params(axis='x', rotation=45)
ax.axhline(y=data['num_cycles'].mean(), color='black', linestyle='--', label='Overall mean')
ax.legend()

plt.tight_layout()
plt.savefig(fig_dir / 'fig04_dataset_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig04_dataset_comparison.png")

# ============================================================
# FIGURE 5: Cycle count distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(data['num_cycles'], bins=range(0, 20), edgecolor='black', alpha=0.7)
ax.set_xlabel('Cycle Count')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Cycle Counts (All Tokens)')
ax.axvline(x=data['num_cycles'].mean(), color='red', linestyle='--', label=f'Mean={data["num_cycles"].mean():.1f}')
ax.axvline(x=data['num_cycles'].median(), color='green', linestyle='--', label=f'Median={data["num_cycles"].median():.1f}')
ax.legend()

ax = axes[1]
for sex, color in [('F', '#E74C3C'), ('M', '#3498DB')]:
    subset = speaker_data[speaker_data['sex'] == sex]['num_cycles']
    ax.hist(subset, bins=np.arange(0, 15, 0.5), alpha=0.5, label=sex, color=color, edgecolor='black')
ax.set_xlabel('Mean Cycle Count (per speaker)')
ax.set_ylabel('Frequency')
ax.set_title('Cycle Count Distribution by Sex')
ax.legend()

plt.tight_layout()
plt.savefig(fig_dir / 'fig05_cycle_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig05_cycle_distribution.png")

# ============================================================
# FIGURE 6: Duration vs Cycles scatter
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
for sex, color, marker in [('F', '#E74C3C', 'o'), ('M', '#3498DB', 's')]:
    subset = speaker_data[speaker_data['sex'] == sex]
    ax.scatter(subset['duration_ms'], subset['num_cycles'],
               c=color, alpha=0.6, s=30, marker=marker, label=sex)

ax.set_xlabel('Mean Duration (ms)')
ax.set_ylabel('Mean Cycle Count')
ax.set_title('Duration vs Cycle Count by Speaker')

slope, intercept, r, p, se = stats.linregress(speaker_data['duration_ms'], speaker_data['num_cycles'])
x_line = np.linspace(speaker_data['duration_ms'].min(), speaker_data['duration_ms'].max(), 100)
ax.plot(x_line, slope * x_line + intercept, 'k--', label=f'r={r:.2f}')
ax.legend()

plt.savefig(fig_dir / 'fig06_duration_vs_cycles.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig06_duration_vs_cycles.png")

# ============================================================
# FIGURE 7: F0 vs Cycle Rate
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
for sex, color, marker in [('F', '#E74C3C', 'o'), ('M', '#3498DB', 's')]:
    subset = speaker_data[speaker_data['sex'] == sex]
    ax.scatter(subset['mean_f0_hz'], subset['cycle_rate_hz'],
               c=color, alpha=0.6, s=30, marker=marker, label=sex)

ax.set_xlabel('Mean F0 (Hz)')
ax.set_ylabel('Cycle Rate (Hz)')
ax.set_title('F0 vs Cycle Rate by Sex')
ax.legend()

plt.savefig(fig_dir / 'fig07_f0_vs_cyclerate.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig07_f0_vs_cyclerate.png")

# ============================================================
# FIGURE 8: Effect sizes bar chart
# ============================================================
tests = pd.read_csv('outputs/tables/statistical_tests.csv')
sex_tests = tests[tests['predictor_variable'] == 'sex'].copy()

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green' if x > 0.14 else 'orange' if x > 0.06 else 'gray' for x in sex_tests['effect_size']]
bars = ax.barh(sex_tests['outcome_variable'], sex_tests['effect_size'], color=colors)
ax.set_xlabel('Effect Size (epsilon-squared)')
ax.set_ylabel('Acoustic Measure')
ax.set_title('Effect Sizes for Sex Differences')
ax.axvline(x=0.14, color='green', linestyle='--', alpha=0.5, label='Large (>0.14)')
ax.axvline(x=0.06, color='orange', linestyle='--', alpha=0.5, label='Medium (>0.06)')
ax.legend()

for bar, val in zip(bars, sex_tests['effect_size']):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center')

plt.tight_layout()
plt.savefig(fig_dir / 'fig08_effect_sizes.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig08_effect_sizes.png")

# ============================================================
# FIGURE 9: Voicing percentage by context
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(data=context_data, x='context_label', y='voicing_pct', ax=ax,
               order=['intervocalic_rr', 'word_initial', 'after_nls', 'post_vocalic'])
ax.set_xlabel('Phonological Context')
ax.set_ylabel('Voicing (%)')
ax.set_title('Voicing Percentage by Phonological Context')
ax.tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig(fig_dir / 'fig09_voicing_by_context.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig09_voicing_by_context.png")

# ============================================================
# FIGURE 10: HNR by sex (single plot)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 6))
sns.violinplot(data=speaker_data, x='sex', y='mean_hnr_db', ax=ax,
               palette={'F': '#E74C3C', 'M': '#3498DB'})
ax.set_xlabel('Sex')
ax.set_ylabel('Mean HNR (dB)')
ax.set_title('Harmonics-to-Noise Ratio by Sex')

plt.tight_layout()
plt.savefig(fig_dir / 'fig10_hnr_by_sex.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig10_hnr_by_sex.png")

# ============================================================
# FIGURE 11: Cycle count by sex (publication ready)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 6))
sns.boxplot(data=speaker_data, x='sex', y='num_cycles', ax=ax,
            palette={'F': '#E74C3C', 'M': '#3498DB'}, width=0.5)
sns.stripplot(data=speaker_data, x='sex', y='num_cycles', ax=ax,
              color='black', alpha=0.3, size=3)
ax.set_xlabel('Sex')
ax.set_ylabel('Mean Cycle Count')
ax.set_title('Trill Cycle Count by Sex')

for i, sex in enumerate(['F', 'M']):
    mean = speaker_data[speaker_data['sex'] == sex]['num_cycles'].mean()
    ax.text(i, mean + 0.3, f'M={mean:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(fig_dir / 'fig11_cycles_by_sex.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig11_cycles_by_sex.png")

# ============================================================
# FIGURE 12: Dataset x Sex interaction
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=speaker_data, x='dataset', y='num_cycles', hue='sex', ax=ax,
            palette={'F': '#E74C3C', 'M': '#3498DB'})
ax.set_xlabel('Dataset')
ax.set_ylabel('Mean Cycle Count')
ax.set_title('Cycle Count by Dataset and Sex')
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Sex')

plt.tight_layout()
plt.savefig(fig_dir / 'fig12_dataset_sex_interaction.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig12_dataset_sex_interaction.png")

# ============================================================
# FIGURE 13: Correlation heatmap
# ============================================================
corr_vars = ['num_cycles', 'duration_ms', 'voicing_pct', 'mean_f0_hz', 'cycle_rate_hz', 'mean_hnr_db']
corr_labels = ['Cycles', 'Duration', 'Voicing', 'F0', 'Cycle Rate', 'HNR']
corr_matrix = speaker_data[corr_vars].corr()

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=corr_labels, yticklabels=corr_labels, ax=ax,
            vmin=-1, vmax=1)
ax.set_title('Correlation Matrix of Acoustic Measures')

plt.tight_layout()
plt.savefig(fig_dir / 'fig13_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig13_correlation_heatmap.png")

# ============================================================
# FIGURE 14: Context means bar chart
# ============================================================
context_means = context_data.groupby('context_label').agg({
    'num_cycles': ['mean', 'std', 'count'],
    'voicing_pct': ['mean', 'std']
}).reset_index()
context_means.columns = ['context', 'cycles_mean', 'cycles_std', 'cycles_n', 'voicing_mean', 'voicing_std']
context_means = context_means[context_means['context'].isin(trill_contexts)]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

order = ['intervocalic_rr', 'word_initial', 'after_nls', 'post_vocalic']
ctx_ordered = context_means.set_index('context').loc[order].reset_index()

ax = axes[0]
bars = ax.bar(range(len(order)), ctx_ordered['cycles_mean'],
              yerr=ctx_ordered['cycles_std']/np.sqrt(ctx_ordered['cycles_n']), capsize=5, color='steelblue')
ax.set_xticks(range(len(order)))
ax.set_xticklabels(['Intervocalic\nrr', 'Word\nInitial', 'After\nn/l/s', 'Post\nVocalic'])
ax.set_ylabel('Mean Cycle Count')
ax.set_title('Cycle Count by Phonological Context')

ax = axes[1]
bars = ax.bar(range(len(order)), ctx_ordered['voicing_mean'],
              yerr=ctx_ordered['voicing_std']/np.sqrt(ctx_ordered['cycles_n']), capsize=5, color='coral')
ax.set_xticks(range(len(order)))
ax.set_xticklabels(['Intervocalic\nrr', 'Word\nInitial', 'After\nn/l/s', 'Post\nVocalic'])
ax.set_ylabel('Mean Voicing (%)')
ax.set_title('Voicing by Phonological Context')

plt.tight_layout()
plt.savefig(fig_dir / 'fig14_context_means.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig14_context_means.png")

# ============================================================
# FIGURE 15: Speaker density plot
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))
sns.kdeplot(data=speaker_data, x='mean_f0_hz', y='num_cycles', hue='sex',
            fill=True, alpha=0.5, levels=5, ax=ax,
            palette={'F': '#E74C3C', 'M': '#3498DB'})
ax.set_xlabel('Mean F0 (Hz)')
ax.set_ylabel('Mean Cycle Count')
ax.set_title('Speaker Distribution: F0 vs Cycle Count')

plt.tight_layout()
plt.savefig(fig_dir / 'fig15_speaker_density.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: fig15_speaker_density.png")

print("\n=== DONE ===")
print(f"Generated 15 figures in {fig_dir}")
