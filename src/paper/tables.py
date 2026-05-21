"""Generate formatted tables for the research paper."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def generate_corpus_table(output_path: Path) -> str:
    """
    Generate Table 1: Corpus overview.

    Args:
        output_path: Output CSV path

    Returns:
        Path to saved table
    """
    data = {
        'Corpus': ['ALBAYZIN', 'PRESEEA', 'DIMEx100', 'Total'],
        'Speech Style': ['Read', 'Spontaneous', 'Controlled', '-'],
        'Speakers': ['55', '16', '-', '71*'],
        'Trill Tokens': ['694', '12,969', '46,340', '60,003'],
        'Country': ['Spain', 'Multiple', 'Mexico', '-'],
    }

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved corpus table: {output_path}")
    return str(output_path)


def generate_descriptive_table(
    data_path: Path,
    output_path: Path
) -> str:
    """
    Generate Table 2: Overall descriptive statistics.

    Args:
        data_path: Path to descriptive_stats.csv
        output_path: Output CSV path

    Returns:
        Path to saved table
    """
    df = pd.read_csv(data_path)

    # Filter to overall statistics
    overall = df[df['group_by'] == 'overall'].copy()

    if len(overall) == 0:
        logger.error("No overall statistics found")
        return ""

    # Format variable names
    var_names = {
        'num_cycles': 'Number of cycles',
        'duration_ms': 'Duration (ms)',
        'voicing_pct': 'Voicing (%)',
        'mean_f0_hz': 'Mean F0 (Hz)',
        'cycle_rate_hz': 'Cycle rate (Hz)',
        'mean_hnr_db': 'Mean HNR (dB)',
    }

    # Select and rename columns
    result = overall[['variable', 'n', 'mean', 'std', 'median', 'min', 'max']].copy()
    result['variable'] = result['variable'].map(var_names)
    result = result.dropna(subset=['variable'])

    # Format numbers
    for col in ['mean', 'std', 'median', 'min', 'max']:
        result[col] = result[col].apply(lambda x: f'{x:.2f}')

    result.columns = ['Variable', 'N', 'Mean', 'SD', 'Median', 'Min', 'Max']
    result.to_csv(output_path, index=False)

    logger.info(f"Saved descriptive table: {output_path}")
    return str(output_path)


def generate_sex_comparison_table(
    desc_path: Path,
    tests_path: Path,
    output_path: Path
) -> str:
    """
    Generate Table 3: Sex differences with statistics.

    Args:
        desc_path: Path to descriptive_stats.csv
        tests_path: Path to statistical_tests.csv
        output_path: Output CSV path

    Returns:
        Path to saved table
    """
    desc = pd.read_csv(desc_path)
    tests = pd.read_csv(tests_path)

    # Filter descriptive stats to sex comparisons
    sex_desc = desc[(desc['group_by'] == 'sex') & (desc['group_level'].isin(['F', 'M']))]

    # Filter tests to sex comparisons
    sex_tests = tests[tests['predictor_variable'] == 'sex']

    if len(sex_desc) == 0 or len(sex_tests) == 0:
        logger.error("No sex comparison data found")
        return ""

    # Variables of interest (significant ones)
    variables = ['num_cycles', 'duration_ms', 'voicing_pct', 'mean_f0_hz', 'cycle_rate_hz', 'mean_hnr_db']
    var_names = {
        'num_cycles': 'Number of cycles',
        'duration_ms': 'Duration (ms)',
        'voicing_pct': 'Voicing (%)',
        'mean_f0_hz': 'Mean F0 (Hz)',
        'cycle_rate_hz': 'Cycle rate (Hz)',
        'mean_hnr_db': 'Mean HNR (dB)',
    }

    rows = []
    for var in variables:
        female = sex_desc[(sex_desc['variable'] == var) & (sex_desc['group_level'] == 'F')]
        male = sex_desc[(sex_desc['variable'] == var) & (sex_desc['group_level'] == 'M')]
        test = sex_tests[sex_tests['outcome_variable'] == var]

        if len(female) == 0 or len(male) == 0 or len(test) == 0:
            continue

        female = female.iloc[0]
        male = male.iloc[0]
        test = test.iloc[0]

        # Significance stars
        p_adj = test['p_adjusted'] if pd.notna(test['p_adjusted']) else test['p_value']
        if p_adj < 0.001:
            sig = '***'
        elif p_adj < 0.01:
            sig = '**'
        elif p_adj < 0.05:
            sig = '*'
        else:
            sig = ''

        rows.append({
            'Variable': var_names.get(var, var),
            'Female (n=30)': f"{female['mean']:.2f} ({female['std']:.2f})",
            'Male (n=38)': f"{male['mean']:.2f} ({male['std']:.2f})",
            'U': f"{test['statistic']:.1f}",
            'p': f"{p_adj:.4f}{sig}",
            'r': f"{test['effect_size']:.2f}",
        })

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)

    logger.info(f"Saved sex comparison table: {output_path}")
    return str(output_path)


def generate_education_table(
    desc_path: Path,
    tests_path: Path,
    output_path: Path
) -> str:
    """
    Generate Table 4: Education effects with statistics.

    Args:
        desc_path: Path to descriptive_stats.csv
        tests_path: Path to statistical_tests.csv
        output_path: Output CSV path

    Returns:
        Path to saved table
    """
    desc = pd.read_csv(desc_path)
    tests = pd.read_csv(tests_path)

    # Filter to education comparisons
    edu_desc = desc[(desc['group_by'] == 'education_bin') &
                    (desc['group_level'].isin(['low', 'mid', 'high']))]
    edu_tests = tests[tests['predictor_variable'] == 'education_bin']

    if len(edu_desc) == 0 or len(edu_tests) == 0:
        logger.error("No education comparison data found")
        return ""

    # Variables
    variables = ['num_cycles', 'duration_ms', 'voicing_pct', 'mean_f0_hz', 'cycle_rate_hz', 'mean_hnr_db']
    var_names = {
        'num_cycles': 'Number of cycles',
        'duration_ms': 'Duration (ms)',
        'voicing_pct': 'Voicing (%)',
        'mean_f0_hz': 'Mean F0 (Hz)',
        'cycle_rate_hz': 'Cycle rate (Hz)',
        'mean_hnr_db': 'Mean HNR (dB)',
    }

    rows = []
    for var in variables:
        low = edu_desc[(edu_desc['variable'] == var) & (edu_desc['group_level'] == 'low')]
        mid = edu_desc[(edu_desc['variable'] == var) & (edu_desc['group_level'] == 'mid')]
        high = edu_desc[(edu_desc['variable'] == var) & (edu_desc['group_level'] == 'high')]
        test = edu_tests[edu_tests['outcome_variable'] == var]

        if len(low) == 0 or len(mid) == 0 or len(high) == 0 or len(test) == 0:
            continue

        low = low.iloc[0]
        mid = mid.iloc[0]
        high = high.iloc[0]
        test = test.iloc[0]

        # Significance stars
        p_adj = test['p_adjusted'] if pd.notna(test['p_adjusted']) else test['p_value']
        if p_adj < 0.001:
            sig = '***'
        elif p_adj < 0.01:
            sig = '**'
        elif p_adj < 0.05:
            sig = '*'
        else:
            sig = ''

        rows.append({
            'Variable': var_names.get(var, var),
            'Low (n=14)': f"{low['mean']:.2f} ({low['std']:.2f})",
            'Mid (n=42)': f"{mid['mean']:.2f} ({mid['std']:.2f})",
            'High (n=12)': f"{high['mean']:.2f} ({high['std']:.2f})",
            'H': f"{test['statistic']:.2f}",
            'p': f"{p_adj:.4f}{sig}",
            'ε²': f"{test['effect_size']:.3f}",
        })

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)

    logger.info(f"Saved education table: {output_path}")
    return str(output_path)


def generate_all_tables(
    output_dir: Path,
    data_dir: Path = Path('outputs/tables')
) -> dict:
    """
    Generate all paper tables.

    Args:
        output_dir: Directory to save tables
        data_dir: Directory containing data files

    Returns:
        Dict mapping table names to paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {}

    # Table 1: Corpus overview
    corpus_path = output_dir / 'table1_corpora.csv'
    generate_corpus_table(corpus_path)
    tables['table1_corpora'] = str(corpus_path)

    # Table 2: Descriptive statistics
    desc_data = data_dir / 'descriptive_stats.csv'
    if desc_data.exists():
        desc_path = output_dir / 'table2_descriptive.csv'
        generate_descriptive_table(desc_data, desc_path)
        tables['table2_descriptive'] = str(desc_path)

    # Table 3: Sex differences
    tests_data = data_dir / 'statistical_tests.csv'
    if desc_data.exists() and tests_data.exists():
        sex_path = output_dir / 'table3_sex.csv'
        generate_sex_comparison_table(desc_data, tests_data, sex_path)
        tables['table3_sex'] = str(sex_path)

        # Table 4: Education effects
        edu_path = output_dir / 'table4_education.csv'
        generate_education_table(desc_data, tests_data, edu_path)
        tables['table4_education'] = str(edu_path)

    return tables


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tables = generate_all_tables(
        output_dir=Path('paper/tables'),
        data_dir=Path('outputs/tables')
    )
    print(f"Generated {len(tables)} tables:")
    for name, path in tables.items():
        print(f"  {name}: {path}")
