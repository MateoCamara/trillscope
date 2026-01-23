"""CLI for statistical analysis (Block H)."""

import argparse
import logging
import sys
from pathlib import Path

from .base import AnalysisConfig, AnalysisResult
from .data_loader import load_analysis_data
from .descriptive import compute_all_descriptive_stats, get_group_counts
from .inferential import run_all_tests
from .visualization import create_all_visualizations
from .report_generator import generate_report, save_tables

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_analysis(
    measurements_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    reports_dir: Path,
    config: AnalysisConfig,
    datasets: list = None,
    skip_viz: bool = False
) -> AnalysisResult:
    """
    Run complete statistical analysis.

    Args:
        measurements_dir: Directory with acoustic_measurements_*.parquet
        metadata_path: Path to metadata_unified.parquet
        output_dir: Directory for figures and tables
        reports_dir: Directory for report
        config: Analysis configuration
        datasets: List of datasets to include
        skip_viz: Skip visualization generation

    Returns:
        AnalysisResult object
    """
    result = AnalysisResult(config=config)

    # 1. Load and prepare data
    logger.info("Loading analysis data...")
    try:
        df, data_stats = load_analysis_data(
            measurements_dir,
            metadata_path,
            datasets=datasets,
            aggregate_by_speaker=True
        )
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        result.add_issue(str(e))
        return result

    logger.info(f"Loaded {len(df)} records for analysis")

    # 2. Compute descriptive statistics
    logger.info("Computing descriptive statistics...")
    result.descriptive_stats = compute_all_descriptive_stats(
        df,
        config.outcome_variables,
        config.predictor_variables
    )

    # Log group counts
    group_counts = get_group_counts(df, config.predictor_variables)
    if len(group_counts) > 0:
        logger.info("Group counts:")
        for _, row in group_counts.iterrows():
            logger.info(f"  {row['variable']}: {row['level']} = {row['n']} ({row['percentage']:.1f}%)")

    # 3. Run inferential tests
    logger.info("Running statistical tests...")
    result.test_results = run_all_tests(
        df,
        config.outcome_variables,
        config.predictor_variables,
        config
    )
    logger.info(f"Completed {len(result.test_results)} tests")

    # 4. Generate visualizations
    if not skip_viz:
        logger.info("Generating visualizations...")
        figures_dir = output_dir / 'figures'
        result.figures = create_all_visualizations(
            df,
            config.outcome_variables,
            config.predictor_variables,
            figures_dir,
            config.figsize,
            config.dpi
        )
        logger.info(f"Generated {len(result.figures)} figures")

    # 5. Save tables
    tables_dir = output_dir / 'tables'
    save_tables(result, tables_dir)

    # 6. Generate report
    logger.info("Generating report...")
    report_path = reports_dir / 'statistical_analysis.md'
    generate_report(result, report_path, data_stats)

    # Summary
    sig_tests = result.get_significant_tests()
    logger.info(f"Analysis complete: {len(sig_tests)}/{len(result.test_results)} significant tests")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Statistical analysis of trill /r/ acoustic properties'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run statistical analysis')
    analyze_parser.add_argument(
        'factors',
        nargs='?',
        default='all',
        choices=['all', 'sex', 'age', 'education', 'country'],
        help='Factors to analyze'
    )
    analyze_parser.add_argument(
        '--datasets',
        nargs='+',
        default=['albayzin', 'dimex100', 'preseea'],
        help='Datasets to include'
    )
    analyze_parser.add_argument(
        '--measurements-dir',
        type=Path,
        default=Path('outputs/tables'),
        help='Directory with acoustic measurements'
    )
    analyze_parser.add_argument(
        '--metadata-path',
        type=Path,
        default=Path('metadata/metadata_unified.parquet'),
        help='Path to unified metadata'
    )
    analyze_parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs'),
        help='Output directory for figures and tables'
    )
    analyze_parser.add_argument(
        '--reports-dir',
        type=Path,
        default=Path('reports'),
        help='Directory for reports'
    )
    analyze_parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level'
    )
    analyze_parser.add_argument(
        '--min-group-size',
        type=int,
        default=10,
        help='Minimum group size for analysis'
    )
    analyze_parser.add_argument(
        '--correction',
        choices=['bonferroni', 'holm', 'none'],
        default='bonferroni',
        help='Multiple comparison correction method'
    )
    analyze_parser.add_argument(
        '--skip-viz',
        action='store_true',
        help='Skip visualization generation'
    )

    # Report command (just generate report from existing results)
    report_parser = subparsers.add_parser('report', help='Generate report from existing analysis')
    report_parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs'),
        help='Output directory with tables'
    )
    report_parser.add_argument(
        '--reports-dir',
        type=Path,
        default=Path('reports'),
        help='Directory for reports'
    )

    # Show config command
    config_parser = subparsers.add_parser('show-config', help='Show default configuration')

    args = parser.parse_args()

    if args.command == 'analyze':
        # Configure predictors based on factors
        if args.factors == 'all':
            predictor_vars = ['sex', 'age_bin', 'education_bin', 'country', 'context_label']
        elif args.factors == 'sex':
            predictor_vars = ['sex']
        elif args.factors == 'age':
            predictor_vars = ['age_bin']
        elif args.factors == 'education':
            predictor_vars = ['education_bin']
        elif args.factors == 'country':
            predictor_vars = ['country']
        else:
            predictor_vars = ['sex', 'age_bin', 'education_bin']

        config = AnalysisConfig(
            predictor_variables=predictor_vars,
            alpha=args.alpha,
            min_group_size=args.min_group_size,
            correction_method=args.correction if args.correction != 'none' else None,
        )

        result = run_analysis(
            args.measurements_dir,
            args.metadata_path,
            args.output_dir,
            args.reports_dir,
            config,
            datasets=args.datasets,
            skip_viz=args.skip_viz
        )

        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total tests: {len(result.test_results)}")
        print(f"Significant tests: {len(result.get_significant_tests())}")
        print(f"Figures generated: {len(result.figures)}")
        print(f"Report: {args.reports_dir / 'statistical_analysis.md'}")

        if result.get_significant_tests():
            print("\nSignificant findings:")
            for test in result.get_significant_tests():
                print(f"  - {test.outcome_variable} by {test.predictor_variable}: "
                      f"p={test.p_adjusted or test.p_value:.4f}, "
                      f"effect={test.effect_size_interpretation}")

    elif args.command == 'report':
        # Load existing results and regenerate report
        import pandas as pd

        tables_dir = args.output_dir / 'tables'
        desc_path = tables_dir / 'descriptive_stats.csv'
        tests_path = tables_dir / 'statistical_tests.csv'

        if not desc_path.exists():
            print(f"Error: {desc_path} not found. Run 'analyze' first.")
            sys.exit(1)

        config = AnalysisConfig()
        result = AnalysisResult(config=config)
        result.descriptive_stats = pd.read_csv(desc_path)

        # Note: Cannot fully reconstruct TestResult objects from CSV
        print(f"Regenerating report from {tables_dir}")
        report_path = args.reports_dir / 'statistical_analysis.md'
        generate_report(result, report_path)
        print(f"Report saved to {report_path}")

    elif args.command == 'show-config':
        config = AnalysisConfig()
        print("Default Analysis Configuration:")
        print(f"  Outcome variables: {config.outcome_variables}")
        print(f"  Predictor variables: {config.predictor_variables}")
        print(f"  Alpha: {config.alpha}")
        print(f"  Min group size: {config.min_group_size}")
        print(f"  Correction method: {config.correction_method}")
        print(f"  Figure size: {config.figsize}")
        print(f"  DPI: {config.dpi}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
