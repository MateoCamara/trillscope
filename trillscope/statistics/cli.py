"""CLI for statistical analysis."""

import argparse
import logging
import sys
from pathlib import Path

from .base import AnalysisConfig, AnalysisResult
from .data_loader import load_analysis_data, _aggregate_by_speaker
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
    skip_viz: bool = False,
    min_cycles: int = 0,
    min_duration: float = 0.0,
    max_cycles: int = 0,
    exclude_overlap: bool = False,
    run_mixed_effects: bool = False,
    exclude_f0_for_dimex: bool = True
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
        min_cycles: Minimum cycle count filter (filter sensitivity)
        min_duration: Minimum duration filter in ms (filter sensitivity)
        max_cycles: Maximum cycle count filter (outlier removal)
        exclude_overlap: Exclude PRESEEA tokens with overlap markers
        run_mixed_effects: Also run token-level mixed-effects models
            (speaker as random effect)
        exclude_f0_for_dimex: Exclude F0 from DIMEx100 sex analysis

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
            aggregate_by_speaker=False  # Get token-level data first for filtering
        )
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        result.add_issue(str(e))
        return result

    initial_count = len(df)
    logger.info(f"Loaded {initial_count} records")

    # 2. Apply optional token filters (filter sensitivity)
    if min_cycles > 0:
        df = df[df['num_cycles'] >= min_cycles]
        logger.info(f"After min_cycles={min_cycles} filter: {len(df)} records")

    if min_duration > 0:
        df = df[df['duration_ms'] >= min_duration]
        logger.info(f"After min_duration={min_duration}ms filter: {len(df)} records")

    if max_cycles > 0:
        df = df[df['num_cycles'] <= max_cycles]
        logger.info(f"After max_cycles={max_cycles} filter: {len(df)} records")

    if exclude_overlap and 'has_overlap' in df.columns:
        df = df[~df['has_overlap']]
        logger.info(f"After excluding overlap: {len(df)} records")

    # Track filter statistics
    data_stats['filters_applied'] = {
        'min_cycles': min_cycles,
        'min_duration': min_duration,
        'max_cycles': max_cycles,
        'exclude_overlap': exclude_overlap,
        'initial_count': initial_count,
        'filtered_count': len(df),
        'removed_count': initial_count - len(df)
    }

    # 3. Aggregate by speaker for traditional analysis
    # For context analysis, aggregate by speaker+context
    include_context = 'context_label' in config.predictor_variables
    df_aggregated = _aggregate_by_speaker(df, include_context=include_context)
    if include_context:
        logger.info(f"Aggregated to {len(df_aggregated)} speaker-context combinations")
    else:
        logger.info(f"Aggregated to {len(df_aggregated)} speakers")

    # 4. Compute descriptive statistics
    logger.info("Computing descriptive statistics...")
    result.descriptive_stats = compute_all_descriptive_stats(
        df_aggregated,
        config.outcome_variables,
        config.predictor_variables
    )

    # Log group counts
    group_counts = get_group_counts(df_aggregated, config.predictor_variables)
    if len(group_counts) > 0:
        logger.info("Group counts:")
        for _, row in group_counts.iterrows():
            logger.info(f"  {row['variable']}: {row['level']} = {row['n']} ({row['percentage']:.1f}%)")

    # 5. Run inferential tests
    logger.info("Running statistical tests...")
    result.test_results = run_all_tests(
        df_aggregated,
        config.outcome_variables,
        config.predictor_variables,
        config
    )
    logger.info(f"Completed {len(result.test_results)} tests")

    # 6. Run token-level mixed-effects models if requested
    if run_mixed_effects:
        logger.info("Running mixed-effects models...")
        from .inferential import run_mixed_effects_tests
        mixed_results = run_mixed_effects_tests(
            df,  # Use token-level data, not aggregated
            config.outcome_variables,
            config.predictor_variables,
            config
        )
        result.mixed_effects_results = mixed_results
        logger.info(f"Completed {len(mixed_results)} mixed-effects tests")

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
        choices=['all', 'sex', 'age', 'education', 'country', 'region', 'context', 'speech_style'],
        help='Factors to analyze'
    )
    analyze_parser.add_argument(
        '--datasets',
        nargs='+',
        default=['albayzin', 'dimex100', 'preseea', 'glissando', 'mailabs', 'tedx', 'heroico', 'commonvoice'],
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
    # Subsetting and sensitivity-analysis arguments
    analyze_parser.add_argument(
        '--corpus',
        choices=['albayzin', 'preseea', 'dimex100'],
        help='Analyze only a single corpus (for within-corpus analysis)'
    )
    analyze_parser.add_argument(
        '--min-cycles',
        type=int,
        default=0,
        help='Minimum number of cycles to include (for filter-sensitivity analysis)'
    )
    analyze_parser.add_argument(
        '--min-duration',
        type=float,
        default=0.0,
        help='Minimum duration in ms to include (for filter-sensitivity analysis)'
    )
    analyze_parser.add_argument(
        '--max-cycles',
        type=int,
        default=0,
        help='Maximum number of cycles to include (0 = no max, for outlier removal)'
    )
    analyze_parser.add_argument(
        '--exclude-overlap',
        action='store_true',
        help='Exclude PRESEEA tokens with overlap markers'
    )
    analyze_parser.add_argument(
        '--mixed-effects',
        action='store_true',
        help='Also run token-level mixed-effects models with speaker as random effect'
    )
    analyze_parser.add_argument(
        '--exclude-f0-for-dimex',
        action='store_true',
        default=True,
        help='Exclude F0 from DIMEx100 sex analysis (due to circularity)'
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
    subparsers.add_parser('show-config', help='Show default configuration')

    args = parser.parse_args()

    if args.command == 'analyze':
        # Configure predictors based on factors
        if args.factors == 'all':
            # country/region are excluded from 'all': the post-quality-filter
            # sample is España-dominated (n~93 vs Arg/Mex/Ven n=3-6), too
            # unbalanced for a reliable dialectal test.
            predictor_vars = ['sex', 'age_bin', 'education_bin', 'speech_style', 'context_label']
        elif args.factors == 'sex':
            predictor_vars = ['sex']
        elif args.factors == 'age':
            predictor_vars = ['age_bin']
        elif args.factors == 'education':
            predictor_vars = ['education_bin']
        elif args.factors == 'country':
            predictor_vars = ['country']
        elif args.factors == 'region':
            predictor_vars = ['region']
        elif args.factors == 'context':
            predictor_vars = ['context_label']
        elif args.factors == 'speech_style':
            predictor_vars = ['speech_style']
        else:
            predictor_vars = ['sex', 'age_bin', 'education_bin']

        config = AnalysisConfig(
            predictor_variables=predictor_vars,
            alpha=args.alpha,
            min_group_size=args.min_group_size,
            correction_method=args.correction if args.correction != 'none' else None,
        )

        # Handle --corpus shortcut (single corpus = that corpus only)
        if args.corpus:
            datasets = [args.corpus]
            logger.info(f"Running within-corpus analysis for: {args.corpus}")
        else:
            datasets = args.datasets

        result = run_analysis(
            args.measurements_dir,
            args.metadata_path,
            args.output_dir,
            args.reports_dir,
            config,
            datasets=datasets,
            skip_viz=args.skip_viz,
            min_cycles=args.min_cycles,
            min_duration=args.min_duration,
            max_cycles=args.max_cycles,
            exclude_overlap=args.exclude_overlap,
            run_mixed_effects=args.mixed_effects,
            exclude_f0_for_dimex=args.exclude_f0_for_dimex
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
