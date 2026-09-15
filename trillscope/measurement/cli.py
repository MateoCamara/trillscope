"""Command-line interface for acoustic measurement (Block G)."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .base import MeasurementConfig, MeasurementResult
from .processors.dimex100_processor import DIMEx100Processor
from .processors.albayzin_processor import AlbayzinProcessor
from .processors.preseea_processor import PreseeaProcessor
from .processors.glissando_processor import GlissandoProcessor
from .processors.mfa_processor import MFAProcessor
from .utils.report_generator import MeasurementReportGenerator

# Supported datasets
ORIGINAL_DATASETS = ['dimex100', 'albayzin', 'preseea', 'glissando']
MFA_DATASETS = ['mailabs', 'tedx', 'heroico', 'commonvoice']
ALL_DATASETS = ORIGINAL_DATASETS + MFA_DATASETS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def measure_dataset(
    dataset: str,
    candidates_dir: Path,
    output_dir: Path,
    config: Optional[MeasurementConfig] = None,
    skip_invalid_timing: bool = True
) -> Optional[MeasurementResult]:
    """Run measurement on a single dataset."""
    candidates_path = candidates_dir / f'r_candidates_{dataset}.parquet'

    if not candidates_path.exists():
        logger.warning(f"{dataset} candidates not found at {candidates_path}")
        logger.warning(f"Run Block F extraction first: python -m trillscope.extraction.cli extract {dataset}")
        return None

    if dataset == 'dimex100':
        processor = DIMEx100Processor(candidates_path, config)
    elif dataset == 'albayzin':
        processor = AlbayzinProcessor(candidates_path, config)
    elif dataset == 'preseea':
        processor = PreseeaProcessor(candidates_path, config)
    elif dataset == 'glissando':
        processor = GlissandoProcessor(candidates_path, config)
    elif dataset in MFA_DATASETS:
        processor = MFAProcessor(candidates_path, config, dataset_name=dataset)
    else:
        logger.error(f"Unknown dataset: {dataset}")
        return None

    result = processor.process_all(skip_invalid_timing=skip_invalid_timing)

    # Save measurements parquet
    measurements_path = output_dir / 'outputs' / 'tables' / f'acoustic_measurements_{dataset}.parquet'
    processor.save_result(result, measurements_path)

    # Generate report
    report_gen = MeasurementReportGenerator(output_dir / 'reports' / 'measurement')
    report_path = report_gen.generate(result)
    logger.info(f"Generated report: {report_path}")

    return result


def show_config():
    """Display default configuration."""
    config = MeasurementConfig()
    print("\n" + "=" * 60)
    print("  DEFAULT MEASUREMENT CONFIGURATION")
    print("=" * 60 + "\n")

    print("Audio loading:")
    print(f"  Target sample rate    : {config.target_sr} Hz")
    print(f"  Context padding       : {config.context_ms} ms")
    print()

    print("Cycle detection:")
    print(f"  Method                : {config.cycle_method}")
    print(f"  Min cycle duration    : {config.min_cycle_duration_ms} ms")
    print(f"  Max cycle duration    : {config.max_cycle_duration_ms} ms")
    print(f"  Envelope smoothing    : {config.envelope_smoothing_ms} ms")
    print(f"  Min peak prominence   : {config.min_cycle_prominence}")
    print(f"  Band-pass low cutoff  : {config.bp_freq_low_hz} Hz")
    print(f"  Band-pass high cutoff : {config.bp_freq_high_hz} Hz")
    print(f"  Filter order          : {config.bp_filter_order}")
    print(f"  STFT window           : {config.spectrogram_win_ms} ms")
    print(f"  STFT hop              : {config.spectrogram_hop_ms} ms")
    print()

    print("Voicing detection:")
    print(f"  Pitch floor           : {config.pitch_floor_hz} Hz")
    print(f"  Pitch ceiling         : {config.pitch_ceiling_hz} Hz")
    print(f"  Voicing threshold     : {config.voicing_threshold}")
    print()

    print("Quality thresholds:")
    print(f"  Min duration          : {config.min_duration_ms} ms")
    print(f"  Max duration (warning): {config.max_duration_ms} ms")
    print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Acoustic measurement for trill /r/ tokens (Block G)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Measure all datasets
  python -m trillscope.measurement.cli measure all

  # Measure specific dataset
  python -m trillscope.measurement.cli measure dimex100

  # Include tokens without timing (PRESEEA orthographic)
  python -m trillscope.measurement.cli measure all --include-invalid-timing

  # Show default configuration
  python -m trillscope.measurement.cli show-config

Output files:
  outputs/tables/acoustic_measurements_<dataset>.parquet  - Per-token measurements
  reports/measurement/<dataset>.md                        - Summary report
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Measure command
    measure_parser = subparsers.add_parser('measure', help='Run acoustic measurement')
    measure_parser.add_argument(
        'dataset',
        choices=ALL_DATASETS + ['all'],
        help='Dataset to process'
    )
    measure_parser.add_argument(
        '--candidates-dir',
        type=Path,
        default=Path('outputs/tables'),
        help='Directory containing r_candidates_*.parquet files (default: outputs/tables/)'
    )
    measure_parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Output directory (default: current directory)'
    )
    measure_parser.add_argument(
        '--include-invalid-timing',
        action='store_true',
        help='Include tokens without valid timing (will fail measurement)'
    )

    # Configuration arguments
    measure_parser.add_argument(
        '--min-cycle-ms',
        type=float,
        default=10.0,
        help='Minimum cycle duration in ms (default: 10)'
    )
    measure_parser.add_argument(
        '--max-cycle-ms',
        type=float,
        default=60.0,
        help='Maximum cycle duration in ms (default: 60)'
    )
    measure_parser.add_argument(
        '--min-duration-ms',
        type=float,
        default=5.0,
        help='Minimum token duration in ms (default: 5)'
    )
    measure_parser.add_argument(
        '--cycle-method',
        choices=['multi', 'envelope', 'legacy'],
        default='multi',
        help='Cycle detection method (default: multi)'
    )
    measure_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # Show config command
    subparsers.add_parser('show-config', help='Display default configuration')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'show-config':
        show_config()
        sys.exit(0)

    if args.command == 'measure':
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        candidates_dir = args.candidates_dir.resolve()
        output_dir = args.output_dir.resolve()

        logger.info(f"Candidates directory: {candidates_dir}")
        logger.info(f"Output directory: {output_dir}")

        # Create configuration from arguments
        config = MeasurementConfig(
            min_cycle_duration_ms=args.min_cycle_ms,
            max_cycle_duration_ms=args.max_cycle_ms,
            min_duration_ms=args.min_duration_ms,
            cycle_method=args.cycle_method,
        )

        skip_invalid = not args.include_invalid_timing
        all_results = []

        # Determine which datasets to process
        if args.dataset == 'all':
            datasets = ALL_DATASETS
        else:
            datasets = [args.dataset]

        for dataset in datasets:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"  Processing {dataset.upper()}")
            logger.info(f"{'=' * 60}\n")

            result = measure_dataset(
                dataset, candidates_dir, output_dir, config, skip_invalid
            )

            if result:
                all_results.append(result)

                # Print summary
                stats = result.statistics
                logger.info(f"\n{dataset.upper()} Summary:")
                logger.info(f"  Total tokens: {stats.get('total_tokens', 0):,}")
                logger.info(f"  Successful: {stats.get('successful', 0):,}")
                logger.info(f"  Failed: {stats.get('failed', 0):,}")

                # Confidence summary
                conf_dist = stats.get('confidence_distribution', {})
                if conf_dist:
                    logger.info(f"  Confidence: high={conf_dist.get('high', 0):,}, "
                                f"medium={conf_dist.get('medium', 0):,}, "
                                f"low={conf_dist.get('low', 0):,}")

                # Cycle distribution
                cycle_dist = stats.get('cycle_distribution', {})
                if cycle_dist:
                    logger.info("  Cycle distribution:")
                    for cycles in sorted(cycle_dist.keys())[:6]:
                        count = cycle_dist[cycles]
                        logger.info(f"    {cycles} cycles: {count:,}")

        if all_results:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"  MEASUREMENT COMPLETE")
            logger.info(f"{'=' * 60}")

            total_tokens = sum(r.statistics.get('total_tokens', 0) for r in all_results)
            total_success = sum(r.statistics.get('successful', 0) for r in all_results)

            logger.info(f"  Total tokens: {total_tokens:,}")
            logger.info(f"  Total successful: {total_success:,}")

        logger.info("\nAcoustic measurement complete!")


if __name__ == '__main__':
    main()
