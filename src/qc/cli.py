"""Command-line interface for audio quality control (Block E)."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .base import QCResult, QCThresholds
from .albayzin import AlbayzinQCProcessor
from .preseea import PreseeaQCProcessor
from .dimex100 import DIMEx100QCProcessor
from .utils.report_generator import QCReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def qc_dataset(
    dataset: str,
    index_dir: Path,
    data_norm_dir: Path,
    output_dir: Path,
    thresholds: Optional[QCThresholds] = None
) -> Optional[QCResult]:
    """Run QC on a single dataset."""
    index_path = index_dir / f'utt_index_{dataset}.csv'

    if not index_path.exists():
        logger.warning(f"{dataset} index not found at {index_path}")
        logger.warning(f"Run Block B normalization first: python -m src.normalization.cli normalize {dataset}")
        return None

    if dataset == 'albayzin':
        processor = AlbayzinQCProcessor(index_path, data_norm_dir, thresholds)
    elif dataset == 'preseea':
        processor = PreseeaQCProcessor(index_path, data_norm_dir, thresholds)
    elif dataset == 'dimex100':
        processor = DIMEx100QCProcessor(index_path, data_norm_dir, thresholds)
    else:
        logger.error(f"Unknown dataset: {dataset}")
        return None

    result = processor.process_all()

    # Save QC flags CSV
    csv_output = output_dir / 'outputs' / 'tables' / f'qc_flags_{dataset}.csv'
    result.to_csv(csv_output)

    # Generate report
    report_gen = QCReportGenerator(output_dir / 'reports' / 'qc')
    report_path = report_gen.generate(result)
    logger.info(f"Generated QC report: {report_path}")

    return result


def show_thresholds():
    """Display default thresholds."""
    thresholds = QCThresholds()
    print("\n" + "=" * 60)
    print("  DEFAULT QC THRESHOLDS")
    print("=" * 60 + "\n")

    print("Failure thresholds (utterance is discarded if):")
    print(f"  Clipping     > {thresholds.clipping_max_pct}%")
    print(f"  Duration     < {thresholds.duration_min_ms} ms")
    print(f"  SNR          < {thresholds.snr_min_db} dB")
    print(f"  Silence      > {thresholds.silence_max_pct}%")
    print()

    print("Analysis parameters:")
    print(f"  Frame size         : {thresholds.frame_size_ms} ms")
    print(f"  Hop size           : {thresholds.hop_size_ms} ms")
    print(f"  Clipping threshold : {thresholds.clipping_near_max_fraction} (fraction of max)")
    print(f"  Silence threshold  : {thresholds.silence_threshold_db} dB")
    print(f"  VAD speech thresh  : {thresholds.vad_speech_threshold_db} dB")
    print(f"  VAD noise thresh   : {thresholds.vad_noise_threshold_db} dB")
    print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Audio Quality Control for Spanish speech corpora (Block E)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run QC on all datasets with default thresholds
  python -m src.qc.cli qc all

  # Run QC on ALBAYZIN only
  python -m src.qc.cli qc albayzin

  # Run QC with custom thresholds
  python -m src.qc.cli qc all --snr-min 15 --clipping-max 0.05

  # Show default thresholds
  python -m src.qc.cli show-thresholds

Output files:
  outputs/tables/qc_flags_<dataset>.csv  - Per-utterance QC metrics and flags
  reports/qc/<dataset>.md                - Summary report with statistics
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # QC command
    qc_parser = subparsers.add_parser('qc', help='Run audio quality control')
    qc_parser.add_argument(
        'dataset',
        choices=['albayzin', 'preseea', 'dimex100', 'all'],
        help='Dataset to process'
    )
    qc_parser.add_argument(
        '--index-dir',
        type=Path,
        default=Path('outputs/tables'),
        help='Directory containing utt_index_*.csv files (default: outputs/tables/)'
    )
    qc_parser.add_argument(
        '--data-norm-dir',
        type=Path,
        default=Path('data_norm'),
        help='Directory containing normalized audio (default: data_norm/)'
    )
    qc_parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Output directory (default: current directory)'
    )

    # Threshold arguments
    qc_parser.add_argument(
        '--clipping-max',
        type=float,
        default=0.1,
        help='Max clipping percentage (default: 0.1)'
    )
    qc_parser.add_argument(
        '--duration-min',
        type=float,
        default=500.0,
        help='Min duration in ms (default: 500)'
    )
    qc_parser.add_argument(
        '--snr-min',
        type=float,
        default=10.0,
        help='Min SNR in dB (default: 10)'
    )
    qc_parser.add_argument(
        '--silence-max',
        type=float,
        default=60.0,
        help='Max silence percentage (default: 60)'
    )

    qc_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # Show thresholds command
    subparsers.add_parser(
        'show-thresholds',
        help='Display default thresholds'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'show-thresholds':
        show_thresholds()
        sys.exit(0)

    if args.command == 'qc':
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        index_dir = args.index_dir.resolve()
        data_norm_dir = args.data_norm_dir.resolve()
        output_dir = args.output_dir.resolve()

        logger.info(f"Index directory: {index_dir}")
        logger.info(f"Data norm directory: {data_norm_dir}")
        logger.info(f"Output directory: {output_dir}")

        # Create thresholds from arguments
        thresholds = QCThresholds(
            clipping_max_pct=args.clipping_max,
            duration_min_ms=args.duration_min,
            snr_min_db=args.snr_min,
            silence_max_pct=args.silence_max,
        )

        all_results = []

        # Determine which datasets to process
        if args.dataset == 'all':
            datasets = ['albayzin', 'preseea', 'dimex100']
        else:
            datasets = [args.dataset]

        for dataset in datasets:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"  Processing {dataset.upper()}")
            logger.info(f"{'=' * 60}\n")

            result = qc_dataset(dataset, index_dir, data_norm_dir, output_dir, thresholds)
            if result:
                all_results.append(result)

                # Print summary
                stats = result.statistics
                logger.info(f"\n{dataset.upper()} Summary:")
                logger.info(f"  Total utterances: {stats.total_utterances:,}")
                logger.info(f"  Passed QC: {stats.passed_utterances:,} ({stats.passed_utterances / stats.total_utterances * 100 if stats.total_utterances > 0 else 0:.1f}%)")
                logger.info(f"  Failed QC: {stats.failed_utterances:,}")
                logger.info(f"  - Clipping: {stats.failed_by_clipping:,}")
                logger.info(f"  - Duration: {stats.failed_by_duration:,}")
                logger.info(f"  - SNR: {stats.failed_by_snr:,}")
                logger.info(f"  - Silence: {stats.failed_by_silence:,}")

        if all_results:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"  QC COMPLETE")
            logger.info(f"{'=' * 60}")

            total_utt = sum(r.statistics.total_utterances for r in all_results)
            total_passed = sum(r.statistics.passed_utterances for r in all_results)
            total_hours = sum(r.statistics.total_duration_hours for r in all_results)
            passed_hours = sum(r.statistics.passed_duration_hours for r in all_results)

            logger.info(f"  Total utterances: {total_utt:,}")
            logger.info(f"  Total passed: {total_passed:,} ({total_passed / total_utt * 100 if total_utt > 0 else 0:.1f}%)")
            logger.info(f"  Total audio: {total_hours:.2f} hours")
            logger.info(f"  Passed audio: {passed_hours:.2f} hours")

        logger.info("\nAudio QC complete!")


if __name__ == '__main__':
    main()
