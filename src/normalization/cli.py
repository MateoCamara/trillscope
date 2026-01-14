"""Command-line interface for structure normalization (Block B)."""

import argparse
import logging
import sys
from pathlib import Path

from .albayzin import AlbayzinNormalizer
from .preseea import PreseeaNormalizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def normalize_albayzin(parquet_path: Path, output_dir: Path,
                       skip_existing: bool = False,
                       num_workers: int = 1) -> None:
    """Normalize ALBAYZIN corpus."""
    logger.info(f"Normalizing ALBAYZIN")
    logger.info(f"  Input: {parquet_path}")
    logger.info(f"  Output: {output_dir}")

    normalizer = AlbayzinNormalizer(parquet_path, output_dir)
    result = normalizer.normalize(skip_existing=skip_existing, num_workers=num_workers)

    # Save outputs
    index_path = output_dir / 'outputs' / 'tables' / 'utt_index_albayzin.csv'
    result.to_index_csv(index_path)

    report_path = output_dir / 'reports' / 'normalization' / 'albayzin.md'
    result.to_validation_report(report_path)

    logger.info(f"ALBAYZIN normalization complete:")
    logger.info(f"  - Utterances: {result.statistics['total_utterances']:,}")
    logger.info(f"  - Valid: {result.statistics['valid_utterances']:,}")
    logger.info(f"  - Duration: {result.statistics['total_duration_hours']:.1f} hours")
    logger.info(f"  - Index: {index_path}")
    logger.info(f"  - Report: {report_path}")


def normalize_preseea(parquet_path: Path, output_dir: Path,
                      skip_existing: bool = False,
                      num_workers: int = 1,
                      convert_audio: bool = False) -> None:
    """Normalize PRESEEA corpus."""
    logger.info(f"Normalizing PRESEEA")
    logger.info(f"  Input: {parquet_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Convert audio: {convert_audio}")

    # By default, skip audio conversion to save ~12GB disk space
    normalizer = PreseeaNormalizer(parquet_path, output_dir, skip_audio=not convert_audio)
    result = normalizer.normalize(skip_existing=skip_existing, num_workers=num_workers)

    # Save outputs
    index_path = output_dir / 'outputs' / 'tables' / 'utt_index_preseea.csv'
    result.to_index_csv(index_path)

    report_path = output_dir / 'reports' / 'normalization' / 'preseea.md'
    result.to_validation_report(report_path)

    logger.info(f"PRESEEA normalization complete:")
    logger.info(f"  - Utterances: {result.statistics['total_utterances']:,}")
    logger.info(f"  - Valid: {result.statistics['valid_utterances']:,}")
    logger.info(f"  - Duration: {result.statistics['total_duration_hours']:.1f} hours")
    logger.info(f"  - Index: {index_path}")
    logger.info(f"  - Report: {report_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Structure normalization for Spanish speech corpora (Block B)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normalize ALBAYZIN corpus
  python -m src.normalization.cli normalize albayzin

  # Normalize PRESEEA corpus
  python -m src.normalization.cli normalize preseea

  # Normalize all corpora
  python -m src.normalization.cli normalize all

  # Skip existing files (resume interrupted run)
  python -m src.normalization.cli normalize all --skip-existing

Output directories created:
  data_norm/<dataset>/audio_16k/     - Normalized WAV files (16kHz mono)
  data_norm/<dataset>/transcripts/   - Normalized transcripts
  outputs/tables/utt_index_*.csv     - Utterance mapping tables
  reports/normalization/*.md         - Validation reports
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Normalize command
    norm_parser = subparsers.add_parser('normalize', help='Normalize a dataset')
    norm_parser.add_argument(
        'dataset',
        choices=['albayzin', 'preseea', 'all'],
        help='Dataset to normalize'
    )
    norm_parser.add_argument(
        '--metadata-dir',
        type=Path,
        default=Path('metadata'),
        help='Directory containing Block A parquet files (default: metadata/)'
    )
    norm_parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Output directory for normalized data (default: current directory)'
    )
    norm_parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already exist (for resuming interrupted runs)'
    )
    norm_parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of parallel workers for audio conversion (default: 1)'
    )
    norm_parser.add_argument(
        '--convert-audio',
        action='store_true',
        help='Convert PRESEEA MP3 to WAV (default: keep as MP3 to save disk space)'
    )
    norm_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # Dry-run command
    dry_parser = subparsers.add_parser('dry-run', help='Preview normalization without changes')
    dry_parser.add_argument(
        'dataset',
        choices=['albayzin', 'preseea', 'all'],
        help='Dataset to preview'
    )
    dry_parser.add_argument(
        '--metadata-dir',
        type=Path,
        default=Path('metadata'),
        help='Directory containing Block A parquet files'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == 'normalize':
        metadata_dir = args.metadata_dir.resolve()
        output_dir = args.output_dir.resolve()

        logger.info(f"Metadata directory: {metadata_dir}")
        logger.info(f"Output directory: {output_dir}")

        if args.dataset == 'albayzin' or args.dataset == 'all':
            parquet_path = metadata_dir / 'albayzin_raw.parquet'
            if parquet_path.exists():
                normalize_albayzin(
                    parquet_path, output_dir,
                    skip_existing=args.skip_existing,
                    num_workers=args.num_workers
                )
            else:
                logger.warning(f"ALBAYZIN parquet not found at {parquet_path}")
                logger.warning(f"Run Block A ingestion first: python -m src.ingestion.cli ingest albayzin")

        if args.dataset == 'preseea' or args.dataset == 'all':
            parquet_path = metadata_dir / 'preseea_raw.parquet'
            if parquet_path.exists():
                normalize_preseea(
                    parquet_path, output_dir,
                    skip_existing=args.skip_existing,
                    num_workers=args.num_workers,
                    convert_audio=args.convert_audio
                )
            else:
                logger.warning(f"PRESEEA parquet not found at {parquet_path}")
                logger.warning(f"Run Block A ingestion first: python -m src.ingestion.cli ingest preseea")

        logger.info("Normalization complete!")

    elif args.command == 'dry-run':
        import pandas as pd

        metadata_dir = args.metadata_dir.resolve()

        logger.info(f"=== Dry Run: Preview of normalization ===")
        logger.info(f"Metadata directory: {metadata_dir}")

        total_utterances = 0
        total_duration_hrs = 0
        estimated_disk_gb = 0

        if args.dataset == 'albayzin' or args.dataset == 'all':
            parquet_path = metadata_dir / 'albayzin_raw.parquet'
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                duration_hrs = df['duration_ms'].sum() / 3600000
                # WAV at 16kHz mono 16-bit: ~32KB/sec
                disk_gb = (duration_hrs * 3600 * 32000) / (1024**3)

                logger.info(f"\nALBAYZIN:")
                logger.info(f"  Utterances: {len(df):,}")
                logger.info(f"  Duration: {duration_hrs:.2f} hours")
                logger.info(f"  Estimated WAV size: {disk_gb:.2f} GB")

                total_utterances += len(df)
                total_duration_hrs += duration_hrs
                estimated_disk_gb += disk_gb

        if args.dataset == 'preseea' or args.dataset == 'all':
            parquet_path = metadata_dir / 'preseea_raw.parquet'
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                duration_hrs = df['duration_ms'].sum() / 3600000
                disk_gb = (duration_hrs * 3600 * 32000) / (1024**3)

                logger.info(f"\nPRESEEA:")
                logger.info(f"  Utterances: {len(df):,}")
                logger.info(f"  Duration: {duration_hrs:.2f} hours")
                logger.info(f"  Estimated WAV size: {disk_gb:.2f} GB")
                logger.info(f"  Note: MP3 to WAV conversion takes ~1-2x real-time")

                total_utterances += len(df)
                total_duration_hrs += duration_hrs
                estimated_disk_gb += disk_gb

        logger.info(f"\n=== TOTAL ===")
        logger.info(f"  Utterances: {total_utterances:,}")
        logger.info(f"  Duration: {total_duration_hrs:.2f} hours")
        logger.info(f"  Estimated disk space: {estimated_disk_gb:.2f} GB")


if __name__ == '__main__':
    main()
