"""Command-line interface for dataset ingestion."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .albayzin import AlbayzinIngestor
from .preseea import PreseeaIngestor
from .dimex100 import DIMEx100Ingestor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def ingest_albayzin(dataset_root: Path, output_dir: Path) -> None:
    """Ingest ALBAYZIN corpus."""
    logger.info(f"Ingesting ALBAYZIN from {dataset_root}")

    ingestor = AlbayzinIngestor(dataset_root, output_dir)
    result = ingestor.ingest()

    logger.info(f"ALBAYZIN ingestion complete:")
    logger.info(f"  - Utterances: {result.statistics['total_utterances']:,}")
    logger.info(f"  - Speakers: {result.statistics['total_speakers']:,}")
    logger.info(f"  - Duration: {result.statistics['total_duration_hours']:.1f} hours")
    logger.info(f"  - Issues: {len(result.issues)}")


def ingest_preseea(dataset_root: Path, output_dir: Path) -> None:
    """Ingest PRESEEA corpus."""
    logger.info(f"Ingesting PRESEEA from {dataset_root}")

    ingestor = PreseeaIngestor(dataset_root, output_dir)
    result = ingestor.ingest()

    logger.info(f"PRESEEA ingestion complete:")
    logger.info(f"  - Utterances: {result.statistics['total_utterances']:,}")
    logger.info(f"  - Speakers: {result.statistics['total_speakers']:,}")
    logger.info(f"  - Duration: {result.statistics['total_duration_hours']:.1f} hours")
    logger.info(f"  - Issues: {len(result.issues)}")


def ingest_dimex100(dataset_root: Path, output_dir: Path,
                   metadata_csv: Optional[Path] = None) -> None:
    """Ingest DIMEx100 corpus."""
    logger.info(f"Ingesting DIMEx100 from {dataset_root}")

    ingestor = DIMEx100Ingestor(dataset_root, output_dir, metadata_csv)
    result = ingestor.ingest()

    logger.info(f"DIMEx100 ingestion complete:")
    logger.info(f"  - Utterances: {result.statistics['total_utterances']:,}")
    logger.info(f"  - Speakers: {result.statistics['total_speakers']:,}")
    logger.info(f"  - Duration: {result.statistics['total_duration_hours']:.1f} hours")
    logger.info(f"  - Trill segments: {result.statistics.get('total_trill_segments', 0):,}")
    logger.info(f"  - Issues: {len(result.issues)}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Ingest Spanish speech corpora for trill /r/ analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest ALBAYZIN corpus
  python -m src.ingestion.cli ingest albayzin

  # Ingest PRESEEA corpus
  python -m src.ingestion.cli ingest preseea

  # Ingest DIMEx100 with speaker metadata CSV
  python -m src.ingestion.cli ingest dimex100 --metadata-csv dimex_speakers.csv

  # Ingest all corpora
  python -m src.ingestion.cli ingest all
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest a dataset')
    ingest_parser.add_argument(
        'dataset',
        choices=['albayzin', 'preseea', 'dimex100', 'all'],
        help='Dataset to ingest'
    )
    ingest_parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('dataset'),
        help='Root directory containing datasets (default: dataset/)'
    )
    ingest_parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Output directory for metadata and reports (default: current directory)'
    )
    ingest_parser.add_argument(
        '--metadata-csv',
        type=Path,
        help='CSV file with speaker metadata (for DIMEx100)'
    )
    ingest_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == 'ingest':
        dataset_dir = args.dataset_dir.resolve()
        output_dir = args.output_dir.resolve()

        logger.info(f"Dataset directory: {dataset_dir}")
        logger.info(f"Output directory: {output_dir}")

        if args.dataset == 'albayzin' or args.dataset == 'all':
            albayzin_root = dataset_dir / 'ALBAYZIN'
            if albayzin_root.exists():
                ingest_albayzin(albayzin_root, output_dir)
            else:
                logger.warning(f"ALBAYZIN not found at {albayzin_root}")

        if args.dataset == 'preseea' or args.dataset == 'all':
            preseea_root = dataset_dir / 'preseea'
            if preseea_root.exists():
                ingest_preseea(preseea_root, output_dir)
            else:
                logger.warning(f"PRESEEA not found at {preseea_root}")

        if args.dataset == 'dimex100' or args.dataset == 'all':
            dimex_root = dataset_dir / 'CorpusDimex100'
            if dimex_root.exists():
                ingest_dimex100(dimex_root, output_dir, args.metadata_csv)
            else:
                logger.warning(f"DIMEx100 not found at {dimex_root}")

        logger.info("Ingestion complete!")


if __name__ == '__main__':
    main()
