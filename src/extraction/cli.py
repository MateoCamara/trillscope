"""Command-line interface for trill /r/ extraction."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .base import ExtractionResult
from .dimex100_extractor import DIMEx100Extractor
from .albayzin_extractor import AlbayzinExtractor
from .preseea_extractor import PreseeaExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def save_result(result: ExtractionResult, output_path: Path) -> None:
    """Save extraction result to Parquet file."""
    if not result.candidates:
        logger.warning(f"No candidates to save for {result.dataset}")
        return

    # Convert to DataFrame
    rows = [c.to_dict() for c in result.candidates]
    df = pd.DataFrame(rows)

    # Save to Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info(f"Saved {len(df)} candidates to {output_path}")


def extract_dimex100(dataset_root: Path, output_dir: Path) -> None:
    """Extract trill /r/ candidates from DIMEx100."""
    logger.info(f"Extracting from DIMEx100 at {dataset_root}")

    extractor = DIMEx100Extractor(dataset_root)
    result = extractor.extract_all()

    # Log summary
    logger.info(f"DIMEx100 extraction complete:")
    logger.info(f"  - Candidates: {result.statistics.get('total_candidates', 0):,}")
    logger.info(f"  - Speakers: {result.statistics.get('unique_speakers', 0):,}")
    logger.info(f"  - Utterances: {result.statistics.get('unique_utterances', 0):,}")
    logger.info(f"  - Issues: {result.statistics.get('total_issues', 0)}")

    if result.statistics.get('by_context'):
        logger.info("  - By context:")
        for ctx, count in sorted(result.statistics['by_context'].items()):
            logger.info(f"      {ctx}: {count:,}")

    # Save
    output_path = output_dir / 'outputs' / 'tables' / 'r_candidates_dimex100.parquet'
    save_result(result, output_path)


def extract_albayzin(dataset_root: Path, output_dir: Path) -> None:
    """Extract trill /r/ candidates from ALBAYZIN."""
    logger.info(f"Extracting from ALBAYZIN at {dataset_root}")

    extractor = AlbayzinExtractor(dataset_root)
    result = extractor.extract_all()

    # Log summary
    logger.info(f"ALBAYZIN extraction complete:")
    logger.info(f"  - Candidates: {result.statistics.get('total_candidates', 0):,}")
    logger.info(f"  - Speakers: {result.statistics.get('unique_speakers', 0):,}")
    logger.info(f"  - Utterances: {result.statistics.get('unique_utterances', 0):,}")
    logger.info(f"  - Issues: {result.statistics.get('total_issues', 0)}")

    if result.statistics.get('by_context'):
        logger.info("  - By context:")
        for ctx, count in sorted(result.statistics['by_context'].items()):
            logger.info(f"      {ctx}: {count:,}")

    # Save
    output_path = output_dir / 'outputs' / 'tables' / 'r_candidates_albayzin.parquet'
    save_result(result, output_path)


def extract_preseea(dataset_root: Path, output_dir: Path,
                   use_mfa: bool = True, run_mfa: bool = True,
                   mfa_work_dir: Path = None) -> None:
    """Extract trill /r/ candidates from PRESEEA."""
    logger.info(f"Extracting from PRESEEA at {dataset_root}")

    # Create work directory for MFA
    # Default to mfa_preseea for existing outputs, mfa_work/preseea for new runs
    if mfa_work_dir:
        work_dir = mfa_work_dir
    elif use_mfa and not run_mfa:
        # Using existing outputs - look in mfa_preseea
        work_dir = output_dir / 'mfa_preseea'
    elif use_mfa:
        work_dir = output_dir / 'mfa_work' / 'preseea'
    else:
        work_dir = None

    extractor = PreseeaExtractor(dataset_root, work_dir=work_dir, use_mfa=use_mfa)

    if use_mfa:
        result = extractor.extract_all(run_mfa=run_mfa)
    else:
        result = extractor.extract_without_mfa()

    # Log summary
    logger.info(f"PRESEEA extraction complete:")
    logger.info(f"  - Candidates: {result.statistics.get('total_candidates', 0):,}")
    logger.info(f"  - Speakers: {result.statistics.get('unique_speakers', 0):,}")
    logger.info(f"  - Utterances: {result.statistics.get('unique_utterances', 0):,}")
    logger.info(f"  - Issues: {result.statistics.get('total_issues', 0)}")

    if result.statistics.get('by_context'):
        logger.info("  - By context:")
        for ctx, count in sorted(result.statistics['by_context'].items()):
            logger.info(f"      {ctx}: {count:,}")

    # Save
    output_path = output_dir / 'outputs' / 'tables' / 'r_candidates_preseea.parquet'
    save_result(result, output_path)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Extract trill /r/ candidates from Spanish speech corpora',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from DIMEx100 (uses existing .phn alignment)
  python -m src.extraction.cli extract dimex100

  # Extract from ALBAYZIN (uses existing SEO alignment)
  python -m src.extraction.cli extract albayzin

  # Extract from PRESEEA with MFA alignment
  python -m src.extraction.cli extract preseea --use-mfa

  # Extract from PRESEEA without MFA (orthographic only)
  python -m src.extraction.cli extract preseea --no-mfa

  # Extract from all corpora
  python -m src.extraction.cli extract all
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract trill /r/ candidates')
    extract_parser.add_argument(
        'dataset',
        choices=['dimex100', 'albayzin', 'preseea', 'all'],
        help='Dataset to extract from'
    )
    extract_parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('dataset'),
        help='Root directory containing datasets (default: dataset/)'
    )
    extract_parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Output directory (default: current directory)'
    )
    extract_parser.add_argument(
        '--use-mfa',
        action='store_true',
        default=True,
        help='Use MFA for PRESEEA alignment (default: True)'
    )
    extract_parser.add_argument(
        '--no-mfa',
        action='store_true',
        help='Skip MFA for PRESEEA (orthographic only)'
    )
    extract_parser.add_argument(
        '--skip-mfa-run',
        action='store_true',
        help='Use existing MFA outputs without re-running'
    )
    extract_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # Setup MFA command
    mfa_parser = subparsers.add_parser('setup-mfa', help='Download MFA Spanish models')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == 'setup-mfa':
        from .mfa_utils import MFARunner
        runner = MFARunner()
        if runner.check_mfa_installed():
            logger.info("MFA is installed, downloading Spanish models...")
            if runner.download_spanish_model():
                logger.info("Spanish models downloaded successfully")
            else:
                logger.error("Failed to download Spanish models")
                sys.exit(1)
        else:
            logger.error("MFA is not installed. Install with: pip install montreal-forced-aligner")
            sys.exit(1)

    elif args.command == 'extract':
        dataset_dir = args.dataset_dir.resolve()
        output_dir = args.output_dir.resolve()

        logger.info(f"Dataset directory: {dataset_dir}")
        logger.info(f"Output directory: {output_dir}")

        use_mfa = args.use_mfa and not args.no_mfa
        run_mfa = not args.skip_mfa_run

        if args.dataset == 'dimex100' or args.dataset == 'all':
            dimex_root = dataset_dir / 'CorpusDimex100'
            if dimex_root.exists():
                extract_dimex100(dimex_root, output_dir)
            else:
                logger.warning(f"DIMEx100 not found at {dimex_root}")

        if args.dataset == 'albayzin' or args.dataset == 'all':
            albayzin_root = dataset_dir / 'ALBAYZIN' / 'ALBAYZIN' / 'corpora'
            if albayzin_root.exists():
                extract_albayzin(albayzin_root, output_dir)
            else:
                logger.warning(f"ALBAYZIN not found at {albayzin_root}")

        if args.dataset == 'preseea' or args.dataset == 'all':
            preseea_root = dataset_dir / 'preseea'
            if preseea_root.exists():
                extract_preseea(preseea_root, output_dir,
                              use_mfa=use_mfa, run_mfa=run_mfa)
            else:
                logger.warning(f"PRESEEA not found at {preseea_root}")

        logger.info("Extraction complete!")


if __name__ == '__main__':
    main()
