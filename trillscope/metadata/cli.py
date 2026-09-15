"""Command-line interface for metadata normalization."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .albayzin import AlbayzinMetadataNormalizer
from .preseea import PreseeaMetadataNormalizer
from .base import MetadataResult
from .utils.report_generator import MappingReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def normalize_dataset(dataset: str, metadata_dir: Path,
                     output_dir: Path) -> Optional[MetadataResult]:
    """Normalize a single dataset's metadata."""
    parquet_path = metadata_dir / f'{dataset}_raw.parquet'

    if not parquet_path.exists():
        logger.warning(f"{dataset} parquet not found at {parquet_path}")
        logger.warning(f"Run ingestion first: python -m trillscope.ingestion.cli ingest {dataset}")
        return None

    if dataset == 'albayzin':
        normalizer = AlbayzinMetadataNormalizer(parquet_path, output_dir)
    elif dataset == 'preseea':
        normalizer = PreseeaMetadataNormalizer(parquet_path, output_dir)
    else:
        logger.error(f"Unknown dataset: {dataset}")
        return None

    result = normalizer.normalize()

    # Generate mapping report
    report_gen = MappingReportGenerator(output_dir / 'reports')
    report_path = report_gen.generate(result)
    logger.info(f"Generated mapping report: {report_path}")

    return result


def merge_results(results: List[MetadataResult], output_path: Path) -> None:
    """Merge multiple MetadataResult into a single unified parquet."""
    all_records = []
    for result in results:
        all_records.extend(result.records)

    combined = MetadataResult(
        dataset_name='unified',
        records=all_records,
        mapping_stats={},
        issues=[],
    )
    combined.to_parquet(output_path)


def preview_mappings(dataset: str) -> None:
    """Preview mapping rules without processing data."""
    from .utils.education_mapper import EducationMapper

    mapper = EducationMapper()
    print(f"\n{'='*60}")
    print(f"  {dataset.upper()} EDUCATION MAPPING RULES")
    print(f"{'='*60}\n")

    if dataset == 'albayzin':
        print("HIGH (University degree required):\n")
        for pattern, desc in mapper.HIGH_EDUCATION_PATTERNS[:10]:
            print(f"  {pattern:30s} -> {desc}")
        print(f"  ... and {len(mapper.HIGH_EDUCATION_PATTERNS) - 10} more patterns\n")

        print("MID (Technical/vocational training):\n")
        for pattern, desc in mapper.MID_EDUCATION_PATTERNS[:10]:
            print(f"  {pattern:30s} -> {desc}")
        print(f"  ... and {len(mapper.MID_EDUCATION_PATTERNS) - 10} more patterns\n")

        print("LOW (No higher education indicated):\n")
        for pattern, desc in mapper.LOW_EDUCATION_PATTERNS[:10]:
            print(f"  {pattern:30s} -> {desc}")
        print()
    else:
        print("PRESEEA direct mappings:\n")
        for key, val in sorted(mapper.PRESEEA_MAPPINGS.items()):
            print(f"  '{key}' -> {val}")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Metadata normalization for Spanish speech corpora',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normalize ALBAYZIN metadata
  python -m trillscope.metadata.cli normalize albayzin

  # Normalize PRESEEA metadata
  python -m trillscope.metadata.cli normalize preseea

  # Normalize all and merge into unified parquet
  python -m trillscope.metadata.cli normalize all

  # Preview education mapping rules
  python -m trillscope.metadata.cli preview albayzin

Output files:
  metadata/metadata_unified.parquet  - Merged metadata from all datasets
  reports/metadata_mapping_*.md      - Mapping documentation per dataset
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Normalize command
    norm_parser = subparsers.add_parser('normalize', help='Normalize metadata')
    norm_parser.add_argument(
        'dataset',
        choices=['albayzin', 'preseea', 'dimex100', 'all'],
        help='Dataset to normalize'
    )
    norm_parser.add_argument(
        '--metadata-dir',
        type=Path,
        default=Path('metadata'),
        help='Directory containing the raw ingestion parquet files (default: metadata/)'
    )
    norm_parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Output directory (default: current directory)'
    )
    norm_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # Preview command
    preview_parser = subparsers.add_parser('preview', help='Preview mapping rules')
    preview_parser.add_argument(
        'dataset',
        choices=['albayzin', 'preseea'],
        help='Dataset to preview'
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

        all_results = []

        # Determine which datasets to process
        if args.dataset == 'all':
            datasets = ['albayzin', 'preseea']
        elif args.dataset == 'dimex100':
            logger.warning("DIMEx100 not yet ingested. Run ingestion first.")
            datasets = []
        else:
            datasets = [args.dataset]

        for dataset in datasets:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Processing {dataset.upper()}")
            logger.info(f"{'='*60}\n")

            result = normalize_dataset(dataset, metadata_dir, output_dir)
            if result:
                all_results.append(result)

                # Print summary
                stats = result.mapping_stats
                logger.info(f"\n{dataset.upper()} Summary:")
                logger.info(f"  Total records: {len(result.records):,}")
                for field, stat in stats.items():
                    coverage = stat.mapped_successfully / stat.total_records * 100 if stat.total_records else 0
                    logger.info(f"  {field}: {coverage:.1f}% mapped ({stat.mapped_to_unknown:,} unknown)")

        # Merge into unified parquet
        if all_results:
            unified_path = output_dir / 'metadata' / 'metadata_unified.parquet'
            merge_results(all_results, unified_path)

            total = sum(len(r.records) for r in all_results)
            logger.info(f"\n{'='*60}")
            logger.info("  UNIFIED METADATA")
            logger.info(f"{'='*60}")
            logger.info(f"  Total records: {total:,}")
            logger.info(f"  Output: {unified_path}")

        logger.info("\nMetadata normalization complete!")

    elif args.command == 'preview':
        preview_mappings(args.dataset)


if __name__ == '__main__':
    main()
