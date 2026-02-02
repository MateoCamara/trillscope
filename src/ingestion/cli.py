"""Command-line interface for dataset ingestion."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .albayzin import AlbayzinIngestor
from .preseea import PreseeaIngestor
from .dimex100 import DIMEx100Ingestor
from .glissando import GlissandoIngestor
from .mailabs import MAILABSIngestor
from .tedx import TEDxIngestor
from .heroico import HeroicoIngestor
from .commonvoice import CommonVoiceIngestor

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


def ingest_glissando(dataset_root: Path, output_dir: Path) -> None:
    """Ingest Glissando-sp corpus."""
    logger.info(f"Ingesting Glissando-sp from {dataset_root}")

    ingestor = GlissandoIngestor(dataset_root, output_dir)
    result = ingestor.ingest()

    logger.info(f"Glissando-sp ingestion complete:")
    logger.info(f"  - Utterances: {result.statistics['total_utterances']:,}")
    logger.info(f"  - Speakers: {result.statistics['total_speakers']:,}")
    logger.info(f"  - Duration: {result.statistics['total_duration_hours']:.1f} hours")
    logger.info(f"  - With alignment: {result.statistics.get('utterances_with_alignment', 0):,}")
    logger.info(f"  - Issues: {len(result.issues)}")


def ingest_mailabs(dataset_root: Path, output_dir: Path) -> None:
    """Ingest M-AILABS Spanish corpus (all regional variants)."""
    logger.info(f"Ingesting M-AILABS from {dataset_root}")

    ingestor = MAILABSIngestor(dataset_root, output_dir)
    result = ingestor.ingest()

    logger.info(f"M-AILABS ingestion complete:")
    logger.info(f"  - Utterances: {result.statistics['total_utterances']:,}")
    logger.info(f"  - Speakers: {result.statistics['total_speakers']:,}")
    logger.info(f"  - Duration: {result.statistics['total_duration_hours']:.1f} hours")
    logger.info(f"  - Countries: {', '.join(result.statistics.get('countries', []))}")
    logger.info(f"  - Issues: {len(result.issues)}")

    if result.statistics.get('by_country'):
        logger.info("  - By country:")
        for country, count in sorted(result.statistics['by_country'].items()):
            logger.info(f"      {country}: {count:,}")


def ingest_tedx(dataset_root: Path, output_dir: Path) -> None:
    """Ingest TEDx Spanish corpus."""
    logger.info(f"Ingesting TEDx Spanish from {dataset_root}")

    ingestor = TEDxIngestor(dataset_root, output_dir)
    result = ingestor.ingest()

    logger.info(f"TEDx Spanish ingestion complete:")
    logger.info(f"  - Utterances: {result.statistics['total_utterances']:,}")
    logger.info(f"  - Speakers: {result.statistics['total_speakers']:,}")
    logger.info(f"  - Duration: {result.statistics['total_duration_hours']:.1f} hours")
    logger.info(f"  - With transcripts: {result.statistics.get('files_with_transcripts', 0):,}")
    logger.info(f"  - Issues: {len(result.issues)}")

    if result.statistics.get('by_gender'):
        logger.info("  - By gender:")
        for gender, count in sorted(result.statistics['by_gender'].items()):
            logger.info(f"      {gender}: {count}")


def ingest_heroico(dataset_root: Path, output_dir: Path) -> None:
    """Ingest Heroico (LDC2006S37) corpus."""
    logger.info(f"Ingesting Heroico from {dataset_root}")

    ingestor = HeroicoIngestor(dataset_root, output_dir)
    result = ingestor.ingest()

    logger.info(f"Heroico ingestion complete:")
    logger.info(f"  - Utterances: {result.statistics['total_utterances']:,}")
    logger.info(f"  - Speakers: {result.statistics['total_speakers']:,}")
    logger.info(f"  - Duration: {result.statistics['total_duration_hours']:.1f} hours")
    logger.info(f"  - With transcripts: {result.statistics.get('files_with_transcripts', 0):,}")
    logger.info(f"  - Issues: {len(result.issues)}")

    if result.statistics.get('by_subcorpus'):
        logger.info("  - By subcorpus:")
        for subcorpus, count in sorted(result.statistics['by_subcorpus'].items()):
            logger.info(f"      {subcorpus}: {count:,}")

    if result.statistics.get('by_native'):
        logger.info("  - By native status:")
        for status, count in sorted(result.statistics['by_native'].items()):
            logger.info(f"      {status}: {count:,}")


def ingest_commonvoice(dataset_root: Path, output_dir: Path,
                       sample_size: int = 15000) -> None:
    """Ingest Common Voice Spanish corpus with stratified sampling."""
    logger.info(f"Ingesting Common Voice from {dataset_root}")

    ingestor = CommonVoiceIngestor(dataset_root, output_dir, sample_size=sample_size)
    result = ingestor.ingest()

    logger.info(f"Common Voice ingestion complete:")
    logger.info(f"  - Utterances: {result.statistics['total_utterances']:,}")
    logger.info(f"  - Speakers: {result.statistics['total_speakers']:,}")
    logger.info(f"  - Duration: {result.statistics['total_duration_hours']:.1f} hours")
    logger.info(f"  - Total validated: {result.statistics.get('total_validated', 0):,}")
    logger.info(f"  - Issues: {len(result.issues)}")

    if result.statistics.get('by_region'):
        logger.info("  - By region:")
        for region, count in sorted(result.statistics['by_region'].items(), key=lambda x: -x[1]):
            logger.info(f"      {region}: {count:,}")


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
        choices=['albayzin', 'preseea', 'dimex100', 'glissando', 'mailabs', 'tedx', 'heroico', 'commonvoice', 'all'],
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

        if args.dataset == 'glissando' or args.dataset == 'all':
            glissando_root = dataset_dir / '03 glissando-sp'
            if glissando_root.exists():
                ingest_glissando(glissando_root, output_dir)
            else:
                logger.warning(f"Glissando-sp not found at {glissando_root}")

        if args.dataset == 'mailabs' or args.dataset == 'all':
            # M-AILABS directories are at dataset root level (es_ar_female, etc.)
            # Check if any variant exists
            mailabs_variants = ['es_ar_female', 'es_ar_male', 'es_cl_female', 'es_cl_male',
                               'es_co_female', 'es_co_male', 'es_pe_male',
                               'es_ve_female', 'es_ve_male']
            has_mailabs = any((dataset_dir / v).exists() for v in mailabs_variants)
            if has_mailabs:
                ingest_mailabs(dataset_dir, output_dir)
            else:
                logger.warning(f"M-AILABS not found at {dataset_dir}")

        if args.dataset == 'tedx' or args.dataset == 'all':
            tedx_root = dataset_dir / 'tedx_spanish_corpus'
            if tedx_root.exists():
                ingest_tedx(tedx_root, output_dir)
            else:
                logger.warning(f"TEDx Spanish not found at {tedx_root}")

        if args.dataset == 'heroico' or args.dataset == 'all':
            heroico_root = dataset_dir / 'LDC2006S37'
            if heroico_root.exists():
                ingest_heroico(heroico_root, output_dir)
            else:
                logger.warning(f"Heroico (LDC2006S37) not found at {heroico_root}")

        if args.dataset == 'commonvoice' or args.dataset == 'all':
            # Common Voice directory pattern: cv-corpus-*/es
            cv_dirs = list(dataset_dir.glob('cv-corpus-*/es'))
            if cv_dirs:
                cv_root = cv_dirs[0]  # Use first match
                ingest_commonvoice(cv_root, output_dir)
            else:
                logger.warning(f"Common Voice not found at {dataset_dir}/cv-corpus-*/es")

        logger.info("Ingestion complete!")


if __name__ == '__main__':
    main()
