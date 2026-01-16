"""Command-line interface for trill /r/ visualization."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .base import VisualizationConfig
from .plotter import plot_trill
from .selector import select_examples, validate_audio_paths
from .html_viewer import create_viewer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def plot_dataset(
    parquet_path: Path,
    output_dir: Path,
    n_examples: int = 5,
    config: VisualizationConfig = None
) -> int:
    """
    Plot examples from a single dataset's extraction results.

    Returns number of successful plots.
    """
    if config is None:
        config = VisualizationConfig()

    if not parquet_path.exists():
        logger.error(f"Parquet file not found: {parquet_path}")
        logger.info("Run extraction first: python -m src.extraction.cli extract <dataset>")
        return 0

    # Load extraction results
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} candidates from {parquet_path.name}")

    # Get dataset name from filename
    dataset = parquet_path.stem.replace('r_candidates_', '')

    # Validate audio paths
    df = validate_audio_paths(df)
    if len(df) == 0:
        logger.error(f"No valid audio paths found for {dataset}")
        return 0

    # Select diverse examples
    selected = select_examples(df, n_examples=n_examples)
    if len(selected) == 0:
        logger.error(f"Could not select examples for {dataset}")
        return 0

    logger.info(f"Selected {len(selected)} examples for {dataset}")

    # Plot each example
    success_count = 0
    for idx, row in selected.iterrows():
        output_path = output_dir / f"trill_{dataset}_{success_count + 1:03d}.png"

        result = plot_trill(
            audio_path=row['audio_path'],
            start_ms=row['start_ms'],
            end_ms=row['end_ms'],
            output_path=str(output_path),
            word=row.get('word', 'unknown'),
            context_label=row.get('context_label', 'unknown'),
            dataset=dataset,
            speaker_id=row.get('speaker_id', 'unknown'),
            config=config
        )

        if result.success:
            success_count += 1

    return success_count


def generate_viewer(
    parquet_dir: Path,
    output_path: Path,
    datasets: list,
    n_examples: int = 10,
    config: VisualizationConfig = None
) -> bool:
    """Generate HTML viewer with audio playback."""
    if config is None:
        config = VisualizationConfig()

    all_dfs = []
    for dataset in datasets:
        parquet_path = parquet_dir / f'r_candidates_{dataset}.parquet'
        if not parquet_path.exists():
            logger.warning(f"Skipping {dataset}: {parquet_path} not found")
            continue

        df = pd.read_parquet(parquet_path)
        df = validate_audio_paths(df)
        if len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        logger.error("No valid data found")
        return False

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} candidates with valid audio")

    return create_viewer(combined, output_path, n_examples=n_examples, config=config)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate trill /r/ visualization plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot 5 examples from ALBAYZIN
  python -m src.visualization.cli plot albayzin

  # Plot 3 examples from DIMEx100
  python -m src.visualization.cli plot dimex100 --n-examples 3

  # Plot from all datasets
  python -m src.visualization.cli plot all --n-examples 5

  # Generate interactive HTML viewer with audio
  python -m src.visualization.cli viewer --n-examples 10
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Generate visualization plots')
    plot_parser.add_argument(
        'dataset',
        choices=['albayzin', 'dimex100', 'preseea', 'all'],
        help='Dataset to visualize'
    )
    plot_parser.add_argument(
        '--n-examples', '-n',
        type=int,
        default=5,
        help='Number of examples per dataset (default: 5)'
    )
    plot_parser.add_argument(
        '--parquet-dir',
        type=Path,
        default=Path('outputs/tables'),
        help='Directory containing r_candidates parquet files'
    )
    plot_parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/figures'),
        help='Output directory for figures'
    )
    plot_parser.add_argument(
        '--context-ms',
        type=float,
        default=100.0,
        help='Context window around /r/ region in milliseconds'
    )
    plot_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # Viewer command
    viewer_parser = subparsers.add_parser('viewer', help='Generate HTML viewer with audio')
    viewer_parser.add_argument(
        '--dataset',
        choices=['albayzin', 'dimex100', 'preseea', 'all'],
        default='all',
        help='Dataset(s) to include (default: all)'
    )
    viewer_parser.add_argument(
        '--n-examples', '-n',
        type=int,
        default=10,
        help='Number of examples to include (default: 10)'
    )
    viewer_parser.add_argument(
        '--parquet-dir',
        type=Path,
        default=Path('outputs/tables'),
        help='Directory containing r_candidates parquet files'
    )
    viewer_parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/trill_viewer.html'),
        help='Output HTML file path'
    )
    viewer_parser.add_argument(
        '--context-ms',
        type=float,
        default=100.0,
        help='Context window around /r/ region in milliseconds'
    )
    viewer_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == 'plot':
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure visualization
        config = VisualizationConfig(context_ms=args.context_ms)

        datasets = []
        if args.dataset == 'all':
            datasets = ['albayzin', 'dimex100', 'preseea']
        else:
            datasets = [args.dataset]

        total_success = 0
        for dataset in datasets:
            parquet_path = args.parquet_dir / f'r_candidates_{dataset}.parquet'

            if not parquet_path.exists():
                logger.warning(f"Skipping {dataset}: {parquet_path} not found")
                continue

            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {dataset.upper()}")
            logger.info(f"{'='*50}")

            success = plot_dataset(
                parquet_path=parquet_path,
                output_dir=args.output_dir,
                n_examples=args.n_examples,
                config=config
            )

            total_success += success
            logger.info(f"{dataset}: {success} plots generated")

        logger.info(f"\nTotal: {total_success} plots saved to {args.output_dir}")

    elif args.command == 'viewer':
        # Configure visualization
        config = VisualizationConfig(context_ms=args.context_ms)

        datasets = []
        if args.dataset == 'all':
            datasets = ['albayzin', 'dimex100', 'preseea']
        else:
            datasets = [args.dataset]

        success = generate_viewer(
            parquet_dir=args.parquet_dir,
            output_path=args.output,
            datasets=datasets,
            n_examples=args.n_examples,
            config=config
        )

        if success:
            logger.info(f"\nViewer saved to: {args.output}")
            logger.info("Open this file in a web browser to view and listen to examples.")
        else:
            logger.error("Failed to generate viewer")
            sys.exit(1)


if __name__ == '__main__':
    main()
