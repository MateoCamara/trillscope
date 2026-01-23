"""Data loader for statistical analysis - merges measurements with metadata."""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_acoustic_measurements(
    measurements_dir: Path,
    datasets: Optional[list] = None
) -> pd.DataFrame:
    """
    Load acoustic measurements from parquet files.

    Args:
        measurements_dir: Directory containing acoustic_measurements_*.parquet
        datasets: List of datasets to load (default: all available)

    Returns:
        Combined DataFrame of all measurements
    """
    if datasets is None:
        datasets = ['albayzin', 'dimex100', 'preseea']

    dfs = []
    for dataset in datasets:
        path = measurements_dir / f'acoustic_measurements_{dataset}.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            logger.info(f"Loaded {len(df)} measurements from {dataset}")
            dfs.append(df)
        else:
            logger.warning(f"Measurements not found: {path}")

    if not dfs:
        raise FileNotFoundError("No acoustic measurement files found")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total measurements: {len(combined)}")

    return combined


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Load unified metadata.

    Args:
        metadata_path: Path to metadata_unified.parquet

    Returns:
        Metadata DataFrame
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    df = pd.read_parquet(metadata_path)
    logger.info(f"Loaded {len(df)} metadata records")

    return df


def merge_with_metadata(
    measurements: pd.DataFrame,
    metadata: pd.DataFrame,
    aggregate_by_speaker: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Merge acoustic measurements with speaker metadata.

    Args:
        measurements: Acoustic measurements DataFrame
        metadata: Unified metadata DataFrame
        aggregate_by_speaker: If True, aggregate measurements per speaker

    Returns:
        Tuple of (merged DataFrame, merge statistics dict)
    """
    stats = {
        'total_measurements': len(measurements),
        'datasets': {},
    }

    merged_dfs = []

    # Process each dataset separately due to different join strategies
    for dataset in measurements['dataset'].unique():
        dataset_measurements = measurements[measurements['dataset'] == dataset].copy()
        dataset_metadata = metadata[metadata['dataset'] == dataset].copy()

        stats['datasets'][dataset] = {
            'measurements': len(dataset_measurements),
            'metadata_records': len(dataset_metadata),
        }

        if len(dataset_metadata) == 0:
            logger.warning(f"No metadata for {dataset}, skipping merge")
            stats['datasets'][dataset]['merged'] = 0
            continue

        if dataset == 'albayzin':
            # ALBAYZIN: Join on speaker_id (speaker-level metadata)
            merged = _merge_albayzin(dataset_measurements, dataset_metadata)

        elif dataset == 'preseea':
            # PRESEEA: Join on utt_id (need to handle segment suffixes)
            merged = _merge_preseea(dataset_measurements, dataset_metadata)

        else:
            # DIMEx100 or other: No metadata available
            merged = dataset_measurements.copy()
            # Add empty metadata columns
            for col in ['sex', 'age', 'age_bin', 'education_bin', 'country', 'region', 'city']:
                merged[col] = 'unknown'

        stats['datasets'][dataset]['merged'] = len(merged)
        merged_dfs.append(merged)

    if not merged_dfs:
        raise ValueError("No data after merging")

    combined = pd.concat(merged_dfs, ignore_index=True)

    # Optionally aggregate by speaker
    if aggregate_by_speaker:
        combined = _aggregate_by_speaker(combined)
        stats['aggregated_speakers'] = len(combined)

    stats['final_records'] = len(combined)
    logger.info(f"Final dataset: {len(combined)} records")

    return combined, stats


def _merge_albayzin(
    measurements: pd.DataFrame,
    metadata: pd.DataFrame
) -> pd.DataFrame:
    """Merge ALBAYZIN measurements with metadata on speaker_id."""
    # Get unique speaker metadata
    speaker_meta = metadata.drop_duplicates(subset='speaker_id')

    # Select relevant metadata columns
    meta_cols = ['speaker_id', 'sex', 'age', 'age_bin', 'education_bin',
                 'country', 'region', 'city', 'speech_style']
    meta_cols = [c for c in meta_cols if c in speaker_meta.columns]

    merged = pd.merge(
        measurements,
        speaker_meta[meta_cols],
        on='speaker_id',
        how='left'
    )

    # Fill missing values
    for col in ['sex', 'age_bin', 'education_bin', 'country']:
        if col in merged.columns:
            merged[col] = merged[col].fillna('unknown')

    return merged


def _merge_preseea(
    measurements: pd.DataFrame,
    metadata: pd.DataFrame
) -> pd.DataFrame:
    """Merge PRESEEA measurements with metadata on utt_id."""
    # Extract base utt_id from segmented IDs
    # PRE_ALCA_H11_037_seg001 -> PRE_ALCA_H11_037
    measurements = measurements.copy()

    def extract_base_utt_id(utt_id: str) -> str:
        # Remove _segXXX suffix if present
        match = re.match(r'(PRE_.+?)(?:_seg\d+)?$', utt_id)
        if match:
            return match.group(1)
        return utt_id

    measurements['base_utt_id'] = measurements['utt_id'].apply(extract_base_utt_id)

    # Prepare metadata utt_ids
    metadata = metadata.copy()
    if 'utt_id' not in metadata.columns:
        # Create utt_id from file identifiers if needed
        metadata['utt_id'] = 'PRE_' + metadata.get('file_id', metadata.index.astype(str))

    # Select relevant metadata columns
    meta_cols = ['utt_id', 'sex', 'age', 'age_bin', 'education_bin',
                 'country', 'region', 'city', 'speech_style']
    meta_cols = [c for c in meta_cols if c in metadata.columns]

    merged = pd.merge(
        measurements,
        metadata[meta_cols],
        left_on='base_utt_id',
        right_on='utt_id',
        how='left',
        suffixes=('', '_meta')
    )

    # Clean up
    if 'utt_id_meta' in merged.columns:
        merged = merged.drop(columns=['utt_id_meta'])
    if 'base_utt_id' in merged.columns:
        merged = merged.drop(columns=['base_utt_id'])

    # Fill missing values
    for col in ['sex', 'age_bin', 'education_bin', 'country']:
        if col in merged.columns:
            merged[col] = merged[col].fillna('unknown')

    return merged


def _aggregate_by_speaker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate measurements by speaker to avoid pseudo-replication.

    Args:
        df: Merged measurements DataFrame

    Returns:
        DataFrame with one row per speaker (mean of acoustic measures)
    """
    # Columns to aggregate (numeric acoustic measures)
    agg_cols = [
        'num_cycles', 'duration_ms', 'voicing_pct',
        'mean_f0_hz', 'cycle_rate_hz', 'mean_hnr_db',
        'mean_intensity_db', 'cycle_regularity'
    ]
    agg_cols = [c for c in agg_cols if c in df.columns]

    # Columns to keep (take first value per speaker)
    keep_cols = ['speaker_id', 'dataset', 'sex', 'age', 'age_bin',
                 'education_bin', 'country', 'region', 'city', 'speech_style']
    keep_cols = [c for c in keep_cols if c in df.columns]

    # Create aggregation dict
    agg_dict = {col: 'mean' for col in agg_cols}
    agg_dict['utt_id'] = 'count'  # Count of tokens per speaker

    # Group by speaker
    grouped = df.groupby(['speaker_id', 'dataset']).agg(agg_dict).reset_index()
    grouped = grouped.rename(columns={'utt_id': 'n_tokens'})

    # Merge back metadata (first value per speaker)
    if keep_cols:
        meta = df[keep_cols].drop_duplicates(subset=['speaker_id', 'dataset'])
        grouped = pd.merge(grouped, meta, on=['speaker_id', 'dataset'], how='left')

    return grouped


def load_analysis_data(
    measurements_dir: Path,
    metadata_path: Path,
    datasets: Optional[list] = None,
    aggregate_by_speaker: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Load and prepare data for statistical analysis.

    Args:
        measurements_dir: Directory containing acoustic measurement files
        metadata_path: Path to unified metadata parquet
        datasets: List of datasets to include
        aggregate_by_speaker: Whether to aggregate by speaker

    Returns:
        Tuple of (analysis DataFrame, statistics dict)
    """
    measurements = load_acoustic_measurements(measurements_dir, datasets)
    metadata = load_metadata(metadata_path)

    df, stats = merge_with_metadata(measurements, metadata, aggregate_by_speaker)

    return df, stats
