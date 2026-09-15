"""Data loader for statistical analysis - merges measurements with metadata."""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_acoustic_measurements(
    measurements_dir: Path,
    datasets: Optional[list] = None,
    include_overlap_info: bool = True
) -> pd.DataFrame:
    """
    Load acoustic measurements from parquet files.

    Args:
        measurements_dir: Directory containing acoustic_measurements_*.parquet
        datasets: List of datasets to load (default: all available)
        include_overlap_info: If True, merge has_overlap from candidates (PRESEEA only)

    Returns:
        Combined DataFrame of all measurements
    """
    if datasets is None:
        datasets = ['albayzin', 'dimex100', 'preseea', 'glissando']

    dfs = []
    for dataset in datasets:
        path = measurements_dir / f'acoustic_measurements_{dataset}.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            logger.info(f"Loaded {len(df)} measurements from {dataset}")

            # Add has_overlap from candidates for PRESEEA
            if include_overlap_info and dataset == 'preseea':
                df = _add_overlap_info(df, measurements_dir, dataset)

            dfs.append(df)
        else:
            logger.warning(f"Measurements not found: {path}")

    if not dfs:
        raise FileNotFoundError("No acoustic measurement files found")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total measurements: {len(combined)}")

    return combined


def _add_overlap_info(
    measurements: pd.DataFrame,
    measurements_dir: Path,
    dataset: str
) -> pd.DataFrame:
    """Add has_overlap field from candidates to measurements."""
    candidates_path = measurements_dir / f'r_candidates_{dataset}.parquet'
    if not candidates_path.exists():
        logger.warning(f"Candidates not found for overlap info: {candidates_path}")
        measurements['has_overlap'] = False
        return measurements

    candidates = pd.read_parquet(candidates_path)

    if 'has_overlap' not in candidates.columns:
        logger.warning(f"has_overlap not in candidates for {dataset}")
        measurements['has_overlap'] = False
        return measurements

    # Create join key from utt_id, word, start_ms (unique identifier for each token)
    # Using rounded start_ms to handle floating point differences
    candidates['_join_key'] = (
        candidates['utt_id'] + '_' +
        candidates['word'] + '_' +
        candidates['start_ms'].round(1).astype(str)
    )
    measurements['_join_key'] = (
        measurements['utt_id'] + '_' +
        measurements['word'] + '_' +
        measurements['start_ms'].round(1).astype(str)
    )

    # Get unique overlap status per join key
    overlap_map = candidates[['_join_key', 'has_overlap']].drop_duplicates('_join_key')
    overlap_map = overlap_map.set_index('_join_key')['has_overlap']

    # Map to measurements
    measurements['has_overlap'] = measurements['_join_key'].map(overlap_map).fillna(False)
    measurements = measurements.drop(columns=['_join_key'])

    overlap_count = measurements['has_overlap'].sum()
    logger.info(f"  - {dataset}: {overlap_count}/{len(measurements)} tokens with overlap markers")

    return measurements


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

        if dataset == 'dimex100':
            # DIMEx100: Infer sex from F0 (no external metadata available)
            merged = _merge_dimex100_with_inferred_sex(dataset_measurements)
            stats['datasets'][dataset]['merged'] = len(merged)
            merged_dfs.append(merged)
            continue

        if dataset == 'glissando':
            # Glissando: Load metadata from raw parquet (not in unified metadata)
            merged = _merge_glissando(dataset_measurements)
            stats['datasets'][dataset]['merged'] = len(merged)
            merged_dfs.append(merged)
            continue

        if dataset in MFA_DATASETS:
            # MFA-aligned datasets: Load metadata from raw parquet
            merged = _merge_mfa_dataset(dataset_measurements, dataset)
            stats['datasets'][dataset]['merged'] = len(merged)
            merged_dfs.append(merged)
            continue

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
            # Other datasets: No metadata available
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
    # PRE_CITY_H11_001_seg001 -> PRE_CITY_H11_001
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


def _merge_glissando(measurements: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Glissando measurements with metadata from raw parquet.

    Glissando has speaker-level metadata (sex, age, origin) in the raw ingestion parquet.
    """
    measurements = measurements.copy()

    # Load Glissando raw metadata
    metadata_path = Path('metadata/glissando_raw.parquet')
    if not metadata_path.exists():
        logger.warning(f"Glissando metadata not found at {metadata_path}")
        for col in ['sex', 'age', 'age_bin', 'education_bin', 'country', 'region', 'city']:
            measurements[col] = 'unknown'
        return measurements

    metadata = pd.read_parquet(metadata_path)

    # Get unique speaker metadata
    speaker_meta = metadata.drop_duplicates(subset='speaker_id')[
        ['speaker_id', 'sex_raw', 'age_raw', 'origin_raw', 'speech_style']
    ].copy()

    # Normalize metadata fields
    speaker_meta['sex'] = speaker_meta['sex_raw'].fillna('unknown')

    # Age binning
    def bin_age(age):
        if pd.isna(age):
            return 'unknown'
        try:
            age = int(age)
            if age < 30:
                return '<30'
            elif age <= 55:
                return '30-55'
            else:
                return '>55'
        except (ValueError, TypeError):
            return 'unknown'

    speaker_meta['age'] = speaker_meta['age_raw']
    speaker_meta['age_bin'] = speaker_meta['age_raw'].apply(bin_age)

    # Education (not available for Glissando)
    speaker_meta['education_bin'] = 'unknown'

    # Country/region from origin (all Spain - Valladolid area)
    speaker_meta['country'] = 'España'
    speaker_meta['region'] = speaker_meta['origin_raw'].fillna('unknown')
    speaker_meta['city'] = speaker_meta['origin_raw'].fillna('unknown')

    # Merge with measurements
    merge_cols = ['speaker_id', 'sex', 'age', 'age_bin', 'education_bin',
                  'country', 'region', 'city', 'speech_style']
    merged = pd.merge(measurements, speaker_meta[merge_cols], on='speaker_id', how='left')

    # Fill missing values
    for col in ['sex', 'age_bin', 'education_bin', 'country']:
        if col in merged.columns:
            merged[col] = merged[col].fillna('unknown')

    logger.info(f"Glissando: merged {len(merged)} measurements with speaker metadata")

    return merged


def _merge_dimex100_with_inferred_sex(
    measurements: pd.DataFrame
) -> pd.DataFrame:
    """
    Add inferred sex metadata to DIMEx100 measurements using F0.

    Since DIMEx100 has no speaker metadata, we infer sex from mean F0:
    - F0 > 165 Hz → Female (typical female range: 180-250 Hz)
    - F0 < 145 Hz → Male (typical male range: 85-155 Hz)
    - 145-165 Hz → Ambiguous (excluded from sex analysis)

    WARNING: Do not use F0-inferred sex to analyze F0 differences - this is circular!
    Only use for non-F0 measures (cycles, duration, voicing, HNR, etc.)
    """

    measurements = measurements.copy()

    # Calculate mean F0 per speaker
    speaker_f0 = measurements.groupby('speaker_id')['mean_f0_hz'].mean()

    # Infer sex from F0
    def classify_sex(f0):
        if pd.isna(f0):
            return 'unknown'
        elif f0 > 165:
            return 'F'
        elif f0 < 145:
            return 'M'
        else:
            return 'ambiguous'

    speaker_sex = speaker_f0.apply(classify_sex)
    speaker_sex_df = speaker_sex.reset_index()
    speaker_sex_df.columns = ['speaker_id', 'sex']

    # Log inference statistics
    sex_counts = speaker_sex.value_counts()
    total_speakers = len(speaker_sex)
    logger.info("DIMEx100 sex inference from F0:")
    logger.info(f"  - Total speakers: {total_speakers}")
    logger.info(f"  - Inferred Female (F0 > 165 Hz): {sex_counts.get('F', 0)}")
    logger.info(f"  - Inferred Male (F0 < 145 Hz): {sex_counts.get('M', 0)}")
    logger.info(f"  - Ambiguous (145-165 Hz): {sex_counts.get('ambiguous', 0)}")
    logger.info(f"  - Unknown (no F0 data): {sex_counts.get('unknown', 0)}")

    # Merge inferred sex back to measurements
    merged = pd.merge(measurements, speaker_sex_df, on='speaker_id', how='left')

    # Add flag indicating sex is inferred (for circularity warning)
    merged['sex_inferred_from_f0'] = True

    # Add other metadata columns as unknown
    for col in ['age', 'age_bin', 'education_bin', 'country', 'region', 'city']:
        merged[col] = 'unknown'

    return merged


def _merge_mfa_dataset(measurements: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Merge MFA-aligned dataset measurements with metadata from raw parquet.

    Works for: mailabs, tedx, heroico, commonvoice
    """
    measurements = measurements.copy()

    # Load raw metadata
    metadata_path = Path(f'metadata/{dataset}_raw.parquet')
    if not metadata_path.exists():
        logger.warning(f"{dataset} metadata not found at {metadata_path}")
        for col in ['sex', 'age', 'age_bin', 'education_bin', 'country', 'region', 'city']:
            measurements[col] = 'unknown'
        return measurements

    metadata = pd.read_parquet(metadata_path)

    # Get unique speaker metadata
    meta_cols = ['speaker_id']
    if 'sex_raw' in metadata.columns:
        meta_cols.append('sex_raw')
    if 'age_raw' in metadata.columns:
        meta_cols.append('age_raw')
    if 'origin_raw' in metadata.columns:
        meta_cols.append('origin_raw')
    if 'speech_style' in metadata.columns:
        meta_cols.append('speech_style')

    speaker_meta = metadata.drop_duplicates(subset='speaker_id')[meta_cols].copy()

    # Normalize sex
    if 'sex_raw' in speaker_meta.columns:
        speaker_meta['sex'] = speaker_meta['sex_raw'].fillna('unknown')
    else:
        speaker_meta['sex'] = 'unknown'

    # Age binning - handles both numeric ages and text labels (twenties, thirties, etc.)
    def bin_age(age):
        if pd.isna(age) or age == 'unknown':
            return 'unknown'
        # Handle Common Voice text labels
        age_str = str(age).lower()
        if age_str in ['teens', 'twenties']:
            return '<30'
        elif age_str in ['thirties', 'fourties', 'forties', 'fifties']:
            return '30-55'
        elif age_str in ['sixties', 'seventies', 'eighties', 'nineties']:
            return '>55'
        # Handle numeric ages
        try:
            age_num = int(age)
            if age_num < 30:
                return '<30'
            elif age_num <= 55:
                return '30-55'
            else:
                return '>55'
        except (ValueError, TypeError):
            return 'unknown'

    if 'age_raw' in speaker_meta.columns:
        speaker_meta['age'] = speaker_meta['age_raw']
        speaker_meta['age_bin'] = speaker_meta['age_raw'].apply(bin_age)
    else:
        speaker_meta['age'] = 'unknown'
        speaker_meta['age_bin'] = 'unknown'

    # Education (typically not available for MFA datasets)
    speaker_meta['education_bin'] = 'unknown'

    # Country/region from origin
    if 'origin_raw' in speaker_meta.columns:
        speaker_meta['country'] = speaker_meta['origin_raw'].fillna('unknown')
        speaker_meta['region'] = speaker_meta['origin_raw'].fillna('unknown')
        speaker_meta['city'] = speaker_meta['origin_raw'].fillna('unknown')
    else:
        speaker_meta['country'] = 'unknown'
        speaker_meta['region'] = 'unknown'
        speaker_meta['city'] = 'unknown'

    # Merge with measurements
    merge_cols = ['speaker_id', 'sex', 'age', 'age_bin', 'education_bin',
                  'country', 'region', 'city']
    if 'speech_style' in speaker_meta.columns:
        merge_cols.append('speech_style')

    merged = pd.merge(measurements, speaker_meta[merge_cols], on='speaker_id', how='left')

    # Fill missing values
    for col in ['sex', 'age_bin', 'education_bin', 'country']:
        if col in merged.columns:
            merged[col] = merged[col].fillna('unknown')

    logger.info(f"{dataset}: merged {len(merged)} measurements with speaker metadata")

    return merged


# MFA-aligned datasets that use _merge_mfa_dataset
MFA_DATASETS = ['mailabs', 'tedx', 'heroico', 'commonvoice']


def _aggregate_by_speaker(df: pd.DataFrame, include_context: bool = False) -> pd.DataFrame:
    """
    Aggregate measurements by speaker to avoid pseudo-replication.

    Args:
        df: Merged measurements DataFrame
        include_context: If True, aggregate by speaker+context_label (for context analysis)

    Returns:
        DataFrame with one row per speaker (or per speaker+context if include_context=True)
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
                 'education_bin', 'country', 'region', 'city', 'speech_style',
                 'sex_inferred_from_f0']
    keep_cols = [c for c in keep_cols if c in df.columns]

    # Create aggregation dict
    agg_dict = {col: 'mean' for col in agg_cols}
    agg_dict['utt_id'] = 'count'  # Count of tokens per speaker

    # Determine grouping columns
    group_cols = ['speaker_id', 'dataset']
    if include_context and 'context_label' in df.columns:
        group_cols.append('context_label')

    # Group by speaker (and optionally context)
    grouped = df.groupby(group_cols).agg(agg_dict).reset_index()
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
