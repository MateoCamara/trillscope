"""Sample selection for diverse trill /r/ examples."""

import logging
from pathlib import Path
from typing import List, Optional
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Default paths for normalized audio
DEFAULT_NORM_ROOTS = {
    'albayzin': Path('data_norm/albayzin/audio_16k'),
    'dimex100': Path('dataset/CorpusDimex100'),  # DIMEx100 uses original paths
}


def map_to_normalized_audio(audio_path: str, dataset: str) -> Optional[str]:
    """
    Map original audio path to normalized audio path.

    For ALBAYZIN, maps .SES files to normalized .wav files in data_norm/.
    For DIMEx100, returns the original path (already usable).
    """
    if not audio_path:
        return None

    path = Path(audio_path)

    if dataset == 'albayzin':
        # Map ALBAYZIN .SES files to normalized .wav files
        # Original: dataset/ALBAYZIN/.../BVFA0101.SES
        # Normalized: data_norm/albayzin/audio_16k/ALB_BVFA0101.wav
        stem = path.stem  # e.g., 'BVFA0101'
        norm_path = DEFAULT_NORM_ROOTS['albayzin'] / f"ALB_{stem}.wav"
        if norm_path.exists():
            return str(norm_path)
        # Fallback: try without ALB_ prefix
        norm_path = DEFAULT_NORM_ROOTS['albayzin'] / f"{stem}.wav"
        if norm_path.exists():
            return str(norm_path)
        return None

    # DIMEx100 and others use original paths
    return audio_path


def select_examples(
    df: pd.DataFrame,
    n_examples: int = 5,
    min_duration_ms: float = 20.0,
    max_duration_ms: float = 150.0,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Select diverse examples from extraction results.

    Ensures variety by:
    - Sampling from different context_labels
    - Filtering reasonable durations
    - Sampling from different speakers

    Args:
        df: DataFrame with extraction results (from r_candidates parquet)
        n_examples: Number of examples to select
        min_duration_ms: Minimum /r/ duration to include
        max_duration_ms: Maximum /r/ duration to include
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with selected examples
    """
    if len(df) == 0:
        logger.warning("Empty DataFrame, no examples to select")
        return df

    # Filter by duration
    valid = df[
        (df['duration_ms'] >= min_duration_ms) &
        (df['duration_ms'] <= max_duration_ms)
    ].copy()

    if len(valid) == 0:
        logger.warning(f"No examples with duration {min_duration_ms}-{max_duration_ms}ms")
        return df.head(n_examples)

    # Filter out rows without valid audio_path
    if 'audio_path' in valid.columns:
        valid = valid[valid['audio_path'].notna()]
        valid = valid[valid['audio_path'].apply(lambda p: Path(p).exists() if p else False)]

    if len(valid) == 0:
        logger.warning("No examples with valid audio paths")
        return pd.DataFrame()

    selected = []

    # Priority context labels (trill contexts)
    priority_contexts = ['intervocalic_rr', 'word_initial', 'after_nls']

    # Try to get at least one from each context type
    for context in priority_contexts:
        context_df = valid[valid['context_label'] == context]
        if len(context_df) > 0:
            # Sample one, preferring different speakers
            sample = context_df.sample(1, random_state=random_seed)
            selected.append(sample)
            # Remove this utterance from pool to avoid duplicates
            valid = valid[valid.index != sample.index[0]]

    # Fill remaining slots
    remaining = n_examples - len(selected)
    if remaining > 0 and len(valid) > 0:
        # Try to sample from different speakers
        additional = valid.sample(
            min(remaining, len(valid)),
            random_state=random_seed + 1
        )
        selected.append(additional)

    if not selected:
        return pd.DataFrame()

    result = pd.concat(selected, ignore_index=True)

    # Limit to requested number
    result = result.head(n_examples)

    logger.info(f"Selected {len(result)} examples")
    return result


def validate_audio_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that audio_path exists for each row.

    Returns DataFrame with only valid audio paths.
    For ALBAYZIN, maps original .SES paths to normalized .wav paths.
    """
    if 'audio_path' not in df.columns:
        logger.error("No audio_path column in DataFrame")
        return pd.DataFrame()

    df = df.copy()

    # Map to normalized paths for datasets that need it
    if 'dataset' in df.columns:
        df['audio_path'] = df.apply(
            lambda row: map_to_normalized_audio(row['audio_path'], row['dataset']),
            axis=1
        )

    valid_mask = df['audio_path'].apply(
        lambda p: p is not None and Path(p).exists()
    )

    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning(f"Filtered out {n_invalid} rows with invalid audio paths")

    return df[valid_mask].copy()
