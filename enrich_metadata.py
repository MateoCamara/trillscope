"""Enrich metadata for new datasets with additional demographic information."""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def enrich_commonvoice():
    """Enrich Common Voice metadata with age and accent from validated.tsv."""
    logger.info("Enriching Common Voice metadata...")

    # Load current metadata
    meta_path = Path('metadata/commonvoice_raw.parquet')
    df = pd.read_parquet(meta_path)
    logger.info(f"Loaded {len(df)} records")

    # Load TSV with extra data
    tsv_path = Path('dataset/cv-corpus-24.0-2025-12-05/es/validated.tsv')
    if not tsv_path.exists():
        logger.warning(f"TSV not found: {tsv_path}")
        return

    tsv = pd.read_csv(tsv_path, sep='\t', usecols=['path', 'age', 'gender', 'accents'])
    logger.info(f"Loaded {len(tsv)} TSV records")

    # Extract filename from path for joining
    tsv['filename'] = tsv['path'].str.replace('.mp3', '', regex=False)

    # Extract filename from utt_id (CVS_common_voice_es_12345 -> common_voice_es_12345)
    df['filename'] = df['utt_id'].str.replace('CVS_', '', regex=False)

    # Merge
    df = df.merge(tsv[['filename', 'age', 'accents']], on='filename', how='left')

    # Update metadata fields
    df['age_raw'] = df['age'].fillna('unknown')
    df['origin_raw'] = df['accents'].fillna('unknown')

    # Clean up
    df = df.drop(columns=['filename', 'age', 'accents'])

    # Save
    df.to_parquet(meta_path, index=False)

    # Stats
    age_counts = df['age_raw'].value_counts()
    origin_counts = df[df['origin_raw'] != 'unknown']['origin_raw'].nunique()
    logger.info(f"Age distribution: {dict(age_counts.head())}")
    logger.info(f"Unique accents: {origin_counts}")


def normalize_mailabs_countries():
    """Normalize M-AILABS country names for better statistics."""
    logger.info("Normalizing M-AILABS metadata...")

    meta_path = Path('metadata/mailabs_raw.parquet')
    df = pd.read_parquet(meta_path)

    # Countries are already good (Argentina, Chile, Colombia, Peru, Venezuela)
    # But let's add region = country for Latin American countries
    df['region_raw'] = df['origin_raw'].fillna('unknown')

    df.to_parquet(meta_path, index=False)
    logger.info(f"Updated {len(df)} records")


def normalize_heroico_countries():
    """Normalize Heroico country/origin data."""
    logger.info("Normalizing Heroico metadata...")

    meta_path = Path('metadata/heroico_raw.parquet')
    df = pd.read_parquet(meta_path)

    # Normalize country names
    country_map = {
        'Mexico': 'México',
        'Mexico.Texas': 'México',
        'Spain': 'España',
        'Venezuela': 'Venezuela',
        'Argentina': 'Argentina',
        'Argentina.France': 'Argentina',
        'Puerto.Rico': 'Puerto Rico',
        'NA': 'unknown',
        'mic2': 'unknown',
        'Castillian': 'España',
    }

    df['country_normalized'] = df['origin_raw'].map(country_map).fillna(df['origin_raw'])
    df['origin_raw'] = df['country_normalized']
    df = df.drop(columns=['country_normalized'], errors='ignore')

    df.to_parquet(meta_path, index=False)
    logger.info(f"Updated {len(df)} records")


def add_speech_style_labels():
    """Add speech style labels to datasets."""
    logger.info("Adding speech style labels...")

    # TEDx is spontaneous speech (TED talks)
    meta_path = Path('metadata/tedx_raw.parquet')
    if meta_path.exists():
        df = pd.read_parquet(meta_path)
        df['speech_style'] = 'spontaneous'
        df.to_parquet(meta_path, index=False)
        logger.info(f"TEDx: set to spontaneous ({len(df)} records)")

    # M-AILABS is read speech (audiobooks)
    meta_path = Path('metadata/mailabs_raw.parquet')
    if meta_path.exists():
        df = pd.read_parquet(meta_path)
        df['speech_style'] = 'read'
        df.to_parquet(meta_path, index=False)
        logger.info(f"M-AILABS: set to read ({len(df)} records)")

    # Heroico has read + free response
    meta_path = Path('metadata/heroico_raw.parquet')
    if meta_path.exists():
        df = pd.read_parquet(meta_path)
        # Default to read (recordings subcorpus), could refine based on file paths
        df['speech_style'] = 'read'
        df.to_parquet(meta_path, index=False)
        logger.info(f"Heroico: set to read ({len(df)} records)")

    # Common Voice is read speech (sentences)
    meta_path = Path('metadata/commonvoice_raw.parquet')
    if meta_path.exists():
        df = pd.read_parquet(meta_path)
        df['speech_style'] = 'read'
        df.to_parquet(meta_path, index=False)
        logger.info(f"Common Voice: set to read ({len(df)} records)")


if __name__ == '__main__':
    enrich_commonvoice()
    normalize_mailabs_countries()
    normalize_heroico_countries()
    add_speech_style_labels()

    logger.info("\nMetadata enrichment complete!")
