"""PRESEEA metadata normalizer."""

from pathlib import Path
from typing import Optional, Tuple
import logging

import pandas as pd

from ..base import (
    DatasetMetadataNormalizer,
    UnifiedMetadata,
    MetadataResult,
    MappingStatistics,
)
from ..utils.education_mapper import EducationMapper

logger = logging.getLogger(__name__)


# Sex mapping for PRESEEA
SEX_MAPPINGS = {
    # Male variants
    'hombre': 'M',
    'h': 'M',
    'masculino': 'M',
    'male': 'M',
    'hOmbre': 'M',
    'Hombre': 'M',
    'm': 'M',  # Note: 'm' could be male or mujer, context-dependent
    # Female variants
    'mujer': 'F',
    'f': 'F',
    'femenino': 'F',
    'female': 'F',
    'Mujer': 'F',
}


# City code to location mapping (from PRESEEA filename patterns and metadata)
CITY_CODES = {
    # Spain
    'ALCA': {'city': 'Alcalá de Henares', 'region': 'Comunidad de Madrid', 'country': 'España'},
    'BARC': {'city': 'Barcelona', 'region': 'Cataluña', 'country': 'España'},
    'CADI': {'city': 'Cádiz', 'region': 'Andalucía', 'country': 'España'},
    'GRAN': {'city': 'Granada', 'region': 'Andalucía', 'country': 'España'},
    'LASP': {'city': 'Las Palmas', 'region': 'Canarias', 'country': 'España'},
    'MADR': {'city': 'Madrid', 'region': 'Comunidad de Madrid', 'country': 'España'},
    'MALA': {'city': 'Málaga', 'region': 'Andalucía', 'country': 'España'},
    'PALM': {'city': 'Palma de Mallorca', 'region': 'Islas Baleares', 'country': 'España'},
    'SANT': {'city': 'Santiago de Compostela', 'region': 'Galicia', 'country': 'España'},
    'SEVI': {'city': 'Sevilla', 'region': 'Andalucía', 'country': 'España'},
    'VALE': {'city': 'Valencia', 'region': 'Comunidad Valenciana', 'country': 'España'},
    'XIXO': {'city': 'Gijón', 'region': 'Asturias', 'country': 'España'},
    # Latin America
    'BAIR': {'city': 'Buenos Aires', 'region': 'Buenos Aires', 'country': 'Argentina'},
    'BARR': {'city': 'Barranquilla', 'region': 'Atlántico', 'country': 'Colombia'},
    'BOGO': {'city': 'Bogotá', 'region': 'Cundinamarca', 'country': 'Colombia'},
    'CALI': {'city': 'Cali', 'region': 'Valle del Cauca', 'country': 'Colombia'},
    'CARA': {'city': 'Caracas', 'region': 'Distrito Capital', 'country': 'Venezuela'},
    'CART': {'city': 'Cartagena de Indias', 'region': 'Bolívar', 'country': 'Colombia'},
    'CHIC': {'city': 'Chiclayo', 'region': 'Lambayeque', 'country': 'Perú'},
    'GUAD': {'city': 'Guadalajara', 'region': 'Jalisco', 'country': 'México'},
    'GUAT': {'city': 'Ciudad de Guatemala', 'region': 'Guatemala', 'country': 'Guatemala'},
    'LHAB': {'city': 'La Habana', 'region': 'La Habana', 'country': 'Cuba'},
    'LIMA': {'city': 'Lima', 'region': 'Lima', 'country': 'Perú'},
    'LPAZ': {'city': 'La Paz', 'region': 'La Paz', 'country': 'Bolivia'},
    'MEDE': {'city': 'Medellín', 'region': 'Antioquia', 'country': 'Colombia'},
    'MEVE': {'city': 'Mérida', 'region': 'Mérida', 'country': 'Venezuela'},
    'MEXI': {'city': 'Ciudad de México', 'region': 'CDMX', 'country': 'México'},
    'MONR': {'city': 'Monterrey', 'region': 'Nuevo León', 'country': 'México'},
    'MONV': {'city': 'Montevideo', 'region': 'Montevideo', 'country': 'Uruguay'},
    'MXLI': {'city': 'Ciudad de México', 'region': 'CDMX', 'country': 'México'},
    'NYOC': {'city': 'Nueva York', 'region': 'Nueva York', 'country': 'Estados Unidos'},
    'PERE': {'city': 'Pereira', 'region': 'Risaralda', 'country': 'Colombia'},
    'PUEB': {'city': 'Puebla', 'region': 'Puebla', 'country': 'México'},
    'QUIT': {'city': 'Quito', 'region': 'Pichincha', 'country': 'Ecuador'},
    'SCHI': {'city': 'Santiago', 'region': 'Región Metropolitana', 'country': 'Chile'},
    'SCOM': {'city': 'Santa Cruz de Tenerife', 'region': 'Canarias', 'country': 'España'},
}

# Direct city name mapping (for origin_raw values)
CITY_NAMES = {
    # Fix encoding issues and map to structured location
    'Guadalajara': {'city': 'Guadalajara', 'region': 'Jalisco', 'country': 'México'},
    'Madrid': {'city': 'Madrid', 'region': 'Comunidad de Madrid', 'country': 'España'},
    'Alcalá de Henares': {'city': 'Alcalá de Henares', 'region': 'Comunidad de Madrid', 'country': 'España'},
    'Valencia': {'city': 'Valencia', 'region': 'Comunidad Valenciana', 'country': 'España'},
    'Barcelona': {'city': 'Barcelona', 'region': 'Cataluña', 'country': 'España'},
    'Barranquilla': {'city': 'Barranquilla', 'region': 'Atlántico', 'country': 'Colombia'},
    'Bogotá': {'city': 'Bogotá', 'region': 'Cundinamarca', 'country': 'Colombia'},
    'Cali': {'city': 'Cali', 'region': 'Valle del Cauca', 'country': 'Colombia'},
    'Caracas': {'city': 'Caracas', 'region': 'Distrito Capital', 'country': 'Venezuela'},
}


class PreseeaMetadataNormalizer(DatasetMetadataNormalizer):
    """
    Normalize PRESEEA metadata to unified schema.

    Input: metadata/preseea_raw.parquet
    Output: Contributes to metadata/metadata_unified.parquet
    Report: reports/metadata_mapping_preseea.md
    """

    DATASET_NAME = 'preseea'

    def __init__(self, parquet_path: Path, output_root: Path):
        super().__init__(parquet_path, output_root)
        self.education_mapper = EducationMapper()

    def load_raw_data(self) -> pd.DataFrame:
        """Load Block A parquet."""
        logger.info(f"Loading PRESEEA data from {self.parquet_path}")
        df = pd.read_parquet(self.parquet_path)
        logger.info(f"Loaded {len(df):,} records")
        return df

    def normalize_sex(self, raw_value: Optional[str]) -> str:
        """Normalize sex value to F/M/unknown."""
        if not raw_value:
            return 'unknown'

        value_lower = raw_value.lower().strip()

        # Direct mapping
        if value_lower in SEX_MAPPINGS:
            return SEX_MAPPINGS[value_lower]

        # Handle 'H' (hombre) explicitly
        if value_lower == 'h':
            return 'M'

        # First letter heuristics
        if value_lower.startswith('h') and not value_lower.startswith('he'):
            return 'M'  # hombre
        if value_lower.startswith('muj') or value_lower.startswith('fem'):
            return 'F'

        return 'unknown'

    def normalize_age(self, raw_value: Optional[str]) -> Tuple[Optional[int], str]:
        """Convert age string to numeric and bin."""
        if not raw_value:
            return None, 'unknown'

        try:
            age = int(raw_value)
            if age < 0 or age > 120:
                return None, 'unknown'

            if age < 30:
                age_bin = '<30'
            elif age <= 55:
                age_bin = '30-55'
            else:
                age_bin = '>55'

            return age, age_bin
        except (ValueError, TypeError):
            return None, 'unknown'

    def normalize_education(self, raw_value: Optional[str]) -> Tuple[str, str]:
        """Map PRESEEA education string to level."""
        return self.education_mapper.map_preseea(raw_value)

    def normalize_location(self, raw_origin: Optional[str],
                          raw_residence: Optional[str] = None) -> Tuple[str, str, str, str]:
        """Normalize PRESEEA location from origin or city code."""
        if not raw_origin:
            return 'unknown', 'unknown', 'unknown', 'empty_value'

        # Clean encoding issues
        origin_clean = raw_origin.replace('\ufffd', 'a').replace('�', 'a')

        # Try direct city name match
        if origin_clean in CITY_NAMES:
            loc = CITY_NAMES[origin_clean]
            return loc['country'], loc['region'], loc['city'], f'direct: {raw_origin}'

        # Try case-insensitive match
        for name, loc in CITY_NAMES.items():
            if name.lower() == origin_clean.lower():
                return loc['country'], loc['region'], loc['city'], f'case_insensitive: {raw_origin}'

        # Try city code (first 4 chars uppercase)
        code = origin_clean[:4].upper()
        if code in CITY_CODES:
            loc = CITY_CODES[code]
            return loc['country'], loc['region'], loc['city'], f'code: {code}'

        # Unknown location
        return 'unknown', 'unknown', origin_clean, f'no_match: {raw_origin}'

    def normalize(self) -> MetadataResult:
        """Run full normalization."""
        logger.info("Starting PRESEEA metadata normalization")

        df = self.load_raw_data()
        records = []

        # Track mapping statistics
        sex_values = {}
        age_values = {}
        education_values = {}
        education_unmapped = []
        sex_unmapped = []

        for _, row in df.iterrows():
            # Normalize each field
            sex = self.normalize_sex(row.get('sex_raw'))
            age, age_bin = self.normalize_age(row.get('age_raw'))
            education_bin, edu_rule = self.normalize_education(row.get('education_raw'))
            country, region, city, loc_rule = self.normalize_location(
                row.get('origin_raw'), row.get('residence_raw')
            )

            # Track statistics
            sex_values[sex] = sex_values.get(sex, 0) + 1
            age_values[age_bin] = age_values.get(age_bin, 0) + 1
            education_values[education_bin] = education_values.get(education_bin, 0) + 1

            if education_bin == 'unknown' and row.get('education_raw'):
                raw_edu = row['education_raw']
                if raw_edu not in education_unmapped:
                    education_unmapped.append(raw_edu)

            if sex == 'unknown' and row.get('sex_raw'):
                raw_sex = row['sex_raw']
                if raw_sex not in sex_unmapped:
                    sex_unmapped.append(raw_sex)

            # Get speech style, default to 'spontaneous' for PRESEEA
            speech_style = row.get('speech_style', 'spontaneous')
            if not speech_style or speech_style == 'unknown':
                speech_style = 'spontaneous'

            records.append(UnifiedMetadata(
                utt_id=row['utt_id'],
                speaker_id=row['speaker_id'],
                dataset=self.DATASET_NAME,
                sex=sex,
                age=age,
                age_bin=age_bin,
                education_bin=education_bin,
                country=country,
                region=region,
                city=city,
                speech_style=speech_style,
                sex_raw=row.get('sex_raw'),
                age_raw=row.get('age_raw'),
                education_raw=row.get('education_raw'),
                origin_raw=row.get('origin_raw'),
                education_mapping_rule=edu_rule,
                location_mapping_rule=loc_rule,
            ))

        # Build statistics
        total = len(records)
        mapping_stats = {
            'sex': MappingStatistics(
                field_name='sex',
                total_records=total,
                mapped_successfully=total - sex_values.get('unknown', 0),
                mapped_to_unknown=sex_values.get('unknown', 0),
                unique_raw_values=len(df['sex_raw'].dropna().unique()) if 'sex_raw' in df else 0,
                value_distribution=sex_values,
                unmapped_values=sex_unmapped,
            ),
            'age_bin': MappingStatistics(
                field_name='age_bin',
                total_records=total,
                mapped_successfully=total - age_values.get('unknown', 0),
                mapped_to_unknown=age_values.get('unknown', 0),
                unique_raw_values=len(df['age_raw'].dropna().unique()) if 'age_raw' in df else 0,
                value_distribution=age_values,
                unmapped_values=[],
            ),
            'education_bin': MappingStatistics(
                field_name='education_bin',
                total_records=total,
                mapped_successfully=total - education_values.get('unknown', 0),
                mapped_to_unknown=education_values.get('unknown', 0),
                unique_raw_values=len(df['education_raw'].dropna().unique()) if 'education_raw' in df else 0,
                value_distribution=education_values,
                unmapped_values=education_unmapped,
            ),
        }

        logger.info(f"PRESEEA normalization complete: {total:,} records")
        logger.info(f"  Sex: {sex_values}")
        logger.info(f"  Age bins: {age_values}")
        logger.info(f"  Education: {education_values}")

        return MetadataResult(
            dataset_name=self.DATASET_NAME,
            records=records,
            mapping_stats=mapping_stats,
            issues=self.issues,
        )
