"""ALBAYZIN metadata normalizer."""

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


# Spanish location mapping: origin -> (country, region, city)
SPAIN_LOCATIONS = {
    'MADRID': ('España', 'Comunidad de Madrid', 'Madrid'),
    'BARCELONA': ('España', 'Cataluña', 'Barcelona'),
    'VALENCIA': ('España', 'Comunidad Valenciana', 'Valencia'),
    'SEVILLA': ('España', 'Andalucía', 'Sevilla'),
    'ZARAGOZA': ('España', 'Aragón', 'Zaragoza'),
    'MURCIA': ('España', 'Región de Murcia', 'Murcia'),
    'TOLEDO': ('España', 'Castilla-La Mancha', 'Toledo'),
    'GUADALAJARA': ('España', 'Castilla-La Mancha', 'Guadalajara'),
    'CACERES': ('España', 'Extremadura', 'Cáceres'),
    'SEGOVIA': ('España', 'Castilla y León', 'Segovia'),
    'CUENCA': ('España', 'Castilla-La Mancha', 'Cuenca'),
    'CIUDAD REAL': ('España', 'Castilla-La Mancha', 'Ciudad Real'),
    'CARTAGENA': ('España', 'Región de Murcia', 'Cartagena'),
    'TENERIFE': ('España', 'Canarias', 'Santa Cruz de Tenerife'),
    'GRANADA': ('España', 'Andalucía', 'Granada'),
    'MALAGA': ('España', 'Andalucía', 'Málaga'),
    'CEDEIRA': ('España', 'Galicia', 'Cedeira'),
    'RIOJA': ('España', 'La Rioja', 'Logroño'),
    'BURGOS': ('España', 'Castilla y León', 'Burgos'),
    'CANTABRIA': ('España', 'Cantabria', 'Santander'),
    'EIBAR': ('España', 'País Vasco', 'Eibar'),
    'JAEN': ('España', 'Andalucía', 'Jaén'),
    'LEON': ('España', 'Castilla y León', 'León'),
    'NAVARRA': ('España', 'Navarra', 'Pamplona'),
    'PALENCIA': ('España', 'Castilla y León', 'Palencia'),
    'SALAMANCA': ('España', 'Castilla y León', 'Salamanca'),
    'VALLADOLID': ('España', 'Castilla y León', 'Valladolid'),
    'VIZCAYA': ('España', 'País Vasco', 'Bilbao'),
    # Compound locations
    'LORCA (MURCIA)': ('España', 'Región de Murcia', 'Lorca'),
    'MADRIDEJOS (TOLEDO)': ('España', 'Castilla-La Mancha', 'Madridejos'),
    # Foreign locations
    'COGNAC (FRANCIA)': ('Francia', 'Nueva Aquitania', 'Cognac'),
    'LISBOA': ('Portugal', 'Área Metropolitana de Lisboa', 'Lisboa'),
}


class AlbayzinMetadataNormalizer(DatasetMetadataNormalizer):
    """
    Normalize ALBAYZIN metadata to unified schema.

    Input: metadata/albayzin_raw.parquet
    Output: Contributes to metadata/metadata_unified.parquet
    Report: reports/metadata_mapping_albayzin.md
    """

    DATASET_NAME = 'albayzin'

    def __init__(self, parquet_path: Path, output_root: Path):
        super().__init__(parquet_path, output_root)
        self.education_mapper = EducationMapper()

    def load_raw_data(self) -> pd.DataFrame:
        """Load the raw parquet written by ingestion."""
        logger.info(f"Loading ALBAYZIN data from {self.parquet_path}")
        df = pd.read_parquet(self.parquet_path)
        logger.info(f"Loaded {len(df):,} records")
        return df

    def normalize_sex(self, raw_value: Optional[str]) -> str:
        """ALBAYZIN sex is already F/M format."""
        if not raw_value:
            return 'unknown'
        value = raw_value.strip().upper()
        if value in ('F', 'M'):
            return value
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
        """Map ALBAYZIN job titles to education levels."""
        return self.education_mapper.map_albayzin(raw_value)

    def normalize_location(self, raw_origin: Optional[str],
                          raw_residence: Optional[str] = None) -> Tuple[str, str, str, str]:
        """Normalize Spanish locations."""
        if not raw_origin:
            return 'unknown', 'unknown', 'unknown', 'empty_value'

        origin_upper = raw_origin.strip().upper()

        # Direct match
        if origin_upper in SPAIN_LOCATIONS:
            country, region, city = SPAIN_LOCATIONS[origin_upper]
            return country, region, city, f'direct: {raw_origin}'

        # Partial match for compound locations
        for key, (country, region, city) in SPAIN_LOCATIONS.items():
            if origin_upper in key or key in origin_upper:
                return country, region, city, f'partial: {raw_origin} -> {key}'

        # Assume Spain if no match (ALBAYZIN is primarily Spanish)
        return 'España', 'unknown', raw_origin.title(), f'assumed_spain: {raw_origin}'

    def normalize(self) -> MetadataResult:
        """Run full normalization."""
        logger.info("Starting ALBAYZIN metadata normalization")

        df = self.load_raw_data()
        records = []

        # Track mapping statistics
        sex_values = {}
        age_values = {}
        education_values = {}
        education_unmapped = []

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

            # Get speech style, default to 'read' for ALBAYZIN
            speech_style = row.get('speech_style', 'read')
            if not speech_style or speech_style == 'unknown':
                speech_style = 'read'

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
                unmapped_values=[],
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

        logger.info(f"ALBAYZIN normalization complete: {total:,} records")
        logger.info(f"  Sex: {sex_values}")
        logger.info(f"  Age bins: {age_values}")
        logger.info(f"  Education: {education_values}")

        return MetadataResult(
            dataset_name=self.DATASET_NAME,
            records=records,
            mapping_stats=mapping_stats,
            issues=self.issues,
        )
