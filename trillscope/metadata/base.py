"""Base classes and data types for metadata normalization (Block C)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class UnifiedMetadata:
    """Unified speaker/utterance metadata record."""
    # Identifiers
    utt_id: str
    speaker_id: str
    dataset: str  # 'albayzin', 'preseea', 'dimex100'

    # Demographics (normalized)
    sex: str  # 'F', 'M', 'unknown'
    age: Optional[int]  # Numeric age or None
    age_bin: str  # '<30', '30-55', '>55', 'unknown'
    education_bin: str  # 'low', 'mid', 'high', 'unknown'

    # Location (normalized)
    country: str  # 'España', 'México', etc. or 'unknown'
    region: str  # Province/state or 'unknown'
    city: str  # City name or 'unknown'

    # Recording context
    speech_style: str  # 'read', 'spontaneous', 'unknown'

    # Raw values (preserved for auditing)
    sex_raw: Optional[str] = None
    age_raw: Optional[str] = None
    education_raw: Optional[str] = None
    origin_raw: Optional[str] = None

    # Mapping info
    education_mapping_rule: str = ''
    location_mapping_rule: str = ''


@dataclass
class MappingStatistics:
    """Statistics for a single field mapping."""
    field_name: str
    total_records: int
    mapped_successfully: int
    mapped_to_unknown: int
    unique_raw_values: int
    value_distribution: Dict[str, int]
    unmapped_values: List[str]


@dataclass
class MetadataResult:
    """Result of metadata normalization for a dataset."""
    dataset_name: str
    records: List[UnifiedMetadata]
    mapping_stats: Dict[str, MappingStatistics]
    issues: List[Dict[str, Any]]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to DataFrame."""
        records_dicts = []
        for rec in self.records:
            records_dicts.append({
                'utt_id': rec.utt_id,
                'speaker_id': rec.speaker_id,
                'dataset': rec.dataset,
                'sex': rec.sex,
                'age': rec.age,
                'age_bin': rec.age_bin,
                'education_bin': rec.education_bin,
                'country': rec.country,
                'region': rec.region,
                'city': rec.city,
                'speech_style': rec.speech_style,
                'sex_raw': rec.sex_raw,
                'age_raw': rec.age_raw,
                'education_raw': rec.education_raw,
                'origin_raw': rec.origin_raw,
                'education_mapping_rule': rec.education_mapping_rule,
                'location_mapping_rule': rec.location_mapping_rule,
            })
        return pd.DataFrame(records_dicts)

    def to_parquet(self, output_path: Path) -> None:
        """Save unified metadata to parquet."""
        df = self.to_dataframe()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, engine='pyarrow', index=False)
        logger.info(f"Saved {len(df)} records to {output_path}")


class DatasetMetadataNormalizer(ABC):
    """Abstract base class for dataset metadata normalization."""

    DATASET_NAME: str = ''

    def __init__(self, parquet_path: Path, output_root: Path):
        self.parquet_path = Path(parquet_path)
        self.output_root = Path(output_root)
        self.issues: List[Dict[str, Any]] = []

    def log_issue(self, severity: str, category: str,
                  message: str, context: Optional[Dict] = None) -> None:
        """Log an issue discovered during normalization."""
        self.issues.append({
            'severity': severity,
            'category': category,
            'message': message,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        })

    @abstractmethod
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw parquet from Block A."""
        pass

    @abstractmethod
    def normalize_sex(self, raw_value: Optional[str]) -> str:
        """Normalize sex to F/M/unknown."""
        pass

    @abstractmethod
    def normalize_age(self, raw_value: Optional[str]) -> Tuple[Optional[int], str]:
        """Normalize age to (numeric_age, age_bin)."""
        pass

    @abstractmethod
    def normalize_education(self, raw_value: Optional[str]) -> Tuple[str, str]:
        """Normalize education to (education_bin, mapping_rule)."""
        pass

    @abstractmethod
    def normalize_location(self, raw_origin: Optional[str],
                          raw_residence: Optional[str] = None) -> Tuple[str, str, str, str]:
        """Normalize location to (country, region, city, mapping_rule)."""
        pass

    @abstractmethod
    def normalize(self) -> MetadataResult:
        """Run full metadata normalization."""
        pass


class MetadataError(Exception):
    """Base exception for metadata normalization errors."""
    pass
