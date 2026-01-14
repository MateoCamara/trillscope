"""Base classes and data types for structure normalization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import csv
import logging

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NormalizedUtterance:
    """Record for a normalized utterance."""
    utt_id: str
    speaker_id: str
    dataset: str
    # Original paths
    original_audio_path: str
    original_transcript_path: Optional[str]
    # Normalized paths (relative to data_norm root)
    normalized_audio_path: str
    normalized_transcript_path: Optional[str]
    # Audio properties
    duration_ms: float
    sample_rate: int = 16000
    # Transcript properties
    transcript_text: Optional[str] = None
    has_timing: bool = False
    # Validation status
    is_valid: bool = True
    validation_issues: List[str] = field(default_factory=list)


@dataclass
class NormalizationResult:
    """Result of normalization for a dataset."""
    dataset_name: str
    utterances: List[NormalizedUtterance]
    issues: List[Dict[str, Any]]
    statistics: Dict[str, Any]

    def to_index_csv(self, output_path: Path) -> None:
        """Save mapping table to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'utt_id',
                'speaker_id',
                'dataset',
                'original_audio_path',
                'normalized_audio_path',
                'original_transcript_path',
                'normalized_transcript_path',
                'duration_ms',
                'has_timing',
                'is_valid',
                'validation_notes'
            ])

            for utt in self.utterances:
                writer.writerow([
                    utt.utt_id,
                    utt.speaker_id,
                    utt.dataset,
                    utt.original_audio_path,
                    utt.normalized_audio_path,
                    utt.original_transcript_path or '',
                    utt.normalized_transcript_path or '',
                    f'{utt.duration_ms:.1f}',
                    utt.has_timing,
                    utt.is_valid,
                    '; '.join(utt.validation_issues) if utt.validation_issues else ''
                ])

        logger.info(f"Saved utterance index to {output_path}")

    def to_validation_report(self, output_path: Path) -> None:
        """Generate validation report markdown."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate statistics
        total = len(self.utterances)
        valid = sum(1 for u in self.utterances if u.is_valid)
        invalid = total - valid
        total_duration_hrs = sum(u.duration_ms for u in self.utterances) / 3600000
        with_timing = sum(1 for u in self.utterances if u.has_timing)

        # Group issues by category
        issue_counts: Dict[str, int] = {}
        for issue in self.issues:
            cat = issue.get('category', 'unknown')
            issue_counts[cat] = issue_counts.get(cat, 0) + 1

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Normalization Report: {self.dataset_name}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Summary\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total utterances | {total:,} |\n")
            f.write(f"| Valid utterances | {valid:,} ({100*valid/total:.1f}%) |\n")
            f.write(f"| Invalid utterances | {invalid:,} ({100*invalid/total:.1f}%) |\n")
            f.write(f"| Total duration | {total_duration_hrs:.2f} hours |\n")
            f.write(f"| With phoneme timing | {with_timing:,} ({100*with_timing/total:.1f}%) |\n")
            f.write("\n")

            if issue_counts:
                f.write("## Issues by Category\n\n")
                f.write("| Category | Count |\n")
                f.write("|----------|-------|\n")
                for cat, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
                    f.write(f"| {cat} | {count} |\n")
                f.write("\n")

            # Show sample issues
            if self.issues:
                f.write("## Sample Issues (first 20)\n\n")
                for issue in self.issues[:20]:
                    severity = issue.get('severity', 'info')
                    category = issue.get('category', 'unknown')
                    message = issue.get('message', '')
                    f.write(f"- **[{severity.upper()}]** `{category}`: {message}\n")

        logger.info(f"Saved validation report to {output_path}")


class DatasetNormalizer(ABC):
    """Abstract base class for dataset normalization."""

    def __init__(self, parquet_path: Path, output_root: Path):
        """
        Initialize normalizer.

        Args:
            parquet_path: Path to Block A parquet file (metadata/<dataset>_raw.parquet)
            output_root: Root directory for outputs (usually project root)
        """
        self.parquet_path = Path(parquet_path)
        self.output_root = Path(output_root)
        self.issues: List[Dict[str, Any]] = []

    def log_issue(self, severity: str, category: str,
                  message: str, context: Optional[Dict] = None) -> None:
        """
        Log an issue discovered during normalization.

        Args:
            severity: 'error', 'warning', or 'info'
            category: Issue category (e.g., 'audio_conversion_failed', 'missing_transcript')
            message: Human-readable description
            context: Optional additional context
        """
        self.issues.append({
            'severity': severity,
            'category': category,
            'message': message,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        })

    @abstractmethod
    def load_ingestion_data(self) -> pd.DataFrame:
        """Load data from Block A parquet file."""
        pass

    @abstractmethod
    def normalize_audio(self, row: pd.Series, output_dir: Path) -> tuple:
        """
        Convert audio to normalized format.

        Args:
            row: Row from ingestion parquet
            output_dir: Directory to save normalized audio

        Returns:
            Tuple of (output_path, duration_ms, issues_list)
        """
        pass

    @abstractmethod
    def normalize_transcript(self, row: pd.Series, output_dir: Path) -> tuple:
        """
        Normalize transcript.

        Args:
            row: Row from ingestion parquet
            output_dir: Directory to save normalized transcript

        Returns:
            Tuple of (output_path, has_timing, transcript_text, issues_list)
        """
        pass

    @abstractmethod
    def normalize(self, skip_existing: bool = False,
                  num_workers: int = 1) -> NormalizationResult:
        """
        Run full normalization pipeline.

        Args:
            skip_existing: Skip files that already exist
            num_workers: Number of parallel workers

        Returns:
            NormalizationResult with all normalized utterances
        """
        pass


class NormalizationError(Exception):
    """Base exception for normalization errors."""
    pass


class AudioConversionError(NormalizationError):
    """Raised when audio conversion fails."""
    pass


class TranscriptNormalizationError(NormalizationError):
    """Raised when transcript normalization fails."""
    pass
