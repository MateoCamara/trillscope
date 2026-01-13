"""Base classes and data types for dataset ingestion."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class SpeakerMetadata:
    """Raw speaker metadata from any dataset."""
    speaker_id: str
    sex: Optional[str] = None
    age: Optional[int] = None
    birth_date: Optional[str] = None
    birth_place: Optional[str] = None
    residence: Optional[str] = None
    education: Optional[str] = None
    profession: Optional[str] = None
    native_language: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UtteranceRecord:
    """Record for a single utterance."""
    utt_id: str
    speaker_id: str
    audio_path: str
    audio_format: str
    sample_rate: int
    duration_ms: float
    transcript_path: Optional[str] = None
    transcript_text: Optional[str] = None
    transcript_format: str = 'unknown'
    has_phoneme_alignment: bool = False
    alignment_path: Optional[str] = None
    original_filename: str = ''
    extra_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionResult:
    """Result of dataset ingestion."""
    dataset_name: str
    speakers: List[SpeakerMetadata]
    utterances: List[UtteranceRecord]
    issues: List[Dict[str, Any]]
    statistics: Dict[str, Any]

    def to_raw_parquet(self, output_path: Path) -> None:
        """Export raw metadata to parquet format."""
        records = []

        # Build speaker lookup
        speaker_lookup = {s.speaker_id: s for s in self.speakers}

        for utt in self.utterances:
            speaker = speaker_lookup.get(utt.speaker_id)

            record = {
                'utt_id': utt.utt_id,
                'speaker_id': utt.speaker_id,
                'dataset': self.dataset_name,
                'audio_path': utt.audio_path,
                'audio_format': utt.audio_format,
                'sample_rate_original': utt.sample_rate,
                'duration_ms': utt.duration_ms,
                'transcript_path': utt.transcript_path,
                'transcript_text': utt.transcript_text,
                'transcript_format': utt.transcript_format,
                'has_phoneme_alignment': utt.has_phoneme_alignment,
                'alignment_path': utt.alignment_path,
                'original_filename': utt.original_filename,
                # Speaker fields (raw, not normalized)
                'sex_raw': speaker.sex if speaker else None,
                'age_raw': str(speaker.age) if speaker and speaker.age else None,
                'education_raw': speaker.education if speaker else None,
                'origin_raw': speaker.birth_place if speaker else None,
                'residence_raw': speaker.residence if speaker else None,
                # Extra fields from utterance
                'subcorpus': utt.extra_fields.get('subcorpus'),
                'speech_style': utt.extra_fields.get('speech_style'),
                'has_overlap_markers': utt.extra_fields.get('has_overlap_markers', False),
                'task_type': utt.extra_fields.get('task_type'),
                'recording_date': utt.extra_fields.get('recording_date'),
            }
            records.append(record)

        df = pd.DataFrame(records)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, engine='pyarrow', index=False)


class DatasetIngestor(ABC):
    """Abstract base class for dataset ingestion."""

    def __init__(self, dataset_root: Path, output_dir: Path):
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.issues: List[Dict[str, Any]] = []

    def log_issue(self, severity: str, category: str,
                  message: str, context: Optional[Dict] = None) -> None:
        """
        Log an issue discovered during ingestion.

        Args:
            severity: 'error', 'warning', or 'info'
            category: Issue category (e.g., 'missing_transcript', 'encoding_error')
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
    def discover_structure(self) -> Dict[str, Any]:
        """Discover and document dataset file structure."""
        pass

    @abstractmethod
    def discover_audio(self) -> Dict[str, Any]:
        """Discover audio files, formats, and sample rates."""
        pass

    @abstractmethod
    def discover_transcripts(self) -> Dict[str, Any]:
        """Discover transcript files and formats."""
        pass

    @abstractmethod
    def discover_metadata(self) -> List[SpeakerMetadata]:
        """Extract speaker/session metadata."""
        pass

    @abstractmethod
    def link_audio_transcript(self) -> List[UtteranceRecord]:
        """Link audio files with transcripts."""
        pass

    @abstractmethod
    def ingest(self) -> IngestionResult:
        """Run full ingestion pipeline."""
        pass


class IngestionError(Exception):
    """Base exception for ingestion errors."""
    pass


class AudioFormatError(IngestionError):
    """Raised when audio format is invalid or unsupported."""
    pass


class TranscriptParseError(IngestionError):
    """Raised when transcript parsing fails."""
    pass


class MetadataError(IngestionError):
    """Raised when metadata is missing or invalid."""
    pass
