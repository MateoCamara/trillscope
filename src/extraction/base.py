"""Base classes and data types for trill /r/ extraction."""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class TrillCandidate:
    """Single trill /r/ occurrence."""

    # Identity
    utt_id: str
    speaker_id: str
    dataset: str

    # Word context
    word: str
    word_idx: int  # 0-indexed position in utterance

    # Timing (milliseconds)
    start_ms: float
    end_ms: float

    # Phonetic context
    phoneme_label: str  # Original label: 'r(', 'rr', 'r'

    # Fields with defaults must come after fields without defaults
    r_idx_in_word: int = 0  # Which /r/ in this word (0 if only one)
    prev_phoneme: Optional[str] = None
    next_phoneme: Optional[str] = None
    context_label: str = 'unknown'  # 'intervocalic_rr', 'word_initial', 'after_nls'
    alignment_source: str = 'unknown'  # 'phn', 'seo', 'textgrid', 'mfa'
    audio_path: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Duration of /r/ segment in milliseconds."""
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'utt_id': self.utt_id,
            'speaker_id': self.speaker_id,
            'dataset': self.dataset,
            'word': self.word,
            'word_idx': self.word_idx,
            'r_idx_in_word': self.r_idx_in_word,
            'start_ms': self.start_ms,
            'end_ms': self.end_ms,
            'duration_ms': self.duration_ms,
            'phoneme_label': self.phoneme_label,
            'prev_phoneme': self.prev_phoneme,
            'next_phoneme': self.next_phoneme,
            'context_label': self.context_label,
            'alignment_source': self.alignment_source,
            'audio_path': self.audio_path,
        }


@dataclass
class ExtractionResult:
    """Result of extraction for a dataset."""

    dataset: str
    candidates: List[TrillCandidate] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)

    def add_candidate(self, candidate: TrillCandidate) -> None:
        """Add a trill candidate."""
        self.candidates.append(candidate)

    def add_issue(self, issue: str) -> None:
        """Log an issue."""
        self.issues.append(issue)

    def compute_statistics(self) -> None:
        """Compute summary statistics."""
        self.statistics = {
            'total_candidates': len(self.candidates),
            'total_issues': len(self.issues),
        }

        # Count by context_label
        context_counts = {}
        for c in self.candidates:
            label = c.context_label
            context_counts[label] = context_counts.get(label, 0) + 1
        self.statistics['by_context'] = context_counts

        # Count unique speakers and utterances
        speakers = set(c.speaker_id for c in self.candidates)
        utterances = set(c.utt_id for c in self.candidates)
        self.statistics['unique_speakers'] = len(speakers)
        self.statistics['unique_utterances'] = len(utterances)

        # Duration stats
        if self.candidates:
            durations = [c.duration_ms for c in self.candidates]
            self.statistics['mean_duration_ms'] = sum(durations) / len(durations)
            self.statistics['min_duration_ms'] = min(durations)
            self.statistics['max_duration_ms'] = max(durations)
