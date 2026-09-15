"""Stable utterance ID generation."""

from pathlib import Path
from typing import Optional
import hashlib


class IDGenerator:
    """Generate stable, reproducible utterance IDs."""

    def __init__(self, dataset_prefix: str):
        """
        Args:
            dataset_prefix: 'ALB', 'PRE', or 'DIM' for dataset identification
        """
        self.prefix = dataset_prefix

    def generate_utt_id(self, speaker_id: str, recording_id: str,
                        sequence_num: Optional[int] = None) -> str:
        """
        Generate deterministic utterance ID.

        Format: {prefix}_{speaker}_{recording}[_{seq}]
        Example: ALB_AA_FA0001, PRE_CITY_H11_001, DIM_s001_01

        Args:
            speaker_id: Speaker identifier
            recording_id: Recording or utterance identifier
            sequence_num: Optional sequence number for multi-utterance recordings

        Returns:
            Stable utterance ID string
        """
        # Clean identifiers
        speaker_clean = self._sanitize(speaker_id)
        recording_clean = self._sanitize(recording_id)

        if sequence_num is not None:
            return f"{self.prefix}_{speaker_clean}_{recording_clean}_{sequence_num:04d}"
        else:
            return f"{self.prefix}_{speaker_clean}_{recording_clean}"

    def hash_based_id(self, *components: str) -> str:
        """
        Generate hash-based ID for edge cases.

        Useful when components don't follow a clean pattern.

        Args:
            *components: Strings to hash together

        Returns:
            ID in format: {prefix}_{hash8}
        """
        combined = '_'.join(str(c) for c in components)
        hash_val = hashlib.md5(combined.encode()).hexdigest()[:8]
        return f"{self.prefix}_{hash_val}"

    def from_filename(self, filename: str) -> str:
        """
        Generate ID from a filename, preserving the stem.

        Args:
            filename: Filename with or without extension

        Returns:
            ID in format: {prefix}_{filename_stem}
        """
        stem = Path(filename).stem
        return f"{self.prefix}_{self._sanitize(stem)}"

    def _sanitize(self, s: str) -> str:
        """Sanitize string for use in ID (remove special chars, lowercase)."""
        # Replace common separators with underscore
        result = s.replace('-', '_').replace(' ', '_').replace('.', '_')
        # Keep only alphanumeric and underscore
        result = ''.join(c for c in result if c.isalnum() or c == '_')
        # Remove consecutive underscores
        while '__' in result:
            result = result.replace('__', '_')
        return result.strip('_')
