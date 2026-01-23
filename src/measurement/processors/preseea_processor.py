"""PRESEEA acoustic measurement processor."""

from pathlib import Path
from typing import Optional

from ..base import MeasurementConfig
from .base_processor import BaseMeasurementProcessor


class PreseeaProcessor(BaseMeasurementProcessor):
    """
    Acoustic measurement processor for PRESEEA corpus.

    PRESEEA has two alignment sources:
    - 'mfa': Montreal Forced Aligner with millisecond timing
    - 'orthographic': No timing (will be skipped)

    Audio is in MP3 format.
    """

    DATASET = "preseea"

    def __init__(
        self,
        candidates_path: Path,
        config: Optional[MeasurementConfig] = None
    ):
        """
        Initialize PRESEEA processor.

        Args:
            candidates_path: Path to r_candidates_preseea.parquet
            config: Measurement configuration
        """
        super().__init__(candidates_path, config)
