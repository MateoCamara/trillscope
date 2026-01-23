"""ALBAYZIN acoustic measurement processor."""

from pathlib import Path
from typing import Optional

from ..base import MeasurementConfig
from .base_processor import BaseMeasurementProcessor


class AlbayzinProcessor(BaseMeasurementProcessor):
    """
    Acoustic measurement processor for ALBAYZIN corpus.

    ALBAYZIN has phoneme alignment from SEO files with sample-level
    timing, and audio in SES format (raw 16-bit PCM, 16kHz).
    """

    DATASET = "albayzin"

    def __init__(
        self,
        candidates_path: Path,
        config: Optional[MeasurementConfig] = None
    ):
        """
        Initialize ALBAYZIN processor.

        Args:
            candidates_path: Path to r_candidates_albayzin.parquet
            config: Measurement configuration
        """
        super().__init__(candidates_path, config)
