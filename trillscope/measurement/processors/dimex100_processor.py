"""DIMEx100 acoustic measurement processor."""

from pathlib import Path
from typing import Optional

from ..base import MeasurementConfig
from .base_processor import BaseMeasurementProcessor


class DIMEx100Processor(BaseMeasurementProcessor):
    """
    Acoustic measurement processor for DIMEx100 corpus.

    DIMEx100 has phoneme-level alignment from .phn files with
    millisecond precision timing.
    """

    DATASET = "dimex100"

    def __init__(
        self,
        candidates_path: Path,
        config: Optional[MeasurementConfig] = None
    ):
        """
        Initialize DIMEx100 processor.

        Args:
            candidates_path: Path to r_candidates_dimex100.parquet
            config: Measurement configuration
        """
        super().__init__(candidates_path, config)
