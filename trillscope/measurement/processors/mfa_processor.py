"""Generic MFA-aligned dataset acoustic measurement processor."""

from pathlib import Path
from typing import Optional

from ..base import MeasurementConfig
from .base_processor import BaseMeasurementProcessor


class MFAProcessor(BaseMeasurementProcessor):
    """
    Acoustic measurement processor for MFA-aligned datasets.

    Works with any dataset that has been processed through Montreal Forced
    Aligner (mailabs, tedx, heroico, commonvoice, etc.).
    """

    def __init__(
        self,
        candidates_path: Path,
        config: Optional[MeasurementConfig] = None,
        dataset_name: str = "mfa"
    ):
        """
        Initialize MFA processor.

        Args:
            candidates_path: Path to r_candidates_{dataset}.parquet
            config: Measurement configuration
            dataset_name: Name of the dataset
        """
        self.DATASET = dataset_name
        super().__init__(candidates_path, config)
