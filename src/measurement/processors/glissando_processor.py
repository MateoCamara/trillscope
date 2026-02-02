"""Glissando-sp acoustic measurement processor."""

from pathlib import Path
from typing import Optional

from ..base import MeasurementConfig
from .base_processor import BaseMeasurementProcessor


class GlissandoProcessor(BaseMeasurementProcessor):
    """
    Acoustic measurement processor for Glissando-sp corpus.

    Glissando-sp has phoneme alignment from TextGrid files with SAMPA
    notation (rr = trill), and audio in WAV format (44kHz stereo).
    """

    DATASET = "glissando"

    def __init__(
        self,
        candidates_path: Path,
        config: Optional[MeasurementConfig] = None
    ):
        """
        Initialize Glissando processor.

        Args:
            candidates_path: Path to r_candidates_glissando.parquet
            config: Measurement configuration
        """
        super().__init__(candidates_path, config)
