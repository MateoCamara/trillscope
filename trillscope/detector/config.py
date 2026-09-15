"""Detector configuration loading.

The detector parameters used for the published analysis ship with the package
in ``trillscope/config/detector.yaml``. ``DetectorConfig()`` defaults are more
conservative; use :func:`load_config` to reproduce the published analysis.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .cycle_detector import DetectorConfig

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "detector.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG) -> DetectorConfig:
    """Load detector parameters from a YAML file (default: the packaged config)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"detector config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    kwargs = {key: tuple(value) if isinstance(value, list) else value
              for key, value in data.items()}
    return DetectorConfig(**kwargs)
