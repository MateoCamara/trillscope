"""erres command-line interface.

Currently exposes a single sub-command:
    python -m erres.cli detect <dataset|all>

which runs detector v2 over the v1 r_candidates parquet tables and writes
closures_v2_<dataset>.parquet next to them.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from .detector import DetectorConfig
from .measure_v2 import run_dataset


DATASETS = [
    "dimex100",
    "albayzin",
    "glissando",
    "preseea",
    "tedx",
    "heroico",
    "commonvoice",
    "mailabs",
]


def load_config(path: Path) -> DetectorConfig:
    """Load detector parameters from YAML. Falls back to defaults if missing."""
    if not path.exists():
        logging.warning("config file %s not found; using defaults", path)
        return DetectorConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    kwargs: dict = {}
    for key, value in data.items():
        if isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value
    return DetectorConfig(**kwargs)


def _cmd_detect(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    targets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in targets:
        cand = args.candidates_dir / f"r_candidates_{ds}.parquet"
        out = args.out_dir / f"closures_v2_{ds}.parquet"
        if not cand.exists():
            logging.warning("skip %s: %s not found", ds, cand)
            continue
        logging.info("-> %s", ds)
        run_dataset(cand, out, cfg=cfg)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="erres")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("detect", help="Run detector v2 on a dataset")
    p.add_argument("dataset", choices=["all", *DATASETS])
    p.add_argument(
        "--candidates-dir",
        type=Path,
        default=Path("outputs/tables"),
        help="Directory containing r_candidates_<dataset>.parquet (default: outputs/tables)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/tables"),
        help="Output directory for closures_v2_<dataset>.parquet (default: outputs/tables)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config/detector.yaml"),
        help="Detector YAML config (default: config/detector.yaml)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=_cmd_detect)

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
