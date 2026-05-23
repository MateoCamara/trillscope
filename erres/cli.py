"""erres command-line interface.

Subcommands:
    python -m erres.cli detect         <dataset|all>   run v2 closure detector
    python -m erres.cli cross-validate <dataset|all>   run independent period
                                                       detector and compare to v2
    python -m erres.cli validate       <dataset|all>   write validation report
                                                       against literature targets
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from .cross_validate import run_dataset as run_xval
from .detector import DetectorConfig
from .measure_v2 import run_dataset as run_detect


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


def _resolve_targets(name: str) -> list[str]:
    return DATASETS if name == "all" else [name]


def _cmd_detect(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    for ds in _resolve_targets(args.dataset):
        cand = args.candidates_dir / f"r_candidates_{ds}.parquet"
        out = args.out_dir / f"closures_v2_{ds}.parquet"
        if not cand.exists():
            logging.warning("skip %s: %s not found", ds, cand)
            continue
        logging.info("-> detect %s", ds)
        run_detect(cand, out, cfg=cfg)


def _cmd_cross_validate(args: argparse.Namespace) -> None:
    for ds in _resolve_targets(args.dataset):
        src = args.candidates_dir / f"closures_v2_{ds}.parquet"
        out = args.out_dir / f"cross_validation_{ds}.parquet"
        if not src.exists():
            logging.warning("skip %s: %s not found (run `detect` first)", ds, src)
            continue
        logging.info("-> cross-validate %s", ds)
        run_xval(src, out)


def _cmd_validate(args: argparse.Namespace) -> None:
    from .validate import run_dataset as run_validate
    for ds in _resolve_targets(args.dataset):
        closures = args.candidates_dir / f"closures_v2_{ds}.parquet"
        xval = args.candidates_dir / f"cross_validation_{ds}.parquet"
        if not closures.exists() or not xval.exists():
            logging.warning("skip %s: closures or cross-validation missing", ds)
            continue
        logging.info("-> validate %s", ds)
        run_validate(ds, closures, xval, args.reports_dir)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="erres")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common_paths = dict(
        candidates_dir=("--candidates-dir", Path("outputs/tables"),
                        "Directory containing input parquets (default: outputs/tables)"),
        out_dir=("--out-dir", Path("outputs/tables"),
                 "Output directory for parquets (default: outputs/tables)"),
    )

    p = sub.add_parser("detect", help="Run detector v2 on a dataset")
    p.add_argument("dataset", choices=["all", *DATASETS])
    p.add_argument(common_paths["candidates_dir"][0], type=Path,
                   default=common_paths["candidates_dir"][1],
                   help=common_paths["candidates_dir"][2])
    p.add_argument(common_paths["out_dir"][0], type=Path,
                   default=common_paths["out_dir"][1],
                   help=common_paths["out_dir"][2])
    p.add_argument("--config", type=Path, default=Path("config/detector.yaml"),
                   help="Detector YAML config (default: config/detector.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=_cmd_detect)

    p = sub.add_parser("cross-validate", help="Run independent period detector and compare")
    p.add_argument("dataset", choices=["all", *DATASETS])
    p.add_argument(common_paths["candidates_dir"][0], type=Path,
                   default=common_paths["candidates_dir"][1],
                   help="Directory with closures_v2_<dataset>.parquet")
    p.add_argument(common_paths["out_dir"][0], type=Path,
                   default=common_paths["out_dir"][1])
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=_cmd_cross_validate)

    p = sub.add_parser("validate", help="Write literature-anchored validation report")
    p.add_argument("dataset", choices=["all", *DATASETS])
    p.add_argument(common_paths["candidates_dir"][0], type=Path,
                   default=common_paths["candidates_dir"][1])
    p.add_argument("--reports-dir", type=Path, default=Path("reports/validation"),
                   help="Where to write markdown reports + histograms")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=_cmd_validate)

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
