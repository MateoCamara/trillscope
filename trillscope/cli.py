"""trillscope command-line interface.

Subcommands:
    python -m trillscope.cli detect         <dataset|all>   run v2 closure detector
    python -m trillscope.cli cross-validate <dataset|all>   run independent period
                                                       detector and compare to v2
    python -m trillscope.cli validate       <dataset|all>   write validation report
                                                       against literature targets
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .cross_validate import run_dataset as run_xval
from .detector import DEFAULT_CONFIG, load_config
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


def _resolve_targets(name: str) -> list[str]:
    return DATASETS if name == "all" else [name]


def _cmd_detect(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    keep_keys_by_corpus = _load_filter_keys(args)
    filter_on = bool(keep_keys_by_corpus)
    for ds in _resolve_targets(args.dataset):
        cand = args.candidates_dir / f"r_candidates_{ds}.parquet"
        out = args.out_dir / f"closures_v2_{ds}.parquet"
        if not cand.exists():
            logging.warning("skip %s: %s not found", ds, cand)
            continue
        if filter_on and ds not in keep_keys_by_corpus:
            logging.info("-> skip %s (not in tokens.parquet, filter excludes it)", ds)
            continue
        logging.info("-> detect %s", ds)
        run_detect(cand, out, cfg=cfg, keep_keys=keep_keys_by_corpus.get(ds))


def _load_filter_keys(args: argparse.Namespace) -> dict[str, set]:
    """If tokens.parquet exists, apply the quality filter and return the set
    of (utt_id, start_ms, end_ms) tuples per dataset."""
    if args.no_filter:
        return {}
    tokens_path = args.candidates_dir / "tokens.parquet"
    if not tokens_path.exists():
        logging.warning("tokens.parquet not found; running without quality filter")
        return {}
    import pandas as pd
    from .build_tokens import CORPUS_CANONICAL
    from .quality import apply_quality_filter

    tokens = pd.read_parquet(tokens_path)
    if "periodicity_score" not in tokens.columns:
        logging.warning("tokens.parquet lacks periodicity_score; running without filter")
        return {}
    filt = apply_quality_filter(tokens)
    logging.info("quality filter: %d/%d tokens survive", len(filt), len(tokens))
    canonical_to_key = {v: k for k, v in CORPUS_CANONICAL.items()}
    out: dict[str, set] = {}
    for corpus, group in filt.groupby("corpus"):
        ds = canonical_to_key.get(corpus)
        if ds is None:
            continue
        # Recover utt_id from token_id: "<ds>-<utt_id>-<word_idx>-<r_idx>"
        # Better: parse from group; tokens.parquet drops utt_id. We need it.
        # The token_id has format ds-UTT_ID-wi-ri. Split by '-' but utt_ids can contain '-'.
        # We split off the prefix (ds + "-") and the last two '-N' chunks.
        utt_ids = []
        for tid in group["token_id"]:
            tail = tid[len(ds) + 1:]
            # strip "-<word_idx>-<r_idx>"
            utt = tail.rsplit("-", 2)[0]
            utt_ids.append(utt)
        keys = set(
            zip(utt_ids, group["t0_ms"].round(4), group["t1_ms"].round(4))
        )
        out[ds] = keys
        logging.info("  %s: %d filtered tokens", ds, len(keys))
    return out


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
        xval = (args.xval_dir or args.candidates_dir) / f"cross_validation_{ds}.parquet"
        if not closures.exists() or not xval.exists():
            logging.warning("skip %s: closures or cross-validation missing", ds)
            continue
        logging.info("-> validate %s", ds)
        run_validate(ds, closures, xval, args.reports_dir)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="trillscope")
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
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="Detector YAML config (default: packaged config/detector.yaml)")
    p.add_argument("--no-filter", action="store_true",
                   help="Skip the upstream quality filter; run detect on every timed token.")
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
    p.add_argument("--xval-dir", type=Path, default=None,
                   help="Directory with cross_validation_<dataset>.parquet "
                        "(default: --candidates-dir)")
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
