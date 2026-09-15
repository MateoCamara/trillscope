"""B0: detector v2 on the FULL token pool, with NO quality filter.

The production ``closures_v2_*.parquet`` only contain tokens that pass the
production quality filter (periodicity >= 0.40, voicing >= 80, duration in
[50, 200]). A filter-sensitivity sweep therefore cannot go *looser* than the
production thresholds without closure counts for the excluded tokens.

This script runs detector v2 with the production config on every token in
``tokens.parquet`` (the sweep universe, ~16.8k tokens) — NOT the full
``r_candidates_*`` (which include tens of thousands of un-measured tokens) —
and writes the result to ``outputs/tables_unfiltered/``, leaving the production
tables in ``outputs/tables/`` untouched. Token membership is restricted with
the same (utt_id, start_ms, end_ms) key reconstruction the production filter
path uses in ``trillscope.cli``.

Run:
    python -m trillscope.detect_unfiltered
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .build_tokens import CORPUS_CANONICAL
from .detector import DEFAULT_CONFIG, load_config
from .measure_v2 import run_dataset

log = logging.getLogger(__name__)

CORPORA = ["dimex100", "albayzin", "glissando", "preseea", "tedx", "heroico"]
TOKENS = Path("outputs/tables/tokens.parquet")
CAND_DIR = Path("outputs/tables")
OUT_DIR = Path("outputs/tables_unfiltered")


def token_keys_by_corpus(tokens_path: Path) -> dict[str, set]:
    """All tokens in tokens.parquet keyed by (utt_id, start_ms, end_ms) per ds.

    Mirrors trillscope.cli._load_filter_keys but applies NO quality filter, so every
    token in the pool is retained. token_id format is "<ds>-<utt_id>-<wi>-<ri>";
    utt_ids may contain '-', so we strip the ds prefix and the trailing two
    numeric chunks only.
    """
    tokens = pd.read_parquet(tokens_path)
    canonical_to_key = {v: k for k, v in CORPUS_CANONICAL.items()}
    out: dict[str, set] = {}
    for corpus, group in tokens.groupby("corpus"):
        ds = canonical_to_key.get(corpus)
        if ds is None:
            continue
        utt_ids = [tid[len(ds) + 1:].rsplit("-", 2)[0] for tid in group["token_id"]]
        out[ds] = set(zip(utt_ids, group["t0_ms"].round(4), group["t1_ms"].round(4)))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(DEFAULT_CONFIG)
    keys = token_keys_by_corpus(TOKENS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in CORPORA:
        cand = CAND_DIR / f"r_candidates_{ds}.parquet"
        if not cand.exists():
            log.warning("skip %s: %s missing", ds, cand)
            continue
        log.info("-> detect (unfiltered) %s: %d tokens in pool", ds, len(keys.get(ds, set())))
        run_dataset(cand, OUT_DIR / f"closures_v2_{ds}.parquet", cfg=cfg, keep_keys=keys.get(ds))
    log.info("done; unfiltered closures in %s", OUT_DIR)


if __name__ == "__main__":
    main()
