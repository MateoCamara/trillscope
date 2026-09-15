"""Run the closure detector over extracted trill candidates.

Reads r_candidates_<dataset>.parquet (produced by trillscope.extraction), loads
the original audio referenced by each row, runs the closure detector on a window
padded by DetectorConfig.boundary_expand_ms around the token, and writes
closures_v2_<dataset>.parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .audio_io import load_audio_full
from .detector import DetectorConfig, DetectionResult, detect_closures

log = logging.getLogger(__name__)


CARRY_COLUMNS = (
    "utt_id",
    "speaker_id",
    "dataset",
    "word",
    "word_idx",
    "r_idx_in_word",
    "start_ms",
    "end_ms",
    "duration_ms",
    "context_label",
    "phoneme_label",
    "prev_phoneme",
    "next_phoneme",
    "alignment_source",
    "audio_path",
)


def _detect_one(
    audio: np.ndarray,
    sr: int,
    start_ms: float,
    end_ms: float,
    cfg: DetectorConfig,
) -> DetectionResult | None:
    """Slice an audio chunk around the token boundaries and run the detector."""
    pad_ms = cfg.boundary_expand_ms
    chunk_start_ms = max(0.0, float(start_ms) - pad_ms)
    chunk_end_ms = float(end_ms) + pad_ms
    i0 = int(round(chunk_start_ms / 1000.0 * sr))
    i1 = int(round(chunk_end_ms / 1000.0 * sr))
    chunk = audio[i0:i1]
    if len(chunk) < int(sr * 0.030):
        return None
    roi = (float(start_ms) - chunk_start_ms, float(end_ms) - chunk_start_ms)
    return detect_closures(chunk, sr, cfg=cfg, roi_ms=roi)


def _success_row(row: pd.Series, result: DetectionResult) -> dict:
    closures = result.closures
    closure_times = [c.closure_t_ms for c in closures]
    intervals = np.array(
        [c.interval_to_next_ms for c in closures if not np.isnan(c.interval_to_next_ms)]
    )
    if intervals.size:
        mean_period = float(np.mean(intervals))
        period_cv = (
            float(np.std(intervals) / mean_period)
            if intervals.size > 1 and mean_period > 0
            else float("nan")
        )
    else:
        mean_period = float("nan")
        period_cv = float("nan")
    depths = [c.depth_db for c in closures]
    out = {k: row.get(k) for k in CARRY_COLUMNS}
    out.update(
        n_closures_v2=int(result.n_closures),
        confidence_v2=float(result.confidence),
        mean_period_ms_v2=mean_period,
        period_cv_v2=period_cv,
        mean_depth_db_v2=float(np.mean(depths)) if depths else float("nan"),
        closure_times_ms_v2=closure_times,
        status_v2="ok",
        notes_v2="; ".join(result.notes),
    )
    return out


def _failed_row(row: pd.Series, reason: str) -> dict:
    out = {k: row.get(k) for k in CARRY_COLUMNS}
    out.update(
        n_closures_v2=0,
        confidence_v2=0.0,
        mean_period_ms_v2=float("nan"),
        period_cv_v2=float("nan"),
        mean_depth_db_v2=float("nan"),
        closure_times_ms_v2=[],
        status_v2="failed",
        notes_v2=reason,
    )
    return out


def run_dataset(
    candidates_path: Path,
    output_path: Path,
    cfg: DetectorConfig | None = None,
    target_sr: int = 16_000,
    progress_every: int = 500,
    keep_keys: set[tuple] | None = None,
) -> pd.DataFrame:
    """Run the closure detector on every timed token in `candidates_path`.

    Tokens without timing (alignment_source == "orthographic" or duration_ms <= 0)
    are dropped before the loop. If `keep_keys` is given, only tokens whose
    (utt_id, start_ms_rounded, end_ms_rounded) tuple is in the set are processed
    (used to apply the upstream quality filter). Returns the output DataFrame
    and also writes it to `output_path` as parquet.
    """
    cfg = cfg or DetectorConfig()
    df = pd.read_parquet(candidates_path)

    n_total = len(df)
    df = df[df["duration_ms"] > 0].copy()
    if "alignment_source" in df.columns:
        df = df[df["alignment_source"] != "orthographic"]
    # r_candidates from trillscope.extraction contains exact-duplicate rows for some
    # corpora (ALBAYZIN every token twice, DIMEx100 more), so detect each token
    # once. (utt_id, start_ms, end_ms) uniquely identifies a token.
    before_dedup = len(df)
    df = df.drop_duplicates(["utt_id", "start_ms", "end_ms"]).reset_index(drop=True)
    if len(df) != before_dedup:
        log.info("dropped %d duplicate candidate rows", before_dedup - len(df))

    n_timed = len(df)
    log.info("%s: %d/%d tokens have valid timing", candidates_path.name, n_timed, n_total)

    if keep_keys is not None:
        before = len(df)
        df["_k"] = list(zip(df["utt_id"], df["start_ms"].round(4), df["end_ms"].round(4)))
        df = df[df["_k"].isin(keep_keys)].drop(columns=["_k"])
        log.info("quality filter: %d/%d tokens retained", len(df), before)

    out_rows: list[dict] = []
    processed = 0
    failed_audio = 0

    for audio_path, group in df.groupby("audio_path", sort=False):
        try:
            audio, sr = load_audio_full(audio_path, target_sr=target_sr)
        except Exception as exc:
            log.warning("audio load failed: %s: %s", audio_path, exc)
            failed_audio += 1
            for _, row in group.iterrows():
                out_rows.append(_failed_row(row, f"audio_load_failed: {exc}"))
                processed += 1
            continue
        for _, row in group.iterrows():
            try:
                result = _detect_one(
                    audio, sr, row["start_ms"], row["end_ms"], cfg
                )
                if result is None:
                    out_rows.append(_failed_row(row, "chunk_too_short"))
                else:
                    out_rows.append(_success_row(row, result))
            except Exception as exc:
                log.warning("token %s: %s", row.get("utt_id"), exc)
                out_rows.append(_failed_row(row, f"detector_error: {exc}"))
            processed += 1
            if progress_every and processed % progress_every == 0:
                log.info("  %d/%d processed", processed, n_timed)

    out_df = pd.DataFrame(out_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    log.info(
        "wrote %s: %d rows, %d audio-load failures",
        output_path,
        len(out_df),
        failed_audio,
    )
    return out_df
