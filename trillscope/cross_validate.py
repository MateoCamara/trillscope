"""Cross-validation: run the independent period detector over every token
already measured by the closure detector (detect_closures), and report agreement."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .audio_io import load_audio_full
from .detector import PeriodDetectorConfig, detect_n_closures_by_period

log = logging.getLogger(__name__)


def run_dataset(
    closures_v2_path: Path,
    output_path: Path,
    target_sr: int = 16_000,
    cfg: PeriodDetectorConfig | None = None,
    progress_every: int = 500,
) -> pd.DataFrame:
    """For each row of `closures_v2_path`, run the period detector on the same
    audio + ROI and write a per-token agreement table."""
    cfg = cfg or PeriodDetectorConfig()
    src = pd.read_parquet(closures_v2_path)
    src = src[src["status_v2"] == "ok"].copy()
    log.info("%s: %d tokens with status_v2=ok to cross-validate",
             closures_v2_path.name, len(src))

    rows: list[dict] = []
    processed = 0
    failed_audio = 0
    for audio_path, group in src.groupby("audio_path", sort=False):
        try:
            audio, sr = load_audio_full(audio_path, target_sr=target_sr)
        except Exception as exc:
            log.warning("audio load failed: %s: %s", audio_path, exc)
            failed_audio += 1
            for _, row in group.iterrows():
                rows.append(_failed_row(row, f"audio_load_failed: {exc}"))
                processed += 1
            continue
        for _, row in group.iterrows():
            try:
                # Use the same ±boundary padding as the closure detector
                # (DetectorConfig.boundary_expand_ms) so both detectors see the
                # same audio window. ROI inside that window is the labelled token.
                pad_ms = 20.0
                chunk_start_ms = max(0.0, float(row["start_ms"]) - pad_ms)
                chunk_end_ms = float(row["end_ms"]) + pad_ms
                i0 = int(round(chunk_start_ms / 1000.0 * sr))
                i1 = int(round(chunk_end_ms / 1000.0 * sr))
                chunk = audio[i0:i1]
                roi = (float(row["start_ms"]) - chunk_start_ms,
                       float(row["end_ms"]) - chunk_start_ms)
                result = detect_n_closures_by_period(chunk, sr, roi_ms=roi, cfg=cfg)
                rows.append(_success_row(row, result))
            except Exception as exc:
                log.warning("token %s: %s", row.get("utt_id"), exc)
                rows.append(_failed_row(row, f"period_error: {exc}"))
            processed += 1
            if progress_every and processed % progress_every == 0:
                log.info("  %d/%d processed", processed, len(src))

    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    n_agree = int(out["agreement"].sum())
    log.info(
        "wrote %s: %d rows, %d audio fail, agreement %d/%d = %.1f%%",
        output_path, len(out), failed_audio, n_agree, len(out),
        100.0 * n_agree / max(1, len(out)),
    )
    return out


def _success_row(row: pd.Series, result) -> dict:
    v2 = int(row["n_closures_v2"])
    p = int(result.n_closures)
    return {
        "utt_id": row.get("utt_id"),
        "speaker_id": row.get("speaker_id"),
        "dataset": row.get("dataset"),
        "start_ms": row.get("start_ms"),
        "end_ms": row.get("end_ms"),
        "duration_ms": row.get("duration_ms"),
        "context_label": row.get("context_label"),
        "audio_path": row.get("audio_path"),
        "n_closures_v2": v2,
        "n_closures_period": p,
        "diff": p - v2,
        "agreement": int(abs(p - v2) <= 1),
        "period_ms_v2": row.get("mean_period_ms_v2"),
        "period_ms_period": result.period_ms,
        "regularity_period": result.regularity,
        "confidence_v2": row.get("confidence_v2"),
        "notes_period": "; ".join(result.notes),
    }


def _failed_row(row: pd.Series, reason: str) -> dict:
    return {
        "utt_id": row.get("utt_id"),
        "speaker_id": row.get("speaker_id"),
        "dataset": row.get("dataset"),
        "start_ms": row.get("start_ms"),
        "end_ms": row.get("end_ms"),
        "duration_ms": row.get("duration_ms"),
        "context_label": row.get("context_label"),
        "audio_path": row.get("audio_path"),
        "n_closures_v2": int(row["n_closures_v2"]),
        "n_closures_period": 0,
        "diff": -int(row["n_closures_v2"]),
        "agreement": 0,
        "period_ms_v2": row.get("mean_period_ms_v2"),
        "period_ms_period": float("nan"),
        "regularity_period": 0.0,
        "confidence_v2": row.get("confidence_v2"),
        "notes_period": reason,
    }
