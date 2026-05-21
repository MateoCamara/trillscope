"""Audio loading that copes with the corpus formats we have to deal with.

Albayzin ships raw-PCM .SES files that no general decoder reads; MP3 (PRESEEA)
needs an audioread/pydub backend; everything else is plain WAV. This module
dispatches by suffix.
"""

from __future__ import annotations

import struct
from pathlib import Path

import librosa
import numpy as np


SES_SAMPLE_RATE = 16_000  # ALBAYZIN .SES = raw 16-bit signed PCM, little-endian, mono


def load_audio_full(audio_path: str | Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load an audio file in full, mono float32, resampled to `target_sr`.

    Raises FileNotFoundError if the path does not exist and any decoder-specific
    error if the file is malformed.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()

    if suffix in (".ses", ".sam"):
        with open(path, "rb") as fh:
            raw = fh.read()
        n = len(raw) // 2
        samples = struct.unpack(f"<{n}h", raw[: n * 2])
        audio = np.asarray(samples, dtype=np.float32) / 32768.0
        sr = SES_SAMPLE_RATE
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return audio, sr

    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32), sr
