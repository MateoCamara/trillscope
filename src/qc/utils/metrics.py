"""Audio metric calculations for QC."""

from pathlib import Path
from typing import Tuple
import struct
import logging

import numpy as np

logger = logging.getLogger(__name__)

# 16-bit signed PCM max value
MAX_16BIT = 32767


def load_wav_samples(audio_path: Path) -> Tuple[np.ndarray, int]:
    """
    Load WAV file and return normalized samples.

    Reads WAV files directly using struct for minimal dependencies.
    Expects 16kHz, 16-bit, mono WAV files (normalized format from Block B).

    Args:
        audio_path: Path to WAV file

    Returns:
        Tuple of (samples as float32 normalized to [-1, 1], sample_rate)

    Raises:
        ValueError: If file is not a valid WAV or has unsupported format
    """
    audio_path = Path(audio_path)

    with open(audio_path, 'rb') as f:
        # Read RIFF header
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError(f"Invalid WAV file: missing RIFF header in {audio_path}")

        f.read(4)  # file size
        wave = f.read(4)
        if wave != b'WAVE':
            raise ValueError(f"Invalid WAV file: missing WAVE marker in {audio_path}")

        # Find fmt and data chunks
        sample_rate = 0
        channels = 0
        bit_depth = 0
        data_bytes = b''

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break

            chunk_size = struct.unpack('<I', f.read(4))[0]

            if chunk_id == b'fmt ':
                audio_format = struct.unpack('<H', f.read(2))[0]
                if audio_format != 1:  # PCM
                    raise ValueError(f"Unsupported audio format {audio_format} in {audio_path}")
                channels = struct.unpack('<H', f.read(2))[0]
                sample_rate = struct.unpack('<I', f.read(4))[0]
                f.read(4)  # byte rate
                f.read(2)  # block align
                bit_depth = struct.unpack('<H', f.read(2))[0]
                remaining = chunk_size - 16
                if remaining > 0:
                    f.seek(remaining, 1)
            elif chunk_id == b'data':
                data_bytes = f.read(chunk_size)
                break
            else:
                f.seek(chunk_size, 1)

    if not data_bytes:
        raise ValueError(f"No audio data found in {audio_path}")

    if bit_depth != 16:
        raise ValueError(f"Expected 16-bit audio, got {bit_depth}-bit in {audio_path}")

    # Convert bytes to numpy array
    samples = np.frombuffer(data_bytes, dtype=np.int16)

    # Handle stereo by averaging channels
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
        logger.warning(f"Converted {channels}-channel audio to mono: {audio_path}")

    # Normalize to [-1, 1]
    samples_float = samples.astype(np.float32) / MAX_16BIT

    return samples_float, sample_rate


def calculate_clipping(
    samples: np.ndarray,
    near_max_fraction: float = 0.99
) -> Tuple[float, int, int]:
    """
    Calculate clipping percentage.

    Samples are considered "clipped" if they are at or above the near_max_fraction
    of the maximum possible value (1.0 for normalized samples).

    Args:
        samples: Audio samples normalized to [-1, 1]
        near_max_fraction: Threshold fraction of max value to consider "clipped"

    Returns:
        Tuple of (clipping_percentage, num_clipped_samples, total_samples)
    """
    threshold = near_max_fraction
    num_clipped = np.sum(np.abs(samples) >= threshold)
    total = len(samples)

    clipping_pct = (num_clipped / total * 100) if total > 0 else 0.0

    return clipping_pct, int(num_clipped), total


def calculate_rms_db(samples: np.ndarray, ref: float = 1.0) -> float:
    """
    Calculate RMS level in dB.

    Args:
        samples: Audio samples normalized to [-1, 1]
        ref: Reference value for dB calculation

    Returns:
        RMS level in dB (20 * log10(rms / ref))
        Returns -100.0 for silence (to avoid -inf)
    """
    rms = np.sqrt(np.mean(samples ** 2))

    if rms < 1e-10:
        return -100.0

    return 20 * np.log10(rms / ref)


def calculate_frame_energies(
    samples: np.ndarray,
    sample_rate: int,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0
) -> np.ndarray:
    """
    Calculate frame-wise energy levels in dB.

    Args:
        samples: Audio samples
        sample_rate: Sample rate
        frame_size_ms: Frame length in milliseconds
        hop_size_ms: Hop length in milliseconds

    Returns:
        Array of frame energies in dB
    """
    frame_size = int(sample_rate * frame_size_ms / 1000)
    hop_size = int(sample_rate * hop_size_ms / 1000)

    if len(samples) < frame_size:
        # Audio too short, return single energy value
        energy = np.mean(samples ** 2)
        if energy < 1e-20:
            return np.array([-100.0])
        return np.array([10 * np.log10(energy)])

    # Calculate number of frames
    num_frames = 1 + (len(samples) - frame_size) // hop_size
    energies = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = samples[start:end]
        energy = np.mean(frame ** 2)
        if energy < 1e-20:
            energies[i] = -100.0
        else:
            energies[i] = 10 * np.log10(energy)

    return energies


def calculate_silence_percentage(
    frame_energies_db: np.ndarray,
    threshold_db: float = -40.0
) -> float:
    """
    Calculate percentage of frames classified as silence.

    Args:
        frame_energies_db: Frame energies in dB
        threshold_db: Energy threshold below which a frame is silence

    Returns:
        Percentage of silent frames (0-100)
    """
    if len(frame_energies_db) == 0:
        return 0.0

    silent_frames = np.sum(frame_energies_db < threshold_db)
    return (silent_frames / len(frame_energies_db)) * 100
