"""Audio loading utilities for acoustic measurement."""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
import struct

logger = logging.getLogger(__name__)


def load_audio_segment(
    audio_path: str,
    start_ms: float,
    end_ms: float,
    target_sr: int = 16000,
    context_ms: float = 50.0
) -> Tuple[Optional[np.ndarray], int]:
    """
    Load audio segment with optional context padding.

    Args:
        audio_path: Path to audio file (WAV, SES, or MP3)
        start_ms: Segment start time in milliseconds
        end_ms: Segment end time in milliseconds
        target_sr: Target sample rate
        context_ms: Extra context before/after segment (for analysis stability)

    Returns:
        Tuple of (audio_array, sample_rate) or (None, 0) on error
    """
    path = Path(audio_path)

    if not path.exists():
        logger.warning(f"Audio file not found: {audio_path}")
        return None, 0

    try:
        suffix = path.suffix.lower()

        if suffix == '.wav':
            return _load_wav_segment(path, start_ms, end_ms, target_sr, context_ms)
        elif suffix in ('.ses', '.sam'):
            return _load_ses_segment(path, start_ms, end_ms, context_ms)
        elif suffix == '.mp3':
            return _load_mp3_segment(path, start_ms, end_ms, target_sr, context_ms)
        else:
            logger.warning(f"Unsupported audio format: {suffix}")
            return None, 0

    except Exception as e:
        logger.error(f"Error loading audio {audio_path}: {e}")
        return None, 0


def _load_wav_segment(
    path: Path,
    start_ms: float,
    end_ms: float,
    target_sr: int,
    context_ms: float
) -> Tuple[Optional[np.ndarray], int]:
    """Load segment from WAV file."""
    try:
        import soundfile as sf

        # Get file info
        info = sf.info(str(path))
        sr = info.samplerate
        total_frames = info.frames

        # Calculate frame indices with context
        start_with_ctx = max(0, start_ms - context_ms)
        end_with_ctx = end_ms + context_ms

        start_frame = int(start_with_ctx * sr / 1000)
        end_frame = int(end_with_ctx * sr / 1000)
        end_frame = min(end_frame, total_frames)

        # Read segment
        audio, sr = sf.read(str(path), start=start_frame, stop=end_frame)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != target_sr:
            audio = _resample(audio, sr, target_sr)
            sr = target_sr

        return audio.astype(np.float32), sr

    except ImportError:
        logger.warning("soundfile not installed, trying scipy")
        return _load_wav_scipy(path, start_ms, end_ms, target_sr, context_ms)
    except Exception as e:
        logger.error(f"Error loading WAV {path}: {e}")
        return None, 0


def _load_wav_scipy(
    path: Path,
    start_ms: float,
    end_ms: float,
    target_sr: int,
    context_ms: float
) -> Tuple[Optional[np.ndarray], int]:
    """Load WAV using scipy as fallback."""
    try:
        from scipy.io import wavfile

        sr, audio = wavfile.read(str(path))

        # Convert to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Extract segment with context
        start_with_ctx = max(0, start_ms - context_ms)
        end_with_ctx = end_ms + context_ms

        start_sample = int(start_with_ctx * sr / 1000)
        end_sample = int(end_with_ctx * sr / 1000)
        end_sample = min(end_sample, len(audio))

        audio = audio[start_sample:end_sample]

        # Resample if needed
        if sr != target_sr:
            audio = _resample(audio, sr, target_sr)
            sr = target_sr

        return audio.astype(np.float32), sr

    except Exception as e:
        logger.error(f"Error loading WAV with scipy {path}: {e}")
        return None, 0


def _load_ses_segment(
    path: Path,
    start_ms: float,
    end_ms: float,
    context_ms: float
) -> Tuple[Optional[np.ndarray], int]:
    """
    Load segment from SES/SAM file (raw 16-bit PCM, 16kHz mono).

    ALBAYZIN SES format: raw 16-bit signed PCM, little-endian, 16kHz, mono
    """
    sr = 16000  # SES files are always 16kHz

    try:
        # Calculate sample indices with context
        start_with_ctx = max(0, start_ms - context_ms)
        end_with_ctx = end_ms + context_ms

        start_sample = int(start_with_ctx * sr / 1000)
        end_sample = int(end_with_ctx * sr / 1000)

        # Read raw bytes
        with open(path, 'rb') as f:
            # Seek to start position (2 bytes per sample)
            f.seek(start_sample * 2)

            # Read segment
            num_samples = end_sample - start_sample
            data = f.read(num_samples * 2)

        # Unpack as 16-bit signed integers
        num_samples_read = len(data) // 2
        samples = struct.unpack(f'<{num_samples_read}h', data)

        # Convert to float32 normalized to [-1, 1]
        audio = np.array(samples, dtype=np.float32) / 32768.0

        return audio, sr

    except Exception as e:
        logger.error(f"Error loading SES {path}: {e}")
        return None, 0


def _load_mp3_segment(
    path: Path,
    start_ms: float,
    end_ms: float,
    target_sr: int,
    context_ms: float
) -> Tuple[Optional[np.ndarray], int]:
    """Load segment from MP3 file."""
    try:
        from pydub import AudioSegment

        # Load full MP3
        audio_seg = AudioSegment.from_mp3(str(path))

        # Extract segment with context (pydub uses ms)
        start_with_ctx = max(0, start_ms - context_ms)
        end_with_ctx = min(end_ms + context_ms, len(audio_seg))

        segment = audio_seg[start_with_ctx:end_with_ctx]

        # Convert to mono
        segment = segment.set_channels(1)

        # Get sample rate and resample if needed
        sr = segment.frame_rate
        if sr != target_sr:
            segment = segment.set_frame_rate(target_sr)
            sr = target_sr

        # Convert to numpy array
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)

        # Normalize to [-1, 1]
        if segment.sample_width == 2:  # 16-bit
            samples = samples / 32768.0
        elif segment.sample_width == 4:  # 32-bit
            samples = samples / 2147483648.0

        return samples, sr

    except ImportError:
        logger.warning("pydub not installed, trying soundfile")
        # Try soundfile as fallback (works for some MP3s)
        try:
            return _load_wav_segment(path, start_ms, end_ms, target_sr, context_ms)
        except Exception:
            return None, 0
    except Exception as e:
        logger.error(f"Error loading MP3 {path}: {e}")
        return None, 0


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio

    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Simple linear interpolation as fallback
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)


def load_full_audio(
    audio_path: str,
    target_sr: int = 16000
) -> Tuple[Optional[np.ndarray], int]:
    """
    Load full audio file (for batch processing).

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Tuple of (audio_array, sample_rate) or (None, 0) on error
    """
    path = Path(audio_path)

    if not path.exists():
        return None, 0

    try:
        suffix = path.suffix.lower()

        if suffix == '.wav':
            import soundfile as sf
            audio, sr = sf.read(str(path))
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != target_sr:
                audio = _resample(audio, sr, target_sr)
                sr = target_sr
            return audio.astype(np.float32), sr

        elif suffix in ('.ses', '.sam'):
            sr = 16000
            with open(path, 'rb') as f:
                data = f.read()
            num_samples = len(data) // 2
            samples = struct.unpack(f'<{num_samples}h', data)
            audio = np.array(samples, dtype=np.float32) / 32768.0
            return audio, sr

        elif suffix == '.mp3':
            from pydub import AudioSegment
            audio_seg = AudioSegment.from_mp3(str(path))
            audio_seg = audio_seg.set_channels(1).set_frame_rate(target_sr)
            samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
            if audio_seg.sample_width == 2:
                samples = samples / 32768.0
            return samples, target_sr

        else:
            return None, 0

    except Exception as e:
        logger.error(f"Error loading audio {audio_path}: {e}")
        return None, 0
