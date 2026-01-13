"""Audio metadata extraction utilities."""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import struct
import logging

logger = logging.getLogger(__name__)


@dataclass
class AudioMetadata:
    """Container for audio file metadata."""
    file_path: str
    format: str  # 'wav', 'mp3', 'sam'
    sample_rate: int
    bit_depth: int
    channels: int
    duration_seconds: float
    file_size_bytes: int
    is_valid: bool
    error_message: Optional[str] = None


class AudioInspector:
    """Extract metadata from various audio formats."""

    def inspect(self, file_path: Path) -> AudioMetadata:
        """
        Inspect audio file and return metadata.

        Automatically detects format based on extension.
        """
        ext = file_path.suffix.lower()

        if ext == '.wav':
            return self.inspect_wav(file_path)
        elif ext == '.mp3':
            return self.inspect_mp3(file_path)
        elif ext in ['.ses', '.sam']:
            return self.inspect_sam_ses(file_path)
        else:
            return AudioMetadata(
                file_path=str(file_path),
                format='unknown',
                sample_rate=0,
                bit_depth=0,
                channels=0,
                duration_seconds=0.0,
                file_size_bytes=file_path.stat().st_size if file_path.exists() else 0,
                is_valid=False,
                error_message=f"Unknown audio format: {ext}"
            )

    def inspect_wav(self, file_path: Path) -> AudioMetadata:
        """Parse WAV header for metadata."""
        try:
            with open(file_path, 'rb') as f:
                # RIFF header
                riff = f.read(4)
                if riff != b'RIFF':
                    raise ValueError("Not a valid RIFF file")

                file_size = struct.unpack('<I', f.read(4))[0]
                wave = f.read(4)
                if wave != b'WAVE':
                    raise ValueError("Not a valid WAVE file")

                # Find fmt chunk
                sample_rate = 0
                bit_depth = 0
                channels = 0
                byte_rate = 0

                while True:
                    chunk_id = f.read(4)
                    if len(chunk_id) < 4:
                        break

                    chunk_size = struct.unpack('<I', f.read(4))[0]

                    if chunk_id == b'fmt ':
                        audio_format = struct.unpack('<H', f.read(2))[0]
                        channels = struct.unpack('<H', f.read(2))[0]
                        sample_rate = struct.unpack('<I', f.read(4))[0]
                        byte_rate = struct.unpack('<I', f.read(4))[0]
                        block_align = struct.unpack('<H', f.read(2))[0]
                        bit_depth = struct.unpack('<H', f.read(2))[0]
                        # Skip rest of fmt chunk
                        remaining = chunk_size - 16
                        if remaining > 0:
                            f.seek(remaining, 1)
                    elif chunk_id == b'data':
                        data_size = chunk_size
                        break
                    else:
                        f.seek(chunk_size, 1)

            file_size_bytes = file_path.stat().st_size
            duration = data_size / byte_rate if byte_rate > 0 else 0.0

            return AudioMetadata(
                file_path=str(file_path),
                format='wav',
                sample_rate=sample_rate,
                bit_depth=bit_depth,
                channels=channels,
                duration_seconds=duration,
                file_size_bytes=file_size_bytes,
                is_valid=True
            )

        except Exception as e:
            return AudioMetadata(
                file_path=str(file_path),
                format='wav',
                sample_rate=0,
                bit_depth=0,
                channels=0,
                duration_seconds=0.0,
                file_size_bytes=file_path.stat().st_size if file_path.exists() else 0,
                is_valid=False,
                error_message=str(e)
            )

    def inspect_mp3(self, file_path: Path) -> AudioMetadata:
        """Extract MP3 metadata using pydub."""
        try:
            from pydub import AudioSegment
            from pydub.utils import mediainfo

            # Get media info
            info = mediainfo(str(file_path))

            sample_rate = int(info.get('sample_rate', 0))
            channels = int(info.get('channels', 0))
            duration = float(info.get('duration', 0))
            bit_depth = 16  # MP3 is typically decoded to 16-bit

            return AudioMetadata(
                file_path=str(file_path),
                format='mp3',
                sample_rate=sample_rate,
                bit_depth=bit_depth,
                channels=channels,
                duration_seconds=duration,
                file_size_bytes=file_path.stat().st_size,
                is_valid=True
            )

        except ImportError:
            # Fallback: estimate from file size
            file_size = file_path.stat().st_size
            # Assume 128kbps, mono
            estimated_duration = (file_size * 8) / 128000

            return AudioMetadata(
                file_path=str(file_path),
                format='mp3',
                sample_rate=44100,  # Common MP3 rate
                bit_depth=16,
                channels=1,
                duration_seconds=estimated_duration,
                file_size_bytes=file_size,
                is_valid=True,
                error_message="pydub not available, using estimates"
            )

        except Exception as e:
            return AudioMetadata(
                file_path=str(file_path),
                format='mp3',
                sample_rate=0,
                bit_depth=0,
                channels=0,
                duration_seconds=0.0,
                file_size_bytes=file_path.stat().st_size if file_path.exists() else 0,
                is_valid=False,
                error_message=str(e)
            )

    def inspect_sam_ses(self, file_path: Path) -> AudioMetadata:
        """
        Parse ALBAYZIN SAM format (.SES files).

        SAM format: Raw 16-bit signed PCM, little-endian, mono, 16kHz.
        No header - file contains only samples.

        File size (bytes) = samples * 2
        Duration = samples / 16000
        """
        file_size = file_path.stat().st_size
        num_samples = file_size // 2
        duration = num_samples / 16000.0

        return AudioMetadata(
            file_path=str(file_path),
            format='sam',
            sample_rate=16000,
            bit_depth=16,
            channels=1,
            duration_seconds=duration,
            file_size_bytes=file_size,
            is_valid=True
        )

    def sample_directory(self, dir_path: Path, pattern: str,
                        sample_n: int = 10) -> List[AudioMetadata]:
        """
        Sample N files from directory to verify format consistency.

        Args:
            dir_path: Directory to search
            pattern: Glob pattern (e.g., '*.wav', '**/*.mp3')
            sample_n: Number of files to sample

        Returns:
            List of AudioMetadata for sampled files
        """
        files = list(dir_path.glob(pattern))[:sample_n]
        return [self.inspect(f) for f in files]
