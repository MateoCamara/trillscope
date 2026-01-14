"""Audio conversion utilities for structure normalization."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import struct
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of audio conversion."""
    success: bool
    output_path: str
    duration_ms: float
    sample_rate: int
    issues: List[str] = field(default_factory=list)


class AudioConverter:
    """
    Convert audio files to normalized format.

    Target format: WAV mono, 16 kHz, 16-bit PCM.
    """

    TARGET_SAMPLE_RATE = 16000
    TARGET_BIT_DEPTH = 16
    TARGET_CHANNELS = 1

    def convert(self, input_path: Path, output_path: Path,
                skip_existing: bool = False) -> ConversionResult:
        """
        Convert audio file to normalized WAV format.

        Automatically detects input format based on extension.

        Args:
            input_path: Path to input audio file
            output_path: Path for output WAV file
            skip_existing: Skip if output already exists

        Returns:
            ConversionResult with status and metadata
        """
        if skip_existing and output_path.exists():
            # Read existing file to get duration
            try:
                duration_ms = self._get_wav_duration_ms(output_path)
                return ConversionResult(
                    success=True,
                    output_path=str(output_path),
                    duration_ms=duration_ms,
                    sample_rate=self.TARGET_SAMPLE_RATE,
                    issues=['skipped_existing']
                )
            except Exception as e:
                logger.warning(f"Could not read existing file {output_path}: {e}")

        ext = input_path.suffix.lower()

        if ext in ['.ses', '.sam']:
            return self.convert_sam_to_wav(input_path, output_path)
        elif ext == '.mp3':
            return self.convert_mp3_to_wav(input_path, output_path)
        elif ext == '.wav':
            return self.convert_wav_to_wav(input_path, output_path)
        else:
            return ConversionResult(
                success=False,
                output_path='',
                duration_ms=0,
                sample_rate=0,
                issues=[f"Unsupported audio format: {ext}"]
            )

    def convert_sam_to_wav(self, input_path: Path, output_path: Path) -> ConversionResult:
        """
        Convert ALBAYZIN SAM/SES format to WAV.

        SAM format: Raw 16-bit signed PCM, little-endian, mono, 16kHz.
        No header - file contains only audio samples.
        """
        issues = []

        try:
            # Read raw PCM data
            with open(input_path, 'rb') as f:
                pcm_data = f.read()

            num_samples = len(pcm_data) // 2
            duration_seconds = num_samples / self.TARGET_SAMPLE_RATE
            duration_ms = duration_seconds * 1000

            # Create WAV file with header
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                # Write WAV header
                self._write_wav_header(
                    f,
                    num_samples=num_samples,
                    sample_rate=self.TARGET_SAMPLE_RATE,
                    bit_depth=self.TARGET_BIT_DEPTH,
                    channels=self.TARGET_CHANNELS
                )
                # Write PCM data
                f.write(pcm_data)

            return ConversionResult(
                success=True,
                output_path=str(output_path),
                duration_ms=duration_ms,
                sample_rate=self.TARGET_SAMPLE_RATE,
                issues=issues
            )

        except Exception as e:
            logger.error(f"SAM conversion failed for {input_path}: {e}")
            return ConversionResult(
                success=False,
                output_path='',
                duration_ms=0,
                sample_rate=0,
                issues=[f"SAM conversion error: {str(e)}"]
            )

    def convert_mp3_to_wav(self, input_path: Path, output_path: Path) -> ConversionResult:
        """
        Convert MP3 to WAV with resampling to 16kHz mono.

        Uses pydub with ffmpeg backend.
        """
        issues = []

        try:
            from pydub import AudioSegment

            # Load MP3
            audio = AudioSegment.from_mp3(str(input_path))

            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
                issues.append('converted_to_mono')

            # Resample to 16kHz
            if audio.frame_rate != self.TARGET_SAMPLE_RATE:
                original_rate = audio.frame_rate
                audio = audio.set_frame_rate(self.TARGET_SAMPLE_RATE)
                issues.append(f'resampled_from_{original_rate}')

            # Set sample width to 16-bit
            if audio.sample_width != 2:
                audio = audio.set_sample_width(2)
                issues.append('converted_to_16bit')

            duration_ms = len(audio)

            # Export as WAV
            output_path.parent.mkdir(parents=True, exist_ok=True)
            audio.export(str(output_path), format='wav')

            return ConversionResult(
                success=True,
                output_path=str(output_path),
                duration_ms=duration_ms,
                sample_rate=self.TARGET_SAMPLE_RATE,
                issues=issues
            )

        except ImportError:
            return ConversionResult(
                success=False,
                output_path='',
                duration_ms=0,
                sample_rate=0,
                issues=['pydub not installed - cannot convert MP3']
            )

        except Exception as e:
            logger.error(f"MP3 conversion failed for {input_path}: {e}")
            return ConversionResult(
                success=False,
                output_path='',
                duration_ms=0,
                sample_rate=0,
                issues=[f"MP3 conversion error: {str(e)}"]
            )

    def convert_wav_to_wav(self, input_path: Path, output_path: Path) -> ConversionResult:
        """
        Normalize existing WAV file (resample/rechannelize if needed).

        If already 16kHz mono 16-bit, just copy.
        """
        issues = []

        try:
            # First check if conversion is needed
            needs_conversion = False
            with open(input_path, 'rb') as f:
                # Read WAV header
                riff = f.read(4)
                if riff != b'RIFF':
                    return ConversionResult(
                        success=False,
                        output_path='',
                        duration_ms=0,
                        sample_rate=0,
                        issues=['Invalid WAV file: missing RIFF header']
                    )

                f.read(4)  # file size
                wave = f.read(4)
                if wave != b'WAVE':
                    return ConversionResult(
                        success=False,
                        output_path='',
                        duration_ms=0,
                        sample_rate=0,
                        issues=['Invalid WAV file: missing WAVE marker']
                    )

                # Find fmt chunk
                sample_rate = 0
                channels = 0
                bit_depth = 0

                while True:
                    chunk_id = f.read(4)
                    if len(chunk_id) < 4:
                        break

                    chunk_size = struct.unpack('<I', f.read(4))[0]

                    if chunk_id == b'fmt ':
                        f.read(2)  # audio format
                        channels = struct.unpack('<H', f.read(2))[0]
                        sample_rate = struct.unpack('<I', f.read(4))[0]
                        f.read(4)  # byte rate
                        f.read(2)  # block align
                        bit_depth = struct.unpack('<H', f.read(2))[0]
                        remaining = chunk_size - 16
                        if remaining > 0:
                            f.seek(remaining, 1)
                    elif chunk_id == b'data':
                        break
                    else:
                        f.seek(chunk_size, 1)

            # Check if conversion needed
            if sample_rate != self.TARGET_SAMPLE_RATE:
                needs_conversion = True
                issues.append(f'resampled_from_{sample_rate}')
            if channels != self.TARGET_CHANNELS:
                needs_conversion = True
                issues.append('converted_to_mono')
            if bit_depth != self.TARGET_BIT_DEPTH:
                needs_conversion = True
                issues.append(f'converted_to_16bit_from_{bit_depth}bit')

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if not needs_conversion:
                # Just copy the file
                import shutil
                shutil.copy2(input_path, output_path)
                duration_ms = self._get_wav_duration_ms(output_path)
                return ConversionResult(
                    success=True,
                    output_path=str(output_path),
                    duration_ms=duration_ms,
                    sample_rate=self.TARGET_SAMPLE_RATE,
                    issues=['already_normalized']
                )

            # Need to convert - use pydub
            from pydub import AudioSegment

            audio = AudioSegment.from_wav(str(input_path))

            if audio.channels > 1:
                audio = audio.set_channels(1)

            if audio.frame_rate != self.TARGET_SAMPLE_RATE:
                audio = audio.set_frame_rate(self.TARGET_SAMPLE_RATE)

            if audio.sample_width != 2:
                audio = audio.set_sample_width(2)

            duration_ms = len(audio)
            audio.export(str(output_path), format='wav')

            return ConversionResult(
                success=True,
                output_path=str(output_path),
                duration_ms=duration_ms,
                sample_rate=self.TARGET_SAMPLE_RATE,
                issues=issues
            )

        except Exception as e:
            logger.error(f"WAV conversion failed for {input_path}: {e}")
            return ConversionResult(
                success=False,
                output_path='',
                duration_ms=0,
                sample_rate=0,
                issues=[f"WAV conversion error: {str(e)}"]
            )

    def _write_wav_header(self, f, num_samples: int, sample_rate: int,
                          bit_depth: int, channels: int) -> None:
        """Write WAV file header."""
        bytes_per_sample = bit_depth // 8
        byte_rate = sample_rate * channels * bytes_per_sample
        block_align = channels * bytes_per_sample
        data_size = num_samples * bytes_per_sample
        file_size = 36 + data_size

        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', file_size))
        f.write(b'WAVE')

        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # chunk size
        f.write(struct.pack('<H', 1))   # audio format (PCM)
        f.write(struct.pack('<H', channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', byte_rate))
        f.write(struct.pack('<H', block_align))
        f.write(struct.pack('<H', bit_depth))

        # data chunk header
        f.write(b'data')
        f.write(struct.pack('<I', data_size))

    def _get_wav_duration_ms(self, wav_path: Path) -> float:
        """Get duration of WAV file in milliseconds."""
        try:
            with open(wav_path, 'rb') as f:
                f.read(4)  # RIFF
                f.read(4)  # file size
                f.read(4)  # WAVE

                byte_rate = 0
                data_size = 0

                while True:
                    chunk_id = f.read(4)
                    if len(chunk_id) < 4:
                        break

                    chunk_size = struct.unpack('<I', f.read(4))[0]

                    if chunk_id == b'fmt ':
                        f.read(2)  # audio format
                        f.read(2)  # channels
                        f.read(4)  # sample rate
                        byte_rate = struct.unpack('<I', f.read(4))[0]
                        remaining = chunk_size - 12
                        if remaining > 0:
                            f.seek(remaining, 1)
                    elif chunk_id == b'data':
                        data_size = chunk_size
                        break
                    else:
                        f.seek(chunk_size, 1)

                if byte_rate > 0:
                    return (data_size / byte_rate) * 1000
                return 0

        except Exception:
            return 0
