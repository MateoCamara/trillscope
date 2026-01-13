"""Encoding detection and normalization utilities."""

from pathlib import Path
from typing import Tuple, Optional, List
import codecs
import logging

import chardet

logger = logging.getLogger(__name__)


class EncodingDetector:
    """Detects and normalizes text file encodings to UTF-8."""

    # Fallback chain for Spanish text files
    SPANISH_ENCODINGS = ['utf-8', 'iso-8859-1', 'iso-8859-15', 'cp1252', 'latin-1']

    def detect_encoding(self, file_path: Path, sample_size: int = 65536) -> Tuple[str, float]:
        """
        Detect file encoding with confidence score.

        Args:
            file_path: Path to text file
            sample_size: Bytes to sample for detection

        Returns:
            Tuple of (encoding_name, confidence_0_to_1)
        """
        with open(file_path, 'rb') as f:
            raw_data = f.read(sample_size)

        result = chardet.detect(raw_data)
        encoding = result.get('encoding', 'utf-8') or 'utf-8'
        confidence = result.get('confidence', 0.0) or 0.0

        return encoding, confidence

    def read_with_fallback(self, file_path: Path,
                          encodings: Optional[List[str]] = None) -> Tuple[str, str]:
        """
        Read file content with fallback encoding chain.

        Args:
            file_path: Path to text file
            encodings: List of encodings to try (defaults to SPANISH_ENCODINGS)

        Returns:
            Tuple of (content, detected_encoding)

        Raises:
            UnicodeDecodeError: If all encodings fail
        """
        if encodings is None:
            encodings = self.SPANISH_ENCODINGS

        # First try detected encoding
        detected, confidence = self.detect_encoding(file_path)
        if confidence > 0.8 and detected.lower() not in [e.lower() for e in encodings]:
            encodings = [detected] + encodings

        last_error = None
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                return content, encoding
            except (UnicodeDecodeError, LookupError) as e:
                last_error = e
                continue

        # If all fail, try with errors='replace'
        logger.warning(f"All encodings failed for {file_path}, using replacement characters")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return content, 'utf-8-replace'

    def normalize_to_utf8(self, content: str) -> str:
        """
        Normalize text content to UTF-8, handling Spanish special chars.

        Args:
            content: Text content to normalize

        Returns:
            UTF-8 normalized string
        """
        # Handle common legacy accent notations
        replacements = {
            "'a": "á", "'e": "é", "'i": "í", "'o": "ó", "'u": "ú",
            "'A": "Á", "'E": "É", "'I": "Í", "'O": "Ó", "'U": "Ú",
            "~n": "ñ", "~N": "Ñ",
            "?'": "¿",
            "!'": "¡",
        }

        for old, new in replacements.items():
            content = content.replace(old, new)

        return content

    def detect_line_endings(self, file_path: Path) -> str:
        """
        Detect line ending style in a file.

        Returns:
            'CRLF', 'LF', 'CR', or 'mixed'
        """
        with open(file_path, 'rb') as f:
            content = f.read(8192)

        has_crlf = b'\r\n' in content
        has_lf = b'\n' in content and not has_crlf
        has_cr = b'\r' in content and b'\r\n' not in content

        if has_crlf and not has_lf and not has_cr:
            return 'CRLF'
        elif has_lf and not has_cr:
            return 'LF'
        elif has_cr and not has_lf:
            return 'CR'
        else:
            return 'mixed'
