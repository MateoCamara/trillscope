"""Transcript normalization utilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class NormalizedTranscript:
    """Normalized transcript with optional timing."""
    utt_id: str
    text: str
    encoding: str = 'utf-8'
    has_timing: bool = False
    phoneme_labels: Optional[List[Dict[str, Any]]] = None
    source_format: str = 'unknown'


class TranscriptNormalizer:
    """Normalize transcripts to consistent UTF-8 format."""

    # ALBAYZIN accent notation mapping
    ACCENT_MAP = {
        "'a": "á", "'e": "é", "'i": "í", "'o": "ó", "'u": "ú",
        "'A": "Á", "'E": "É", "'I": "Í", "'O": "Ó", "'U": "Ú",
        "~n": "ñ", "~N": "Ñ",
    }

    def normalize_seo_to_json(self, seo_path: Path, output_path: Path,
                              utt_id: str) -> NormalizedTranscript:
        """
        Normalize ALBAYZIN SEO transcript to JSON with timing.

        Args:
            seo_path: Path to .SEO file
            output_path: Path for output .json file
            utt_id: Utterance ID

        Returns:
            NormalizedTranscript with text and phoneme timing
        """
        try:
            with open(seo_path, 'r', encoding='latin-1', errors='replace') as f:
                content = f.read()

            # Parse SEO fields
            lbr_text = ''
            ext_text = ''
            phoneme_labels = []

            for line in content.split('\n'):
                line = line.rstrip('\r\n')
                if not line or ':' not in line:
                    continue

                key, _, value = line.partition(':')
                key = key.strip()
                value = value.strip()

                if key == 'LBR':
                    # Format: start, end, gain, min, max, text
                    parts = value.split(',', 5)
                    if len(parts) >= 6:
                        lbr_text = parts[5].strip()

                elif key == 'EXT':
                    ext_text = value

                elif key == 'LBA':
                    # Format: sample_position, phoneme
                    parts = value.split(',')
                    if len(parts) >= 2:
                        try:
                            sample_pos = int(parts[0].strip())
                            phoneme = parts[1].strip()
                            phoneme_labels.append({
                                'start_sample': sample_pos,
                                'phoneme': phoneme
                            })
                        except ValueError:
                            pass

            # Combine LBR and EXT
            full_text = lbr_text
            if ext_text:
                full_text = full_text.rstrip() + ' ' + ext_text if full_text else ext_text

            # Normalize accents
            text = self._normalize_albayzin_text(full_text)

            # Create normalized transcript
            transcript = NormalizedTranscript(
                utt_id=utt_id,
                text=text,
                encoding='utf-8',
                has_timing=len(phoneme_labels) > 0,
                phoneme_labels=phoneme_labels if phoneme_labels else None,
                source_format='seo'
            )

            # Save as JSON
            self._save_as_json(transcript, output_path)

            return transcript

        except Exception as e:
            logger.error(f"Failed to normalize SEO {seo_path}: {e}")
            return NormalizedTranscript(
                utt_id=utt_id,
                text='',
                has_timing=False,
                source_format='seo'
            )

    def normalize_xml_to_txt(self, xml_path: Path, output_path: Path,
                             utt_id: str) -> NormalizedTranscript:
        """
        Normalize PRESEEA XML transcript to plain text.

        Args:
            xml_path: Path to XML transcript file
            output_path: Path for output .txt file
            utt_id: Utterance ID

        Returns:
            NormalizedTranscript with plain text (no timing)
        """
        try:
            # Read with encoding detection
            content = self._read_file_with_encoding(xml_path)

            # Extract text from XML
            text = self._extract_preseea_text(content)

            # Create normalized transcript
            transcript = NormalizedTranscript(
                utt_id=utt_id,
                text=text,
                encoding='utf-8',
                has_timing=False,
                phoneme_labels=None,
                source_format='xml'
            )

            # Save as plain text
            self._save_as_txt(transcript, output_path)

            return transcript

        except Exception as e:
            logger.error(f"Failed to normalize XML {xml_path}: {e}")
            return NormalizedTranscript(
                utt_id=utt_id,
                text='',
                has_timing=False,
                source_format='xml'
            )

    def _normalize_albayzin_text(self, text: str) -> str:
        """Normalize ALBAYZIN accent notation to UTF-8."""
        result = text
        for old, new in self.ACCENT_MAP.items():
            result = result.replace(old, new)
        return result.strip()

    def _read_file_with_encoding(self, path: Path) -> str:
        """Read file with encoding detection."""
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # Last resort
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    def _extract_preseea_text(self, content: str) -> str:
        """
        Extract plain text from PRESEEA XML content.

        Removes XML tags but keeps text content.
        Focuses on informant speech (I:) rather than interviewer (E:).
        """
        # Find body after </Trans> or </Hablantes>
        body_start = content.find('</Trans>')
        if body_start == -1:
            body_start = content.find('</Hablantes>')
        if body_start == -1:
            body_start = 0

        body = content[body_start:]

        # Extract all turns
        text_parts = []
        current_speaker = None
        current_text = []

        for line in body.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Check for speaker turn
            turn_match = re.match(r'^([IE]):\s*(.*)$', line)
            if turn_match:
                # Save previous turn if it's from informant
                if current_speaker == 'I' and current_text:
                    text_parts.append(' '.join(current_text))

                current_speaker = turn_match.group(1)
                current_text = [turn_match.group(2)]
            elif current_speaker:
                current_text.append(line)

        # Don't forget last turn
        if current_speaker == 'I' and current_text:
            text_parts.append(' '.join(current_text))

        # Join all informant text
        full_text = ' '.join(text_parts)

        # Clean XML tags
        full_text = self._clean_xml_tags(full_text)

        return full_text.strip()

    def _clean_xml_tags(self, text: str) -> str:
        """Remove XML tags from text, keeping content."""
        # Remove self-closing tags
        text = re.sub(r'<[^>]+/>', '', text)
        # Remove paired tags but keep content
        text = re.sub(r'<simult[áa]neo>|</simult[áa]neo>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<simultaneo>|</simultaneo>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<cita>|</cita>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<extranjero>|</extranjero>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<risas>|</risas>', '', text, flags=re.IGNORECASE)
        # Remove any remaining tags
        text = re.sub(r'<[^>]+>', '', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _save_as_json(self, transcript: NormalizedTranscript, output_path: Path) -> None:
        """Save transcript as JSON with timing information."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'utt_id': transcript.utt_id,
            'text': transcript.text,
            'encoding': transcript.encoding,
            'has_timing': transcript.has_timing,
            'source_format': transcript.source_format
        }

        if transcript.phoneme_labels:
            data['phoneme_labels'] = transcript.phoneme_labels

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_as_txt(self, transcript: NormalizedTranscript, output_path: Path) -> None:
        """Save transcript as plain text."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript.text)
