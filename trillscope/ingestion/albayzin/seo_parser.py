"""SEO orthographic label parser for ALBAYZIN corpus."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import re


@dataclass
class PhonemeLabel:
    """Single phoneme label from SEO file."""
    sample_position: int
    phoneme: str


@dataclass
class SEORecord:
    """Parsed SEO (orthographic label) file."""
    version: str = ''              # LHD field
    label_type: str = ''           # TYP field
    database: str = ''             # DBN field
    source_file: str = ''          # SRC field
    sample_rate: int = 16000       # SAM field
    start_sample: int = 0          # BEG field
    end_sample: int = 0            # END field
    recording_date: str = ''       # RED field
    recording_time: str = ''       # RET field
    recording_place: str = ''      # REP field
    bytes_per_sample: int = 2      # SNB field
    sample_format: str = ''        # SBF field
    bits_per_sample: int = 16      # SSB field
    num_channels: int = 1          # NCH field
    orthographic_text: str = ''    # Combined LBR + EXT fields
    phoneme_labels: List[PhonemeLabel] = field(default_factory=list)  # LBA fields

    @property
    def duration_seconds(self) -> float:
        """Calculate duration in seconds."""
        samples = self.end_sample - self.start_sample
        return samples / self.sample_rate if self.sample_rate > 0 else 0.0

    @property
    def speaker_code(self) -> str:
        """Extract 2-letter speaker code from source filename."""
        if self.source_file:
            # Format: XXFA0001.SES where XX is speaker code
            return self.source_file[:2]
        return ''


class SEOParser:
    """
    Parse ALBAYZIN .SEO label files.

    Format example:
    LHD: V4.1
    TYP: orthographic
    DBN: ALBAYZIN
    SRC: BRFA0001.SES
    SAM: 16000
    BEG: 0
    END: 50415
    RED: 22/Apr/96
    RET: 17:34:03
    REP: MADRID
    SNB: 2
    SBF: 01
    SSB: 16
    NCH: 1
    LBR: 0, 50415,  0, -18043, 10775, Francia, Suiza y Hungr'ia...
    EXT: com'un.
    LBA: 4619, SIL
    LBA: 5917, f
    ...
    ELF:
    """

    # Mapping of legacy accent notation to Unicode
    ACCENT_MAP = {
        "'a": "á", "'e": "é", "'i": "í", "'o": "ó", "'u": "ú",
        "'A": "Á", "'E": "É", "'I": "Í", "'O": "Ó", "'U": "Ú",
        "~n": "ñ", "~N": "Ñ",
    }

    def parse(self, seo_path: Path) -> SEORecord:
        """
        Parse SEO file and return structured record.

        Args:
            seo_path: Path to .SEO file

        Returns:
            SEORecord with parsed data
        """
        record = SEORecord()
        lbr_text = ''
        ext_text = ''
        phoneme_labels = []

        with open(seo_path, 'r', encoding='latin-1', errors='replace') as f:
            for line in f:
                line = line.rstrip('\r\n')

                if not line or ':' not in line:
                    continue

                key, _, value = line.partition(':')
                key = key.strip()
                value = value.strip()

                if key == 'LHD':
                    record.version = value
                elif key == 'TYP':
                    record.label_type = value
                elif key == 'DBN':
                    record.database = value
                elif key == 'SRC':
                    record.source_file = value
                elif key == 'SAM':
                    record.sample_rate = self._parse_int(value, 16000)
                elif key == 'BEG':
                    record.start_sample = self._parse_int(value, 0)
                elif key == 'END':
                    record.end_sample = self._parse_int(value, 0)
                elif key == 'RED':
                    record.recording_date = value
                elif key == 'RET':
                    record.recording_time = value
                elif key == 'REP':
                    record.recording_place = value
                elif key == 'SNB':
                    record.bytes_per_sample = self._parse_int(value, 2)
                elif key == 'SBF':
                    record.sample_format = value
                elif key == 'SSB':
                    record.bits_per_sample = self._parse_int(value, 16)
                elif key == 'NCH':
                    record.num_channels = self._parse_int(value, 1)
                elif key == 'LBR':
                    lbr_text = self._parse_lbr(value)
                elif key == 'EXT':
                    ext_text = value
                elif key == 'LBA':
                    label = self._parse_lba(value)
                    if label:
                        phoneme_labels.append(label)

        # Combine LBR and EXT for full orthographic text
        full_text = lbr_text
        if ext_text:
            full_text = full_text.rstrip() + ' ' + ext_text if full_text else ext_text

        record.orthographic_text = self.normalize_text(full_text)
        record.phoneme_labels = phoneme_labels

        return record

    def _parse_int(self, value: str, default: int = 0) -> int:
        """Safely parse integer value."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _parse_lbr(self, value: str) -> str:
        """
        Parse LBR line to extract orthographic text.

        Format: start, end, gain, min, max, text
        Example: 0, 50415,  0, -18043, 10775, Francia, Suiza y Hungr'ia ya hicieron causa
        """
        parts = value.split(',', 5)  # Split into max 6 parts
        if len(parts) >= 6:
            return parts[5].strip()
        return ''

    def _parse_lba(self, value: str) -> Optional[PhonemeLabel]:
        """
        Parse LBA line to extract phoneme label.

        Format: sample_position, phoneme
        Example: 5917, f
        """
        parts = value.split(',')
        if len(parts) >= 2:
            try:
                sample_pos = int(parts[0].strip())
                phoneme = parts[1].strip()
                return PhonemeLabel(sample_position=sample_pos, phoneme=phoneme)
            except ValueError:
                pass
        return None

    def normalize_text(self, text: str) -> str:
        """
        Normalize ALBAYZIN accent notation to UTF-8.

        Examples:
        - 'a -> á  (acute accent marker)
        - ~n -> ñ  (tilde marker for ñ)
        """
        result = text
        for old, new in self.ACCENT_MAP.items():
            result = result.replace(old, new)
        return result

    def extract_words_with_r(self, record: SEORecord) -> List[Tuple[str, int, int]]:
        """
        Extract words containing 'r' from orthographic text.

        Returns:
            List of (word, start_char, end_char) tuples
        """
        text = record.orthographic_text
        words = []

        # Find all words
        for match in re.finditer(r'\b\w+\b', text, re.UNICODE):
            word = match.group()
            if 'r' in word.lower() or 'R' in word:
                words.append((word, match.start(), match.end()))

        return words

    def get_phoneme_at_time(self, record: SEORecord, sample_position: int) -> Optional[str]:
        """
        Get phoneme label at a given sample position.

        Args:
            record: Parsed SEO record
            sample_position: Sample position to query

        Returns:
            Phoneme label or None
        """
        if not record.phoneme_labels:
            return None

        # Find the phoneme whose interval contains this position
        for i, label in enumerate(record.phoneme_labels):
            # Next label's position marks the end of current phoneme
            if i + 1 < len(record.phoneme_labels):
                next_pos = record.phoneme_labels[i + 1].sample_position
                if label.sample_position <= sample_position < next_pos:
                    return label.phoneme
            else:
                # Last phoneme extends to end
                if label.sample_position <= sample_position:
                    return label.phoneme

        return None

    def find_trill_phonemes(self, record: SEORecord) -> List[PhonemeLabel]:
        """
        Find trill /r/ phonemes in the SEO record.

        ALBAYZIN uses 'rr' for trill and 'r' for tap.

        Returns:
            List of PhonemeLabel objects for trills
        """
        trills = []
        for label in record.phoneme_labels:
            # 'rr' is the trill phoneme in ALBAYZIN notation
            if label.phoneme.lower() == 'rr':
                trills.append(label)
        return trills
