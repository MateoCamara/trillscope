"""DBF metadata parser for ALBAYZIN corpus.

Note: ALBAYZIN DBF files are text-formatted, not binary dBASE format.
"""

from pathlib import Path
from typing import List, Dict, Optional
import re
from datetime import datetime

from ..base import SpeakerMetadata


class AlbayzinDBFParser:
    """
    Parse ALBAYZIN speaker metadata from text-formatted DBF files.

    File structure (from SPEAKER.DBF):
    - SCD: Speaker code (2 letters, e.g., 'AA')
    - SNM: Surname
    - SBN: First name
    - SEX: Sex (F/M)
    - DOB: Date of birth (DD/MM/YYYY or D/M/YYYY)
    - LNA: Place of birth
    - LRE: Place of residence
    - LRA: Previous residence
    - EPR: Education/profession
    - LNP: Father's birthplace
    - LNM: Mother's birthplace
    - PAU: Hearing difficulties (NO/YES)

    Records are separated by lines starting with '-'.
    """

    FIELD_KEYS = ['SCD', 'SNM', 'SBN', 'SEX', 'DOB', 'LNA', 'LRE', 'LRA', 'EPR', 'LNP', 'LNM', 'PAU']

    def parse_speaker_dbf(self, dbf_path: Path) -> List[SpeakerMetadata]:
        """
        Parse SPEAKER.DBF file.

        Args:
            dbf_path: Path to SPEAKER.DBF file

        Returns:
            List of SpeakerMetadata objects
        """
        speakers = []
        current_record: Dict[str, str] = {}

        with open(dbf_path, 'r', encoding='latin-1', errors='replace') as f:
            for line in f:
                line = line.rstrip('\r\n')

                # Record separator
                if line.startswith('-') and current_record:
                    speaker = self._record_to_speaker(current_record)
                    if speaker:
                        speakers.append(speaker)
                    current_record = {}
                    continue

                # Skip header/documentation lines
                if not line or line.startswith('*') or line.startswith(' ') and ':' not in line:
                    continue

                # Parse key-value pair
                if ':' in line:
                    key, _, value = line.partition(':')
                    key = key.strip()
                    value = value.strip()

                    if key in self.FIELD_KEYS:
                        current_record[key] = value

        # Don't forget last record
        if current_record:
            speaker = self._record_to_speaker(current_record)
            if speaker:
                speakers.append(speaker)

        return speakers

    def _record_to_speaker(self, record: Dict[str, str]) -> Optional[SpeakerMetadata]:
        """Convert parsed record dict to SpeakerMetadata."""
        speaker_code = record.get('SCD', '').strip()
        if not speaker_code:
            return None

        # Parse date of birth to extract age
        dob = record.get('DOB', '')
        age = self._extract_age_from_dob(dob, recording_year=1996)

        return SpeakerMetadata(
            speaker_id=speaker_code,
            sex=self._normalize_sex(record.get('SEX', '')),
            age=age,
            birth_date=dob,
            birth_place=record.get('LNA', '').strip() or None,
            residence=record.get('LRE', '').strip() or None,
            education=record.get('EPR', '').strip() or None,
            profession=record.get('EPR', '').strip() or None,  # Same field in ALBAYZIN
            extra_fields={
                'surname': record.get('SNM', '').strip(),
                'first_name': record.get('SBN', '').strip(),
                'previous_residence': record.get('LRA', '').strip(),
                'father_birthplace': record.get('LNP', '').strip(),
                'mother_birthplace': record.get('LNM', '').strip(),
                'hearing_difficulties': record.get('PAU', '').strip(),
            }
        )

    def _normalize_sex(self, value: str) -> Optional[str]:
        """Normalize sex value to F/M/None."""
        value = value.strip().upper()
        if value in ['F', 'FEMALE', 'MUJER']:
            return 'F'
        elif value in ['M', 'MALE', 'HOMBRE']:
            return 'M'
        return None

    def _extract_age_from_dob(self, dob: str, recording_year: int = 1996) -> Optional[int]:
        """
        Calculate speaker age at time of recording.

        Args:
            dob: Date of birth string (D/M/YYYY or DD/MM/YYYY)
            recording_year: Year of recording (default 1996 for ALBAYZIN)

        Returns:
            Age in years or None if DOB cannot be parsed
        """
        if not dob:
            return None

        # Try to parse date in various formats
        formats = ['%d/%m/%Y', '%d/%m/%y', '%d/%b/%y', '%d/%B/%Y']

        for fmt in formats:
            try:
                birth_date = datetime.strptime(dob.strip(), fmt)
                return recording_year - birth_date.year
            except ValueError:
                continue

        # Try regex for flexible parsing
        match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', dob)
        if match:
            try:
                year = int(match.group(3))
                if year < 100:
                    year += 1900 if year > 50 else 2000
                return recording_year - year
            except ValueError:
                pass

        return None

    def parse_speakf_dbf(self, dbf_path: Path) -> Dict[str, Dict]:
        """
        Parse SPEAKF1.DBF/SPEAKF2.DBF for additional speaker info.

        These contain per-recording session information.

        Args:
            dbf_path: Path to SPEAKF1.DBF or SPEAKF2.DBF

        Returns:
            Dict mapping speaker code to session info
        """
        sessions = {}
        current_record: Dict[str, str] = {}

        with open(dbf_path, 'r', encoding='latin-1', errors='replace') as f:
            for line in f:
                line = line.rstrip('\r\n')

                if line.startswith('-') and current_record:
                    speaker_code = current_record.get('SCD', '').strip()
                    if speaker_code:
                        sessions[speaker_code] = dict(current_record)
                    current_record = {}
                    continue

                if ':' in line:
                    key, _, value = line.partition(':')
                    key = key.strip()
                    value = value.strip()
                    current_record[key] = value

        if current_record:
            speaker_code = current_record.get('SCD', '').strip()
            if speaker_code:
                sessions[speaker_code] = dict(current_record)

        return sessions

    def get_all_speakers(self, locutor_dir: Path) -> List[SpeakerMetadata]:
        """
        Get all speakers from LOCUTOR directory.

        Args:
            locutor_dir: Path to LOCUTOR directory

        Returns:
            List of deduplicated SpeakerMetadata
        """
        speaker_dbf = locutor_dir / 'SPEAKER.DBF'
        if speaker_dbf.exists():
            return self.parse_speaker_dbf(speaker_dbf)
        return []
