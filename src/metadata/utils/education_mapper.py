"""Education level mapping from job titles and raw values."""

from typing import Tuple
import re


class EducationMapper:
    """
    Map education/profession strings to low/mid/high/unknown.

    ALBAYZIN uses job titles (EPR field) instead of education levels.
    PRESEEA uses explicit levels but with inconsistent formatting.
    """

    # ALBAYZIN job title patterns -> education level
    # HIGH: University degrees (4+ years higher education)
    HIGH_EDUCATION_PATTERNS = [
        (r'^ING\.?\s', 'Engineering degree'),
        (r'INGENIER', 'Engineering degree'),
        (r'^ABOGAD[OA]$', 'Law degree'),
        (r'^DERECHO$', 'Law degree'),
        (r'^LICENCIAD[OA]', 'Licenciatura'),
        (r'^DR\.?\s', 'Doctorate'),
        (r'^DOCTOR', 'Doctorate'),
        (r'^PROFESOR[A]?$', 'Professor'),
        (r'^ARQUITECT', 'Architecture degree'),
        (r'^PSICOLOG[OA]$', 'Psychology degree'),
        (r'^BIOLOG', 'Biology degree'),
        (r'^PERIODIS', 'Journalism degree'),
        (r'^ECONOMIS', 'Economics degree'),
        (r'^FISIC[OA]$', 'Physics degree'),
        (r'^QUIMIC', 'Chemistry degree'),
        (r'^MEDICIN', 'Medical degree'),
        (r'^FARMAC', 'Pharmacy degree'),
        (r'^VETERINAR', 'Veterinary degree'),
        (r'^NOTARI[OA]$', 'Notary'),
        (r'^MAGISTRAD', 'Judge'),
        (r'^FILOLOG', 'Philology degree'),
        (r'^FILOSOFI', 'Philosophy degree'),
        (r'^GEOGRAF', 'Geography degree'),
        (r'^PEDAGOG', 'Pedagogy degree'),
        (r'^MAGISTERIO$', 'Teaching degree'),
    ]

    # MID: Technical/vocational (2-3 years post-secondary or professional training)
    MID_EDUCATION_PATTERNS = [
        (r'^TEC\.?\s', 'Technical degree'),
        (r'^TECNIC[OA]', 'Technical degree'),
        (r'^ADMINISTRATIV', 'Administrative training'),
        (r'^SECRETARI[OA]$', 'Secretarial training'),
        (r'^COMERCIAL$', 'Commercial training'),
        (r'^CONTAB', 'Accounting'),
        (r'^ENFERM', 'Nursing'),
        (r'DOBLAJE', 'Voice acting'),
        (r'^ACTRI[Z]?$', 'Acting'),
        (r'^ACTOR', 'Acting'),
        (r'^GUIA\s+DE\s+TURISMO$', 'Tourism'),
        (r'^MILITAR$', 'Military'),
        (r'^ANALISTA', 'Analyst'),
        (r'^PROGRAMADOR', 'Programming'),
        (r'^INFORMATICA$', 'IT'),
        (r'^INFORMATICO$', 'IT'),
        (r'^ELECTRONICA$', 'Electronics'),
        (r'^DELINEANTE$', 'Drafting'),
        (r'^EMPRESARIO$', 'Business'),
        (r'^AUX\.?\s', 'Auxiliary'),
        (r'^AUXILIAR', 'Auxiliary'),
        (r'^EDUCADOR', 'Education'),
        (r'^ESTETICIEN', 'Aesthetics'),
        (r'^ESTETICISTA$', 'Aesthetics'),
        (r'^CORREDOR', 'Agent'),
        (r'^DIRECTOR', 'Director'),
        (r'^RADIO$', 'Radio'),
        (r'^INDUSTRIAL$', 'Industrial'),
        (r'^MAESTRO$', 'Teacher'),
        (r'^SACERDOTE$', 'Priest'),
        (r'^COMPOSITOR$', 'Composer'),
        (r'^PIANO$', 'Musician'),
    ]

    # LOW: No higher education indicated
    LOW_EDUCATION_PATTERNS = [
        (r'^AMA\s+DE\s+CASA$', 'Homemaker'),
        (r'^PENSION', 'Retired'),
        (r'^JUBILAD', 'Retired'),
        (r'^OBRER[OA]$', 'Manual laborer'),
        (r'^ALBAÑIL$', 'Construction'),
        (r'^CONDUCTOR$', 'Driver'),
        (r'^CAMARER[OA]?$', 'Waiter'),
        (r'^LIMPIEZ', 'Cleaning'),
        (r'^PEÓN$', 'Unskilled laborer'),
        (r'^TAXISTA$', 'Taxi driver'),
        (r'^MECANICO$', 'Mechanic'),
        (r'^CONSERJE$', 'Janitor'),
        (r'^ORDENANZA$', 'Orderly'),
        (r'^CONFECCINISTA$', 'Garment worker'),
        (r'^PELUQUER', 'Hairdresser'),
        (r'^PINTOR$', 'Painter'),
        (r'^SERIGRAFO$', 'Screen printer'),
        (r'^VENTAS$', 'Sales'),
        (r'^EMPLEADO$', 'Employee'),
    ]

    # Students - map to mid (ALBAYZIN participants were university-age)
    STUDENT_PATTERNS = [
        (r'^ESTUDIANTE$', 'Student'),
        (r'^ALUMNO$', 'Student'),
    ]

    # PRESEEA explicit education mappings
    PRESEEA_MAPPINGS = {
        # Low
        '1': 'low',
        'bajo': 'low',
        'primaria': 'low',
        'basico': 'low',
        'básico': 'low',
        'primarios': 'low',
        'sin estudios': 'low',
        'elemental': 'low',
        # Mid
        '2': 'mid',
        'medio': 'mid',
        'secundaria': 'mid',
        'bachillerato': 'mid',
        'medios': 'mid',
        'media': 'mid',
        'nivel': 'mid',
        # High
        '3': 'high',
        'alto': 'high',
        'superior': 'high',
        'universitario': 'high',
        'universitarios': 'high',
        'universidad': 'high',
        'licenciatura': 'high',
        'licenciado': 'high',
        'superiores': 'high',
    }

    def map_albayzin(self, job_title: str) -> Tuple[str, str]:
        """
        Map ALBAYZIN job title to education level.

        Args:
            job_title: Raw education/profession from ALBAYZIN

        Returns:
            Tuple of (education_bin, mapping_rule)
        """
        if not job_title:
            return 'unknown', 'empty_value'

        title_upper = job_title.strip().upper()

        # Check for students first
        for pattern, desc in self.STUDENT_PATTERNS:
            if re.search(pattern, title_upper):
                return 'mid', f'student: {desc}'

        # Check high education patterns
        for pattern, desc in self.HIGH_EDUCATION_PATTERNS:
            if re.search(pattern, title_upper):
                return 'high', f'high: {desc}'

        # Check mid education patterns
        for pattern, desc in self.MID_EDUCATION_PATTERNS:
            if re.search(pattern, title_upper):
                return 'mid', f'mid: {desc}'

        # Check low education patterns
        for pattern, desc in self.LOW_EDUCATION_PATTERNS:
            if re.search(pattern, title_upper):
                return 'low', f'low: {desc}'

        # Unknown - log for review
        return 'unknown', f'no_match: {job_title}'

    def map_preseea(self, education_value: str) -> Tuple[str, str]:
        """
        Map PRESEEA education string to level.

        Args:
            education_value: Raw education level from PRESEEA

        Returns:
            Tuple of (education_bin, mapping_rule)
        """
        if not education_value:
            return 'unknown', 'empty_value'

        # Clean value: strip whitespace, lowercase, fix encoding
        value_clean = education_value.strip().lower()
        # Handle encoding issues (e.g., 'b�sico' -> 'basico')
        value_clean = value_clean.replace('\ufffd', 'a')
        value_clean = value_clean.replace('�', 'a')

        # Direct lookup
        if value_clean in self.PRESEEA_MAPPINGS:
            return self.PRESEEA_MAPPINGS[value_clean], f'direct: {value_clean}'

        # Keyword matching for partial matches
        if any(k in value_clean for k in ['primari', 'basic', 'elemental']):
            return 'low', f'keyword_low: {education_value}'
        if any(k in value_clean for k in ['secundari', 'bachiller', 'medio', 'media']):
            return 'mid', f'keyword_mid: {education_value}'
        if any(k in value_clean for k in ['universit', 'superior', 'licenci', 'doctor', 'master', 'alto']):
            return 'high', f'keyword_high: {education_value}'

        return 'unknown', f'no_match: {education_value}'
