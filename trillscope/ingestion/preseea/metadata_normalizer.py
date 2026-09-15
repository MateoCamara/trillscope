"""Metadata normalization for PRESEEA corpus."""

from typing import Dict
import re


class PreseeaMetadataNormalizer:
    """
    Normalize inconsistent metadata values in PRESEEA corpus.

    Known variations:
    - Sex: 'hombre', 'H', 'masculino', 'male' -> 'M'
           'mujer', 'M', 'femenino', 'female' -> 'F'
    - Age groups: '1', 'joven', '<30' -> '<30'
    - Education: 'bajo', 'primaria', '1' -> 'low'
    """

    SEX_MAPPINGS = {
        # Male variants
        'hombre': 'M',
        'h': 'M',
        'masculino': 'M',
        'male': 'M',
        'hOmbre': 'M',
        'Hombre': 'M',
        # Female variants
        'mujer': 'F',
        'f': 'F',
        'femenino': 'F',
        'female': 'F',
        'Mujer': 'F',
    }

    AGE_GROUP_MAPPINGS = {
        '1': '<30',
        'joven': '<30',
        '<30': '<30',
        'menor de 30': '<30',
        '2': '30-55',
        'adulto': '30-55',
        '30-55': '30-55',
        'medio': '30-55',
        '3': '>55',
        'mayor': '>55',
        '>55': '>55',
        'mayor de 55': '>55',
    }

    EDUCATION_MAPPINGS = {
        # Low education
        '1': 'low',
        'bajo': 'low',
        'primaria': 'low',
        'basico': 'low',
        'básico': 'low',
        'primarios': 'low',
        'sin estudios': 'low',
        # Mid education
        '2': 'mid',
        'medio': 'mid',
        'secundaria': 'mid',
        'bachillerato': 'mid',
        'medios': 'mid',
        # High education
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

    UNKNOWN_VALUES = {'desc', 'desconocido', 'unknown', '', 'na', 'n/a', 'no aplica'}

    def normalize_sex(self, value: str) -> str:
        """
        Normalize sex value to F/M/unknown.

        Args:
            value: Raw sex value from transcript

        Returns:
            Normalized value: 'F', 'M', or 'unknown'
        """
        if not value:
            return 'unknown'

        value_lower = value.lower().strip()

        if value_lower in self.UNKNOWN_VALUES:
            return 'unknown'

        normalized = self.SEX_MAPPINGS.get(value_lower)
        if normalized:
            return normalized

        # Try matching first letter
        if value_lower.startswith('h') or value_lower.startswith('m'):
            if value_lower.startswith('h'):
                return 'M'  # hombre
            elif value_lower.startswith('m') and not value_lower.startswith('ma'):
                return 'F'  # mujer (but not 'male' or 'masculino')

        return 'unknown'

    def normalize_age_group(self, value: str) -> str:
        """
        Normalize age group to <30/30-55/>55/unknown.

        Args:
            value: Raw age group value

        Returns:
            Normalized value: '<30', '30-55', '>55', or 'unknown'
        """
        if not value:
            return 'unknown'

        value_lower = value.lower().strip()

        if value_lower in self.UNKNOWN_VALUES:
            return 'unknown'

        normalized = self.AGE_GROUP_MAPPINGS.get(value_lower)
        if normalized:
            return normalized

        return 'unknown'

    def normalize_education(self, value: str) -> str:
        """
        Normalize education to low/mid/high/unknown.

        Args:
            value: Raw education level value

        Returns:
            Normalized value: 'low', 'mid', 'high', or 'unknown'
        """
        if not value:
            return 'unknown'

        value_lower = value.lower().strip()

        if value_lower in self.UNKNOWN_VALUES:
            return 'unknown'

        normalized = self.EDUCATION_MAPPINGS.get(value_lower)
        if normalized:
            return normalized

        # Try keyword matching
        if any(k in value_lower for k in ['primari', 'básic', 'basico', 'elemental']):
            return 'low'
        if any(k in value_lower for k in ['secundari', 'bachiller', 'medio']):
            return 'mid'
        if any(k in value_lower for k in ['universit', 'superior', 'licenci', 'doctor', 'master']):
            return 'high'

        return 'unknown'

    def extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """
        Extract metadata from PRESEEA filename pattern.

        Pattern: {CITY}_{SEX}{AGE}{EDU}_{ID}.{ext}
        Example: ALCA_H11_037 -> city=ALCA, sex=H, age_group=1, edu=1, id=037

        Args:
            filename: Filename with or without extension

        Returns:
            Dict with extracted metadata
        """
        stem = filename.rsplit('.', 1)[0] if '.' in filename else filename

        result = {
            'city': '',
            'sex': 'unknown',
            'age_group': 'unknown',
            'education': 'unknown',
            'id': '',
        }

        # Pattern: CITY_CODE_ID
        match = re.match(r'^([A-Z]{4})_([HM])(\d)(\d)_(\d+)$', stem)
        if match:
            result['city'] = match.group(1)
            result['sex'] = 'M' if match.group(2) == 'H' else 'F'
            result['age_group'] = self.normalize_age_group(match.group(3))
            result['education'] = self.normalize_education(match.group(4))
            result['id'] = match.group(5)

        return result

    def is_unknown(self, value: str) -> bool:
        """Check if a value represents unknown/missing data."""
        if not value:
            return True
        return value.lower().strip() in self.UNKNOWN_VALUES


# City code to full name mapping
CITY_CODES = {
    'ALCA': {'city': 'Alcalá de Henares', 'country': 'España'},
    'BAIR': {'city': 'Buenos Aires', 'country': 'Argentina'},
    'BARC': {'city': 'Barcelona', 'country': 'España'},
    'BARR': {'city': 'Barranquilla', 'country': 'Colombia'},
    'BOGO': {'city': 'Bogotá', 'country': 'Colombia'},
    'CADI': {'city': 'Cádiz', 'country': 'España'},
    'CALI': {'city': 'Cali', 'country': 'Colombia'},
    'CARA': {'city': 'Caracas', 'country': 'Venezuela'},
    'CART': {'city': 'Cartagena de Indias', 'country': 'Colombia'},
    'CHIC': {'city': 'Chiclayo', 'country': 'Perú'},
    'GRAN': {'city': 'Granada', 'country': 'España'},
    'GUAD': {'city': 'Guadalajara', 'country': 'México'},
    'GUAT': {'city': 'Ciudad de Guatemala', 'country': 'Guatemala'},
    'LASP': {'city': 'Las Palmas', 'country': 'España'},
    'LHAB': {'city': 'La Habana', 'country': 'Cuba'},
    'LIMA': {'city': 'Lima', 'country': 'Perú'},
    'LPAZ': {'city': 'La Paz', 'country': 'Bolivia'},
    'MADR': {'city': 'Madrid', 'country': 'España'},
    'MALA': {'city': 'Málaga', 'country': 'España'},
    'MEDE': {'city': 'Medellín', 'country': 'Colombia'},
    'MEVE': {'city': 'Mérida', 'country': 'Venezuela'},
    'MEXI': {'city': 'Ciudad de México', 'country': 'México'},
    'MONR': {'city': 'Monterrey', 'country': 'México'},
    'MONV': {'city': 'Montevideo', 'country': 'Uruguay'},
    'MXLI': {'city': 'Ciudad de México', 'country': 'México'},
    'NYOC': {'city': 'Nueva York', 'country': 'Estados Unidos'},
    'PALM': {'city': 'Palma de Mallorca', 'country': 'España'},
    'PERE': {'city': 'Pereira', 'country': 'Colombia'},
    'PUEB': {'city': 'Puebla', 'country': 'México'},
    'QUIT': {'city': 'Quito', 'country': 'Ecuador'},
    'SANT': {'city': 'Santiago de Compostela', 'country': 'España'},
    'SCHI': {'city': 'Santiago', 'country': 'Chile'},
    'SCOM': {'city': 'Santa Cruz de Tenerife', 'country': 'España'},
    'SEVI': {'city': 'Sevilla', 'country': 'España'},
    'VALE': {'city': 'Valencia', 'country': 'España'},
    'XIXO': {'city': 'Gijón', 'country': 'España'},
}


def get_city_info(code: str) -> Dict[str, str]:
    """Get city and country from code."""
    return CITY_CODES.get(code.upper(), {'city': code, 'country': 'unknown'})
