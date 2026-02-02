"""Speaker metadata for Glissando-sp corpus.

Speaker information extracted from Glissando_sp_doc.pdf.
Total: 28 speakers (14 male, 14 female)
- 4 'news broadcaster' professionals (2M, 2F)
- 4 'advertising' professionals (2M, 2F)
- 20 non-professionals (10M, 10F)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from ..base import SpeakerMetadata


@dataclass
class GlissandoSpeaker:
    """Glissando speaker information."""
    speaker_id: str
    sex: str  # 'F' or 'M'
    age: int
    profile: str  # 'news_broadcaster', 'advertising', 'non_professional'
    birth_place: str
    category: str  # A, B, C, or D (indicates which subcorpora they recorded)


# Speaker data from documentation (Table in Section 3)
GLISSANDO_SPEAKERS = [
    # Category A: Prosodic news + Task dialogues + Informal dialogue
    GlissandoSpeaker('f11r', 'F', 41, 'news_broadcaster', 'Ceuta', 'A'),
    GlissandoSpeaker('m12r', 'M', 48, 'news_broadcaster', 'Zaragoza', 'A'),
    GlissandoSpeaker('m09a', 'M', 45, 'advertising', 'Valladolid', 'A'),
    GlissandoSpeaker('m10a', 'M', 34, 'advertising', 'Valladolid', 'A'),

    # Category B: Prosodic news + Phonetic news
    GlissandoSpeaker('f13r', 'F', 41, 'news_broadcaster', 'Valladolid', 'B'),
    GlissandoSpeaker('m14r', 'M', 49, 'news_broadcaster', 'Valladolid', 'B'),
    GlissandoSpeaker('f15a', 'F', 46, 'advertising', 'Valladolid', 'B'),
    GlissandoSpeaker('f16a', 'F', 40, 'advertising', 'Valladolid', 'B'),

    # Category C: Task dialogues + Informal dialogue
    GlissandoSpeaker('m17s', 'M', 19, 'non_professional', 'Valladolid', 'C'),
    GlissandoSpeaker('m18s', 'M', 18, 'non_professional', 'Burgos', 'C'),
    GlissandoSpeaker('f19s', 'F', 18, 'non_professional', 'Avila', 'C'),
    GlissandoSpeaker('m20s', 'M', 18, 'non_professional', 'Valladolid', 'C'),
    GlissandoSpeaker('f21s', 'F', 19, 'non_professional', 'Valladolid', 'C'),
    GlissandoSpeaker('f22s', 'F', 18, 'non_professional', 'Miranda de Ebro', 'C'),
    GlissandoSpeaker('f23s', 'F', 22, 'non_professional', 'Valladolid', 'C'),
    GlissandoSpeaker('f24s', 'F', 22, 'non_professional', 'Valladolid', 'C'),

    # Category D: Task dialogues only
    GlissandoSpeaker('m25s', 'M', 18, 'non_professional', 'Valladolid', 'D'),
    GlissandoSpeaker('m26s', 'M', 18, 'non_professional', 'Valladolid', 'D'),
    GlissandoSpeaker('m27s', 'M', 22, 'non_professional', 'Leon', 'D'),
    GlissandoSpeaker('m28s', 'M', 18, 'non_professional', 'Valladolid', 'D'),
    GlissandoSpeaker('f29s', 'F', 20, 'non_professional', 'Medina del Campo', 'D'),
    GlissandoSpeaker('m30s', 'M', 18, 'non_professional', 'Valladolid', 'D'),
    GlissandoSpeaker('f31s', 'F', 22, 'non_professional', 'Valladolid', 'D'),
    GlissandoSpeaker('m32s', 'M', 18, 'non_professional', 'Palencia', 'D'),
    GlissandoSpeaker('f33s', 'F', 22, 'non_professional', 'Valladolid', 'D'),
    GlissandoSpeaker('f34s', 'F', 19, 'non_professional', 'Valladolid', 'D'),
    GlissandoSpeaker('f35s', 'F', 18, 'non_professional', 'Palencia', 'D'),
    GlissandoSpeaker('f36s', 'F', 22, 'non_professional', 'Valladolid', 'D'),
]

# Build lookup by speaker ID
SPEAKER_LOOKUP: Dict[str, GlissandoSpeaker] = {s.speaker_id: s for s in GLISSANDO_SPEAKERS}


def get_speaker(speaker_id: str) -> Optional[GlissandoSpeaker]:
    """Get speaker metadata by ID."""
    return SPEAKER_LOOKUP.get(speaker_id)


def parse_speaker_id(speaker_id: str) -> Dict[str, str]:
    """
    Parse speaker ID to extract components.

    Format: [sex][number][profile]
    - sex: f (female) or m (male)
    - number: 2 digits
    - profile: r (news broadcaster), a (advertising), s (non-professional)

    Args:
        speaker_id: e.g., 'f11r', 'm25s'

    Returns:
        Dict with sex, number, profile
    """
    if len(speaker_id) < 3:
        return {}

    sex_char = speaker_id[0].lower()
    number = speaker_id[1:3] if len(speaker_id) >= 3 else ''
    profile_char = speaker_id[3] if len(speaker_id) > 3 else ''

    sex = 'F' if sex_char == 'f' else 'M' if sex_char == 'm' else None

    profile_map = {
        'r': 'news_broadcaster',
        'a': 'advertising',
        's': 'non_professional',
    }
    profile = profile_map.get(profile_char, 'unknown')

    return {
        'sex': sex,
        'number': number,
        'profile': profile,
    }


def extract_speaker_from_path(path_str: str) -> Optional[str]:
    """
    Extract speaker ID from a file path.

    Glissando paths contain 'sp_[speaker_id]' pattern.
    Examples:
        - sp_f11r/Prosodic/sp_f11r_prn01.wav -> f11r
        - sp_f11r_m12r/Transport/... -> f11r, m12r (dialogue pair)

    For Turn files, the speaker ID is in the filename itself.
    """
    import re

    # Pattern for speaker ID: f/m + 2 digits + r/a/s
    pattern = r'(?:^|_)((?:f|m)\d{2}[ras])(?:_|$|\.)'

    matches = re.findall(pattern, path_str.lower())
    if matches:
        return matches[0]  # Return first speaker ID found
    return None


def get_all_speakers() -> List[SpeakerMetadata]:
    """
    Get all Glissando speakers as SpeakerMetadata objects.

    Returns:
        List of SpeakerMetadata for all 28 speakers
    """
    speakers = []
    for gs in GLISSANDO_SPEAKERS:
        speakers.append(SpeakerMetadata(
            speaker_id=gs.speaker_id,
            sex=gs.sex,
            age=gs.age,
            birth_place=gs.birth_place,
            education=None,  # Not available
            profession=gs.profile,  # Use profile as profession
            extra_fields={
                'profile': gs.profile,
                'category': gs.category,
                'corpus': 'glissando',
            }
        ))
    return speakers
