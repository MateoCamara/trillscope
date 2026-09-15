"""XML transcript parser for PRESEEA corpus."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreseeaSpeaker:
    """Speaker information from PRESEEA transcript."""
    speaker_id: str
    name: str
    code: str        # 'I' for informant, 'E' for interviewer
    sex: str         # raw value (may be 'hombre', 'H', 'masculino', etc.)
    age_group: str
    age: Optional[int]
    education_level: str
    studies: str
    profession: str
    origin: str
    role: str        # 'informante' or 'entrevistador'


@dataclass
class PreseeaTranscriptMetadata:
    """Metadata extracted from PRESEEA XML transcript."""
    audio_filename: str
    corpus: str
    subcorpus: str
    city: str
    country: str
    duration: str
    recording_date: str
    transcription_date: str
    word_count: int
    speakers: List[PreseeaSpeaker] = field(default_factory=list)


@dataclass
class PreseeaUtterance:
    """Single utterance/turn from transcript."""
    speaker_code: str  # 'I' or 'E'
    timestamp: Optional[str]
    text: str
    has_overlap: bool
    has_hesitation: bool
    is_unintelligible: bool


class PreseeaXMLParser:
    """
    Parse PRESEEA XML transcript format.

    XML Structure:
    <Trans audio_filename="..." xml:lang="espanol">
      <Datos clave_texto="..." tipo_texto="entrevista_semidirigida">
        <Corpus corpus="PRESEEA" subcorpus="..." ciudad="..." pais="..."/>
        <Grabacion resp_grab="..." duracion="..." fecha_grab="..."/>
        <Transcripcion resp_trans="..." fecha_trans="..." numero_palabras="..."/>
      </Datos>
      <Hablantes>
        <Hablante id="hab1" nombre="..." sexo="hombre" grupo_edad="1" edad="33"
                  nivel_edu="bajo" estudios="..." profesion="..." origen="..."
                  papel="informante"/>
        <Hablante id="hab2" ... papel="entrevistador"/>
      </Hablantes>
    </Trans>

    Transcript body:
    E: <tiempo = "00:04"/> text with <vacilacion/> <silencio/>
       <simultaneo> overlap </simultaneo> markers
    I: response text
    """

    # Regex patterns for parsing
    TRANS_PATTERN = re.compile(r'<Trans\s+audio_filename="([^"]*)"[^>]*>', re.IGNORECASE)
    DATOS_PATTERN = re.compile(r'<Datos[^>]*clave_texto="([^"]*)"[^>]*tipo_texto="([^"]*)"[^>]*>', re.IGNORECASE)
    CORPUS_PATTERN = re.compile(r'<Corpus[^>]*corpus="([^"]*)"[^>]*subcorpus="([^"]*)"[^>]*ciudad="([^"]*)"[^>]*pais="([^"]*)"[^>]*/>', re.IGNORECASE)
    GRABACION_PATTERN = re.compile(r'<Grabacion[^>]*duracion="([^"]*)"[^>]*fecha_grab="([^"]*)"[^>]*/?\s*>', re.IGNORECASE)
    TRANSCRIPCION_PATTERN = re.compile(r'<Transcripcion[^>]*fecha_trans="([^"]*)"[^>]*numero_palabras="([^"]*)"[^>]*/?\s*>', re.IGNORECASE)
    HABLANTE_PATTERN = re.compile(
        r'<Hablante\s+id="([^"]*)"\s+nombre="([^"]*)"\s+codigo_hab="([^"]*)"\s+sexo="([^"]*)"\s*'
        r'grupo_edad="([^"]*)"\s+edad="([^"]*)"\s+nivel_edu="([^"]*)"\s+estudios="([^"]*)"\s+'
        r'profesion="([^"]*)"\s+origen="([^"]*)"\s*papel="([^"]*)"[^>]*/?>',
        re.IGNORECASE | re.DOTALL
    )
    TIEMPO_PATTERN = re.compile(r'<tiempo\s*=\s*"([^"]*)"[^>]*/>', re.IGNORECASE)
    TURN_PATTERN = re.compile(r'^([IE]):\s*(.*)$', re.MULTILINE)

    def parse(self, xml_path: Path) -> Tuple[PreseeaTranscriptMetadata, List[PreseeaUtterance]]:
        """
        Parse complete transcript file.

        Args:
            xml_path: Path to transcript file

        Returns:
            Tuple of (metadata, list of utterances)
        """
        content = self._read_file(xml_path)

        metadata = self._parse_metadata(content)
        utterances = self._parse_body(content)

        return metadata, utterances

    def _read_file(self, path: Path) -> str:
        """Read file with encoding detection."""
        # Try UTF-8 first, fall back to latin-1
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # Last resort: read with errors replaced
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    def _parse_metadata(self, content: str) -> PreseeaTranscriptMetadata:
        """Extract metadata from XML header."""
        metadata = PreseeaTranscriptMetadata(
            audio_filename='',
            corpus='PRESEEA',
            subcorpus='',
            city='',
            country='',
            duration='',
            recording_date='',
            transcription_date='',
            word_count=0,
            speakers=[]
        )

        # Parse Trans tag
        trans_match = self.TRANS_PATTERN.search(content)
        if trans_match:
            metadata.audio_filename = trans_match.group(1)

        # Parse Corpus tag
        corpus_match = self.CORPUS_PATTERN.search(content)
        if corpus_match:
            metadata.corpus = corpus_match.group(1)
            metadata.subcorpus = corpus_match.group(2)
            metadata.city = corpus_match.group(3)
            metadata.country = corpus_match.group(4)

        # Parse Grabacion tag
        grab_match = self.GRABACION_PATTERN.search(content)
        if grab_match:
            metadata.duration = grab_match.group(1)
            metadata.recording_date = grab_match.group(2)

        # Parse Transcripcion tag
        trans_info_match = self.TRANSCRIPCION_PATTERN.search(content)
        if trans_info_match:
            metadata.transcription_date = trans_info_match.group(1)
            try:
                metadata.word_count = int(trans_info_match.group(2))
            except ValueError:
                pass

        # Parse speakers
        for match in self.HABLANTE_PATTERN.finditer(content):
            age = None
            age_str = match.group(6)
            if age_str and age_str.lower() not in ['desc', 'desconocido', '']:
                try:
                    age = int(age_str)
                except ValueError:
                    pass

            speaker = PreseeaSpeaker(
                speaker_id=match.group(1),
                name=match.group(2).strip(),
                code=match.group(3),
                sex=match.group(4),
                age_group=match.group(5),
                age=age,
                education_level=match.group(7),
                studies=match.group(8),
                profession=match.group(9),
                origin=match.group(10),
                role=match.group(11)
            )
            metadata.speakers.append(speaker)

        return metadata

    def _parse_body(self, content: str) -> List[PreseeaUtterance]:
        """Parse transcript body text after </Trans> closing tag."""
        utterances = []

        # Find the body (after </Trans> or </Hablantes>)
        body_start = content.find('</Trans>')
        if body_start == -1:
            body_start = content.find('</Hablantes>')
        if body_start == -1:
            return utterances

        body = content[body_start:]

        # Parse each turn
        current_speaker = None
        current_text = []
        current_timestamp = None

        for line in body.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Check for speaker turn start
            turn_match = re.match(r'^([IE]):\s*(.*)$', line)
            if turn_match:
                # Save previous utterance
                if current_speaker and current_text:
                    full_text = ' '.join(current_text)
                    utterances.append(PreseeaUtterance(
                        speaker_code=current_speaker,
                        timestamp=current_timestamp,
                        text=self._clean_text(full_text),
                        has_overlap=self._has_overlap(full_text),
                        has_hesitation=self._has_hesitation(full_text),
                        is_unintelligible='<ininteligible/>' in full_text.lower()
                    ))

                current_speaker = turn_match.group(1)
                current_text = [turn_match.group(2)]

                # Extract timestamp if present
                tiempo_match = self.TIEMPO_PATTERN.search(turn_match.group(2))
                current_timestamp = tiempo_match.group(1) if tiempo_match else None

            elif current_speaker:
                # Continuation of current turn
                current_text.append(line)

        # Don't forget the last utterance
        if current_speaker and current_text:
            full_text = ' '.join(current_text)
            utterances.append(PreseeaUtterance(
                speaker_code=current_speaker,
                timestamp=current_timestamp,
                text=self._clean_text(full_text),
                has_overlap=self._has_overlap(full_text),
                has_hesitation=self._has_hesitation(full_text),
                is_unintelligible='<ininteligible/>' in full_text.lower()
            ))

        return utterances

    def _clean_text(self, text: str) -> str:
        """Remove XML tags from text, keeping only content."""
        # Remove self-closing tags
        text = re.sub(r'<[^>]+/>', '', text)
        # Remove paired tags but keep content
        text = re.sub(r'<simultaneo>|</simultaneo>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<simult[áa]neo>|</simult[áa]neo>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<cita>|</cita>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<extranjero>|</extranjero>', '', text, flags=re.IGNORECASE)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _has_overlap(self, text: str) -> bool:
        """Check if text contains overlap markers."""
        return bool(re.search(r'<simult[áa]neo>', text, re.IGNORECASE))

    def _has_hesitation(self, text: str) -> bool:
        """Check if text contains hesitation markers."""
        return '<vacilación/>' in text.lower() or '<vacilacion/>' in text.lower()

    def get_informant(self, metadata: PreseeaTranscriptMetadata) -> Optional[PreseeaSpeaker]:
        """Get the informant speaker from metadata."""
        for speaker in metadata.speakers:
            if speaker.role.lower() == 'informante':
                return speaker
        return None

    def extract_timestamps(self, content: str) -> List[Tuple[str, int]]:
        """
        Extract all timestamps from content.

        Returns:
            List of (timestamp_string, character_position) tuples
        """
        timestamps = []
        for match in self.TIEMPO_PATTERN.finditer(content):
            timestamps.append((match.group(1), match.start()))
        return timestamps

    def parse_duration(self, duration_str: str) -> Optional[float]:
        """
        Parse duration string to seconds.

        Format examples: "50'00''", "61'48''"
        """
        if not duration_str:
            return None

        # Pattern: MM'SS'' or HH'MM'SS''
        match = re.match(r"(\d+)'(\d+)''", duration_str)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return minutes * 60 + seconds

        return None
