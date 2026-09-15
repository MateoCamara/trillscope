"""Main Heroico (LDC2006S37) corpus ingestor."""

from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

from tqdm import tqdm

from ..base import DatasetIngestor, IngestionResult, SpeakerMetadata, UtteranceRecord
from ..utils.audio import AudioInspector
from ..utils.id_generator import IDGenerator
from ..utils.report_writer import AuditReportWriter

logger = logging.getLogger(__name__)


class HeroicoIngestor(DatasetIngestor):
    """
    Ingest Heroico (LDC2006S37) corpus.

    Structure:
    - data/speech/heroico/
      - Recordings_Spanish/[speaker]/[prompt].wav - Read speech (Latin American history)
      - Answers_Spanish/[speaker]/[answer].wav - Spontaneous answers
    - data/speech/usma/
      - [native/nonnative]-[sex]-[origin]-[metrics]-[id]/s[prompt].wav - Read prompts
    - data/transcripts/
      - heroico-recordings.txt - Transcripts for recordings (ID<TAB>text)
      - heroico-answers.txt - Transcripts for answers (speaker/question<TAB>text)
      - usma-prompts.txt - Transcripts for USMA prompts (ID<TAB>text)
      - questions.txt - Questions for answers

    Audio:
    - Heroico: 22kHz mono WAV (needs conversion)
    - USMA: 20kHz mono WAV (needs conversion)

    No phoneme alignment - requires MFA.
    """

    def __init__(self, dataset_root: Path, output_dir: Path):
        """
        Initialize ingestor.

        Args:
            dataset_root: Path to LDC2006S37 directory
            output_dir: Output directory for metadata and reports
        """
        super().__init__(dataset_root, output_dir)
        self.audio_inspector = AudioInspector()
        self.id_gen = IDGenerator('HER')

        # Cache
        self._structure_cache: Optional[Dict] = None
        self._speakers_cache: Optional[List[SpeakerMetadata]] = None
        self._recordings_transcripts: Dict[str, str] = {}  # prompt_id -> text
        self._answers_transcripts: Dict[str, str] = {}  # speaker/question -> text
        self._usma_transcripts: Dict[str, str] = {}  # prompt_id -> text

    @property
    def data_dir(self) -> Path:
        """Path to data directory."""
        # Handle nested LDC2006S37/LDC2006S37 structure
        nested = self.dataset_root / 'LDC2006S37' / 'data'
        if nested.exists():
            return nested
        return self.dataset_root / 'data'

    @property
    def heroico_speech_dir(self) -> Path:
        """Path to heroico speech directory."""
        return self.data_dir / 'speech' / 'heroico'

    @property
    def usma_speech_dir(self) -> Path:
        """Path to USMA speech directory."""
        return self.data_dir / 'speech' / 'usma'

    @property
    def transcripts_dir(self) -> Path:
        """Path to transcripts directory."""
        return self.data_dir / 'transcripts'

    def discover_structure(self) -> Dict[str, Any]:
        """Discover Heroico directory structure."""
        if self._structure_cache:
            return self._structure_cache

        structure = {
            'heroico': {
                'recordings_speakers': 0,
                'recordings_files': 0,
                'answers_speakers': 0,
                'answers_files': 0,
            },
            'usma': {
                'native_speakers': 0,
                'nonnative_speakers': 0,
                'total_files': 0,
                'speakers': [],
            },
            'transcripts': {
                'recordings': False,
                'answers': False,
                'usma_prompts': False,
            },
            'total_speakers': 0,
            'total_files': 0,
        }

        # Heroico Recordings
        recordings_dir = self.heroico_speech_dir / 'Recordings_Spanish'
        if recordings_dir.exists():
            speakers = [d for d in recordings_dir.iterdir() if d.is_dir()]
            structure['heroico']['recordings_speakers'] = len(speakers)
            for speaker_dir in speakers:
                wav_count = len(list(speaker_dir.glob('*.wav')))
                structure['heroico']['recordings_files'] += wav_count

        # Heroico Answers
        answers_dir = self.heroico_speech_dir / 'Answers_Spanish'
        if answers_dir.exists():
            speakers = [d for d in answers_dir.iterdir() if d.is_dir()]
            structure['heroico']['answers_speakers'] = len(speakers)
            for speaker_dir in speakers:
                wav_count = len(list(speaker_dir.glob('*.wav')))
                structure['heroico']['answers_files'] += wav_count

        # USMA
        if self.usma_speech_dir.exists():
            for speaker_dir in self.usma_speech_dir.iterdir():
                if speaker_dir.is_dir():
                    info = self._parse_usma_folder(speaker_dir.name)
                    if info:
                        structure['usma']['speakers'].append(info)
                        if info['is_native']:
                            structure['usma']['native_speakers'] += 1
                        else:
                            structure['usma']['nonnative_speakers'] += 1
                        wav_count = len(list(speaker_dir.glob('*.wav')))
                        structure['usma']['total_files'] += wav_count

        # Transcripts
        structure['transcripts']['recordings'] = (
            self.transcripts_dir / 'heroico-recordings.txt').exists()
        structure['transcripts']['answers'] = (
            self.transcripts_dir / 'heroico-answers.txt').exists()
        structure['transcripts']['usma_prompts'] = (
            self.transcripts_dir / 'usma-prompts.txt').exists()

        # Totals
        structure['total_speakers'] = (
            structure['heroico']['recordings_speakers'] +
            len(structure['usma']['speakers'])
        )
        structure['total_files'] = (
            structure['heroico']['recordings_files'] +
            structure['heroico']['answers_files'] +
            structure['usma']['total_files']
        )

        self._structure_cache = structure
        return structure

    def _parse_usma_folder(self, folder_name: str) -> Optional[Dict[str, Any]]:
        """
        Parse USMA folder name to extract speaker metadata.

        Format: [native/nonnative]-[sex]-[origin]-[metric1]-[metric2]-[metric3]-[metric4]-[id]
        Example: native-m-Mexico-162-41-68-NA-cgl7958
        """
        parts = folder_name.split('-')
        if len(parts) < 3:
            return None

        is_native = parts[0] == 'native'
        sex = 'M' if parts[1] == 'm' else 'F' if parts[1] == 'f' else 'unknown'

        # Origin can contain dots (e.g., "Mexico.Texas", "Argentina.France")
        origin = parts[2] if len(parts) > 2 else 'unknown'

        # Speaker ID is the last part
        speaker_id = parts[-1] if len(parts) > 3 else folder_name

        # Map origin to country
        country = self._normalize_origin(origin)

        return {
            'folder_name': folder_name,
            'speaker_id': speaker_id,
            'is_native': is_native,
            'sex': sex,
            'origin': origin,
            'country': country,
        }

    def _normalize_origin(self, origin: str) -> str:
        """Normalize origin string to country name."""
        origin_lower = origin.lower()

        # Handle compound origins (take first part)
        if '.' in origin_lower:
            origin_lower = origin_lower.split('.')[0]

        # Map to standard country names
        country_map = {
            'mexico': 'mexico',
            'argentina': 'argentina',
            'venezuela': 'venezuela',
            'spain': 'spain',
            'castillian': 'spain',
            'puerto': 'puerto_rico',
            'mic2': 'unknown',  # Microphone 2 - not origin
            'na': 'unknown',
        }

        return country_map.get(origin_lower, 'unknown')

    def discover_audio(self) -> Dict[str, Any]:
        """Discover and validate audio files."""
        audio_info = {
            'formats': defaultdict(int),
            'sample_rates': defaultdict(int),
            'channels': defaultdict(int),
            'sampled_files': [],
            'issues': [],
        }

        # Sample files from different sections
        sample_files = []

        # Heroico recordings
        recordings_dir = self.heroico_speech_dir / 'Recordings_Spanish'
        if recordings_dir.exists():
            for speaker_dir in list(recordings_dir.iterdir())[:3]:
                if speaker_dir.is_dir():
                    wav_files = list(speaker_dir.glob('*.wav'))[:2]
                    sample_files.extend(wav_files)

        # Heroico answers
        answers_dir = self.heroico_speech_dir / 'Answers_Spanish'
        if answers_dir.exists():
            for speaker_dir in list(answers_dir.iterdir())[:3]:
                if speaker_dir.is_dir():
                    wav_files = list(speaker_dir.glob('*.wav'))[:2]
                    sample_files.extend(wav_files)

        # USMA
        if self.usma_speech_dir.exists():
            for speaker_dir in list(self.usma_speech_dir.iterdir())[:3]:
                if speaker_dir.is_dir():
                    wav_files = list(speaker_dir.glob('*.wav'))[:2]
                    sample_files.extend(wav_files)

        for audio_file in sample_files[:20]:
            try:
                metadata = self.audio_inspector.inspect(audio_file)
                audio_info['formats'][metadata.format] += 1
                audio_info['sample_rates'][metadata.sample_rate] += 1
                audio_info['channels'][metadata.channels] += 1
                audio_info['sampled_files'].append({
                    'path': str(audio_file),
                    'format': metadata.format,
                    'sample_rate': metadata.sample_rate,
                    'duration': metadata.duration_seconds,
                })

                if not metadata.is_valid:
                    audio_info['issues'].append(f"Invalid: {audio_file}")
            except Exception as e:
                audio_info['issues'].append(f"Error: {audio_file} - {e}")

        return audio_info

    def discover_transcripts(self) -> Dict[str, Any]:
        """Discover transcript files."""
        transcript_info = {
            'recordings': {'exists': False, 'lines': 0},
            'answers': {'exists': False, 'lines': 0},
            'usma_prompts': {'exists': False, 'lines': 0},
            'issues': [],
        }

        # Heroico recordings
        recordings_file = self.transcripts_dir / 'heroico-recordings.txt'
        if recordings_file.exists():
            transcript_info['recordings']['exists'] = True
            try:
                with open(recordings_file, 'r', encoding='latin-1') as f:
                    transcript_info['recordings']['lines'] = sum(1 for _ in f)
            except Exception as e:
                transcript_info['issues'].append(f"Error reading recordings: {e}")

        # Heroico answers
        answers_file = self.transcripts_dir / 'heroico-answers.txt'
        if answers_file.exists():
            transcript_info['answers']['exists'] = True
            try:
                with open(answers_file, 'r', encoding='latin-1') as f:
                    transcript_info['answers']['lines'] = sum(1 for _ in f)
            except Exception as e:
                transcript_info['issues'].append(f"Error reading answers: {e}")

        # USMA prompts
        usma_file = self.transcripts_dir / 'usma-prompts.txt'
        if usma_file.exists():
            transcript_info['usma_prompts']['exists'] = True
            try:
                with open(usma_file, 'r', encoding='latin-1') as f:
                    transcript_info['usma_prompts']['lines'] = sum(1 for _ in f)
            except Exception as e:
                transcript_info['issues'].append(f"Error reading USMA prompts: {e}")

        return transcript_info

    def _load_recordings_transcripts(self) -> Dict[str, str]:
        """Load heroico recordings transcripts."""
        if self._recordings_transcripts:
            return self._recordings_transcripts

        transcripts = {}
        trans_file = self.transcripts_dir / 'heroico-recordings.txt'

        if trans_file.exists():
            try:
                with open(trans_file, 'r', encoding='latin-1') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Format: ID<TAB>text
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            prompt_id = parts[0].strip()
                            text = parts[1].strip()
                            transcripts[prompt_id] = text
            except Exception as e:
                self.log_issue('error', 'transcript_load',
                              f"Failed to load recordings transcripts: {e}")

        self._recordings_transcripts = transcripts
        return transcripts

    def _load_answers_transcripts(self) -> Dict[str, str]:
        """Load heroico answers transcripts."""
        if self._answers_transcripts:
            return self._answers_transcripts

        transcripts = {}
        trans_file = self.transcripts_dir / 'heroico-answers.txt'

        if trans_file.exists():
            try:
                with open(trans_file, 'r', encoding='latin-1') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Format: speaker/question<TAB>text
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()  # e.g., "100/10"
                            text = parts[1].strip()
                            transcripts[key] = text
            except Exception as e:
                self.log_issue('error', 'transcript_load',
                              f"Failed to load answers transcripts: {e}")

        self._answers_transcripts = transcripts
        return transcripts

    def _load_usma_transcripts(self) -> Dict[str, str]:
        """Load USMA prompts transcripts."""
        if self._usma_transcripts:
            return self._usma_transcripts

        transcripts = {}
        trans_file = self.transcripts_dir / 'usma-prompts.txt'

        if trans_file.exists():
            try:
                with open(trans_file, 'r', encoding='latin-1') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Format: ID<TAB>text (e.g., "s1<TAB>vivo en una casa")
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            prompt_id = parts[0].strip()  # e.g., "s1"
                            text = parts[1].strip()
                            transcripts[prompt_id] = text
            except Exception as e:
                self.log_issue('error', 'transcript_load',
                              f"Failed to load USMA transcripts: {e}")

        self._usma_transcripts = transcripts
        return transcripts

    def discover_metadata(self) -> List[SpeakerMetadata]:
        """Get speaker metadata."""
        if self._speakers_cache:
            return self._speakers_cache

        speakers = []
        structure = self.discover_structure()

        # Heroico speakers (numeric IDs, no metadata available)
        recordings_dir = self.heroico_speech_dir / 'Recordings_Spanish'
        if recordings_dir.exists():
            for speaker_dir in recordings_dir.iterdir():
                if speaker_dir.is_dir():
                    speaker_id = f"heroico_{speaker_dir.name}"
                    speakers.append(SpeakerMetadata(
                        speaker_id=speaker_id,
                        sex='unknown',  # Not available
                        age=None,
                        birth_place=None,
                        education=None,
                        profession=None,
                        extra_fields={
                            'corpus': 'heroico',
                            'subcorpus': 'heroico',
                            'is_native': True,  # Assumed native
                        }
                    ))

        # USMA speakers (metadata from folder names)
        for speaker_info in structure['usma']['speakers']:
            speakers.append(SpeakerMetadata(
                speaker_id=f"usma_{speaker_info['speaker_id']}",
                sex=speaker_info['sex'],
                age=None,
                birth_place=speaker_info['origin'],
                education=None,
                profession=None,
                extra_fields={
                    'corpus': 'heroico',
                    'subcorpus': 'usma',
                    'is_native': speaker_info['is_native'],
                    'country': speaker_info['country'],
                    'folder_name': speaker_info['folder_name'],
                }
            ))

        self._speakers_cache = speakers
        return speakers

    def link_audio_transcript(self) -> List[UtteranceRecord]:
        """Link audio files with transcripts."""
        utterances = []

        # Load all transcripts
        recordings_trans = self._load_recordings_transcripts()
        answers_trans = self._load_answers_transcripts()
        usma_trans = self._load_usma_transcripts()

        logger.info(f"Loaded transcripts: {len(recordings_trans)} recordings, "
                   f"{len(answers_trans)} answers, {len(usma_trans)} USMA prompts")

        # Process Heroico recordings
        utterances.extend(self._process_heroico_recordings(recordings_trans))

        # Process Heroico answers
        utterances.extend(self._process_heroico_answers(answers_trans))

        # Process USMA
        utterances.extend(self._process_usma(usma_trans))

        return utterances

    def _process_heroico_recordings(self, transcripts: Dict[str, str]) -> List[UtteranceRecord]:
        """Process Heroico recordings section."""
        utterances = []
        recordings_dir = self.heroico_speech_dir / 'Recordings_Spanish'

        if not recordings_dir.exists():
            return utterances

        speaker_dirs = list(recordings_dir.iterdir())
        for speaker_dir in tqdm(speaker_dirs, desc="Processing Heroico Recordings"):
            if not speaker_dir.is_dir():
                continue

            speaker_id = f"heroico_{speaker_dir.name}"

            for wav_file in speaker_dir.glob('*.wav'):
                prompt_id = wav_file.stem  # e.g., "1", "2"

                # Get transcript
                transcript_text = transcripts.get(prompt_id)
                if not transcript_text:
                    self.log_issue('warning', 'missing_transcript',
                                  f"No transcript for recording {wav_file}")

                # Get audio metadata
                try:
                    audio_meta = self.audio_inspector.inspect(wav_file)
                except Exception as e:
                    self.log_issue('error', 'audio_error',
                                  f"Failed to inspect {wav_file}: {e}")
                    continue

                utt_id = self.id_gen.from_filename(f"rec_{speaker_dir.name}_{prompt_id}")

                utterances.append(UtteranceRecord(
                    utt_id=utt_id,
                    speaker_id=speaker_id,
                    audio_path=str(wav_file),
                    audio_format='wav',
                    sample_rate=audio_meta.sample_rate,
                    duration_ms=audio_meta.duration_seconds * 1000,
                    transcript_path=str(self.transcripts_dir / 'heroico-recordings.txt'),
                    transcript_text=transcript_text,
                    transcript_format='txt',
                    has_phoneme_alignment=False,
                    alignment_path=None,
                    original_filename=wav_file.name,
                    extra_fields={
                        'subcorpus': 'heroico_recordings',
                        'speech_style': 'read',
                        'prompt_id': prompt_id,
                        'is_native': True,
                    }
                ))

        return utterances

    def _process_heroico_answers(self, transcripts: Dict[str, str]) -> List[UtteranceRecord]:
        """Process Heroico answers section."""
        utterances = []
        answers_dir = self.heroico_speech_dir / 'Answers_Spanish'

        if not answers_dir.exists():
            return utterances

        speaker_dirs = list(answers_dir.iterdir())
        for speaker_dir in tqdm(speaker_dirs, desc="Processing Heroico Answers"):
            if not speaker_dir.is_dir():
                continue

            speaker_id = f"heroico_{speaker_dir.name}"

            for wav_file in speaker_dir.glob('*.wav'):
                answer_id = wav_file.stem  # e.g., "1", "2"
                # Transcript key format: speaker/question
                trans_key = f"{speaker_dir.name}/{answer_id}"

                # Get transcript
                transcript_text = transcripts.get(trans_key)
                if not transcript_text:
                    self.log_issue('warning', 'missing_transcript',
                                  f"No transcript for answer {wav_file}")

                # Get audio metadata
                try:
                    audio_meta = self.audio_inspector.inspect(wav_file)
                except Exception as e:
                    self.log_issue('error', 'audio_error',
                                  f"Failed to inspect {wav_file}: {e}")
                    continue

                utt_id = self.id_gen.from_filename(f"ans_{speaker_dir.name}_{answer_id}")

                utterances.append(UtteranceRecord(
                    utt_id=utt_id,
                    speaker_id=speaker_id,
                    audio_path=str(wav_file),
                    audio_format='wav',
                    sample_rate=audio_meta.sample_rate,
                    duration_ms=audio_meta.duration_seconds * 1000,
                    transcript_path=str(self.transcripts_dir / 'heroico-answers.txt'),
                    transcript_text=transcript_text,
                    transcript_format='txt',
                    has_phoneme_alignment=False,
                    alignment_path=None,
                    original_filename=wav_file.name,
                    extra_fields={
                        'subcorpus': 'heroico_answers',
                        'speech_style': 'spontaneous',
                        'answer_id': answer_id,
                        'is_native': True,
                    }
                ))

        return utterances

    def _process_usma(self, transcripts: Dict[str, str]) -> List[UtteranceRecord]:
        """Process USMA section."""
        utterances = []

        if not self.usma_speech_dir.exists():
            return utterances

        speaker_dirs = list(self.usma_speech_dir.iterdir())
        for speaker_dir in tqdm(speaker_dirs, desc="Processing USMA"):
            if not speaker_dir.is_dir():
                continue

            # Parse folder name for metadata
            info = self._parse_usma_folder(speaker_dir.name)
            if not info:
                self.log_issue('warning', 'folder_parse',
                              f"Could not parse USMA folder: {speaker_dir.name}")
                continue

            speaker_id = f"usma_{info['speaker_id']}"

            for wav_file in speaker_dir.glob('*.wav'):
                prompt_id = wav_file.stem  # e.g., "s1", "s2"

                # Get transcript
                transcript_text = transcripts.get(prompt_id)
                if not transcript_text:
                    self.log_issue('warning', 'missing_transcript',
                                  f"No transcript for USMA {wav_file}")

                # Get audio metadata
                try:
                    audio_meta = self.audio_inspector.inspect(wav_file)
                except Exception as e:
                    self.log_issue('error', 'audio_error',
                                  f"Failed to inspect {wav_file}: {e}")
                    continue

                utt_id = self.id_gen.from_filename(f"usma_{info['speaker_id']}_{prompt_id}")

                utterances.append(UtteranceRecord(
                    utt_id=utt_id,
                    speaker_id=speaker_id,
                    audio_path=str(wav_file),
                    audio_format='wav',
                    sample_rate=audio_meta.sample_rate,
                    duration_ms=audio_meta.duration_seconds * 1000,
                    transcript_path=str(self.transcripts_dir / 'usma-prompts.txt'),
                    transcript_text=transcript_text,
                    transcript_format='txt',
                    has_phoneme_alignment=False,
                    alignment_path=None,
                    original_filename=wav_file.name,
                    extra_fields={
                        'subcorpus': 'usma',
                        'speech_style': 'read',
                        'prompt_id': prompt_id,
                        'is_native': info['is_native'],
                        'country': info['country'],
                        'origin': info['origin'],
                    }
                ))

        return utterances

    def ingest(self) -> IngestionResult:
        """Run complete Heroico ingestion."""
        logger.info("Starting Heroico (LDC2006S37) ingestion...")

        # Discover structure
        logger.info("Discovering structure...")
        structure = self.discover_structure()

        # Discover audio
        logger.info("Discovering audio...")
        audio_info = self.discover_audio()

        # Discover transcripts
        logger.info("Discovering transcripts...")
        transcript_info = self.discover_transcripts()

        # Get metadata
        logger.info("Loading metadata...")
        speakers = self.discover_metadata()

        # Link everything
        logger.info("Linking audio and transcripts...")
        utterances = self.link_audio_transcript()

        # Calculate statistics
        total_duration_hours = sum(u.duration_ms for u in utterances) / 1000 / 3600
        files_with_transcripts = sum(1 for u in utterances if u.transcript_text)

        # Count by subcorpus
        subcorpus_counts = defaultdict(int)
        for u in utterances:
            subcorpus = u.extra_fields.get('subcorpus', 'unknown')
            subcorpus_counts[subcorpus] += 1

        # Count by native/nonnative
        native_counts = defaultdict(int)
        for u in utterances:
            is_native = u.extra_fields.get('is_native', True)
            native_counts['native' if is_native else 'nonnative'] += 1

        # Count by country (USMA only)
        country_counts = defaultdict(int)
        for u in utterances:
            if u.extra_fields.get('subcorpus') == 'usma':
                country = u.extra_fields.get('country', 'unknown')
                country_counts[country] += 1

        statistics = {
            'total_utterances': len(utterances),
            'total_speakers': len(speakers),
            'total_duration_hours': total_duration_hours,
            'files_with_transcripts': files_with_transcripts,
            'audio_formats': dict(audio_info['formats']),
            'sample_rates': dict(audio_info['sample_rates']),
            'by_subcorpus': dict(subcorpus_counts),
            'by_native': dict(native_counts),
            'by_country': dict(country_counts),
        }

        # Generate audit report
        logger.info("Generating audit report...")
        self._generate_report(structure, audio_info, transcript_info, speakers, utterances)

        result = IngestionResult(
            dataset_name='heroico',
            speakers=speakers,
            utterances=utterances,
            issues=self.issues,
            statistics=statistics
        )

        # Save parquet
        parquet_path = self.output_dir / 'metadata' / 'heroico_raw.parquet'
        logger.info(f"Saving parquet to {parquet_path}...")
        result.to_raw_parquet(parquet_path)

        return result

    def _generate_report(self, structure: Dict, audio_info: Dict,
                        transcript_info: Dict, speakers: List[SpeakerMetadata],
                        utterances: List[UtteranceRecord]) -> None:
        """Generate audit report."""
        report = AuditReportWriter('heroico', self.output_dir / 'reports' / 'dataset_audit')

        # Summary
        total_duration = sum(u.duration_ms for u in utterances) / 1000 / 3600
        files_with_transcripts = sum(1 for u in utterances if u.transcript_text)

        report.add_summary(
            total_files=len(utterances),
            total_speakers=len(speakers),
            total_duration_hours=total_duration,
            issues_count=len(self.issues),
            files_with_transcripts=files_with_transcripts,
            files_with_alignments=0  # No alignment in Heroico
        )

        # File structure
        tree = "LDC2006S37/ (Heroico)\n"
        tree += "  data/speech/heroico/\n"
        tree += f"    Recordings_Spanish/ ({structure['heroico']['recordings_speakers']} speakers, "
        tree += f"{structure['heroico']['recordings_files']} files)\n"
        tree += f"    Answers_Spanish/ ({structure['heroico']['answers_speakers']} speakers, "
        tree += f"{structure['heroico']['answers_files']} files)\n"
        tree += "  data/speech/usma/\n"
        tree += f"    Native: {structure['usma']['native_speakers']} speakers\n"
        tree += f"    Non-native: {structure['usma']['nonnative_speakers']} speakers\n"
        tree += f"    Total files: {structure['usma']['total_files']}\n"

        file_counts = {
            '.wav (total)': structure['total_files'],
            'heroico_recordings': structure['heroico']['recordings_files'],
            'heroico_answers': structure['heroico']['answers_files'],
            'usma': structure['usma']['total_files'],
        }
        report.add_file_structure(tree, file_counts)

        # Audio analysis
        report.add_audio_analysis(
            format_stats=dict(audio_info['formats']),
            sample_rate_distribution=dict(audio_info['sample_rates']),
            channel_distribution=dict(audio_info['channels']),
            issues=audio_info['issues']
        )

        # Transcript analysis
        coverage = (files_with_transcripts / len(utterances) * 100) if utterances else 0
        report.add_transcript_analysis(
            format_type='Tab-separated (ID<TAB>text)',
            encoding_stats={'latin-1': 3},  # 3 transcript files
            linking_method='Numeric ID matching (file ID ↔ transcript line ID)',
            coverage_percent=coverage,
            special_markers={
                'recordings_lines': transcript_info['recordings']['lines'],
                'answers_lines': transcript_info['answers']['lines'],
                'usma_prompts_lines': transcript_info['usma_prompts']['lines'],
            }
        )

        # Metadata analysis
        sex_dist = defaultdict(int)
        native_dist = defaultdict(int)

        for s in speakers:
            sex_dist[s.sex or 'unknown'] += 1
            is_native = s.extra_fields.get('is_native', True) if s.extra_fields else True
            native_dist['native' if is_native else 'nonnative'] += 1

        fields = ['sex', 'is_native', 'country']
        coverage_pct = {
            'sex': sum(1 for s in speakers if s.sex != 'unknown') / len(speakers) * 100 if speakers else 0,
            'is_native': 100.0,
            'country': sum(1 for s in speakers
                          if s.extra_fields and s.extra_fields.get('country') != 'unknown') / len(speakers) * 100 if speakers else 0,
        }
        report.add_metadata_analysis(fields, coverage_pct, {
            'sex': dict(sex_dist),
            'is_native': dict(native_dist),
        })

        # Issues
        report.add_issues(self.issues)

        # Write report
        report.write()
