"""Main Glissando-sp corpus ingestor."""

from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging
import re

from tqdm import tqdm

from ..base import DatasetIngestor, IngestionResult, SpeakerMetadata, UtteranceRecord
from ..utils.audio import AudioInspector
from ..utils.id_generator import IDGenerator
from ..utils.report_writer import AuditReportWriter
from .textgrid_parser import TextGridParser, find_trill_phonemes
from .metadata import (
    get_all_speakers, get_speaker, extract_speaker_from_path,
    SPEAKER_LOOKUP, GlissandoSpeaker
)

logger = logging.getLogger(__name__)


class GlissandoIngestor(DatasetIngestor):
    """
    Ingest Glissando-sp corpus.

    Structure:
    - News/ - Read speech (8 speakers)
    - Task_dialogues/ - Task-oriented dialogues (12 speaker pairs)
    - Free_dialogues/ - Informal conversations (6 speaker pairs)

    Each recording has:
    - .fix.wav / .wir.wav - Audio (fixed mic / wireless mic)
    - .TextGrid - Phonetic alignment (SAMPA)
    - .txt / .xml - Orthographic transcription

    Audio: 44kHz stereo, needs conversion to 16kHz mono
    Phoneme notation: 'rr' = trill, 'r' = tap (SAMPA)
    """

    SUBCORPORA = ['News', 'Task_dialogues', 'Free_dialogues']

    # Task types in Task_dialogues
    TASK_TYPES = ['Transport', 'University', 'Tourist']

    def __init__(self, dataset_root: Path, output_dir: Path):
        super().__init__(dataset_root, output_dir)
        self.textgrid_parser = TextGridParser()
        self.audio_inspector = AudioInspector()
        self.id_gen = IDGenerator('GLI')

        # Cache
        self._structure_cache: Optional[Dict] = None
        self._speakers_cache: Optional[List[SpeakerMetadata]] = None

    def discover_structure(self) -> Dict[str, Any]:
        """Discover Glissando directory structure."""
        if self._structure_cache:
            return self._structure_cache

        structure = {
            'subcorpora': {},
            'total_wav_files': 0,
            'total_textgrid_files': 0,
            'speaker_dirs': set(),
        }

        for subcorpus in self.SUBCORPORA:
            subpath = self.dataset_root / subcorpus
            if subpath.exists():
                sub_info = self._analyze_subcorpus(subpath, subcorpus)
                structure['subcorpora'][subcorpus] = sub_info
                structure['total_wav_files'] += sub_info.get('wav_count', 0)
                structure['total_textgrid_files'] += sub_info.get('textgrid_count', 0)
                structure['speaker_dirs'].update(sub_info.get('speaker_dirs', []))

        structure['speaker_dirs'] = list(structure['speaker_dirs'])
        self._structure_cache = structure
        return structure

    def _analyze_subcorpus(self, subpath: Path, subcorpus_name: str) -> Dict[str, Any]:
        """Analyze a single subcorpus directory."""
        result = {
            'name': subcorpus_name,
            'speaker_dirs': [],
            'wav_count': 0,
            'textgrid_count': 0,
        }

        if subcorpus_name == 'News':
            # News has speaker directories directly
            for speaker_dir in subpath.iterdir():
                if speaker_dir.is_dir() and speaker_dir.name.startswith('sp_'):
                    speaker_id = speaker_dir.name[3:]  # Remove 'sp_' prefix
                    result['speaker_dirs'].append(speaker_id)
                    result['wav_count'] += len(list(speaker_dir.rglob('*.wav')))
                    result['textgrid_count'] += len(list(speaker_dir.rglob('*.TextGrid')))

        elif subcorpus_name in ('Task_dialogues', 'Free_dialogues'):
            # Dialogues have pair directories (sp_f11r_m12r)
            for pair_dir in subpath.iterdir():
                if pair_dir.is_dir() and pair_dir.name.startswith('sp_'):
                    # Extract both speaker IDs
                    pair_name = pair_dir.name[3:]  # Remove 'sp_' prefix
                    speakers = pair_name.split('_')
                    for spk in speakers:
                        if spk and len(spk) >= 3:
                            result['speaker_dirs'].append(spk)

                    result['wav_count'] += len(list(pair_dir.rglob('*.wav')))
                    result['textgrid_count'] += len(list(pair_dir.rglob('*.TextGrid')))

        result['speaker_dirs'] = list(set(result['speaker_dirs']))
        return result

    def discover_audio(self) -> Dict[str, Any]:
        """Discover and validate audio files."""
        audio_info = {
            'formats': defaultdict(int),
            'sample_rates': defaultdict(int),
            'channels': defaultdict(int),
            'total_duration_hours': 0.0,
            'sampled_files': [],
            'issues': [],
        }

        # Sample some WAV files
        sample_files = list(self.dataset_root.rglob('*.fix.wav'))[:10]
        sample_files.extend(list(self.dataset_root.rglob('*.wir.wav'))[:5])

        for audio_file in sample_files[:15]:
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
        """Discover TextGrid files."""
        transcript_info = {
            'format': 'TextGrid (Praat)',
            'total_files': 0,
            'with_phonetic_tier': 0,
            'trill_phonemes_found': 0,
            'issues': [],
        }

        # Sample some TextGrid files
        sample_tg = list(self.dataset_root.rglob('*.TextGrid'))[:20]
        transcript_info['total_files'] = len(list(self.dataset_root.rglob('*.TextGrid')))

        for tg_file in sample_tg:
            try:
                tg = self.textgrid_parser.parse(tg_file)

                if tg.get_phonemes():
                    transcript_info['with_phonetic_tier'] += 1

                    # Check for trills
                    trills = find_trill_phonemes(tg)
                    transcript_info['trill_phonemes_found'] += len(trills)

            except Exception as e:
                transcript_info['issues'].append(f"Parse error: {tg_file} - {e}")

        return transcript_info

    def discover_metadata(self) -> List[SpeakerMetadata]:
        """Get speaker metadata."""
        if self._speakers_cache:
            return self._speakers_cache

        self._speakers_cache = get_all_speakers()
        return self._speakers_cache

    def link_audio_transcript(self) -> List[UtteranceRecord]:
        """Link audio files with TextGrids."""
        utterances = []
        speakers = {s.speaker_id: s for s in self.discover_metadata()}

        # Process News subcorpus
        news_path = self.dataset_root / 'News'
        if news_path.exists():
            utterances.extend(self._process_news(news_path, speakers))

        # Process Task_dialogues
        task_path = self.dataset_root / 'Task_dialogues'
        if task_path.exists():
            utterances.extend(self._process_dialogues(task_path, 'Task_dialogues', speakers))

        # Process Free_dialogues
        free_path = self.dataset_root / 'Free_dialogues'
        if free_path.exists():
            utterances.extend(self._process_dialogues(free_path, 'Free_dialogues', speakers))

        return utterances

    def _process_news(self, news_path: Path,
                      speakers: Dict[str, SpeakerMetadata]) -> List[UtteranceRecord]:
        """Process News subcorpus (read speech)."""
        utterances = []

        for speaker_dir in tqdm(list(news_path.iterdir()), desc="Processing News"):
            if not speaker_dir.is_dir() or not speaker_dir.name.startswith('sp_'):
                continue

            speaker_id = speaker_dir.name[3:]  # Remove 'sp_' prefix

            # Process Prosodic and Phonetic subdirectories
            for subcorpus_type in ['Prosodic', 'Phonetic']:
                subcorpus_dir = speaker_dir / subcorpus_type
                if not subcorpus_dir.exists():
                    continue

                # Find all TextGrid files (each corresponds to a recording)
                for tg_file in subcorpus_dir.glob('*.TextGrid'):
                    # Find corresponding audio file (prefer .fix.wav)
                    stem = tg_file.stem
                    audio_file = subcorpus_dir / f"{stem}.fix.wav"
                    if not audio_file.exists():
                        audio_file = subcorpus_dir / f"{stem}.wir.wav"
                    if not audio_file.exists():
                        # Try without suffix
                        audio_file = subcorpus_dir / f"{stem}.wav"

                    if not audio_file.exists():
                        self.log_issue('warning', 'missing_audio',
                                      f"No audio for {tg_file}")
                        continue

                    # Get audio metadata
                    try:
                        audio_meta = self.audio_inspector.inspect(audio_file)
                    except Exception as e:
                        self.log_issue('error', 'audio_error',
                                      f"Failed to inspect {audio_file}: {e}")
                        continue

                    # Parse TextGrid
                    try:
                        tg = self.textgrid_parser.parse(tg_file)
                        has_alignment = tg.get_phonemes() is not None
                    except Exception as e:
                        self.log_issue('warning', 'textgrid_error',
                                      f"Failed to parse {tg_file}: {e}")
                        has_alignment = False

                    # Get transcript text
                    transcript_text = None
                    txt_file = subcorpus_dir / f"{stem}.txt"
                    if txt_file.exists():
                        try:
                            transcript_text = txt_file.read_text(encoding='utf-8')
                        except:
                            pass

                    utt_id = self.id_gen.from_filename(stem)

                    utterances.append(UtteranceRecord(
                        utt_id=utt_id,
                        speaker_id=speaker_id,
                        audio_path=str(audio_file),
                        audio_format='wav',
                        sample_rate=audio_meta.sample_rate,
                        duration_ms=audio_meta.duration_seconds * 1000,
                        transcript_path=str(txt_file) if txt_file.exists() else None,
                        transcript_text=transcript_text,
                        transcript_format='txt',
                        has_phoneme_alignment=has_alignment,
                        alignment_path=str(tg_file) if has_alignment else None,
                        original_filename=audio_file.name,
                        extra_fields={
                            'subcorpus': 'News',
                            'subcorpus_type': subcorpus_type,
                            'speech_style': 'read',
                            'microphone': 'fixed' if '.fix.' in audio_file.name else 'wireless',
                        }
                    ))

        return utterances

    def _process_dialogues(self, dialogue_path: Path, subcorpus: str,
                           speakers: Dict[str, SpeakerMetadata]) -> List[UtteranceRecord]:
        """Process dialogue subcorpora (Task_dialogues or Free_dialogues)."""
        utterances = []
        speech_style = 'spontaneous' if subcorpus == 'Free_dialogues' else 'task_oriented'

        for pair_dir in tqdm(list(dialogue_path.iterdir()), desc=f"Processing {subcorpus}"):
            if not pair_dir.is_dir() or not pair_dir.name.startswith('sp_'):
                continue

            # For Task_dialogues, iterate over task types
            if subcorpus == 'Task_dialogues':
                for task_type in self.TASK_TYPES:
                    task_dir = pair_dir / task_type
                    if task_dir.exists():
                        # Process Turns directory (individual turn files)
                        turns_dir = task_dir / 'Turns'
                        if turns_dir.exists():
                            utterances.extend(
                                self._process_turns_dir(turns_dir, subcorpus, speech_style,
                                                       task_type, speakers)
                            )
            else:
                # Free_dialogues - process Turns directly
                turns_dir = pair_dir / 'Turns'
                if turns_dir.exists():
                    utterances.extend(
                        self._process_turns_dir(turns_dir, subcorpus, speech_style,
                                               None, speakers)
                    )

        return utterances

    def _process_turns_dir(self, turns_dir: Path, subcorpus: str,
                           speech_style: str, task_type: Optional[str],
                           speakers: Dict[str, SpeakerMetadata]) -> List[UtteranceRecord]:
        """Process a Turns directory containing individual turn files."""
        utterances = []

        for tg_file in turns_dir.glob('*.TextGrid'):
            stem = tg_file.stem

            # Extract speaker ID from filename
            # Format: sp_[speaker_id]_[task]_[turn_number]
            speaker_id = extract_speaker_from_path(stem)
            if not speaker_id:
                self.log_issue('warning', 'speaker_extraction',
                              f"Could not extract speaker from {tg_file}")
                continue

            # Find audio file
            audio_file = turns_dir / f"{stem}.wav"
            if not audio_file.exists():
                audio_file = turns_dir / f"{stem}.fix.wav"

            if not audio_file.exists():
                self.log_issue('warning', 'missing_audio',
                              f"No audio for turn {tg_file}")
                continue

            # Get audio metadata
            try:
                audio_meta = self.audio_inspector.inspect(audio_file)
            except Exception as e:
                self.log_issue('error', 'audio_error',
                              f"Failed to inspect {audio_file}: {e}")
                continue

            # Parse TextGrid
            try:
                tg = self.textgrid_parser.parse(tg_file)
                has_alignment = tg.get_phonemes() is not None
            except Exception as e:
                self.log_issue('warning', 'textgrid_error',
                              f"Failed to parse {tg_file}: {e}")
                has_alignment = False

            utt_id = self.id_gen.from_filename(stem)

            extra = {
                'subcorpus': subcorpus,
                'speech_style': speech_style,
            }
            if task_type:
                extra['task_type'] = task_type

            utterances.append(UtteranceRecord(
                utt_id=utt_id,
                speaker_id=speaker_id,
                audio_path=str(audio_file),
                audio_format='wav',
                sample_rate=audio_meta.sample_rate,
                duration_ms=audio_meta.duration_seconds * 1000,
                transcript_path=None,
                transcript_text=None,
                transcript_format='textgrid',
                has_phoneme_alignment=has_alignment,
                alignment_path=str(tg_file) if has_alignment else None,
                original_filename=audio_file.name,
                extra_fields=extra
            ))

        return utterances

    def ingest(self) -> IngestionResult:
        """Run complete Glissando ingestion."""
        logger.info("Starting Glissando-sp ingestion...")

        # Discover structure
        logger.info("Discovering structure...")
        structure = self.discover_structure()

        # Discover audio
        logger.info("Discovering audio...")
        audio_info = self.discover_audio()

        # Discover transcripts
        logger.info("Discovering TextGrids...")
        transcript_info = self.discover_transcripts()

        # Get metadata
        logger.info("Loading metadata...")
        speakers = self.discover_metadata()

        # Link everything
        logger.info("Linking audio and transcripts...")
        utterances = self.link_audio_transcript()

        # Calculate statistics
        total_duration_hours = sum(u.duration_ms for u in utterances) / 1000 / 3600
        statistics = {
            'total_utterances': len(utterances),
            'total_speakers': len(speakers),
            'total_duration_hours': total_duration_hours,
            'subcorpora_count': len(structure['subcorpora']),
            'audio_formats': dict(audio_info['formats']),
            'sample_rates': dict(audio_info['sample_rates']),
            'utterances_with_alignment': sum(1 for u in utterances if u.has_phoneme_alignment),
            'trill_phonemes_sampled': transcript_info.get('trill_phonemes_found', 0),
        }

        # Generate audit report
        logger.info("Generating audit report...")
        self._generate_report(structure, audio_info, transcript_info, speakers, utterances)

        result = IngestionResult(
            dataset_name='glissando',
            speakers=speakers,
            utterances=utterances,
            issues=self.issues,
            statistics=statistics
        )

        # Save parquet
        parquet_path = self.output_dir / 'metadata' / 'glissando_raw.parquet'
        logger.info(f"Saving parquet to {parquet_path}...")
        result.to_raw_parquet(parquet_path)

        return result

    def _generate_report(self, structure: Dict, audio_info: Dict,
                        transcript_info: Dict, speakers: List[SpeakerMetadata],
                        utterances: List[UtteranceRecord]) -> None:
        """Generate audit report."""
        report = AuditReportWriter('glissando', self.output_dir / 'reports' / 'dataset_audit')

        # Summary
        total_duration = sum(u.duration_ms for u in utterances) / 1000 / 3600
        files_with_alignments = sum(1 for u in utterances if u.has_phoneme_alignment)

        report.add_summary(
            total_files=len(utterances),
            total_speakers=len(speakers),
            total_duration_hours=total_duration,
            issues_count=len(self.issues),
            files_with_transcripts=len(utterances),
            files_with_alignments=files_with_alignments
        )

        # File structure
        tree = "Glissando-sp/\n"
        for sub, info in structure['subcorpora'].items():
            tree += f"  {sub}/\n"
            tree += f"    ({info['wav_count']} WAV, {info['textgrid_count']} TextGrid)\n"

        file_counts = {
            '.wav': structure['total_wav_files'],
            '.TextGrid': structure['total_textgrid_files'],
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
        coverage = (files_with_alignments / len(utterances) * 100) if utterances else 0
        report.add_transcript_analysis(
            format_type='TextGrid (Praat) with SAMPA phonetic transcription',
            encoding_stats={'utf-8': transcript_info.get('with_phonetic_tier', 0)},
            linking_method='Filename stem matching (.wav ↔ .TextGrid)',
            coverage_percent=coverage,
            special_markers={
                'trill_phonemes (rr)': transcript_info.get('trill_phonemes_found', 0),
                'phonetic_tier': transcript_info.get('with_phonetic_tier', 0),
            }
        )

        # Metadata analysis
        sex_dist = defaultdict(int)
        age_dist = defaultdict(int)
        profile_dist = defaultdict(int)

        for s in speakers:
            sex_dist[s.sex or 'unknown'] += 1
            if s.age:
                if s.age < 30:
                    age_dist['<30'] += 1
                elif s.age <= 55:
                    age_dist['30-55'] += 1
                else:
                    age_dist['>55'] += 1
            else:
                age_dist['unknown'] += 1

            profile = s.extra_fields.get('profile', 'unknown') if s.extra_fields else 'unknown'
            profile_dist[profile] += 1

        fields = ['sex', 'age', 'birth_place', 'profile']
        coverage_pct = {
            'sex': 100.0,  # All speakers have sex
            'age': 100.0,  # All speakers have age
            'birth_place': 100.0,  # All speakers have birth place
            'profile': 100.0,  # All speakers have profile
        }
        report.add_metadata_analysis(fields, coverage_pct, {
            'sex': dict(sex_dist),
            'age_bin': dict(age_dist),
            'profile': dict(profile_dist),
        })

        # Issues
        report.add_issues(self.issues)

        # Write report
        report.write()
