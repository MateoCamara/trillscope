"""Main TEDx Spanish corpus ingestor."""

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

logger = logging.getLogger(__name__)


class TEDxIngestor(DatasetIngestor):
    """
    Ingest TEDx Spanish corpus.

    Structure:
    - speech/ - WAV files (16kHz mono)
    - files/
      - TEDx_Spanish.transcription - transcription file
      - TEDx_Spanish.paths - audio paths
      - Speaker_Info.xls - speaker metadata (Excel)

    Audio naming: TEDX_[F/M]_[speaker_num]_SPA_[file_num].wav
    - F/M: Female/Male
    - speaker_num: 3-digit speaker number (001-142)
    - file_num: 4-digit file number

    Audio: 16kHz mono WAV (ready for MFA)
    No phoneme alignment - requires MFA
    Speech style: Spontaneous (TED talks)
    """

    def __init__(self, dataset_root: Path, output_dir: Path):
        """
        Initialize ingestor.

        Args:
            dataset_root: Path to tedx_spanish_corpus directory
            output_dir: Output directory for metadata and reports
        """
        super().__init__(dataset_root, output_dir)
        self.audio_inspector = AudioInspector()
        self.id_gen = IDGenerator('TEX')

        # Cache
        self._structure_cache: Optional[Dict] = None
        self._speakers_cache: Optional[List[SpeakerMetadata]] = None
        self._transcripts_cache: Dict[str, str] = {}  # audio_id -> text

    @property
    def speech_dir(self) -> Path:
        """Path to speech directory."""
        return self.dataset_root / 'speech'

    @property
    def files_dir(self) -> Path:
        """Path to files directory."""
        return self.dataset_root / 'files'

    def discover_structure(self) -> Dict[str, Any]:
        """Discover TEDx directory structure."""
        if self._structure_cache:
            return self._structure_cache

        structure = {
            'total_wav_files': 0,
            'total_speakers': 0,
            'speakers': set(),
            'has_transcription': False,
            'has_paths': False,
            'has_speaker_info': False,
        }

        # Count WAV files and discover speakers
        if self.speech_dir.exists():
            for wav_file in self.speech_dir.glob('*.wav'):
                structure['total_wav_files'] += 1
                speaker_info = self._parse_filename(wav_file.stem)
                if speaker_info:
                    structure['speakers'].add(speaker_info['speaker_id'])

        structure['total_speakers'] = len(structure['speakers'])
        structure['speakers'] = list(structure['speakers'])

        # Check for metadata files
        structure['has_transcription'] = (self.files_dir / 'TEDx_Spanish.transcription').exists()
        structure['has_paths'] = (self.files_dir / 'TEDx_Spanish.paths').exists()
        structure['has_speaker_info'] = (self.files_dir / 'Speaker_Info.xls').exists()

        self._structure_cache = structure
        return structure

    def _parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """
        Parse audio filename to extract metadata.

        Format: TEDX_[F/M]_[speaker_num]_SPA_[file_num]
        Example: TEDX_F_001_SPA_0001 -> {gender: F, speaker_num: 001, file_num: 0001}
        """
        pattern = r'TEDX_([FM])_(\d{3})_SPA_(\d{4})'
        match = re.match(pattern, filename, re.IGNORECASE)

        if match:
            gender = match.group(1).upper()
            speaker_num = match.group(2)
            file_num = match.group(3)

            return {
                'gender': gender,
                'speaker_num': speaker_num,
                'file_num': file_num,
                'speaker_id': f"TEDX_{gender}_{speaker_num}",
            }
        return None

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
        sample_files = list(self.speech_dir.glob('*.wav'))[:20]

        for audio_file in sample_files:
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
        """Discover transcript file."""
        transcript_info = {
            'format': 'Plain text (transcription<SPACE>audio_id)',
            'total_lines': 0,
            'matched_count': 0,
            'issues': [],
        }

        trans_file = self.files_dir / 'TEDx_Spanish.transcription'
        if trans_file.exists():
            try:
                with open(trans_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            transcript_info['total_lines'] += 1
            except Exception as e:
                transcript_info['issues'].append(f"Error reading transcription: {e}")

        return transcript_info

    def _load_transcripts(self) -> Dict[str, str]:
        """Load all transcripts."""
        if self._transcripts_cache:
            return self._transcripts_cache

        trans_file = self.files_dir / 'TEDx_Spanish.transcription'
        transcripts = {}

        if trans_file.exists():
            try:
                with open(trans_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        # Format: transcription<SPACE>audio_id
                        # Audio ID is at the end, starts with TEDX_
                        match = re.search(r'(TEDX_[FM]_\d{3}_SPA_\d{4})$', line)
                        if match:
                            audio_id = match.group(1)
                            text = line[:match.start()].strip()
                            transcripts[audio_id] = text
                        else:
                            self.log_issue('warning', 'transcript_parse',
                                          f"Could not parse line: {line[:50]}...")

            except Exception as e:
                self.log_issue('error', 'transcript_load',
                              f"Failed to load transcripts: {e}")

        self._transcripts_cache = transcripts
        return transcripts

    def discover_metadata(self) -> List[SpeakerMetadata]:
        """Get speaker metadata."""
        if self._speakers_cache:
            return self._speakers_cache

        speakers = []
        seen_ids = set()

        # Discover speakers from WAV filenames
        if self.speech_dir.exists():
            for wav_file in self.speech_dir.glob('*.wav'):
                info = self._parse_filename(wav_file.stem)
                if info and info['speaker_id'] not in seen_ids:
                    seen_ids.add(info['speaker_id'])
                    speakers.append(SpeakerMetadata(
                        speaker_id=info['speaker_id'],
                        sex='F' if info['gender'] == 'F' else 'M',
                        age=None,  # Not available without parsing XLS
                        birth_place=None,
                        education=None,
                        profession='ted_speaker',
                        extra_fields={
                            'speaker_num': info['speaker_num'],
                            'corpus': 'tedx',
                        }
                    ))

        self._speakers_cache = speakers
        return speakers

    def link_audio_transcript(self) -> List[UtteranceRecord]:
        """Link audio files with transcripts."""
        utterances = []

        # Load transcripts
        transcripts = self._load_transcripts()
        logger.info(f"Loaded {len(transcripts)} transcripts")

        # Process all WAV files
        wav_files = list(self.speech_dir.glob('*.wav'))

        for wav_file in tqdm(wav_files, desc="Processing TEDx"):
            audio_id = wav_file.stem
            info = self._parse_filename(audio_id)

            if not info:
                self.log_issue('warning', 'filename_parse',
                              f"Could not parse filename: {wav_file}")
                continue

            # Get transcript
            transcript_text = transcripts.get(audio_id)
            if not transcript_text:
                self.log_issue('warning', 'missing_transcript',
                              f"No transcript for {audio_id}")

            # Get audio metadata
            try:
                audio_meta = self.audio_inspector.inspect(wav_file)
            except Exception as e:
                self.log_issue('error', 'audio_error',
                              f"Failed to inspect {wav_file}: {e}")
                continue

            utt_id = self.id_gen.from_filename(audio_id)

            utterances.append(UtteranceRecord(
                utt_id=utt_id,
                speaker_id=info['speaker_id'],
                audio_path=str(wav_file),
                audio_format='wav',
                sample_rate=audio_meta.sample_rate,
                duration_ms=audio_meta.duration_seconds * 1000,
                transcript_path=str(self.files_dir / 'TEDx_Spanish.transcription'),
                transcript_text=transcript_text,
                transcript_format='txt',
                has_phoneme_alignment=False,  # Needs MFA
                alignment_path=None,
                original_filename=wav_file.name,
                extra_fields={
                    'subcorpus': 'tedx',
                    'speech_style': 'spontaneous',  # TED talks are extemporaneous
                    'speaker_num': info['speaker_num'],
                }
            ))

        return utterances

    def ingest(self) -> IngestionResult:
        """Run complete TEDx ingestion."""
        logger.info("Starting TEDx Spanish ingestion...")

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

        # Count by gender
        gender_counts = defaultdict(int)
        for s in speakers:
            gender_counts[s.sex] += 1

        statistics = {
            'total_utterances': len(utterances),
            'total_speakers': len(speakers),
            'total_duration_hours': total_duration_hours,
            'files_with_transcripts': files_with_transcripts,
            'audio_formats': dict(audio_info['formats']),
            'sample_rates': dict(audio_info['sample_rates']),
            'by_gender': dict(gender_counts),
        }

        # Generate audit report
        logger.info("Generating audit report...")
        self._generate_report(structure, audio_info, transcript_info, speakers, utterances)

        result = IngestionResult(
            dataset_name='tedx',
            speakers=speakers,
            utterances=utterances,
            issues=self.issues,
            statistics=statistics
        )

        # Save parquet
        parquet_path = self.output_dir / 'metadata' / 'tedx_raw.parquet'
        logger.info(f"Saving parquet to {parquet_path}...")
        result.to_raw_parquet(parquet_path)

        return result

    def _generate_report(self, structure: Dict, audio_info: Dict,
                        transcript_info: Dict, speakers: List[SpeakerMetadata],
                        utterances: List[UtteranceRecord]) -> None:
        """Generate audit report."""
        report = AuditReportWriter('tedx', self.output_dir / 'reports' / 'dataset_audit')

        # Summary
        total_duration = sum(u.duration_ms for u in utterances) / 1000 / 3600
        files_with_transcripts = sum(1 for u in utterances if u.transcript_text)

        report.add_summary(
            total_files=len(utterances),
            total_speakers=len(speakers),
            total_duration_hours=total_duration,
            issues_count=len(self.issues),
            files_with_transcripts=files_with_transcripts,
            files_with_alignments=0  # No alignment in TEDx
        )

        # File structure
        tree = "tedx_spanish_corpus/\n"
        tree += f"  speech/ ({structure['total_wav_files']} WAV files)\n"
        tree += "  files/\n"
        tree += f"    TEDx_Spanish.transcription ({transcript_info['total_lines']} lines)\n"
        tree += "    Speaker_Info.xls\n"

        file_counts = {
            '.wav': structure['total_wav_files'],
            'transcription': 1 if structure['has_transcription'] else 0,
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
            format_type='Plain text (transcription followed by audio_id)',
            encoding_stats={'utf-8': 1},
            linking_method='Audio ID extraction from line ending (TEDX_[F/M]_[num]_SPA_[num])',
            coverage_percent=coverage,
            special_markers={'total_lines': transcript_info['total_lines']}
        )

        # Metadata analysis
        sex_dist = defaultdict(int)
        for s in speakers:
            sex_dist[s.sex or 'unknown'] += 1

        fields = ['sex', 'speaker_num']
        coverage_pct = {
            'sex': 100.0,  # Derived from filename
            'speaker_num': 100.0,
        }
        report.add_metadata_analysis(fields, coverage_pct, {
            'sex': dict(sex_dist),
        })

        # Issues
        report.add_issues(self.issues)

        # Write report
        report.write()
