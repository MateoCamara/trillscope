"""Main ALBAYZIN corpus ingestor."""

from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

from tqdm import tqdm

from ..base import DatasetIngestor, IngestionResult, SpeakerMetadata, UtteranceRecord
from ..utils.audio import AudioInspector
from ..utils.encoding import EncodingDetector
from ..utils.id_generator import IDGenerator
from ..utils.report_writer import AuditReportWriter
from .sam_parser import SAMParser
from .seo_parser import SEOParser
from .dbf_parser import AlbayzinDBFParser

logger = logging.getLogger(__name__)


class AlbayzinIngestor(DatasetIngestor):
    """
    Ingest ALBAYZIN corpus.

    Structure discovered:
    - 5 subcorpora (Albayzin1-5)
    - Each has: CF/ (corpus fonetico), CG/ (corpus geografico), LOCUTOR/, DOCUMENT/
    - CF contains: SUB_APRE/ (training), SUB_PRUE/ (test), textgrid/ (alignments)
    - Audio: .SES files (SAM format) or .wav files in SUB_APRE_WAV/
    - Transcripts: .SEO files paired with .SES files
    - Metadata: text-formatted DBF files in LOCUTOR/
    """

    SUBCORPORA = ['Albayzin1', 'Albayzin2', 'Albayzin3', 'Albayzin4', 'Albayzin5']
    CORPUS_TYPES = ['CF', 'CG', 'CL']  # Fonetico, Geografico, Lombard

    def __init__(self, dataset_root: Path, output_dir: Path):
        super().__init__(dataset_root, output_dir)
        self.sam_parser = SAMParser()
        self.seo_parser = SEOParser()
        self.dbf_parser = AlbayzinDBFParser()
        self.audio_inspector = AudioInspector()
        self.encoding_detector = EncodingDetector()
        self.id_gen = IDGenerator('ALB')

        # Cache for discovered data
        self._structure_cache: Optional[Dict] = None
        self._speakers_cache: Optional[List[SpeakerMetadata]] = None

    @property
    def corpora_path(self) -> Path:
        """Path to corpora directory."""
        return self.dataset_root / 'ALBAYZIN' / 'corpora'

    def discover_structure(self) -> Dict[str, Any]:
        """
        Discover ALBAYZIN directory structure.

        Returns structure dict with:
        - subcorpora found
        - speaker directories per subcorpus
        - file counts by type (.SES, .SEO, .wav, .TextGrid)
        """
        if self._structure_cache:
            return self._structure_cache

        structure = {
            'subcorpora': {},
            'total_ses_files': 0,
            'total_seo_files': 0,
            'total_wav_files': 0,
            'total_textgrid_files': 0,
            'speaker_dirs': set(),
        }

        for subcorpus in self.SUBCORPORA:
            subpath = self.corpora_path / subcorpus
            if subpath.exists():
                sub_info = self._analyze_subcorpus(subpath, subcorpus)
                structure['subcorpora'][subcorpus] = sub_info
                structure['total_ses_files'] += sub_info.get('ses_count', 0)
                structure['total_seo_files'] += sub_info.get('seo_count', 0)
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
            'ses_count': 0,
            'seo_count': 0,
            'wav_count': 0,
            'textgrid_count': 0,
            'corpus_types': [],
        }

        # Check each corpus type
        for corpus_type in self.CORPUS_TYPES:
            corpus_dir = subpath / corpus_type
            if corpus_dir.exists():
                result['corpus_types'].append(corpus_type)

                # Check SUB_APRE for speaker directories
                sub_apre = corpus_dir / 'SUB_APRE'
                if sub_apre.exists():
                    for speaker_dir in sub_apre.iterdir():
                        if speaker_dir.is_dir() and len(speaker_dir.name) == 2:
                            result['speaker_dirs'].append(speaker_dir.name)
                            result['ses_count'] += len(list(speaker_dir.glob('*.SES')))
                            result['seo_count'] += len(list(speaker_dir.glob('*.SEO')))

                # Check SUB_PRUE for test data
                sub_prue = corpus_dir / 'SUB_PRUE'
                if sub_prue.exists():
                    for speaker_dir in sub_prue.iterdir():
                        if speaker_dir.is_dir() and len(speaker_dir.name) == 2:
                            if speaker_dir.name not in result['speaker_dirs']:
                                result['speaker_dirs'].append(speaker_dir.name)
                            result['ses_count'] += len(list(speaker_dir.glob('*.SES')))
                            result['seo_count'] += len(list(speaker_dir.glob('*.SEO')))

                # Check for WAV files
                sub_apre_wav = corpus_dir / 'SUB_APRE_WAV'
                if sub_apre_wav.exists():
                    result['wav_count'] += len(list(sub_apre_wav.rglob('*.wav')))

                # Check for TextGrid alignments
                textgrid_dir = corpus_dir / 'textgrid'
                if textgrid_dir.exists():
                    result['textgrid_count'] += len(list(textgrid_dir.glob('*.TextGrid')))

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

        # Sample some files to verify format
        sample_files = []

        for subcorpus in self.SUBCORPORA:
            for corpus_type in self.CORPUS_TYPES:
                sub_apre = self.corpora_path / subcorpus / corpus_type / 'SUB_APRE'
                if sub_apre.exists():
                    ses_files = list(sub_apre.rglob('*.SES'))[:5]
                    sample_files.extend(ses_files)

                wav_dir = self.corpora_path / subcorpus / corpus_type / 'SUB_APRE_WAV'
                if wav_dir.exists():
                    wav_files = list(wav_dir.rglob('*.wav'))[:5]
                    sample_files.extend(wav_files)

        # Inspect sampled files
        for audio_file in sample_files[:20]:
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
                audio_info['issues'].append(f"Invalid file: {audio_file} - {metadata.error_message}")

        return audio_info

    def discover_transcripts(self) -> Dict[str, Any]:
        """Discover transcript files and encoding."""
        transcript_info = {
            'format': 'SEO',
            'encodings': defaultdict(int),
            'total_files': 0,
            'linked_count': 0,
            'phoneme_labels_present': 0,
            'issues': [],
        }

        # Sample SEO files
        sample_seo = []
        for subcorpus in self.SUBCORPORA:
            for corpus_type in self.CORPUS_TYPES:
                sub_apre = self.corpora_path / subcorpus / corpus_type / 'SUB_APRE'
                if sub_apre.exists():
                    seo_files = list(sub_apre.rglob('*.SEO'))[:10]
                    sample_seo.extend(seo_files)

        for seo_file in sample_seo[:30]:
            transcript_info['total_files'] += 1

            # Check encoding
            encoding, confidence = self.encoding_detector.detect_encoding(seo_file)
            transcript_info['encodings'][encoding] += 1

            # Parse and check content
            try:
                record = self.seo_parser.parse(seo_file)

                # Check if linked SES file exists
                ses_file = seo_file.with_suffix('.SES')
                if ses_file.exists():
                    transcript_info['linked_count'] += 1

                # Check for phoneme labels
                if record.phoneme_labels:
                    transcript_info['phoneme_labels_present'] += 1

            except Exception as e:
                transcript_info['issues'].append(f"Parse error: {seo_file} - {e}")

        return transcript_info

    def discover_metadata(self) -> List[SpeakerMetadata]:
        """Extract speaker metadata from all DBF files."""
        if self._speakers_cache:
            return self._speakers_cache

        speakers = []
        seen_ids = set()

        for subcorpus in self.SUBCORPORA:
            locutor_dir = self.corpora_path / subcorpus / 'LOCUTOR'
            if locutor_dir.exists():
                speaker_dbf = locutor_dir / 'SPEAKER.DBF'
                if speaker_dbf.exists():
                    try:
                        parsed = self.dbf_parser.parse_speaker_dbf(speaker_dbf)
                        for speaker in parsed:
                            if speaker.speaker_id not in seen_ids:
                                seen_ids.add(speaker.speaker_id)
                                speakers.append(speaker)
                    except Exception as e:
                        self.log_issue('error', 'metadata_parse',
                                      f"Failed to parse {speaker_dbf}: {e}")

        self._speakers_cache = speakers
        return speakers

    def link_audio_transcript(self) -> List[UtteranceRecord]:
        """
        Link audio files with transcripts.

        Linking strategy:
        - SES files match SEO files by filename stem
        - WAV files match by filename pattern
        """
        utterances = []
        speakers = {s.speaker_id: s for s in self.discover_metadata()}

        for subcorpus in tqdm(self.SUBCORPORA, desc="Processing subcorpora"):
            for corpus_type in self.CORPUS_TYPES:
                # Process SUB_APRE
                sub_apre = self.corpora_path / subcorpus / corpus_type / 'SUB_APRE'
                if sub_apre.exists():
                    utterances.extend(
                        self._process_directory(sub_apre, subcorpus, corpus_type, 'training', speakers)
                    )

                # Process SUB_PRUE
                sub_prue = self.corpora_path / subcorpus / corpus_type / 'SUB_PRUE'
                if sub_prue.exists():
                    utterances.extend(
                        self._process_directory(sub_prue, subcorpus, corpus_type, 'test', speakers)
                    )

                # Process WAV files
                wav_dir = self.corpora_path / subcorpus / corpus_type / 'SUB_APRE_WAV'
                if wav_dir.exists():
                    utterances.extend(
                        self._process_wav_directory(wav_dir, subcorpus, corpus_type, speakers)
                    )

        return utterances

    def _process_directory(self, dir_path: Path, subcorpus: str, corpus_type: str,
                          split: str, speakers: Dict[str, SpeakerMetadata]) -> List[UtteranceRecord]:
        """Process a directory of SES/SEO files."""
        utterances = []

        for speaker_dir in dir_path.iterdir():
            if not speaker_dir.is_dir():
                continue

            speaker_code = speaker_dir.name
            if len(speaker_code) != 2:
                continue

            for ses_file in speaker_dir.glob('*.SES'):
                seo_file = ses_file.with_suffix('.SEO')

                # Get audio metadata
                audio_meta = self.sam_parser.get_audio_metadata(ses_file)

                # Parse transcript if exists
                transcript_text = None
                has_alignment = False
                if seo_file.exists():
                    try:
                        seo_record = self.seo_parser.parse(seo_file)
                        transcript_text = seo_record.orthographic_text
                        has_alignment = len(seo_record.phoneme_labels) > 0
                    except Exception as e:
                        self.log_issue('warning', 'transcript_parse',
                                      f"Failed to parse {seo_file}: {e}")

                utt_id = self.id_gen.from_filename(ses_file.stem)

                utterances.append(UtteranceRecord(
                    utt_id=utt_id,
                    speaker_id=speaker_code,
                    audio_path=str(ses_file),
                    audio_format='sam',
                    sample_rate=audio_meta.sample_rate,
                    duration_ms=audio_meta.duration_seconds * 1000,
                    transcript_path=str(seo_file) if seo_file.exists() else None,
                    transcript_text=transcript_text,
                    transcript_format='seo',
                    has_phoneme_alignment=has_alignment,
                    alignment_path=str(seo_file) if has_alignment else None,
                    original_filename=ses_file.name,
                    extra_fields={
                        'subcorpus': subcorpus,
                        'corpus_type': corpus_type,
                        'split': split,
                        'speech_style': 'read',
                    }
                ))

        return utterances

    def _process_wav_directory(self, wav_dir: Path, subcorpus: str, corpus_type: str,
                               speakers: Dict[str, SpeakerMetadata]) -> List[UtteranceRecord]:
        """Process a directory of WAV files."""
        utterances = []

        for wav_file in wav_dir.rglob('*.wav'):
            audio_meta = self.audio_inspector.inspect_wav(wav_file)

            # Extract speaker code from filename (first 2 chars)
            speaker_code = wav_file.stem[:2] if len(wav_file.stem) >= 2 else 'XX'

            # Look for corresponding SEO in SUB_APRE
            seo_path = None
            transcript_text = None
            has_alignment = False

            potential_seo = self.corpora_path / subcorpus / corpus_type / 'SUB_APRE' / speaker_code / f"{wav_file.stem}.SEO"
            if potential_seo.exists():
                seo_path = str(potential_seo)
                try:
                    seo_record = self.seo_parser.parse(potential_seo)
                    transcript_text = seo_record.orthographic_text
                    has_alignment = len(seo_record.phoneme_labels) > 0
                except Exception:
                    pass

            utt_id = self.id_gen.from_filename(wav_file.stem)

            utterances.append(UtteranceRecord(
                utt_id=utt_id,
                speaker_id=speaker_code,
                audio_path=str(wav_file),
                audio_format='wav',
                sample_rate=audio_meta.sample_rate,
                duration_ms=audio_meta.duration_seconds * 1000,
                transcript_path=seo_path,
                transcript_text=transcript_text,
                transcript_format='seo' if seo_path else 'unknown',
                has_phoneme_alignment=has_alignment,
                alignment_path=seo_path if has_alignment else None,
                original_filename=wav_file.name,
                extra_fields={
                    'subcorpus': subcorpus,
                    'corpus_type': corpus_type,
                    'split': 'training',
                    'speech_style': 'read',
                }
            ))

        return utterances

    def ingest(self) -> IngestionResult:
        """Run complete ALBAYZIN ingestion."""
        logger.info("Starting ALBAYZIN ingestion...")

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
        logger.info("Extracting metadata...")
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
        }

        # Generate audit report
        logger.info("Generating audit report...")
        self._generate_report(structure, audio_info, transcript_info, speakers, utterances)

        result = IngestionResult(
            dataset_name='albayzin',
            speakers=speakers,
            utterances=utterances,
            issues=self.issues,
            statistics=statistics
        )

        # Save parquet
        parquet_path = self.output_dir / 'metadata' / 'albayzin_raw.parquet'
        logger.info(f"Saving parquet to {parquet_path}...")
        result.to_raw_parquet(parquet_path)

        return result

    def _generate_report(self, structure: Dict, audio_info: Dict,
                        transcript_info: Dict, speakers: List[SpeakerMetadata],
                        utterances: List[UtteranceRecord]) -> None:
        """Generate audit report."""
        report = AuditReportWriter('albayzin', self.output_dir / 'reports' / 'dataset_audit')

        # Summary
        total_duration = sum(u.duration_ms for u in utterances) / 1000 / 3600
        files_with_transcripts = sum(1 for u in utterances if u.transcript_text)
        files_with_alignments = sum(1 for u in utterances if u.has_phoneme_alignment)

        report.add_summary(
            total_files=len(utterances),
            total_speakers=len(speakers),
            total_duration_hours=total_duration,
            issues_count=len(self.issues),
            files_with_transcripts=files_with_transcripts,
            files_with_alignments=files_with_alignments
        )

        # File structure
        tree = "ALBAYZIN/corpora/\n"
        for sub, info in structure['subcorpora'].items():
            tree += f"  {sub}/\n"
            tree += f"    CF/SUB_APRE/ ({info['ses_count']} SES files)\n"
            if info['wav_count'] > 0:
                tree += f"    CF/SUB_APRE_WAV/ ({info['wav_count']} WAV files)\n"
            if info['textgrid_count'] > 0:
                tree += f"    CF/textgrid/ ({info['textgrid_count']} TextGrid files)\n"

        file_counts = {
            '.SES': structure['total_ses_files'],
            '.SEO': structure['total_seo_files'],
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
        coverage = (files_with_transcripts / len(utterances) * 100) if utterances else 0
        report.add_transcript_analysis(
            format_type='SEO (ALBAYZIN orthographic labels)',
            encoding_stats=dict(transcript_info['encodings']),
            linking_method='Filename stem matching (XXFA0001.SES ↔ XXFA0001.SEO)',
            coverage_percent=coverage,
            special_markers={'phoneme_labels': transcript_info['phoneme_labels_present']}
        )

        # Metadata analysis
        sex_dist = defaultdict(int)
        age_dist = defaultdict(int)
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

        fields = ['sex', 'age', 'birth_place', 'education']
        coverage = {
            'sex': sum(1 for s in speakers if s.sex) / len(speakers) * 100 if speakers else 0,
            'age': sum(1 for s in speakers if s.age) / len(speakers) * 100 if speakers else 0,
            'birth_place': sum(1 for s in speakers if s.birth_place) / len(speakers) * 100 if speakers else 0,
            'education': sum(1 for s in speakers if s.education) / len(speakers) * 100 if speakers else 0,
        }
        report.add_metadata_analysis(fields, coverage, {
            'sex': dict(sex_dist),
            'age_bin': dict(age_dist),
        })

        # Issues
        report.add_issues(self.issues)

        # Write report
        report.write()
