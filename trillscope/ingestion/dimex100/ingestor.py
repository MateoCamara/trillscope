"""Main DIMEx100 corpus ingestor."""

from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging
import csv

from tqdm import tqdm

from ..base import DatasetIngestor, IngestionResult, SpeakerMetadata, UtteranceRecord
from ..utils.audio import AudioInspector
from ..utils.encoding import EncodingDetector
from ..utils.id_generator import IDGenerator
from ..utils.report_writer import AuditReportWriter
from .phn_parser import DIMExPHNParser
from .dic_parser import DIMExDicParser

logger = logging.getLogger(__name__)


class DIMEx100Ingestor(DatasetIngestor):
    """
    Ingest DIMEx100 corpus.

    Key characteristics:
    - 100 speakers (s001-s100)
    - WAV audio (44.1kHz, needs resampling to 16kHz)
    - Phoneme-level alignments with trill marker 'r('
    - Two categories: 'comunes' (common sentences) and 'individuales'
    - Task types: T22, T44, T54
    - Metadata will be provided externally as CSV
    """

    SPEAKER_RANGE = range(1, 101)  # s001 to s100
    TASKS = ['T22', 'T44', 'T54']
    CATEGORIES = ['comunes', 'individuales']

    def __init__(self, dataset_root: Path, output_dir: Path,
                 metadata_csv: Optional[Path] = None):
        super().__init__(dataset_root, output_dir)
        self.metadata_csv = metadata_csv
        self.phn_parser = DIMExPHNParser()
        self.dic_parser = DIMExDicParser()
        self.audio_inspector = AudioInspector()
        self.encoding_detector = EncodingDetector()
        self.id_gen = IDGenerator('DIM')

        # Cache
        self._structure_cache: Optional[Dict] = None
        self._speakers_cache: Optional[List[SpeakerMetadata]] = None

    def discover_structure(self) -> Dict[str, Any]:
        """Discover DIMEx100 directory structure."""
        if self._structure_cache:
            return self._structure_cache

        structure = {
            'speakers': [],
            'tasks': self.TASKS,
            'categories': self.CATEGORIES,
            'total_wav_files': 0,
            'total_phn_files': 0,
            'total_txt_files': 0,
            'dictionaries': [],
        }

        # Find speaker directories
        for speaker_num in self.SPEAKER_RANGE:
            speaker_id = f's{speaker_num:03d}'
            speaker_dir = self.dataset_root / speaker_id

            if speaker_dir.exists():
                structure['speakers'].append(speaker_id)

                # Count files
                structure['total_wav_files'] += len(list(speaker_dir.rglob('*.wav')))
                structure['total_phn_files'] += len(list(speaker_dir.rglob('*.phn')))
                structure['total_txt_files'] += len(list(speaker_dir.rglob('*.txt')))

        # Find dictionaries
        dic_dir = self.dataset_root / 'diccionarios'
        if dic_dir.exists():
            for dic_file in dic_dir.glob('*.dic'):
                structure['dictionaries'].append(dic_file.name)

        self._structure_cache = structure
        return structure

    def discover_audio(self) -> Dict[str, Any]:
        """Analyze WAV audio files."""
        audio_info = {
            'formats': defaultdict(int),
            'sample_rates': defaultdict(int),
            'channels': defaultdict(int),
            'total_duration_hours': 0.0,
            'issues': [],
        }

        # Sample files from a few speakers
        sample_files = []
        for speaker_id in ['s001', 's010', 's050', 's100']:
            speaker_dir = self.dataset_root / speaker_id / 'audio_editado' / 'comunes'
            if speaker_dir.exists():
                wav_files = list(speaker_dir.glob('*.wav'))[:3]
                sample_files.extend(wav_files)

        for wav_file in sample_files[:20]:
            metadata = self.audio_inspector.inspect(wav_file)
            audio_info['formats'][metadata.format] += 1
            audio_info['sample_rates'][metadata.sample_rate] += 1
            audio_info['channels'][metadata.channels] += 1

            if not metadata.is_valid:
                audio_info['issues'].append(f"Invalid: {wav_file} - {metadata.error_message}")

        return audio_info

    def discover_transcripts(self) -> Dict[str, Any]:
        """Discover transcript and alignment files."""
        transcript_info = {
            'format': 'PHN (phoneme alignment) + TXT (text)',
            'encodings': defaultdict(int),
            'total_phn_files': 0,
            'total_txt_files': 0,
            'trill_segments': 0,
            'issues': [],
        }

        # Sample some PHN files
        sample_phn = []
        for speaker_id in ['s001', 's010', 's050']:
            for task in self.TASKS:
                phn_dir = self.dataset_root / speaker_id / task / 'comunes'
                if phn_dir.exists():
                    phn_files = list(phn_dir.glob('*.phn'))[:3]
                    sample_phn.extend(phn_files)

        for phn_file in sample_phn[:20]:
            transcript_info['total_phn_files'] += 1

            # Check encoding
            encoding, _ = self.encoding_detector.detect_encoding(phn_file)
            transcript_info['encodings'][encoding] += 1

            # Count trill segments
            try:
                trills = self.phn_parser.find_trill_segments(phn_file)
                transcript_info['trill_segments'] += len(trills)
            except Exception as e:
                transcript_info['issues'].append(f"Parse error: {phn_file.name} - {e}")

        return transcript_info

    def discover_metadata(self) -> List[SpeakerMetadata]:
        """
        Load metadata from external CSV if provided.

        If no CSV provided, generate placeholder metadata.
        """
        if self._speakers_cache:
            return self._speakers_cache

        speakers = []

        if self.metadata_csv and self.metadata_csv.exists():
            speakers = self._load_metadata_csv()
        else:
            # Generate placeholder metadata
            self.log_issue('warning', 'metadata',
                          'No metadata CSV provided, using placeholder values')
            speakers = self._generate_placeholder_metadata()

        self._speakers_cache = speakers
        return speakers

    def _load_metadata_csv(self) -> List[SpeakerMetadata]:
        """Load speaker metadata from CSV file."""
        speakers = []

        with open(self.metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                speaker_id = row.get('speaker_id', row.get('id', ''))
                if not speaker_id:
                    continue

                # Parse age if present
                age = None
                age_str = row.get('age', row.get('edad', ''))
                if age_str:
                    try:
                        age = int(age_str)
                    except ValueError:
                        pass

                speakers.append(SpeakerMetadata(
                    speaker_id=speaker_id,
                    sex=row.get('sex', row.get('sexo')),
                    age=age,
                    birth_place=row.get('birth_place', row.get('origen')),
                    education=row.get('education', row.get('educacion')),
                    profession=row.get('profession', row.get('profesion')),
                    extra_fields={k: v for k, v in row.items()
                                 if k not in ['speaker_id', 'id', 'sex', 'sexo',
                                            'age', 'edad', 'birth_place', 'origen',
                                            'education', 'educacion', 'profession', 'profesion']}
                ))

        return speakers

    def _generate_placeholder_metadata(self) -> List[SpeakerMetadata]:
        """Generate placeholder metadata for all speakers."""
        speakers = []
        structure = self.discover_structure()

        for speaker_id in structure['speakers']:
            speakers.append(SpeakerMetadata(
                speaker_id=speaker_id,
                sex=None,
                age=None,
                birth_place=None,
                education=None,
                profession=None,
                extra_fields={'placeholder': True}
            ))

        return speakers

    def link_audio_transcript(self) -> List[UtteranceRecord]:
        """
        Link audio files with transcripts and phoneme alignments.

        Linking by filename pattern:
        - audio_editado/comunes/s00101.wav
        - texto/comunes/s00101.txt
        - T22/comunes/s00101.phn
        """
        utterances = []
        structure = self.discover_structure()

        for speaker_id in tqdm(structure['speakers'], desc="Processing speakers"):
            speaker_dir = self.dataset_root / speaker_id

            # Process both categories
            for category in self.CATEGORIES:
                audio_dir = speaker_dir / 'audio_editado' / category
                if not audio_dir.exists():
                    continue

                for wav_file in audio_dir.glob('*.wav'):
                    # Get audio metadata
                    audio_meta = self.audio_inspector.inspect(wav_file)

                    # Look for transcript
                    txt_path = speaker_dir / 'texto' / category / f"{wav_file.stem}.txt"
                    transcript_text = None
                    if txt_path.exists():
                        try:
                            content, _ = self.encoding_detector.read_with_fallback(txt_path)
                            transcript_text = content.strip()
                        except Exception:
                            pass

                    # Look for phoneme alignment (try each task)
                    phn_path = None
                    trill_count = 0
                    for task in self.TASKS:
                        potential_phn = speaker_dir / task / category / f"{wav_file.stem}.phn"
                        if potential_phn.exists():
                            phn_path = potential_phn
                            # Count trills
                            try:
                                trills = self.phn_parser.find_trill_segments(potential_phn)
                                trill_count = len(trills)
                            except Exception:
                                pass
                            break

                    # Also check tp/ directory
                    if not phn_path:
                        tp_phn = speaker_dir / 'tp' / category / f"{wav_file.stem}.phn"
                        if tp_phn.exists():
                            phn_path = tp_phn

                    utt_id = self.id_gen.from_filename(wav_file.stem)

                    utterances.append(UtteranceRecord(
                        utt_id=utt_id,
                        speaker_id=speaker_id,
                        audio_path=str(wav_file),
                        audio_format='wav',
                        sample_rate=audio_meta.sample_rate,
                        duration_ms=audio_meta.duration_seconds * 1000,
                        transcript_path=str(txt_path) if txt_path.exists() else None,
                        transcript_text=transcript_text,
                        transcript_format='txt',
                        has_phoneme_alignment=phn_path is not None,
                        alignment_path=str(phn_path) if phn_path else None,
                        original_filename=wav_file.name,
                        extra_fields={
                            'subcorpus': category,
                            'speech_style': 'read',
                            'trill_count': trill_count,
                        }
                    ))

        return utterances

    def _analyze_dictionaries(self) -> Dict[str, Any]:
        """Analyze pronunciation dictionaries for trill coverage."""
        dic_info = {
            'dictionaries': {},
            'total_trill_words': 0,
            'trill_contexts': {},
        }

        dic_dir = self.dataset_root / 'diccionarios'
        if not dic_dir.exists():
            return dic_info

        for dic_name in ['T22.full.dic', 'T44.full.dic', 'T54.full.dic']:
            dic_path = dic_dir / dic_name
            if dic_path.exists():
                try:
                    trill_words = self.dic_parser.find_trill_words(dic_path)
                    entries = self.dic_parser.parse(dic_path)

                    dic_info['dictionaries'][dic_name] = {
                        'total_entries': len(entries),
                        'trill_words': len(trill_words),
                    }
                    dic_info['total_trill_words'] += len(trill_words)

                    # Analyze trill contexts
                    contexts = self.dic_parser.get_trill_context_patterns(dic_path)
                    for pattern, count in contexts.items():
                        if pattern not in dic_info['trill_contexts']:
                            dic_info['trill_contexts'][pattern] = 0
                        dic_info['trill_contexts'][pattern] += count

                except Exception as e:
                    self.log_issue('warning', 'dictionary_parse', f"Failed to parse {dic_name}: {e}")

        return dic_info

    def ingest(self) -> IngestionResult:
        """Run complete DIMEx100 ingestion."""
        logger.info("Starting DIMEx100 ingestion...")

        # Discover structure
        logger.info("Discovering structure...")
        structure = self.discover_structure()

        # Discover audio
        logger.info("Discovering audio...")
        audio_info = self.discover_audio()

        # Discover transcripts
        logger.info("Discovering transcripts...")
        transcript_info = self.discover_transcripts()

        # Analyze dictionaries
        logger.info("Analyzing dictionaries...")
        dic_info = self._analyze_dictionaries()

        # Get metadata
        logger.info("Extracting metadata...")
        speakers = self.discover_metadata()

        # Link everything
        logger.info("Linking audio and transcripts...")
        utterances = self.link_audio_transcript()

        # Calculate statistics
        total_duration_hours = sum(u.duration_ms for u in utterances) / 1000 / 3600
        total_trills = sum(u.extra_fields.get('trill_count', 0) for u in utterances)

        statistics = {
            'total_utterances': len(utterances),
            'total_speakers': len(speakers),
            'total_duration_hours': total_duration_hours,
            'total_trill_segments': total_trills,
            'audio_formats': dict(audio_info['formats']),
            'sample_rates': dict(audio_info['sample_rates']),
            'dictionaries': dic_info['dictionaries'],
        }

        # Generate audit report
        logger.info("Generating audit report...")
        self._generate_report(structure, audio_info, transcript_info, dic_info, speakers, utterances)

        result = IngestionResult(
            dataset_name='dimex100',
            speakers=speakers,
            utterances=utterances,
            issues=self.issues,
            statistics=statistics
        )

        # Save parquet
        parquet_path = self.output_dir / 'metadata' / 'dimex100_raw.parquet'
        logger.info(f"Saving parquet to {parquet_path}...")
        result.to_raw_parquet(parquet_path)

        return result

    def _generate_report(self, structure: Dict, audio_info: Dict,
                        transcript_info: Dict, dic_info: Dict,
                        speakers: List[SpeakerMetadata],
                        utterances: List[UtteranceRecord]) -> None:
        """Generate audit report."""
        report = AuditReportWriter('dimex100', self.output_dir / 'reports' / 'dataset_audit')

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
        tree = "CorpusDimex100/\n"
        tree += f"  diccionarios/ ({len(structure['dictionaries'])} .dic files)\n"
        tree += f"  s001-s{len(structure['speakers']):03d}/ ({len(structure['speakers'])} speakers)\n"
        tree += "    audio_editado/comunes/, individuales/\n"
        tree += "    texto/comunes/, individuales/\n"
        tree += "    T22/, T44/, T54/ (phoneme alignments)\n"
        tree += "    tp/ (word alignments)\n"

        file_counts = {
            '.wav': structure['total_wav_files'],
            '.phn': structure['total_phn_files'],
            '.txt': structure['total_txt_files'],
            '.dic': len(structure['dictionaries']),
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
            format_type='PHN (millisecond phoneme alignment) + TXT (plain text)',
            encoding_stats=dict(transcript_info['encodings']),
            linking_method='Filename stem matching (s00101.wav ↔ s00101.txt ↔ s00101.phn)',
            coverage_percent=coverage,
            special_markers={
                'phoneme_files': transcript_info['total_phn_files'],
                'trill_segments_sampled': transcript_info['trill_segments'],
            }
        )

        # Dictionary analysis section
        dic_content = "### Pronunciation Dictionaries\n\n"
        dic_content += "| Dictionary | Entries | Trill Words |\n|------------|---------|-------------|\n"
        for dic_name, info in dic_info.get('dictionaries', {}).items():
            dic_content += f"| {dic_name} | {info['total_entries']:,} | {info['trill_words']:,} |\n"

        dic_content += f"\n**Total trill words**: {dic_info.get('total_trill_words', 0):,}\n"

        if dic_info.get('trill_contexts'):
            dic_content += "\n### Top Trill Contexts (phoneme_r(_phoneme)\n\n"
            sorted_contexts = sorted(dic_info['trill_contexts'].items(), key=lambda x: -x[1])[:10]
            for pattern, count in sorted_contexts:
                dic_content += f"- `{pattern}`: {count}\n"

        report.add_custom_section("Dictionary Analysis", dic_content)

        # Metadata analysis
        has_metadata = any(s.sex or s.age for s in speakers)

        if has_metadata:
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

            fields = ['sex', 'age', 'education']
            coverage = {
                'sex': sum(1 for s in speakers if s.sex) / len(speakers) * 100 if speakers else 0,
                'age': sum(1 for s in speakers if s.age) / len(speakers) * 100 if speakers else 0,
                'education': sum(1 for s in speakers if s.education) / len(speakers) * 100 if speakers else 0,
            }
            report.add_metadata_analysis(fields, coverage, {
                'sex': dict(sex_dist),
                'age_bin': dict(age_dist),
            })
        else:
            report.add_custom_section("Metadata Analysis",
                "**Note**: No speaker metadata CSV provided. Metadata fields are empty.\n\n"
                "To add metadata, provide a CSV file with columns: speaker_id, sex, age, education")

        # Issues
        report.add_issues(self.issues)

        # Write report
        report.write()
