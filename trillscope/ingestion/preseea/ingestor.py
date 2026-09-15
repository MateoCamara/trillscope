"""Main PRESEEA corpus ingestor."""

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
from .xml_parser import PreseeaXMLParser, PreseeaSpeaker
from .metadata_normalizer import PreseeaMetadataNormalizer, get_city_info

logger = logging.getLogger(__name__)


class PreseeaIngestor(DatasetIngestor):
    """
    Ingest PRESEEA corpus.

    Key characteristics:
    - MP3 audio files (44.1kHz, need conversion to 16kHz)
    - XML transcripts with embedded speaker metadata
    - Semi-directed interviews with overlap markers
    - Multiple cities/countries across Spanish-speaking world
    """

    def __init__(self, dataset_root: Path, output_dir: Path):
        super().__init__(dataset_root, output_dir)
        self.xml_parser = PreseeaXMLParser()
        self.normalizer = PreseeaMetadataNormalizer()
        self.audio_inspector = AudioInspector()
        self.encoding_detector = EncodingDetector()
        self.id_gen = IDGenerator('PRE')

        # Cache
        self._structure_cache: Optional[Dict] = None

    def discover_structure(self) -> Dict[str, Any]:
        """Discover PRESEEA file structure."""
        if self._structure_cache:
            return self._structure_cache

        mp3_files = list(self.dataset_root.glob('*.mp3'))
        txt_files = list(self.dataset_root.glob('*.txt'))

        # Analyze filename patterns to discover cities
        cities_found = defaultdict(int)
        for f in mp3_files:
            parts = f.stem.split('_')
            if len(parts) >= 1:
                city_code = parts[0]
                cities_found[city_code] += 1

        # Count paired files
        mp3_stems = {f.stem for f in mp3_files}
        txt_stems = {f.stem for f in txt_files}
        paired = mp3_stems & txt_stems

        structure = {
            'mp3_count': len(mp3_files),
            'txt_count': len(txt_files),
            'paired_count': len(paired),
            'unpaired_mp3': len(mp3_stems - txt_stems),
            'unpaired_txt': len(txt_stems - mp3_stems),
            'cities_found': dict(cities_found),
        }

        self._structure_cache = structure
        return structure

    def discover_audio(self) -> Dict[str, Any]:
        """Analyze MP3 audio files."""
        audio_info = {
            'formats': defaultdict(int),
            'sample_rates': defaultdict(int),
            'channels': defaultdict(int),
            'total_duration_hours': 0.0,
            'issues': [],
        }

        # Sample some files
        mp3_files = list(self.dataset_root.glob('*.mp3'))[:20]

        for mp3_file in mp3_files:
            metadata = self.audio_inspector.inspect(mp3_file)
            audio_info['formats'][metadata.format] += 1
            audio_info['sample_rates'][metadata.sample_rate] += 1
            audio_info['channels'][metadata.channels] += 1

            if not metadata.is_valid:
                audio_info['issues'].append(f"Invalid: {mp3_file.name} - {metadata.error_message}")

        return audio_info

    def discover_transcripts(self) -> Dict[str, Any]:
        """Analyze XML transcripts."""
        transcript_info = {
            'format': 'XML (PRESEEA custom)',
            'encodings': defaultdict(int),
            'total_files': 0,
            'with_speakers': 0,
            'with_timestamps': 0,
            'overlap_markers': 0,
            'issues': [],
        }

        txt_files = list(self.dataset_root.glob('*.txt'))[:30]

        for txt_file in txt_files:
            transcript_info['total_files'] += 1

            # Check encoding
            encoding, _ = self.encoding_detector.detect_encoding(txt_file)
            transcript_info['encodings'][encoding] += 1

            # Parse and analyze
            try:
                metadata, utterances = self.xml_parser.parse(txt_file)

                if metadata.speakers:
                    transcript_info['with_speakers'] += 1

                has_timestamps = any(u.timestamp for u in utterances)
                if has_timestamps:
                    transcript_info['with_timestamps'] += 1

                has_overlap = any(u.has_overlap for u in utterances)
                if has_overlap:
                    transcript_info['overlap_markers'] += 1

            except Exception as e:
                transcript_info['issues'].append(f"Parse error: {txt_file.name} - {e}")
                self.log_issue('warning', 'transcript_parse', str(e), {'file': txt_file.name})

        return transcript_info

    def discover_metadata(self) -> List[SpeakerMetadata]:
        """Extract speaker metadata from all transcripts."""
        speakers = []
        seen_ids = set()

        for txt_file in tqdm(list(self.dataset_root.glob('*.txt')), desc="Extracting metadata"):
            try:
                metadata, _ = self.xml_parser.parse(txt_file)
                informant = self.xml_parser.get_informant(metadata)

                if informant:
                    # Create unique speaker ID from filename
                    speaker_id = txt_file.stem

                    if speaker_id not in seen_ids:
                        seen_ids.add(speaker_id)
                        speakers.append(self._convert_speaker(informant, txt_file, metadata))

            except Exception as e:
                self.log_issue('warning', 'metadata_parse',
                              f"Failed to parse {txt_file.name}: {e}")

        return speakers

    def _convert_speaker(self, preseea_speaker: PreseeaSpeaker,
                        source_file: Path, metadata) -> SpeakerMetadata:
        """Convert PRESEEA speaker to common format."""
        return SpeakerMetadata(
            speaker_id=source_file.stem,
            sex=preseea_speaker.sex,
            age=preseea_speaker.age,
            birth_place=preseea_speaker.origin if preseea_speaker.origin.lower() not in ['desc', 'desconocido'] else None,
            education=preseea_speaker.education_level,
            profession=preseea_speaker.profession if preseea_speaker.profession.lower() not in ['desc', 'desconocido'] else None,
            extra_fields={
                'age_group': preseea_speaker.age_group,
                'studies': preseea_speaker.studies,
                'city': metadata.city,
                'country': metadata.country,
                'subcorpus': metadata.subcorpus,
            }
        )

    def link_audio_transcript(self) -> List[UtteranceRecord]:
        """Link MP3 files with XML transcripts by filename stem."""
        utterances = []

        mp3_files = {f.stem: f for f in self.dataset_root.glob('*.mp3')}
        txt_files = {f.stem: f for f in self.dataset_root.glob('*.txt')}

        for stem in tqdm(mp3_files.keys(), desc="Linking files"):
            mp3_file = mp3_files[stem]
            txt_file = txt_files.get(stem)

            # Get audio metadata
            audio_meta = self.audio_inspector.inspect(mp3_file)

            # Parse transcript if exists
            transcript_text = None
            has_overlap = False
            city = ''
            country = ''
            speaker_id = stem

            if txt_file:
                try:
                    metadata, utts = self.xml_parser.parse(txt_file)

                    # Combine all utterances text
                    transcript_text = ' '.join(u.text for u in utts if u.text)

                    # Check for overlap
                    has_overlap = any(u.has_overlap for u in utts)

                    # Get location info
                    city = metadata.city
                    country = metadata.country

                except Exception as e:
                    self.log_issue('warning', 'transcript_parse',
                                  f"Failed to parse {txt_file.name}: {e}")

            # Extract city from filename if not from XML
            if not city:
                city_code = stem.split('_')[0] if '_' in stem else ''
                city_info = get_city_info(city_code)
                city = city_info.get('city', '')
                country = city_info.get('country', '')

            utt_id = self.id_gen.from_filename(stem)

            utterances.append(UtteranceRecord(
                utt_id=utt_id,
                speaker_id=speaker_id,
                audio_path=str(mp3_file),
                audio_format='mp3',
                sample_rate=audio_meta.sample_rate,
                duration_ms=audio_meta.duration_seconds * 1000,
                transcript_path=str(txt_file) if txt_file else None,
                transcript_text=transcript_text,
                transcript_format='xml',
                has_phoneme_alignment=False,  # PRESEEA has no phoneme alignment
                alignment_path=None,
                original_filename=mp3_file.name,
                extra_fields={
                    'subcorpus': city,
                    'city': city,
                    'country': country,
                    'speech_style': 'spontaneous',
                    'has_overlap_markers': has_overlap,
                }
            ))

        return utterances

    def ingest(self) -> IngestionResult:
        """Run complete PRESEEA ingestion."""
        logger.info("Starting PRESEEA ingestion...")

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
            'cities_count': len(structure['cities_found']),
            'audio_formats': dict(audio_info['formats']),
            'sample_rates': dict(audio_info['sample_rates']),
        }

        # Generate audit report
        logger.info("Generating audit report...")
        self._generate_report(structure, audio_info, transcript_info, speakers, utterances)

        result = IngestionResult(
            dataset_name='preseea',
            speakers=speakers,
            utterances=utterances,
            issues=self.issues,
            statistics=statistics
        )

        # Save parquet
        parquet_path = self.output_dir / 'metadata' / 'preseea_raw.parquet'
        logger.info(f"Saving parquet to {parquet_path}...")
        result.to_raw_parquet(parquet_path)

        return result

    def _generate_report(self, structure: Dict, audio_info: Dict,
                        transcript_info: Dict, speakers: List[SpeakerMetadata],
                        utterances: List[UtteranceRecord]) -> None:
        """Generate audit report."""
        report = AuditReportWriter('preseea', self.output_dir / 'reports' / 'dataset_audit')

        # Summary
        total_duration = sum(u.duration_ms for u in utterances) / 1000 / 3600
        files_with_transcripts = sum(1 for u in utterances if u.transcript_text)

        report.add_summary(
            total_files=len(utterances),
            total_speakers=len(speakers),
            total_duration_hours=total_duration,
            issues_count=len(self.issues),
            files_with_transcripts=files_with_transcripts,
            files_with_alignments=0  # PRESEEA has no phoneme alignment
        )

        # File structure
        tree = f"preseea/\n"
        tree += f"  *.mp3 ({structure['mp3_count']} files)\n"
        tree += f"  *.txt ({structure['txt_count']} transcripts)\n"
        tree += f"\nCities found ({len(structure['cities_found'])}):\n"
        for city, count in sorted(structure['cities_found'].items(), key=lambda x: -x[1])[:10]:
            city_info = get_city_info(city)
            tree += f"  {city}: {count} files ({city_info.get('city', city)}, {city_info.get('country', '?')})\n"

        file_counts = {
            '.mp3': structure['mp3_count'],
            '.txt': structure['txt_count'],
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
            format_type='XML (PRESEEA custom with <Trans>, <Datos>, <Hablantes> tags)',
            encoding_stats=dict(transcript_info['encodings']),
            linking_method='Filename stem matching (ALCA_H11_037.mp3 ↔ ALCA_H11_037.txt)',
            coverage_percent=coverage,
            special_markers={
                'with_speakers': transcript_info['with_speakers'],
                'with_timestamps': transcript_info['with_timestamps'],
                'with_overlap_markers': transcript_info['overlap_markers'],
            }
        )

        # Metadata analysis
        sex_dist = defaultdict(int)
        age_dist = defaultdict(int)
        edu_dist = defaultdict(int)
        country_dist = defaultdict(int)

        for s in speakers:
            sex_norm = self.normalizer.normalize_sex(s.sex or '')
            sex_dist[sex_norm] += 1

            if s.age:
                if s.age < 30:
                    age_dist['<30'] += 1
                elif s.age <= 55:
                    age_dist['30-55'] += 1
                else:
                    age_dist['>55'] += 1
            else:
                age_dist['unknown'] += 1

            edu_norm = self.normalizer.normalize_education(s.education or '')
            edu_dist[edu_norm] += 1

            country = s.extra_fields.get('country', 'unknown') or 'unknown'
            country_dist[country] += 1

        fields = ['sex', 'age', 'education', 'city', 'country']
        coverage = {
            'sex': sum(1 for s in speakers if s.sex and s.sex.lower() not in ['desc', 'desconocido']) / len(speakers) * 100 if speakers else 0,
            'age': sum(1 for s in speakers if s.age) / len(speakers) * 100 if speakers else 0,
            'education': sum(1 for s in speakers if s.education and s.education.lower() not in ['desc', 'desconocido']) / len(speakers) * 100 if speakers else 0,
            'city': sum(1 for s in speakers if s.extra_fields.get('city')) / len(speakers) * 100 if speakers else 0,
            'country': sum(1 for s in speakers if s.extra_fields.get('country')) / len(speakers) * 100 if speakers else 0,
        }
        report.add_metadata_analysis(fields, coverage, {
            'sex': dict(sex_dist),
            'age_bin': dict(age_dist),
            'education': dict(edu_dist),
            'country': dict(country_dist),
        })

        # Issues
        report.add_issues(self.issues)

        # Write report
        report.write()
