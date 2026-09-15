"""Main M-AILABS corpus ingestor for Latin American Spanish variants."""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging
import csv

from tqdm import tqdm

from ..base import DatasetIngestor, IngestionResult, SpeakerMetadata, UtteranceRecord
from ..utils.audio import AudioInspector
from ..utils.id_generator import IDGenerator
from ..utils.report_writer import AuditReportWriter

logger = logging.getLogger(__name__)


# M-AILABS regional variants mapping
MAILABS_VARIANTS = {
    'es_ar_female': {'country': 'argentina', 'sex': 'F', 'code': 'arf'},
    'es_ar_male': {'country': 'argentina', 'sex': 'M', 'code': 'arm'},
    'es_cl_female': {'country': 'chile', 'sex': 'F', 'code': 'clf'},
    'es_cl_male': {'country': 'chile', 'sex': 'M', 'code': 'clm'},
    'es_co_female': {'country': 'colombia', 'sex': 'F', 'code': 'cof'},
    'es_co_male': {'country': 'colombia', 'sex': 'M', 'code': 'com'},
    'es_pe_male': {'country': 'peru', 'sex': 'M', 'code': 'pem'},
    'es_ve_female': {'country': 'venezuela', 'sex': 'F', 'code': 'vef'},
    'es_ve_male': {'country': 'venezuela', 'sex': 'M', 'code': 'vem'},
}

# Reverse mapping from code prefix to variant info
CODE_TO_VARIANT = {v['code']: {'country': v['country'], 'sex': v['sex'], 'dir': k}
                   for k, v in MAILABS_VARIANTS.items()}


class MAILABSIngestor(DatasetIngestor):
    """
    Ingest M-AILABS Spanish corpus.

    Structure:
    - 9 regional variants: es_[country]_[gender]/
    - Each directory contains:
      - WAV files (16kHz mono, ready for MFA)
      - line_index.tsv (audio_id<TAB>transcription)
    - Audio naming: [code]_[speaker_id]_[timestamp].wav
      - code: arf (AR female), arm (AR male), etc.
      - speaker_id: 5-digit number
      - timestamp: 11-digit number

    Audio: 16kHz mono WAV (already correct format!)
    No phoneme alignment - requires MFA
    """

    def __init__(self, dataset_root: Path, output_dir: Path):
        """
        Initialize ingestor.

        Args:
            dataset_root: Path to dataset/ directory containing es_* folders
            output_dir: Output directory for metadata and reports
        """
        super().__init__(dataset_root, output_dir)
        self.audio_inspector = AudioInspector()
        self.id_gen = IDGenerator('MAI')

        # Cache
        self._structure_cache: Optional[Dict] = None
        self._speakers_cache: Optional[List[SpeakerMetadata]] = None
        self._transcripts_cache: Dict[str, Dict[str, str]] = {}  # variant -> {audio_id: text}

    def discover_structure(self) -> Dict[str, Any]:
        """Discover M-AILABS directory structure."""
        if self._structure_cache:
            return self._structure_cache

        structure = {
            'variants': {},
            'total_wav_files': 0,
            'total_speakers': 0,
            'countries': set(),
        }

        for variant_dir, info in MAILABS_VARIANTS.items():
            variant_path = self.dataset_root / variant_dir
            if variant_path.exists():
                variant_info = self._analyze_variant(variant_path, variant_dir, info)
                structure['variants'][variant_dir] = variant_info
                structure['total_wav_files'] += variant_info['wav_count']
                structure['total_speakers'] += variant_info['speaker_count']
                structure['countries'].add(info['country'])

        structure['countries'] = list(structure['countries'])
        self._structure_cache = structure
        return structure

    def _analyze_variant(self, variant_path: Path, variant_dir: str,
                         info: Dict) -> Dict[str, Any]:
        """Analyze a single variant directory."""
        result = {
            'name': variant_dir,
            'country': info['country'],
            'sex': info['sex'],
            'wav_count': 0,
            'speaker_count': 0,
            'has_transcript': False,
            'speakers': set(),
        }

        # Count WAV files and discover speakers
        for wav_file in variant_path.glob('*.wav'):
            result['wav_count'] += 1
            # Extract speaker ID from filename
            speaker_id = self._extract_speaker_id(wav_file.name)
            if speaker_id:
                result['speakers'].add(speaker_id)

        result['speaker_count'] = len(result['speakers'])
        result['speakers'] = list(result['speakers'])

        # Check for transcript file
        tsv_path = variant_path / 'line_index.tsv'
        result['has_transcript'] = tsv_path.exists()

        return result

    def _extract_speaker_id(self, filename: str) -> Optional[str]:
        """
        Extract speaker ID from filename.

        Format: [code]_[speaker_id]_[timestamp].wav
        Example: arf_00295_00000740990.wav -> arf_00295
        """
        parts = filename.replace('.wav', '').split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
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

        # Sample some WAV files from each variant
        sample_files = []
        for variant_dir in MAILABS_VARIANTS.keys():
            variant_path = self.dataset_root / variant_dir
            if variant_path.exists():
                wav_files = list(variant_path.glob('*.wav'))[:3]
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
            'format': 'TSV (audio_id<TAB>transcription)',
            'total_files': 0,
            'variants_with_transcripts': 0,
            'total_lines': 0,
            'issues': [],
        }

        for variant_dir in MAILABS_VARIANTS.keys():
            tsv_path = self.dataset_root / variant_dir / 'line_index.tsv'
            if tsv_path.exists():
                transcript_info['variants_with_transcripts'] += 1
                transcript_info['total_files'] += 1

                # Count lines
                try:
                    with open(tsv_path, 'r', encoding='utf-8') as f:
                        lines = sum(1 for _ in f)
                        transcript_info['total_lines'] += lines
                except Exception as e:
                    transcript_info['issues'].append(f"Error reading {tsv_path}: {e}")

        return transcript_info

    def _load_transcripts(self, variant_dir: str) -> Dict[str, str]:
        """Load transcripts for a variant."""
        if variant_dir in self._transcripts_cache:
            return self._transcripts_cache[variant_dir]

        transcripts = {}
        tsv_path = self.dataset_root / variant_dir / 'line_index.tsv'

        if tsv_path.exists():
            try:
                with open(tsv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='\t')
                    for row in reader:
                        if len(row) >= 2:
                            audio_id = row[0].strip()
                            text = row[1].strip()
                            transcripts[audio_id] = text
            except Exception as e:
                self.log_issue('error', 'transcript_load',
                              f"Failed to load transcripts from {tsv_path}: {e}")

        self._transcripts_cache[variant_dir] = transcripts
        return transcripts

    def discover_metadata(self) -> List[SpeakerMetadata]:
        """Get speaker metadata."""
        if self._speakers_cache:
            return self._speakers_cache

        speakers = []
        seen_ids = set()

        for variant_dir, info in MAILABS_VARIANTS.items():
            variant_path = self.dataset_root / variant_dir
            if not variant_path.exists():
                continue

            # Discover speakers from WAV filenames
            for wav_file in variant_path.glob('*.wav'):
                speaker_id = self._extract_speaker_id(wav_file.name)
                if speaker_id and speaker_id not in seen_ids:
                    seen_ids.add(speaker_id)
                    speakers.append(SpeakerMetadata(
                        speaker_id=speaker_id,
                        sex=info['sex'],
                        age=None,  # Not available
                        birth_place=info['country'].title(),
                        education=None,
                        profession=None,
                        extra_fields={
                            'country': info['country'],
                            'variant': variant_dir,
                            'corpus': 'mailabs',
                        }
                    ))

        self._speakers_cache = speakers
        return speakers

    def link_audio_transcript(self) -> List[UtteranceRecord]:
        """Link audio files with transcripts."""
        utterances = []

        for variant_dir, info in tqdm(MAILABS_VARIANTS.items(), desc="Processing variants"):
            variant_path = self.dataset_root / variant_dir
            if not variant_path.exists():
                continue

            # Load transcripts for this variant
            transcripts = self._load_transcripts(variant_dir)

            # Process all WAV files
            for wav_file in variant_path.glob('*.wav'):
                speaker_id = self._extract_speaker_id(wav_file.name)
                audio_id = wav_file.stem  # Filename without extension

                # Get transcript
                transcript_text = transcripts.get(audio_id)
                if not transcript_text:
                    self.log_issue('warning', 'missing_transcript',
                                  f"No transcript for {wav_file}")

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
                    speaker_id=speaker_id or 'unknown',
                    audio_path=str(wav_file),
                    audio_format='wav',
                    sample_rate=audio_meta.sample_rate,
                    duration_ms=audio_meta.duration_seconds * 1000,
                    transcript_path=str(self.dataset_root / variant_dir / 'line_index.tsv'),
                    transcript_text=transcript_text,
                    transcript_format='tsv',
                    has_phoneme_alignment=False,  # Needs MFA
                    alignment_path=None,
                    original_filename=wav_file.name,
                    extra_fields={
                        'subcorpus': variant_dir,
                        'country': info['country'],
                        'speech_style': 'read',
                    }
                ))

        return utterances

    def ingest(self) -> IngestionResult:
        """Run complete M-AILABS ingestion."""
        logger.info("Starting M-AILABS ingestion...")

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
        statistics = {
            'total_utterances': len(utterances),
            'total_speakers': len(speakers),
            'total_duration_hours': total_duration_hours,
            'variants_count': len(structure['variants']),
            'countries': structure['countries'],
            'audio_formats': dict(audio_info['formats']),
            'sample_rates': dict(audio_info['sample_rates']),
        }

        # Count by country
        country_counts = defaultdict(int)
        for u in utterances:
            country = u.extra_fields.get('country', 'unknown')
            country_counts[country] += 1
        statistics['by_country'] = dict(country_counts)

        # Generate audit report
        logger.info("Generating audit report...")
        self._generate_report(structure, audio_info, transcript_info, speakers, utterances)

        result = IngestionResult(
            dataset_name='mailabs',
            speakers=speakers,
            utterances=utterances,
            issues=self.issues,
            statistics=statistics
        )

        # Save parquet
        parquet_path = self.output_dir / 'metadata' / 'mailabs_raw.parquet'
        logger.info(f"Saving parquet to {parquet_path}...")
        result.to_raw_parquet(parquet_path)

        return result

    def _generate_report(self, structure: Dict, audio_info: Dict,
                        transcript_info: Dict, speakers: List[SpeakerMetadata],
                        utterances: List[UtteranceRecord]) -> None:
        """Generate audit report."""
        report = AuditReportWriter('mailabs', self.output_dir / 'reports' / 'dataset_audit')

        # Summary
        total_duration = sum(u.duration_ms for u in utterances) / 1000 / 3600
        files_with_transcripts = sum(1 for u in utterances if u.transcript_text)

        report.add_summary(
            total_files=len(utterances),
            total_speakers=len(speakers),
            total_duration_hours=total_duration,
            issues_count=len(self.issues),
            files_with_transcripts=files_with_transcripts,
            files_with_alignments=0  # No alignment in M-AILABS
        )

        # File structure
        tree = "M-AILABS Spanish/\n"
        for variant, info in structure['variants'].items():
            tree += f"  {variant}/ ({info['country'].title()}, {info['sex']})\n"
            tree += f"    {info['wav_count']} WAV files, {info['speaker_count']} speakers\n"

        file_counts = {
            '.wav': structure['total_wav_files'],
            'line_index.tsv': transcript_info['variants_with_transcripts'],
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
            format_type='TSV (tab-separated: audio_id<TAB>transcription)',
            encoding_stats={'utf-8': transcript_info['variants_with_transcripts']},
            linking_method='Audio ID matching (filename stem ↔ TSV first column)',
            coverage_percent=coverage,
            special_markers={'total_transcript_lines': transcript_info['total_lines']}
        )

        # Metadata analysis
        sex_dist = defaultdict(int)
        country_dist = defaultdict(int)

        for s in speakers:
            sex_dist[s.sex or 'unknown'] += 1
            country = s.extra_fields.get('country', 'unknown') if s.extra_fields else 'unknown'
            country_dist[country] += 1

        fields = ['sex', 'country']
        coverage_pct = {
            'sex': 100.0,  # Derived from directory name
            'country': 100.0,  # Derived from directory name
        }
        report.add_metadata_analysis(fields, coverage_pct, {
            'sex': dict(sex_dist),
            'country': dict(country_dist),
        })

        # Issues
        report.add_issues(self.issues)

        # Write report
        report.write()
