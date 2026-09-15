"""Main Common Voice Spanish corpus ingestor with stratified sampling."""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging
import csv
import random

from tqdm import tqdm

from ..base import DatasetIngestor, IngestionResult, SpeakerMetadata, UtteranceRecord
from ..utils.audio import AudioInspector
from ..utils.id_generator import IDGenerator
from ..utils.report_writer import AuditReportWriter

logger = logging.getLogger(__name__)


# Accent to region mapping
ACCENT_REGION_MAP = {
    'méxico': 'mexico',
    'ciudad de méxico': 'mexico',
    'méxico centro': 'mexico',
    'españa: sur peninsular': 'spain_south',
    'españa: norte peninsular': 'spain_north',
    'españa: centro-sur peninsular': 'spain_center',
    'españa: islas canarias': 'spain_canarias',
    'españa: cataluña': 'spain_catalonia',
    'españa: comunidad valenciana': 'spain_valencia',
    'españa: galicia': 'spain_galicia',
    'andino-pacífico': 'andean',  # Colombia, Peru, Ecuador, Bolivia, Venezuela
    'rioplatense': 'rioplatense',  # Argentina, Uruguay
    'caribe': 'caribbean',  # Cuba, Venezuela, PR, etc.
    'américa central': 'central_america',
    'chileno': 'chile',
    'español de filipinas': 'philippines',
    'estados unidos': 'usa',
    'costa rica': 'central_america',
}

# Age bin mapping
AGE_BIN_MAP = {
    'teens': '<30',
    'twenties': '<30',
    'thirties': '30-55',
    'fourties': '30-55',
    'fifties': '30-55',
    'sixties': '>55',
    'seventies': '>55',
    'eighties': '>55',
    'nineties': '>55',
}

# Gender mapping
GENDER_MAP = {
    'male_masculine': 'M',
    'female_feminine': 'F',
}


class CommonVoiceIngestor(DatasetIngestor):
    """
    Ingest Common Voice Spanish corpus with stratified sampling.

    Structure:
    - clips/ - MP3 audio files
    - validated.tsv - Validated recordings with metadata

    TSV columns:
    - client_id, path, sentence_id, sentence, sentence_domain
    - up_votes, down_votes, age, gender, accents, variant, locale, segment

    Audio: MP3 (needs conversion to WAV 16kHz)
    No phoneme alignment - requires MFA.
    """

    def __init__(self, dataset_root: Path, output_dir: Path,
                 sample_size: int = 15000, seed: int = 42):
        """
        Initialize ingestor.

        Args:
            dataset_root: Path to cv-corpus-*/es directory
            output_dir: Output directory for metadata and reports
            sample_size: Target number of clips to sample (default: 15000)
            seed: Random seed for reproducible sampling
        """
        super().__init__(dataset_root, output_dir)
        self.audio_inspector = AudioInspector()
        self.id_gen = IDGenerator('CVS')  # Common Voice Spanish
        self.sample_size = sample_size
        self.seed = seed

        # Cache
        self._structure_cache: Optional[Dict] = None
        self._speakers_cache: Optional[List[SpeakerMetadata]] = None
        self._metadata_df: Optional[List[Dict]] = None

    @property
    def clips_dir(self) -> Path:
        """Path to clips directory."""
        return self.dataset_root / 'clips'

    @property
    def validated_tsv(self) -> Path:
        """Path to validated.tsv."""
        return self.dataset_root / 'validated.tsv'

    def discover_structure(self) -> Dict[str, Any]:
        """Discover Common Voice directory structure."""
        if self._structure_cache:
            return self._structure_cache

        structure = {
            'total_validated': 0,
            'has_clips_dir': self.clips_dir.exists(),
            'has_validated_tsv': self.validated_tsv.exists(),
            'accents': defaultdict(int),
            'genders': defaultdict(int),
            'ages': defaultdict(int),
            'with_demographics': 0,
        }

        if self.validated_tsv.exists():
            with open(self.validated_tsv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    structure['total_validated'] += 1
                    structure['accents'][row.get('accents', '')] += 1
                    structure['genders'][row.get('gender', '')] += 1
                    structure['ages'][row.get('age', '')] += 1

                    # Count records with complete demographics
                    if (row.get('age') and row.get('gender') and row.get('accents')):
                        structure['with_demographics'] += 1

        self._structure_cache = structure
        return structure

    def _parse_accent(self, accent_str: str) -> Tuple[str, str]:
        """
        Parse accent string to extract region and country.

        Returns (region, country) tuple.
        """
        if not accent_str:
            return 'unknown', 'unknown'

        accent_lower = accent_str.lower()

        # Check for exact matches first
        for key, region in ACCENT_REGION_MAP.items():
            if key in accent_lower:
                # Map region to country
                country = self._region_to_country(region)
                return region, country

        return 'unknown', 'unknown'

    def _region_to_country(self, region: str) -> str:
        """Map region to country."""
        region_country_map = {
            'mexico': 'mexico',
            'spain_south': 'spain',
            'spain_north': 'spain',
            'spain_center': 'spain',
            'spain_canarias': 'spain',
            'spain_catalonia': 'spain',
            'spain_valencia': 'spain',
            'spain_galicia': 'spain',
            'andean': 'andean',  # Multi-country (Colombia, Peru, Ecuador)
            'rioplatense': 'argentina',  # Primarily Argentina
            'caribbean': 'caribbean',  # Multi-country
            'central_america': 'central_america',
            'chile': 'chile',
            'philippines': 'philippines',
            'usa': 'usa',
        }
        return region_country_map.get(region, 'unknown')

    def _load_metadata(self) -> List[Dict]:
        """Load all metadata from validated.tsv."""
        if self._metadata_df is not None:
            return self._metadata_df

        records = []
        if self.validated_tsv.exists():
            with open(self.validated_tsv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    records.append(dict(row))

        self._metadata_df = records
        return records

    def _stratified_sample(self, records: List[Dict]) -> List[Dict]:
        """
        Perform stratified sampling across regions, genders, and ages.

        Strategy:
        1. Filter for records with complete demographics
        2. Group by (region, gender, age_bin)
        3. Sample proportionally, with minimum representation per group
        """
        random.seed(self.seed)

        # Filter for records with complete demographics
        complete_records = [
            r for r in records
            if r.get('age') and r.get('gender') and r.get('accents')
        ]

        logger.info(f"Records with complete demographics: {len(complete_records):,}")

        # Group by (region, gender, age_bin)
        groups: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)

        for r in complete_records:
            region, _ = self._parse_accent(r['accents'])
            gender = GENDER_MAP.get(r['gender'], 'unknown')
            age_bin = AGE_BIN_MAP.get(r['age'], 'unknown')

            # Skip unknown demographics
            if region == 'unknown' or gender == 'unknown' or age_bin == 'unknown':
                continue

            groups[(region, gender, age_bin)].append(r)

        logger.info(f"Total groups: {len(groups)}")

        # Calculate sample size per group
        # Prioritize underrepresented groups
        samples_per_group = max(1, self.sample_size // len(groups)) if groups else 0

        sampled = []
        for key, group_records in groups.items():
            # Sample up to samples_per_group from each group
            n_sample = min(len(group_records), samples_per_group)
            sampled.extend(random.sample(group_records, n_sample))

        # If we haven't reached target, add more from largest groups
        if len(sampled) < self.sample_size:
            remaining = self.sample_size - len(sampled)
            sampled_ids = {r['path'] for r in sampled}

            # Get remaining records not yet sampled
            remaining_records = [
                r for r in complete_records
                if r['path'] not in sampled_ids
            ]

            if remaining_records:
                additional = random.sample(
                    remaining_records,
                    min(remaining, len(remaining_records))
                )
                sampled.extend(additional)

        logger.info(f"Sampled {len(sampled):,} records")
        return sampled

    def discover_audio(self) -> Dict[str, Any]:
        """Discover and validate audio files."""
        audio_info = {
            'formats': defaultdict(int),
            'sample_rates': defaultdict(int),
            'channels': defaultdict(int),
            'sampled_files': [],
            'issues': [],
        }

        # Sample some MP3 files
        if self.clips_dir.exists():
            mp3_files = list(self.clips_dir.glob('*.mp3'))[:20]

            for audio_file in mp3_files:
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
        """Discover transcript information."""
        transcript_info = {
            'format': 'TSV (validated.tsv)',
            'total_lines': 0,
            'issues': [],
        }

        if self.validated_tsv.exists():
            with open(self.validated_tsv, 'r', encoding='utf-8') as f:
                transcript_info['total_lines'] = sum(1 for _ in f) - 1  # Subtract header

        return transcript_info

    def discover_metadata(self) -> List[SpeakerMetadata]:
        """Get speaker metadata from sampled records."""
        if self._speakers_cache:
            return self._speakers_cache

        # Load and sample
        all_records = self._load_metadata()
        sampled_records = self._stratified_sample(all_records)

        # Extract unique speakers (client_ids)
        speakers_dict = {}
        for r in sampled_records:
            client_id = r['client_id']
            if client_id not in speakers_dict:
                region, country = self._parse_accent(r['accents'])
                gender = GENDER_MAP.get(r['gender'], 'unknown')
                age_bin = AGE_BIN_MAP.get(r['age'], 'unknown')

                speakers_dict[client_id] = SpeakerMetadata(
                    speaker_id=f"cv_{client_id[:12]}",  # Truncate long hash
                    sex=gender,
                    age=None,  # Exact age not available
                    birth_place=None,
                    education=None,
                    profession=None,
                    extra_fields={
                        'corpus': 'commonvoice',
                        'client_id': client_id,
                        'region': region,
                        'country': country,
                        'age_bin': age_bin,
                        'accent_original': r.get('accents', ''),
                    }
                )

        self._speakers_cache = list(speakers_dict.values())
        return self._speakers_cache

    def link_audio_transcript(self) -> List[UtteranceRecord]:
        """Link audio files with transcripts for sampled records."""
        utterances = []

        # Load and sample
        all_records = self._load_metadata()
        sampled_records = self._stratified_sample(all_records)

        logger.info(f"Processing {len(sampled_records):,} sampled records...")

        for r in tqdm(sampled_records, desc="Processing Common Voice"):
            audio_path = self.clips_dir / r['path']

            # Skip if audio doesn't exist
            if not audio_path.exists():
                self.log_issue('warning', 'missing_audio',
                              f"Audio not found: {audio_path}")
                continue

            # Get audio metadata
            try:
                audio_meta = self.audio_inspector.inspect(audio_path)
            except Exception as e:
                self.log_issue('error', 'audio_error',
                              f"Failed to inspect {audio_path}: {e}")
                continue

            # Parse metadata
            region, country = self._parse_accent(r['accents'])
            age_bin = AGE_BIN_MAP.get(r['age'], 'unknown')

            utt_id = self.id_gen.from_filename(r['path'].replace('.mp3', ''))

            utterances.append(UtteranceRecord(
                utt_id=utt_id,
                speaker_id=f"cv_{r['client_id'][:12]}",
                audio_path=str(audio_path),
                audio_format='mp3',
                sample_rate=audio_meta.sample_rate,
                duration_ms=audio_meta.duration_seconds * 1000,
                transcript_path=str(self.validated_tsv),
                transcript_text=r['sentence'],
                transcript_format='tsv',
                has_phoneme_alignment=False,
                alignment_path=None,
                original_filename=r['path'],
                extra_fields={
                    'subcorpus': 'commonvoice',
                    'speech_style': 'read',
                    'region': region,
                    'country': country,
                    'age_bin': age_bin,
                    'accent_original': r.get('accents', ''),
                    'up_votes': int(r.get('up_votes', 0)),
                    'down_votes': int(r.get('down_votes', 0)),
                }
            ))

        return utterances

    def ingest(self) -> IngestionResult:
        """Run complete Common Voice ingestion with sampling."""
        logger.info(f"Starting Common Voice ingestion (sample size: {self.sample_size:,})...")

        # Discover structure
        logger.info("Discovering structure...")
        structure = self.discover_structure()
        logger.info(f"Total validated: {structure['total_validated']:,}")
        logger.info(f"With demographics: {structure['with_demographics']:,}")

        # Discover audio
        logger.info("Discovering audio...")
        audio_info = self.discover_audio()

        # Discover transcripts
        logger.info("Discovering transcripts...")
        transcript_info = self.discover_transcripts()

        # Get metadata (triggers sampling)
        logger.info("Sampling and loading metadata...")
        speakers = self.discover_metadata()

        # Link everything
        logger.info("Linking audio and transcripts...")
        utterances = self.link_audio_transcript()

        # Calculate statistics
        total_duration_hours = sum(u.duration_ms for u in utterances) / 1000 / 3600
        files_with_transcripts = sum(1 for u in utterances if u.transcript_text)

        # Count by region
        region_counts = defaultdict(int)
        for u in utterances:
            region = u.extra_fields.get('region', 'unknown')
            region_counts[region] += 1

        # Count by gender
        gender_counts = defaultdict(int)
        for s in speakers:
            gender_counts[s.sex or 'unknown'] += 1

        # Count by age bin
        age_counts = defaultdict(int)
        for s in speakers:
            age_bin = s.extra_fields.get('age_bin', 'unknown') if s.extra_fields else 'unknown'
            age_counts[age_bin] += 1

        statistics = {
            'total_utterances': len(utterances),
            'total_speakers': len(speakers),
            'total_duration_hours': total_duration_hours,
            'files_with_transcripts': files_with_transcripts,
            'audio_formats': dict(audio_info['formats']),
            'sample_rates': dict(audio_info['sample_rates']),
            'total_validated': structure['total_validated'],
            'sample_size': self.sample_size,
            'by_region': dict(region_counts),
            'by_gender': dict(gender_counts),
            'by_age': dict(age_counts),
        }

        # Generate audit report
        logger.info("Generating audit report...")
        self._generate_report(structure, audio_info, transcript_info, speakers, utterances)

        result = IngestionResult(
            dataset_name='commonvoice',
            speakers=speakers,
            utterances=utterances,
            issues=self.issues,
            statistics=statistics
        )

        # Save parquet
        parquet_path = self.output_dir / 'metadata' / 'commonvoice_raw.parquet'
        logger.info(f"Saving parquet to {parquet_path}...")
        result.to_raw_parquet(parquet_path)

        return result

    def _generate_report(self, structure: Dict, audio_info: Dict,
                        transcript_info: Dict, speakers: List[SpeakerMetadata],
                        utterances: List[UtteranceRecord]) -> None:
        """Generate audit report."""
        report = AuditReportWriter('commonvoice', self.output_dir / 'reports' / 'dataset_audit')

        # Summary
        total_duration = sum(u.duration_ms for u in utterances) / 1000 / 3600
        files_with_transcripts = sum(1 for u in utterances if u.transcript_text)

        report.add_summary(
            total_files=len(utterances),
            total_speakers=len(speakers),
            total_duration_hours=total_duration,
            issues_count=len(self.issues),
            files_with_transcripts=files_with_transcripts,
            files_with_alignments=0  # No alignment
        )

        # File structure
        tree = "cv-corpus (Common Voice Spanish)\n"
        tree += f"  Total validated: {structure['total_validated']:,}\n"
        tree += f"  With demographics: {structure['with_demographics']:,}\n"
        tree += f"  Sampled: {len(utterances):,}\n"
        tree += "  clips/ (MP3 audio files)\n"
        tree += f"  validated.tsv ({transcript_info['total_lines']:,} lines)\n"

        file_counts = {
            '.mp3 (sampled)': len(utterances),
            'validated.tsv': 1,
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
            format_type='TSV (validated.tsv with sentence column)',
            encoding_stats={'utf-8': 1},
            linking_method='Path matching (filename in TSV ↔ clips/ directory)',
            coverage_percent=coverage,
            special_markers={
                'total_validated': structure['total_validated'],
                'sample_size': self.sample_size,
            }
        )

        # Metadata analysis
        sex_dist = defaultdict(int)
        region_dist = defaultdict(int)
        age_dist = defaultdict(int)

        for s in speakers:
            sex_dist[s.sex or 'unknown'] += 1
            if s.extra_fields:
                region_dist[s.extra_fields.get('region', 'unknown')] += 1
                age_dist[s.extra_fields.get('age_bin', 'unknown')] += 1

        fields = ['sex', 'region', 'age_bin']
        coverage_pct = {
            'sex': 100.0,  # All sampled have demographics
            'region': 100.0,
            'age_bin': 100.0,
        }
        report.add_metadata_analysis(fields, coverage_pct, {
            'sex': dict(sex_dist),
            'region': dict(region_dist),
            'age_bin': dict(age_dist),
        })

        # Issues
        report.add_issues(self.issues)

        # Write report
        report.write()
