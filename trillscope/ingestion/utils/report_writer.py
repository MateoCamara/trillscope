"""Markdown audit report generation."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class AuditSection:
    """Section of audit report."""
    title: str
    content: str
    subsections: List['AuditSection'] = field(default_factory=list)


class AuditReportWriter:
    """Generate structured markdown audit reports."""

    def __init__(self, dataset_name: str, output_dir: Path):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.sections: List[AuditSection] = []
        self.generated_at = datetime.now()

    def add_summary(self, total_files: int, total_speakers: int,
                   total_duration_hours: float, issues_count: int,
                   files_with_transcripts: int = 0,
                   files_with_alignments: int = 0) -> None:
        """Add executive summary section."""
        transcript_pct = (files_with_transcripts / total_files * 100) if total_files > 0 else 0
        alignment_pct = (files_with_alignments / total_files * 100) if total_files > 0 else 0

        content = f"""
| Metric | Value |
|--------|-------|
| Total Audio Files | {total_files:,} |
| Total Speakers | {total_speakers:,} |
| Total Duration (hours) | {total_duration_hours:.1f} |
| Files with Transcripts | {files_with_transcripts:,} ({transcript_pct:.1f}%) |
| Files with Alignments | {files_with_alignments:,} ({alignment_pct:.1f}%) |
| Issues Found | {issues_count:,} |
"""
        self.sections.append(AuditSection("Executive Summary", content.strip()))

    def add_file_structure(self, structure_tree: str,
                          file_counts: Dict[str, int]) -> None:
        """Document discovered file structure."""
        counts_table = "| Extension | Count |\n|-----------|-------|\n"
        for ext, count in sorted(file_counts.items()):
            counts_table += f"| {ext} | {count:,} |\n"

        content = f"""
### Discovered Layout

```
{structure_tree}
```

### File Counts by Type

{counts_table}
"""
        self.sections.append(AuditSection("File Structure", content.strip()))

    def add_audio_analysis(self, format_stats: Dict[str, int],
                          sample_rate_distribution: Dict[int, int],
                          channel_distribution: Dict[int, int],
                          issues: List[str]) -> None:
        """Document audio format findings."""
        # Format table
        format_table = "| Format | Count |\n|--------|-------|\n"
        for fmt, count in sorted(format_stats.items()):
            format_table += f"| {fmt} | {count:,} |\n"

        # Sample rate table
        rate_table = "| Rate (Hz) | Count |\n|-----------|-------|\n"
        for rate, count in sorted(sample_rate_distribution.items()):
            rate_table += f"| {rate:,} | {count:,} |\n"

        # Channel table
        channel_table = "| Channels | Count |\n|----------|-------|\n"
        for ch, count in sorted(channel_distribution.items()):
            label = "mono" if ch == 1 else "stereo" if ch == 2 else str(ch)
            channel_table += f"| {label} | {count:,} |\n"

        # Issues list
        issues_list = ""
        if issues:
            issues_list = "\n### Issues Detected\n\n"
            for issue in issues[:10]:  # Limit to first 10
                issues_list += f"- {issue}\n"
            if len(issues) > 10:
                issues_list += f"- ... and {len(issues) - 10} more\n"

        content = f"""
### Format Distribution

{format_table}

### Sample Rate Distribution

{rate_table}

### Channel Distribution

{channel_table}
{issues_list}
"""
        self.sections.append(AuditSection("Audio Analysis", content.strip()))

    def add_transcript_analysis(self, format_type: str,
                               encoding_stats: Dict[str, int],
                               linking_method: str,
                               coverage_percent: float,
                               special_markers: Optional[Dict[str, int]] = None) -> None:
        """Document transcript format and coverage."""
        # Encoding table
        encoding_table = "| Encoding | Count |\n|----------|-------|\n"
        for enc, count in sorted(encoding_stats.items()):
            encoding_table += f"| {enc} | {count:,} |\n"

        markers_section = ""
        if special_markers:
            markers_section = "\n### Special Markers Found\n\n"
            markers_section += "| Marker | Count |\n|--------|-------|\n"
            for marker, count in sorted(special_markers.items()):
                markers_section += f"| {marker} | {count:,} |\n"

        content = f"""
### Format

- **Type**: {format_type}
- **Coverage**: {coverage_percent:.1f}% of audio files have transcripts
- **Linking Method**: {linking_method}

### Encoding Distribution

{encoding_table}
{markers_section}
"""
        self.sections.append(AuditSection("Transcript Analysis", content.strip()))

    def add_metadata_analysis(self, fields_found: List[str],
                             field_coverage: Dict[str, float],
                             value_distributions: Dict[str, Dict[str, int]]) -> None:
        """Document metadata fields and distributions."""
        # Fields table
        fields_table = "| Field | Coverage |\n|-------|----------|\n"
        for fld in fields_found:
            coverage = field_coverage.get(fld, 0)
            fields_table += f"| {fld} | {coverage:.1f}% |\n"

        # Value distributions
        dist_sections = ""
        for field_name, values in value_distributions.items():
            if values:
                dist_sections += f"\n#### {field_name}\n\n"
                dist_sections += "| Value | Count |\n|-------|-------|\n"
                for val, count in sorted(values.items(), key=lambda x: -x[1])[:10]:
                    dist_sections += f"| {val} | {count:,} |\n"
                if len(values) > 10:
                    dist_sections += f"| ... | {len(values) - 10} more values |\n"

        content = f"""
### Fields Discovered

{fields_table}

### Value Distributions
{dist_sections}
"""
        self.sections.append(AuditSection("Metadata Analysis", content.strip()))

    def add_issues(self, issues: List[Dict[str, Any]]) -> None:
        """Document discovered issues and warnings."""
        if not issues:
            self.sections.append(AuditSection("Issues Log", "No issues found."))
            return

        # Group by severity
        errors = [i for i in issues if i.get('severity') == 'error']
        warnings = [i for i in issues if i.get('severity') == 'warning']
        infos = [i for i in issues if i.get('severity') == 'info']

        # Group by category
        by_category: Dict[str, List[Dict]] = {}
        for issue in issues:
            cat = issue.get('category', 'unknown')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(issue)

        content = f"""
### Summary

- **Errors**: {len(errors)}
- **Warnings**: {len(warnings)}
- **Info**: {len(infos)}

### By Category

| Category | Count | Severity |
|----------|-------|----------|
"""
        for cat, items in sorted(by_category.items()):
            severities = set(i.get('severity', 'unknown') for i in items)
            content += f"| {cat} | {len(items)} | {', '.join(severities)} |\n"

        # Sample issues
        if errors:
            content += "\n### Sample Errors\n\n"
            for err in errors[:5]:
                content += f"- **{err.get('category')}**: {err.get('message')}\n"

        if warnings:
            content += "\n### Sample Warnings\n\n"
            for warn in warnings[:5]:
                content += f"- **{warn.get('category')}**: {warn.get('message')}\n"

        self.sections.append(AuditSection("Issues Log", content.strip()))

    def add_custom_section(self, title: str, content: str) -> None:
        """Add a custom section with arbitrary content."""
        self.sections.append(AuditSection(title, content))

    def write(self) -> Path:
        """Write report to markdown file."""
        output_path = self.output_dir / f"{self.dataset_name}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# Dataset Audit Report: {self.dataset_name.upper()}",
            "",
            f"**Generated**: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            ""
        ]

        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")

            for subsection in section.subsections:
                lines.append(f"### {subsection.title}")
                lines.append("")
                lines.append(subsection.content)
                lines.append("")

        output_path.write_text('\n'.join(lines), encoding='utf-8')
        return output_path
