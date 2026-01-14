"""Mapping report generation for metadata normalization."""

from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import MetadataResult


class MappingReportGenerator:
    """Generate markdown reports documenting mapping rules and coverage."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def generate(self, result: 'MetadataResult') -> Path:
        """Generate mapping report for a dataset."""
        output_path = self.output_dir / f'metadata_mapping_{result.dataset_name}.md'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# Metadata Mapping Report: {result.dataset_name.upper()}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Records:** {len(result.records):,}",
            "",
            "---",
            "",
        ]

        # Summary table
        lines.extend([
            "## Summary",
            "",
            "| Field | Mapped Successfully | Unknown | Coverage |",
            "|-------|---------------------|---------|----------|",
        ])

        for field, stats in result.mapping_stats.items():
            coverage = (stats.mapped_successfully / stats.total_records * 100) if stats.total_records > 0 else 0
            lines.append(
                f"| {field} | {stats.mapped_successfully:,} | {stats.mapped_to_unknown:,} | {coverage:.1f}% |"
            )

        lines.append("")

        # Detailed sections per field
        for field, stats in result.mapping_stats.items():
            lines.extend([
                f"## {field.replace('_', ' ').title()}",
                "",
                "### Distribution",
                "",
                "| Value | Count | Percentage |",
                "|-------|-------|------------|",
            ])

            for value, count in sorted(stats.value_distribution.items(), key=lambda x: -x[1]):
                pct = count / stats.total_records * 100 if stats.total_records > 0 else 0
                lines.append(f"| {value} | {count:,} | {pct:.1f}% |")

            lines.append("")

            # Show unmapped values for review
            if stats.unmapped_values:
                lines.extend([
                    "### Unmapped Values (mapped to 'unknown')",
                    "",
                    "The following raw values could not be mapped:",
                    "",
                ])
                for val in stats.unmapped_values[:30]:  # Limit to 30
                    lines.append(f"- `{val}`")
                if len(stats.unmapped_values) > 30:
                    lines.append(f"- ... and {len(stats.unmapped_values) - 30} more")
                lines.append("")

        # Mapping rules documentation
        lines.extend([
            "## Mapping Rules Applied",
            "",
        ])

        if result.dataset_name == 'albayzin':
            lines.extend([
                "### Education Mapping",
                "",
                "ALBAYZIN uses job titles (EPR field) instead of explicit education levels.",
                "Mapping is based on typical educational requirements for each profession:",
                "",
                "- **high**: University degrees (Ingeniería, Abogado, Licenciado, Psicología, etc.)",
                "- **mid**: Technical/vocational training (Técnico, Administrativo, Secretaria, Informática)",
                "- **low**: Occupations without higher education indication (Ama de Casa, Conductor, Camarero)",
                "- **unknown**: Unrecognized job titles",
                "",
                "### Sex Mapping",
                "",
                "ALBAYZIN already uses F/M format, normalized directly.",
                "",
                "### Age Mapping",
                "",
                "- `<30`: Under 30 years",
                "- `30-55`: 30 to 55 years (inclusive)",
                "- `>55`: Over 55 years",
                "",
                "### Location",
                "",
                "All ALBAYZIN recordings are from Spain. City extracted from origin field.",
                "",
            ])
        elif result.dataset_name == 'preseea':
            lines.extend([
                "### Education Mapping",
                "",
                "PRESEEA uses explicit education levels with varying formats.",
                "Mapping normalizes inconsistent formatting:",
                "",
                "- **low**: 'bajo', 'primaria', '1', 'básico', 'elemental'",
                "- **mid**: 'medio', 'secundaria', '2', 'bachillerato', 'media'",
                "- **high**: 'alto', 'superior', '3', 'universitario', 'licenciatura'",
                "",
                "### Sex Mapping",
                "",
                "PRESEEA has inconsistent sex values:",
                "",
                "- Male: 'hombre', 'H', 'masculino' → 'M'",
                "- Female: 'mujer', 'Mujer', 'femenino' → 'F'",
                "",
                "### Location",
                "",
                "PRESEEA spans multiple Spanish-speaking countries. City codes extracted from",
                "filenames and metadata are mapped to full city/country names.",
                "",
            ])

        # Issues section
        if result.issues:
            lines.extend([
                "## Issues Encountered",
                "",
            ])
            for issue in result.issues[:20]:
                lines.append(f"- [{issue['severity']}] {issue['category']}: {issue['message']}")
            if len(result.issues) > 20:
                lines.append(f"- ... and {len(result.issues) - 20} more issues")
            lines.append("")

        output_path.write_text('\n'.join(lines), encoding='utf-8')
        return output_path
