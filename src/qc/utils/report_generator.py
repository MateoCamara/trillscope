"""QC report generation with histograms and statistics."""

from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING, List
import numpy as np

if TYPE_CHECKING:
    from ..base import QCResult


class QCReportGenerator:
    """Generate markdown reports with QC statistics."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def generate(self, result: 'QCResult') -> Path:
        """
        Generate QC report for a dataset.

        Includes:
        - Summary table (total, passed, failed)
        - Failure counts by cause
        - Percentile tables for each metric
        - ASCII histograms for visual distribution
        - List of most problematic files

        Args:
            result: QCResult with computed statistics

        Returns:
            Path to generated markdown file
        """
        output_path = self.output_dir / f'{result.dataset}.md'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = result.statistics
        thresholds = result.thresholds

        lines = [
            f"# Audio QC Report: {result.dataset.upper()}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Utterances:** {stats.total_utterances:,}",
            f"**Passed QC:** {stats.passed_utterances:,} ({self._pct(stats.passed_utterances, stats.total_utterances)})",
            f"**Failed QC:** {stats.failed_utterances:,} ({self._pct(stats.failed_utterances, stats.total_utterances)})",
            "",
            "---",
            "",
        ]

        # Summary section
        lines.extend([
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total utterances | {stats.total_utterances:,} |",
            f"| Passed QC | {stats.passed_utterances:,} ({self._pct(stats.passed_utterances, stats.total_utterances)}) |",
            f"| Failed QC | {stats.failed_utterances:,} ({self._pct(stats.failed_utterances, stats.total_utterances)}) |",
            f"| Total duration | {stats.total_duration_hours:.2f} hours |",
            f"| Passed duration | {stats.passed_duration_hours:.2f} hours |",
            "",
        ])

        # Thresholds section
        lines.extend([
            "## Thresholds Applied",
            "",
            "| Metric | Threshold | Condition |",
            "|--------|-----------|-----------|",
            f"| Clipping | {thresholds.clipping_max_pct}% | Discard if > |",
            f"| Duration | {thresholds.duration_min_ms} ms | Discard if < |",
            f"| SNR | {thresholds.snr_min_db} dB | Discard if < |",
            f"| Silence | {thresholds.silence_max_pct}% | Discard if > |",
            "",
        ])

        # Failures by cause
        lines.extend([
            "## Failures by Cause",
            "",
            "| Cause | Count | Percentage |",
            "|-------|-------|------------|",
            f"| SNR too low | {stats.failed_by_snr:,} | {self._pct(stats.failed_by_snr, stats.total_utterances)} |",
            f"| Duration too short | {stats.failed_by_duration:,} | {self._pct(stats.failed_by_duration, stats.total_utterances)} |",
            f"| Excessive silence | {stats.failed_by_silence:,} | {self._pct(stats.failed_by_silence, stats.total_utterances)} |",
            f"| Clipping detected | {stats.failed_by_clipping:,} | {self._pct(stats.failed_by_clipping, stats.total_utterances)} |",
            "",
            "*Note: Some utterances may fail multiple checks.*",
            "",
        ])

        # Metric distributions
        lines.extend(self._format_metric_section(
            "Duration Distribution",
            stats.duration_percentiles,
            "ms",
            [m.duration_ms for m in result.metrics]
        ))

        # Filter out inf values for SNR
        snr_values = [m.snr_db for m in result.metrics if np.isfinite(m.snr_db)]
        lines.extend(self._format_metric_section(
            "SNR Distribution",
            stats.snr_percentiles,
            "dB",
            snr_values
        ))

        lines.extend(self._format_metric_section(
            "Clipping Distribution",
            stats.clipping_percentiles,
            "%",
            [m.clipping_pct for m in result.metrics]
        ))

        lines.extend(self._format_metric_section(
            "Silence Distribution",
            stats.silence_percentiles,
            "%",
            [m.silence_pct for m in result.metrics]
        ))

        # Most problematic files
        failed_metrics = [m for m in result.metrics if not m.pass_qc]
        if failed_metrics:
            # Sort by number of failed checks, then by SNR (lowest first)
            failed_metrics.sort(key=lambda m: (-len(m.failed_checks), m.snr_db))

            lines.extend([
                "## Most Problematic Files (Top 20)",
                "",
                "| utt_id | Failed Checks | SNR (dB) | Clipping (%) | Silence (%) |",
                "|--------|---------------|----------|--------------|-------------|",
            ])

            for m in failed_metrics[:20]:
                snr_str = f"{m.snr_db:.1f}" if np.isfinite(m.snr_db) else "N/A"
                lines.append(
                    f"| {m.utt_id} | {', '.join(m.failed_checks)} | {snr_str} | {m.clipping_pct:.2f} | {m.silence_pct:.1f} |"
                )

            lines.append("")

        # Issues section
        if result.issues:
            lines.extend([
                "## Issues Encountered",
                "",
            ])
            for issue in result.issues[:30]:
                lines.append(f"- {issue}")
            if len(result.issues) > 30:
                lines.append(f"- ... and {len(result.issues) - 30} more issues")
            lines.append("")

        output_path.write_text('\n'.join(lines), encoding='utf-8')
        return output_path

    def _pct(self, numerator: int, denominator: int) -> str:
        """Format percentage string."""
        if denominator == 0:
            return "0.0%"
        return f"{(numerator / denominator * 100):.1f}%"

    def _format_metric_section(
        self,
        title: str,
        percentiles: dict,
        unit: str,
        values: List[float]
    ) -> List[str]:
        """Format a metric distribution section with percentiles and histogram."""
        lines = [
            f"## {title}",
            "",
        ]

        if not percentiles:
            lines.extend(["*No data available*", ""])
            return lines

        lines.extend([
            "| Percentile | Value |",
            "|------------|-------|",
        ])

        for key in ['p5', 'p25', 'p50', 'p75', 'p95']:
            if key in percentiles:
                val = percentiles[key]
                lines.append(f"| {key.upper().replace('P', '')}th | {val:.1f} {unit} |")

        lines.append("")

        # ASCII histogram
        if values:
            histogram = self._generate_ascii_histogram(values)
            lines.extend([
                "```",
                histogram,
                "```",
                "",
            ])

        return lines

    def _generate_ascii_histogram(
        self,
        values: list,
        bins: int = 15,
        width: int = 40
    ) -> str:
        """Generate ASCII histogram for markdown."""
        if not values:
            return "No data"

        values = np.array(values)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return "No finite values"

        # Calculate histogram
        counts, edges = np.histogram(values, bins=bins)
        max_count = max(counts) if max(counts) > 0 else 1

        lines = []
        for i, count in enumerate(counts):
            bar_len = int(count / max_count * width)
            bar = '#' * bar_len
            edge_start = edges[i]
            edge_end = edges[i + 1]
            lines.append(f"{edge_start:8.1f} - {edge_end:8.1f} | {bar} ({count})")

        return '\n'.join(lines)
