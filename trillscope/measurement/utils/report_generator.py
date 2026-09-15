"""Report generator for acoustic measurement results."""

from pathlib import Path
import logging

from ..base import MeasurementResult, MeasurementStatus

logger = logging.getLogger(__name__)


class MeasurementReportGenerator:
    """Generate markdown reports for acoustic measurement results."""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to write reports to
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, result: MeasurementResult) -> Path:
        """
        Generate markdown report for measurement result.

        Args:
            result: MeasurementResult to report on

        Returns:
            Path to generated report
        """
        report_path = self.output_dir / f"{result.dataset}.md"

        lines = []
        lines.append(f"# Acoustic Measurement Report: {result.dataset.upper()}\n")

        # Summary section
        lines.extend(self._summary_section(result))

        # Cycle distribution
        lines.extend(self._cycle_distribution_section(result))

        # Cycle confidence
        lines.extend(self._confidence_section(result))

        # Voicing statistics
        lines.extend(self._voicing_section(result))

        # Duration statistics
        lines.extend(self._duration_section(result))

        # Quality issues
        lines.extend(self._issues_section(result))

        # Configuration
        lines.extend(self._config_section(result))

        report_path.write_text('\n'.join(lines), encoding='utf-8')
        logger.info(f"Generated report: {report_path}")

        return report_path

    def _summary_section(self, result: MeasurementResult) -> list:
        """Generate summary section."""
        lines = []
        lines.append("## Summary\n")

        stats = result.statistics
        total = stats.get('total_tokens', 0)
        successful = stats.get('successful', 0)
        failed = stats.get('failed', 0)

        success_pct = (successful / total * 100) if total > 0 else 0

        lines.append(f"- **Total tokens**: {total:,}")
        lines.append(f"- **Successfully measured**: {successful:,} ({success_pct:.1f}%)")
        lines.append(f"- **Failed**: {failed:,}")
        lines.append("")

        # Status breakdown
        by_status = stats.get('by_status', {})
        if by_status:
            lines.append("### Status Breakdown\n")
            lines.append("| Status | Count | Percentage |")
            lines.append("|--------|-------|------------|")
            for status, count in sorted(by_status.items()):
                pct = (count / total * 100) if total > 0 else 0
                lines.append(f"| {status} | {count:,} | {pct:.1f}% |")
            lines.append("")

        return lines

    def _cycle_distribution_section(self, result: MeasurementResult) -> list:
        """Generate cycle distribution section."""
        lines = []
        lines.append("## Cycle Count Distribution\n")

        cycle_dist = result.statistics.get('cycle_distribution', {})
        if not cycle_dist:
            lines.append("No cycle data available.\n")
            return lines

        lines.append("| Cycles | Count | Percentage |")
        lines.append("|--------|-------|------------|")

        total = sum(cycle_dist.values())
        for cycles in sorted(cycle_dist.keys()):
            count = cycle_dist[cycles]
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"| {cycles} | {count:,} | {pct:.1f}% |")

        lines.append("")

        # Mode and mean
        if cycle_dist:
            mode_cycles = max(cycle_dist.keys(), key=lambda k: cycle_dist[k])
            weighted_sum = sum(c * n for c, n in cycle_dist.items())
            mean_cycles = weighted_sum / total if total > 0 else 0

            lines.append(f"- **Mode**: {mode_cycles} cycles")
            lines.append(f"- **Mean**: {mean_cycles:.2f} cycles")
            lines.append("")

        return lines

    def _confidence_section(self, result: MeasurementResult) -> list:
        """Generate cycle detection confidence section."""
        lines = []
        lines.append("## Cycle Detection Confidence\n")

        stats = result.statistics
        mean_conf = stats.get('mean_confidence')
        conf_dist = stats.get('confidence_distribution', {})

        if mean_conf is None or not conf_dist:
            lines.append("No confidence data available.\n")
            return lines

        total = conf_dist.get('high', 0) + conf_dist.get('medium', 0) + conf_dist.get('low', 0)
        if total == 0:
            lines.append("No confidence data available.\n")
            return lines

        lines.append(f"- **Mean confidence**: {mean_conf:.3f}")
        lines.append(f"- **High (>0.7)**: {conf_dist.get('high', 0):,} ({conf_dist.get('high', 0)/total*100:.1f}%)")
        lines.append(f"- **Medium (0.3-0.7)**: {conf_dist.get('medium', 0):,} ({conf_dist.get('medium', 0)/total*100:.1f}%)")
        lines.append(f"- **Low (<0.3)**: {conf_dist.get('low', 0):,} ({conf_dist.get('low', 0)/total*100:.1f}%)")
        lines.append("")

        # Add method info from config
        method = result.config.cycle_method
        lines.append(f"- **Detection method**: {method}")
        lines.append("")

        return lines

    def _voicing_section(self, result: MeasurementResult) -> list:
        """Generate voicing statistics section."""
        lines = []
        lines.append("## Voicing Statistics\n")

        # Get successful metrics
        successful = [m for m in result.metrics if m.status == MeasurementStatus.SUCCESS]

        if not successful:
            lines.append("No voicing data available.\n")
            return lines

        voicing_values = [m.voicing_pct for m in successful]
        mean_voicing = sum(voicing_values) / len(voicing_values)

        # Count by voicing category
        fully_voiced = sum(1 for v in voicing_values if v >= 95)
        partially_voiced = sum(1 for v in voicing_values if 50 <= v < 95)
        mostly_unvoiced = sum(1 for v in voicing_values if v < 50)

        total = len(voicing_values)

        lines.append(f"- **Mean voicing**: {mean_voicing:.1f}%")
        lines.append(f"- **Fully voiced (≥95%)**: {fully_voiced:,} ({fully_voiced/total*100:.1f}%)")
        lines.append(f"- **Partially voiced (50-95%)**: {partially_voiced:,} ({partially_voiced/total*100:.1f}%)")
        lines.append(f"- **Mostly unvoiced (<50%)**: {mostly_unvoiced:,} ({mostly_unvoiced/total*100:.1f}%)")
        lines.append("")

        # F0 statistics
        f0_values = [m.mean_f0_hz for m in successful if m.mean_f0_hz is not None]
        if f0_values:
            mean_f0 = sum(f0_values) / len(f0_values)
            min_f0 = min(f0_values)
            max_f0 = max(f0_values)
            lines.append("### F0 Statistics\n")
            lines.append(f"- **Mean F0**: {mean_f0:.1f} Hz")
            lines.append(f"- **Range**: {min_f0:.1f} - {max_f0:.1f} Hz")
            lines.append(f"- **Tokens with F0**: {len(f0_values):,}")
            lines.append("")

        return lines

    def _duration_section(self, result: MeasurementResult) -> list:
        """Generate duration statistics section."""
        lines = []
        lines.append("## Duration Statistics\n")

        successful = [m for m in result.metrics if m.status == MeasurementStatus.SUCCESS]

        if not successful:
            lines.append("No duration data available.\n")
            return lines

        durations = [m.duration_ms for m in successful]
        mean_dur = sum(durations) / len(durations)
        min_dur = min(durations)
        max_dur = max(durations)

        lines.append(f"- **Mean duration**: {mean_dur:.1f} ms")
        lines.append(f"- **Min duration**: {min_dur:.1f} ms")
        lines.append(f"- **Max duration**: {max_dur:.1f} ms")
        lines.append("")

        # Duration by cycle count
        cycle_durations = {}
        for m in successful:
            c = m.num_cycles
            if c not in cycle_durations:
                cycle_durations[c] = []
            cycle_durations[c].append(m.duration_ms)

        if cycle_durations:
            lines.append("### Duration by Cycle Count\n")
            lines.append("| Cycles | Mean (ms) | Min (ms) | Max (ms) | Count |")
            lines.append("|--------|-----------|----------|----------|-------|")

            for cycles in sorted(cycle_durations.keys()):
                durs = cycle_durations[cycles]
                mean_d = sum(durs) / len(durs)
                min_d = min(durs)
                max_d = max(durs)
                lines.append(f"| {cycles} | {mean_d:.1f} | {min_d:.1f} | {max_d:.1f} | {len(durs):,} |")

            lines.append("")

        return lines

    def _issues_section(self, result: MeasurementResult) -> list:
        """Generate quality issues section."""
        lines = []
        lines.append("## Quality Issues\n")

        if not result.issues:
            lines.append("No issues reported.\n")
            return lines

        lines.append(f"**Total issues**: {len(result.issues)}\n")

        # Show first 20 issues
        for issue in result.issues[:20]:
            lines.append(f"- {issue}")

        if len(result.issues) > 20:
            lines.append(f"- ... and {len(result.issues) - 20} more issues")

        lines.append("")
        return lines

    def _config_section(self, result: MeasurementResult) -> list:
        """Generate configuration section."""
        lines = []
        lines.append("## Configuration\n")

        config = result.config
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Target sample rate | {config.target_sr} Hz |")
        lines.append(f"| Context padding | {config.context_ms} ms |")
        lines.append(f"| Cycle detection method | {config.cycle_method} |")
        lines.append(f"| Min cycle duration | {config.min_cycle_duration_ms} ms |")
        lines.append(f"| Max cycle duration | {config.max_cycle_duration_ms} ms |")
        lines.append(f"| Envelope smoothing | {config.envelope_smoothing_ms} ms |")
        lines.append(f"| Band-pass low cutoff | {config.bp_freq_low_hz} Hz |")
        lines.append(f"| Band-pass high cutoff | {config.bp_freq_high_hz} Hz |")
        lines.append(f"| STFT window | {config.spectrogram_win_ms} ms |")
        lines.append(f"| STFT hop | {config.spectrogram_hop_ms} ms |")
        lines.append(f"| Pitch floor | {config.pitch_floor_hz} Hz |")
        lines.append(f"| Pitch ceiling | {config.pitch_ceiling_hz} Hz |")
        lines.append(f"| Min duration threshold | {config.min_duration_ms} ms |")
        lines.append("")

        return lines
