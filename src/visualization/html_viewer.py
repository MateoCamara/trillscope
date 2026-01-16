"""Generate interactive HTML viewer with audio playback for trill /r/ validation."""

import base64
import io
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import resample

from .base import VisualizationConfig
from .plotter import load_audio_segment, compute_spectrogram_data

logger = logging.getLogger(__name__)


@dataclass
class TrillExample:
    """Data for a single trill example in the viewer."""
    utt_id: str
    dataset: str
    word: str
    context_label: str
    duration_ms: float
    speaker_id: str
    start_ms: float
    end_ms: float
    audio_b64: str  # Base64-encoded WAV
    image_b64: str  # Base64-encoded PNG


def extract_audio_clip(
    audio_path: str,
    start_ms: float,
    end_ms: float,
    context_ms: float = 100.0,
    target_sr: int = 16000
) -> bytes:
    """Extract audio clip and return as WAV bytes."""
    audio, sr, _, _ = load_audio_segment(
        audio_path, start_ms, end_ms,
        context_ms=context_ms,
        target_sr=target_sr
    )

    # Write to bytes buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)
    return buffer.read()


def generate_spectrogram_image(
    audio_path: str,
    start_ms: float,
    end_ms: float,
    word: str,
    context_label: str,
    dataset: str,
    speaker_id: str,
    config: VisualizationConfig
) -> bytes:
    """Generate spectrogram plot and return as PNG bytes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Load audio
    audio, sr, r_start, r_end = load_audio_segment(
        audio_path, start_ms, end_ms,
        context_ms=config.context_ms,
        target_sr=config.target_sr
    )

    duration_ms = end_ms - start_ms

    # Create time axis for waveform
    time_axis = np.arange(len(audio)) / sr

    # Compute spectrogram
    f, t, Sxx_db = compute_spectrogram_data(audio, sr, config)

    # Create figure
    fig, (ax_wave, ax_spec) = plt.subplots(
        2, 1,
        figsize=(10, 5),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 2]}
    )

    # Plot waveform
    ax_wave.plot(time_axis, audio, color=config.waveform_color, linewidth=0.5)
    ax_wave.axvspan(r_start, r_end,
                   color=config.highlight_color,
                   alpha=config.highlight_alpha,
                   label='/r/ region')
    ax_wave.axvline(r_start, color=config.highlight_color, linewidth=1.5, linestyle='--')
    ax_wave.axvline(r_end, color=config.highlight_color, linewidth=1.5, linestyle='--')
    ax_wave.set_ylabel('Amplitude', fontsize=config.label_fontsize)
    ax_wave.set_xlim(0, time_axis[-1])
    ax_wave.legend(loc='upper right', fontsize=8)
    ax_wave.grid(True, alpha=0.3)

    # Plot spectrogram
    ax_spec.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax_spec.axvline(r_start, color='white', linewidth=2, linestyle='--')
    ax_spec.axvline(r_end, color='white', linewidth=2, linestyle='--')
    ax_spec.set_ylabel('Frequency (Hz)', fontsize=config.label_fontsize)
    ax_spec.set_xlabel('Time (s)', fontsize=config.label_fontsize)
    ax_spec.set_ylim(0, config.freq_max)

    # Title
    title = f"{dataset.upper()}: '{word}' [{context_label}] - {duration_ms:.1f}ms"
    fig.suptitle(title, fontsize=config.title_fontsize, fontweight='bold')
    ax_wave.set_title(f"Speaker: {speaker_id}", fontsize=9, color='gray')

    plt.tight_layout()

    # Save to bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def create_example(
    row: dict,
    audio_path: str,
    config: VisualizationConfig
) -> Optional[TrillExample]:
    """Create a TrillExample with audio and image data."""
    try:
        # Extract audio clip
        audio_bytes = extract_audio_clip(
            audio_path,
            row['start_ms'],
            row['end_ms'],
            context_ms=config.context_ms,
            target_sr=config.target_sr
        )
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Generate spectrogram image
        image_bytes = generate_spectrogram_image(
            audio_path,
            row['start_ms'],
            row['end_ms'],
            row.get('word', '[unknown]'),
            row.get('context_label', 'unknown'),
            row.get('dataset', 'unknown'),
            row.get('speaker_id', 'unknown'),
            config
        )
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        return TrillExample(
            utt_id=row.get('utt_id', 'unknown'),
            dataset=row.get('dataset', 'unknown'),
            word=row.get('word', '[unknown]'),
            context_label=row.get('context_label', 'unknown'),
            duration_ms=row['end_ms'] - row['start_ms'],
            speaker_id=row.get('speaker_id', 'unknown'),
            start_ms=row['start_ms'],
            end_ms=row['end_ms'],
            audio_b64=audio_b64,
            image_b64=image_b64
        )
    except Exception as e:
        logger.error(f"Failed to create example: {e}")
        return None


def generate_html(examples: List[TrillExample], title: str = "Trill /r/ Viewer") -> str:
    """Generate HTML page with embedded audio and images."""

    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            color: #FF6B6B;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .instructions {{
            background: #16213e;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }}
        .instructions h3 {{
            margin-top: 0;
            color: #FF6B6B;
        }}
        .instructions ul {{
            margin-bottom: 0;
            padding-left: 20px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .card {{
            background: #16213e;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .card img {{
            width: 100%;
            display: block;
        }}
        .card-body {{
            padding: 15px;
        }}
        .card-title {{
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 8px;
            color: #FF6B6B;
        }}
        .card-meta {{
            font-size: 0.9em;
            color: #888;
            margin-bottom: 12px;
        }}
        .card-meta span {{
            margin-right: 15px;
        }}
        audio {{
            width: 100%;
            margin-top: 10px;
        }}
        .tag {{
            display: inline-block;
            background: #0f3460;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-right: 5px;
        }}
        .tag.intervocalic_rr {{ background: #2ecc71; color: #000; }}
        .tag.word_initial {{ background: #3498db; }}
        .tag.after_nls {{ background: #9b59b6; }}
        .play-hint {{
            text-align: center;
            color: #666;
            font-size: 0.85em;
            margin-top: 8px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="subtitle">Interactive validation of trill /r/ extraction</p>

    <div class="instructions">
        <h3>How to use</h3>
        <ul>
            <li>Each card shows a spectrogram with the detected /r/ region highlighted (pink/white dashed lines)</li>
            <li>Click the <strong>play button</strong> to hear the audio segment</li>
            <li>Listen for the characteristic trill sound within the highlighted region</li>
            <li>The trill /r/ should sound like a rolled "rr" with multiple tongue vibrations</li>
        </ul>
    </div>

    <div class="grid">
{cards}
    </div>
</body>
</html>'''

    card_template = '''        <div class="card">
            <img src="data:image/png;base64,{image_b64}" alt="{word}">
            <div class="card-body">
                <div class="card-title">{dataset}: "{word}"</div>
                <div class="card-meta">
                    <span class="tag {context_class}">{context_label}</span>
                    <span>{duration_ms:.1f}ms</span>
                    <span>Speaker: {speaker_id}</span>
                </div>
                <audio controls>
                    <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                    Your browser does not support audio playback.
                </audio>
                <p class="play-hint">Click play to hear the /r/ segment</p>
            </div>
        </div>'''

    cards_html = []
    for ex in examples:
        context_class = ex.context_label.replace(' ', '_')
        card = card_template.format(
            image_b64=ex.image_b64,
            audio_b64=ex.audio_b64,
            word=ex.word,
            dataset=ex.dataset.upper(),
            context_label=ex.context_label,
            context_class=context_class,
            duration_ms=ex.duration_ms,
            speaker_id=ex.speaker_id
        )
        cards_html.append(card)

    return html_template.format(
        title=title,
        cards='\n'.join(cards_html)
    )


def create_viewer(
    df,
    output_path: Path,
    n_examples: int = 10,
    config: Optional[VisualizationConfig] = None
) -> bool:
    """
    Create HTML viewer from extraction results DataFrame.

    Args:
        df: DataFrame with extraction results (must have audio_path column mapped)
        output_path: Path for output HTML file
        n_examples: Number of examples to include
        config: Visualization configuration

    Returns:
        True if successful
    """
    if config is None:
        config = VisualizationConfig()

    # Sample examples
    if len(df) > n_examples:
        df = df.sample(n_examples, random_state=42)

    examples = []
    for _, row in df.iterrows():
        audio_path = row.get('audio_path')
        if not audio_path or not Path(audio_path).exists():
            continue

        example = create_example(row.to_dict(), audio_path, config)
        if example:
            examples.append(example)

    if not examples:
        logger.error("No examples could be created")
        return False

    # Generate HTML
    html = generate_html(examples, title=f"Trill /r/ Viewer ({len(examples)} examples)")

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')

    logger.info(f"Created viewer with {len(examples)} examples: {output_path}")
    return True
