"""Generate spectrograms for the TOP 5 best 'perro' examples."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import pandas as pd

# Load all acoustic measurements
datasets = ['dimex100', 'albayzin', 'glissando', 'preseea', 'mailabs', 'tedx', 'heroico', 'commonvoice']
all_data = []

for ds in datasets:
    try:
        df = pd.read_parquet(f'outputs/tables/acoustic_measurements_{ds}.parquet')
        df['dataset'] = ds
        all_data.append(df)
    except:
        pass

df = pd.concat(all_data, ignore_index=True)

# Find best perro examples - 100% voicing, sorted by cycles
perro = df[(df['word'].str.lower() == 'perro') &
           (df['context_label'] == 'intervocalic_rr') &
           (df['voicing_pct'] == 100) &
           (df['num_cycles'] >= 3)]

perro_sorted = perro.sort_values('num_cycles', ascending=False).drop_duplicates(subset=['audio_path', 'start_ms'])
top5 = perro_sorted.head(5)

print("=== TOP 5 PERRO EXAMPLES ===")
for i, (idx, row) in enumerate(top5.iterrows()):
    print(f"{i+1}. {row['dataset']}: {int(row['num_cycles'])} cycles, {row['duration_ms']:.0f}ms, {row['cycle_rate_hz']:.1f} Hz")


def generate_spectrogram(example, output_name, rank):
    """Generate waveform + spectrogram for one example."""

    audio_path = example['audio_path']

    # Handle mp3
    if audio_path.endswith('.mp3'):
        import subprocess
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(['ffmpeg', '-y', '-i', audio_path, '-ar', '16000', '-ac', '1', tmp_path],
                       capture_output=True)
        sr, audio = wavfile.read(tmp_path)
        os.unlink(tmp_path)
    else:
        sr, audio = wavfile.read(audio_path)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    # Extract segment
    start_sample = int((example['start_ms'] / 1000) * sr)
    end_sample = int((example['end_ms'] / 1000) * sr)

    pad_ms = 25
    pad_samples = int(pad_ms / 1000 * sr)
    start_pad = max(0, start_sample - pad_samples)
    end_pad = min(len(audio), end_sample + pad_samples)

    segment = audio[start_pad:end_pad]
    trill_start_rel = (start_sample - start_pad) / sr * 1000
    trill_end_rel = (end_sample - start_pad) / sr * 1000
    time_ms = np.arange(len(segment)) / sr * 1000

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1.3]})

    # Waveform
    ax1 = axes[0]
    ax1.plot(time_ms, segment, 'k-', linewidth=0.5)
    ax1.axvline(trill_start_rel, color='#E67E22', linewidth=2)
    ax1.axvline(trill_end_rel, color='#E67E22', linewidth=2)
    ax1.axvspan(trill_start_rel, trill_end_rel, alpha=0.15, color='#E67E22')
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title(f'#{rank} - "perro" ({example["dataset"]}) — {int(example["num_cycles"])} cycles, {example["duration_ms"]:.0f}ms',
                  fontsize=13, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(0, time_ms[-1])

    # Wideband spectrogram
    ax2 = axes[1]
    window_ms = 2.5
    nperseg = int(window_ms / 1000 * sr)
    nperseg = max(nperseg, 32)
    noverlap = int(nperseg * 0.95)

    f, t, Sxx = spectrogram(segment, sr, nperseg=nperseg, noverlap=noverlap, window='hann')
    t_ms = t * 1000

    freq_max = 5000
    freq_mask = f <= freq_max

    Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-12)
    vmin = np.percentile(Sxx_db, 5)
    vmax = np.percentile(Sxx_db, 99)

    im = ax2.pcolormesh(t_ms, f[freq_mask], Sxx_db, shading='gouraud',
                         cmap='inferno', vmin=vmin, vmax=vmax)

    ax2.axvline(trill_start_rel, color='white', linewidth=2)
    ax2.axvline(trill_end_rel, color='white', linewidth=2)
    ax2.set_ylabel('Frequency (Hz)', fontsize=11)
    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_ylim(0, freq_max)

    plt.tight_layout()
    plt.savefig(f'paper_materials/figures/{output_name}', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Created: paper_materials/figures/{output_name}")
    plt.close()


# Generate for each
print("\nGenerating spectrograms...")
for i, (idx, row) in enumerate(top5.iterrows()):
    generate_spectrogram(row, f'perro_candidate_{i+1}.png', i+1)

print("\nDone! Review the 5 candidates and pick your favorite.")
