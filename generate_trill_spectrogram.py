"""Generate a clear spectrogram figure showing trill cycles."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import pandas as pd
from pathlib import Path

# Find a good trill example
df = pd.read_parquet('outputs/tables/acoustic_measurements_dimex100.parquet')

# Find a clear example with 3-4 cycles
good = df[(df['num_cycles'] >= 3) & (df['num_cycles'] <= 5) &
          (df['duration_ms'] >= 70) & (df['duration_ms'] <= 100) &
          (df['voicing_pct'] > 80) &
          (df['context_label'] == 'intervocalic_rr')]

if len(good) == 0:
    # Relax criteria
    good = df[(df['num_cycles'] >= 3) & (df['num_cycles'] <= 5) &
              (df['duration_ms'] >= 60) & (df['duration_ms'] <= 120)]

print(f"Found {len(good)} examples")
example = good.iloc[0]
print(f"Using: {example['word']} from {example['utt_id']}")
print(f"  Cycles: {example['num_cycles']}, Duration: {example['duration_ms']:.1f}ms")
print(f"  Time: {example['start_ms']:.1f} - {example['end_ms']:.1f}ms")

# Load audio
audio_path = example['audio_path']
print(f"  Audio: {audio_path}")

sr, audio = wavfile.read(audio_path)
print(f"  Sample rate: {sr}, Duration: {len(audio)/sr:.2f}s")

# Convert to float
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
elif audio.dtype == np.int32:
    audio = audio.astype(np.float32) / 2147483648.0

# Extract the trill segment with context
start_sample = int((example['start_ms'] / 1000) * sr)
end_sample = int((example['end_ms'] / 1000) * sr)

# Add padding for context (50ms before and after)
pad_samples = int(0.05 * sr)
start_with_pad = max(0, start_sample - pad_samples)
end_with_pad = min(len(audio), end_sample + pad_samples)

segment = audio[start_with_pad:end_with_pad]
segment_time = np.arange(len(segment)) / sr * 1000  # in ms

# Relative timing
trill_start_rel = (start_sample - start_with_pad) / sr * 1000
trill_end_rel = (end_sample - start_with_pad) / sr * 1000

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# === Waveform ===
ax1 = axes[0]
ax1.plot(segment_time, segment, 'k-', linewidth=0.5)
ax1.axvline(trill_start_rel, color='#E67E22', linestyle='--', linewidth=2, label='Trill boundaries')
ax1.axvline(trill_end_rel, color='#E67E22', linestyle='--', linewidth=2)
ax1.set_ylabel('Amplitude')
ax1.set_title(f'Trill in "{example["word"]}" - {int(example["num_cycles"])} cycles')
ax1.set_xlim(0, segment_time[-1])

# Shade the trill region
ax1.axvspan(trill_start_rel, trill_end_rel, alpha=0.2, color='#E67E22')

# Add cycle annotations if we can detect peaks
trill_segment = audio[start_sample:end_sample]
trill_time_abs = np.arange(len(trill_segment)) / sr * 1000

# Simple peak detection for annotation
from scipy.signal import find_peaks
# Find peaks (releases) in the envelope
envelope = np.abs(trill_segment)
# Smooth
from scipy.ndimage import uniform_filter1d
envelope_smooth = uniform_filter1d(envelope, size=int(sr*0.005))
peaks, _ = find_peaks(envelope_smooth, distance=int(sr*0.015), prominence=0.01*np.max(envelope_smooth))

if len(peaks) >= 2:
    print(f"  Detected {len(peaks)} peaks")
    # Annotate cycles on waveform
    for i, peak in enumerate(peaks[:int(example['num_cycles'])+1]):
        peak_time = trill_start_rel + (peak / sr * 1000)
        if i < int(example['num_cycles']):
            ax1.annotate(f'{i+1}', xy=(peak_time, segment[int((peak_time/1000)*sr - start_with_pad + start_sample)]),
                        xytext=(peak_time, 0.8*np.max(np.abs(segment))),
                        fontsize=14, fontweight='bold', color='#E67E22',
                        ha='center',
                        arrowprops=dict(arrowstyle='->', color='#E67E22', lw=1.5))

ax1.legend(loc='upper right')

# === Spectrogram ===
ax2 = axes[1]
# Compute spectrogram
nperseg = min(256, len(segment)//4)
f, t, Sxx = spectrogram(segment, sr, nperseg=nperseg, noverlap=nperseg//2)
t = t * 1000  # Convert to ms

# Plot spectrogram (limit frequency range)
freq_mask = f <= 5000
ax2.pcolormesh(t, f[freq_mask], 10*np.log10(Sxx[freq_mask] + 1e-10), shading='gouraud', cmap='Greys')
ax2.axvline(trill_start_rel, color='#E67E22', linestyle='--', linewidth=2)
ax2.axvline(trill_end_rel, color='#E67E22', linestyle='--', linewidth=2)
ax2.set_ylabel('Frequency (Hz)')
ax2.set_xlabel('Time (ms)')
ax2.set_ylim(0, 5000)

plt.tight_layout()
plt.savefig('paper_materials/figures/trill_spectrogram_example.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\nCreated: paper_materials/figures/trill_spectrogram_example.png")
plt.close()

# === Create a cleaner annotated diagram ===
fig, ax = plt.subplots(figsize=(10, 4))

# Just the trill segment with padding
trill_with_pad = audio[start_sample - int(0.03*sr):end_sample + int(0.03*sr)]
trill_time = np.arange(len(trill_with_pad)) / sr * 1000
pad_ms = 30

ax.plot(trill_time, trill_with_pad, 'k-', linewidth=0.8)
ax.axvline(pad_ms, color='#E67E22', linestyle='--', linewidth=2)
ax.axvline(pad_ms + (example['end_ms'] - example['start_ms']), color='#E67E22', linestyle='--', linewidth=2)
ax.axvspan(pad_ms, pad_ms + (example['end_ms'] - example['start_ms']), alpha=0.15, color='#E67E22')

# Annotate closure and release
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_title(f'Spanish Trill /r/ - {int(example["num_cycles"])} Cycles', fontsize=14)

# Add text explanation
textstr = 'One cycle = one closure (valley) + one release (peak)'
ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('paper_materials/figures/trill_waveform_annotated.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: paper_materials/figures/trill_waveform_annotated.png")
plt.close()

print("\nDone!")
