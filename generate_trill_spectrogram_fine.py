"""Generate a fine-resolution spectrogram showing trill cycles clearly."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, stft
import pandas as pd

# Find a good trill example
df = pd.read_parquet('outputs/tables/acoustic_measurements_dimex100.parquet')

# Find a clear example with 3-4 cycles, good voicing
good = df[(df['num_cycles'] >= 3) & (df['num_cycles'] <= 5) &
          (df['duration_ms'] >= 70) & (df['duration_ms'] <= 110) &
          (df['voicing_pct'] > 85)]

example = good.iloc[2]
print(f"Using: {example['word'][:40]}...")
print(f"  Cycles: {int(example['num_cycles'])}, Duration: {example['duration_ms']:.1f}ms")
print(f"  F0: {example['mean_f0_hz']:.1f} Hz, Voicing: {example['voicing_pct']:.1f}%")

# Load audio
sr, audio = wavfile.read(example['audio_path'])
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

# Extract trill segment with context
start_sample = int((example['start_ms'] / 1000) * sr)
end_sample = int((example['end_ms'] / 1000) * sr)

# Add padding (40ms before and after)
pad_ms = 40
pad_samples = int(pad_ms / 1000 * sr)
start_pad = max(0, start_sample - pad_samples)
end_pad = min(len(audio), end_sample + pad_samples)

segment = audio[start_pad:end_pad]
trill_start_rel = (start_sample - start_pad) / sr * 1000
trill_end_rel = (end_sample - start_pad) / sr * 1000
total_duration_ms = len(segment) / sr * 1000

print(f"  Segment duration: {total_duration_ms:.1f}ms")

# === Create high-resolution spectrogram ===
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                          gridspec_kw={'height_ratios': [1, 1.5]})

# Time axis
time_ms = np.arange(len(segment)) / sr * 1000

# === Waveform ===
ax1 = axes[0]
ax1.plot(time_ms, segment, 'k-', linewidth=0.6)
ax1.axvline(trill_start_rel, color='#E67E22', linewidth=2)
ax1.axvline(trill_end_rel, color='#E67E22', linewidth=2)
ax1.axvspan(trill_start_rel, trill_end_rel, alpha=0.15, color='#E67E22')
ax1.set_ylabel('Amplitude', fontsize=11)
ax1.set_title(f'Spanish Trill /r/ — {int(example["num_cycles"])} cycles, {example["duration_ms"]:.0f} ms',
              fontsize=13, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0, total_duration_ms)

# === Fine spectrogram ===
ax2 = axes[1]

# Use short window for good time resolution (captures individual cycles)
# Window of ~5-8ms gives good time resolution for seeing trill cycles (~25-30Hz)
window_ms = 6
nperseg = int(window_ms / 1000 * sr)
noverlap = int(nperseg * 0.9)  # 90% overlap for smooth appearance

f, t, Sxx = spectrogram(segment, sr, nperseg=nperseg, noverlap=noverlap,
                         window='hann', scaling='spectrum')
t_ms = t * 1000

# Limit frequency range
freq_max = 5000
freq_mask = f <= freq_max

# Convert to dB and normalize
Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-10)
vmin = np.percentile(Sxx_db, 5)
vmax = np.percentile(Sxx_db, 99)

# Plot with better contrast
im = ax2.pcolormesh(t_ms, f[freq_mask], Sxx_db,
                     shading='gouraud', cmap='inferno',
                     vmin=vmin, vmax=vmax)

ax2.axvline(trill_start_rel, color='white', linewidth=2, linestyle='-')
ax2.axvline(trill_end_rel, color='white', linewidth=2, linestyle='-')

ax2.set_ylabel('Frequency (Hz)', fontsize=11)
ax2.set_xlabel('Time (ms)', fontsize=11)
ax2.set_ylim(0, freq_max)
ax2.set_xlim(0, total_duration_ms)

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, label='Power (dB)', pad=0.02)

plt.tight_layout()
plt.savefig('paper_materials/figures/trill_spectrogram_fine.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\nCreated: paper_materials/figures/trill_spectrogram_fine.png")
plt.close()


# === Alternative: Even finer with different colormap ===
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                          gridspec_kw={'height_ratios': [1, 1.5]})

# Waveform
ax1 = axes[0]
ax1.plot(time_ms, segment, 'k-', linewidth=0.5)
ax1.axvline(trill_start_rel, color='#E67E22', linewidth=2)
ax1.axvline(trill_end_rel, color='#E67E22', linewidth=2)
ax1.axvspan(trill_start_rel, trill_end_rel, alpha=0.15, color='#E67E22')
ax1.set_ylabel('Amplitude', fontsize=11)
ax1.set_title(f'Spanish Trill /r/ — {int(example["num_cycles"])} cycles, {example["duration_ms"]:.0f} ms',
              fontsize=13, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Very fine spectrogram (3ms window)
window_ms = 4
nperseg = int(window_ms / 1000 * sr)
noverlap = int(nperseg * 0.92)

f, t, Sxx = spectrogram(segment, sr, nperseg=nperseg, noverlap=noverlap,
                         window='hann', scaling='spectrum')
t_ms = t * 1000

freq_max = 5000
freq_mask = f <= freq_max

Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-10)
vmin = np.percentile(Sxx_db, 3)
vmax = np.percentile(Sxx_db, 98)

ax2 = axes[1]
im = ax2.pcolormesh(t_ms, f[freq_mask], Sxx_db,
                     shading='gouraud', cmap='magma',
                     vmin=vmin, vmax=vmax)

ax2.axvline(trill_start_rel, color='white', linewidth=2)
ax2.axvline(trill_end_rel, color='white', linewidth=2)
ax2.set_ylabel('Frequency (Hz)', fontsize=11)
ax2.set_xlabel('Time (ms)', fontsize=11)
ax2.set_ylim(0, freq_max)

cbar = plt.colorbar(im, ax=ax2, label='Power (dB)', pad=0.02)

plt.tight_layout()
plt.savefig('paper_materials/figures/trill_spectrogram_hires.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: paper_materials/figures/trill_spectrogram_hires.png")
plt.close()


# === Zoom in on just the trill ===
trill_only = audio[start_sample:end_sample]
trill_time = np.arange(len(trill_only)) / sr * 1000

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                          gridspec_kw={'height_ratios': [1, 1.5]})

# Waveform - trill only
ax1 = axes[0]
ax1.plot(trill_time, trill_only, 'k-', linewidth=0.7)
ax1.set_ylabel('Amplitude', fontsize=11)
ax1.set_title(f'Trill /r/ Close-up — {int(example["num_cycles"])} cycles', fontsize=13, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0, trill_time[-1])

# Fine spectrogram - trill only
window_ms = 4
nperseg = int(window_ms / 1000 * sr)
noverlap = int(nperseg * 0.9)

f, t, Sxx = spectrogram(trill_only, sr, nperseg=nperseg, noverlap=noverlap,
                         window='hann', scaling='spectrum')
t_ms = t * 1000

freq_max = 5000
freq_mask = f <= freq_max

Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-10)
vmin = np.percentile(Sxx_db, 2)
vmax = np.percentile(Sxx_db, 99)

ax2 = axes[1]
im = ax2.pcolormesh(t_ms, f[freq_mask], Sxx_db,
                     shading='gouraud', cmap='inferno',
                     vmin=vmin, vmax=vmax)
ax2.set_ylabel('Frequency (Hz)', fontsize=11)
ax2.set_xlabel('Time (ms)', fontsize=11)
ax2.set_ylim(0, freq_max)
ax2.set_xlim(0, trill_time[-1])

plt.colorbar(im, ax=ax2, label='Power (dB)', pad=0.02)

plt.tight_layout()
plt.savefig('paper_materials/figures/trill_closeup.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: paper_materials/figures/trill_closeup.png")
plt.close()

print("\nDone!")
