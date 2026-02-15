"""Generate publication-quality trill spectrogram with clear cycle visibility."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.ndimage import uniform_filter1d
import pandas as pd

# Find a good trill example - look for one with clear separation
df = pd.read_parquet('outputs/tables/acoustic_measurements_dimex100.parquet')

# Find examples with moderate cycle rate (easier to see)
good = df[(df['num_cycles'] >= 3) & (df['num_cycles'] <= 4) &
          (df['duration_ms'] >= 80) & (df['duration_ms'] <= 120) &
          (df['voicing_pct'] > 90) &
          (df['cycle_rate_hz'] >= 25) & (df['cycle_rate_hz'] <= 50)]  # Slower = more visible

if len(good) == 0:
    good = df[(df['num_cycles'] >= 3) & (df['num_cycles'] <= 4) &
              (df['duration_ms'] >= 70) & (df['duration_ms'] <= 120) &
              (df['voicing_pct'] > 85)]

print(f"Found {len(good)} examples")
example = good.iloc[0]
print(f"Using: {example['word'][:40]}...")
print(f"  Cycles: {int(example['num_cycles'])}, Duration: {example['duration_ms']:.1f}ms")
print(f"  Cycle rate: {example['cycle_rate_hz']:.1f} Hz")

# Load audio
sr, audio = wavfile.read(example['audio_path'])
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

# Extract trill with minimal padding
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

# === Main figure: Waveform + Wideband Spectrogram ===
fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True,
                          gridspec_kw={'height_ratios': [1, 1.3]})

# Waveform
ax1 = axes[0]
ax1.plot(time_ms, segment, 'k-', linewidth=0.6)
ax1.axvline(trill_start_rel, color='#D35400', linewidth=1.5)
ax1.axvline(trill_end_rel, color='#D35400', linewidth=1.5)
ax1.axvspan(trill_start_rel, trill_end_rel, alpha=0.12, color='#E67E22')
ax1.set_ylabel('Amplitude', fontsize=11)
ax1.set_title(f'Spanish Alveolar Trill /r/ — {int(example["num_cycles"])} cycles',
              fontsize=13, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0, time_ms[-1])

# Wideband spectrogram (very short window = 3ms for time resolution)
ax2 = axes[1]
window_ms = 3
nperseg = int(window_ms / 1000 * sr)
nperseg = max(nperseg, 64)  # minimum size
noverlap = int(nperseg * 0.95)

f, t, Sxx = spectrogram(segment, sr, nperseg=nperseg, noverlap=noverlap, window='hann')
t_ms = t * 1000

# Focus on 0-4000 Hz
freq_max = 4000
freq_mask = f <= freq_max

Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-12)
# Strong contrast
vmin = np.percentile(Sxx_db, 10)
vmax = np.percentile(Sxx_db, 99.5)

im = ax2.pcolormesh(t_ms, f[freq_mask], Sxx_db, shading='gouraud',
                     cmap='hot', vmin=vmin, vmax=vmax)

ax2.axvline(trill_start_rel, color='white', linewidth=1.5)
ax2.axvline(trill_end_rel, color='white', linewidth=1.5)
ax2.set_ylabel('Frequency (Hz)', fontsize=11)
ax2.set_xlabel('Time (ms)', fontsize=11)
ax2.set_ylim(0, freq_max)

plt.tight_layout()
plt.savefig('paper_materials/figures/trill_wideband.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: paper_materials/figures/trill_wideband.png")
plt.close()


# === Alternative with envelope and cycle markers ===
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                          gridspec_kw={'height_ratios': [1, 0.6, 1.3]})

# Waveform
ax1 = axes[0]
ax1.plot(time_ms, segment, 'k-', linewidth=0.5)
ax1.axvline(trill_start_rel, color='#D35400', linewidth=1.5)
ax1.axvline(trill_end_rel, color='#D35400', linewidth=1.5)
ax1.axvspan(trill_start_rel, trill_end_rel, alpha=0.12, color='#E67E22')
ax1.set_ylabel('Amplitude', fontsize=10)
ax1.set_title(f'Spanish Alveolar Trill /r/ — {int(example["num_cycles"])} cycles',
              fontsize=13, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Amplitude envelope with cycle detection
ax2 = axes[1]
# Calculate envelope
trill_seg = segment[int(trill_start_rel/1000*sr):int(trill_end_rel/1000*sr)]
trill_time = np.linspace(trill_start_rel, trill_end_rel, len(trill_seg))
envelope = np.abs(trill_seg)
envelope_smooth = uniform_filter1d(envelope, size=int(sr*0.004))

ax2.fill_between(trill_time, 0, envelope_smooth, color='#3498DB', alpha=0.7)
ax2.plot(trill_time, envelope_smooth, 'k-', linewidth=0.8)

# Mark peaks (releases)
from scipy.signal import find_peaks
peaks, _ = find_peaks(envelope_smooth, distance=int(sr*0.015), prominence=0.02*np.max(envelope_smooth))
for i, pk in enumerate(peaks[:int(example['num_cycles'])+1]):
    pk_time = trill_time[pk]
    ax2.axvline(pk_time, color='#E74C3C', linewidth=1.5, linestyle='-', alpha=0.8)
    ax2.text(pk_time, np.max(envelope_smooth)*1.1, f'{i+1}', ha='center', fontsize=10,
             fontweight='bold', color='#E74C3C')

ax2.set_ylabel('Envelope', fontsize=10)
ax2.set_xlim(trill_start_rel - 5, trill_end_rel + 5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Spectrogram
ax3 = axes[2]
im = ax3.pcolormesh(t_ms, f[freq_mask], Sxx_db, shading='gouraud',
                     cmap='hot', vmin=vmin, vmax=vmax)
ax3.axvline(trill_start_rel, color='white', linewidth=1.5)
ax3.axvline(trill_end_rel, color='white', linewidth=1.5)

# Add cycle markers on spectrogram
for i, pk in enumerate(peaks[:int(example['num_cycles'])+1]):
    pk_time = trill_time[pk]
    ax3.axvline(pk_time, color='white', linewidth=1, linestyle='--', alpha=0.6)

ax3.set_ylabel('Frequency (Hz)', fontsize=10)
ax3.set_xlabel('Time (ms)', fontsize=11)
ax3.set_ylim(0, freq_max)
ax3.set_xlim(0, time_ms[-1])

plt.tight_layout()
plt.savefig('paper_materials/figures/trill_with_envelope.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: paper_materials/figures/trill_with_envelope.png")
plt.close()


# === Simple clean version for paper ===
fig, ax = plt.subplots(figsize=(10, 4))

# Just the spectrogram, zoomed on trill region
trill_only = audio[start_sample:end_sample]
trill_time_only = np.arange(len(trill_only)) / sr * 1000

window_ms = 3
nperseg = int(window_ms / 1000 * sr)
nperseg = max(nperseg, 64)
noverlap = int(nperseg * 0.95)

f, t, Sxx = spectrogram(trill_only, sr, nperseg=nperseg, noverlap=noverlap, window='hann')
t_ms = t * 1000

freq_max = 5000
freq_mask = f <= freq_max

Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-12)
vmin = np.percentile(Sxx_db, 5)
vmax = np.percentile(Sxx_db, 99)

im = ax.pcolormesh(t_ms, f[freq_mask], Sxx_db, shading='gouraud',
                    cmap='inferno', vmin=vmin, vmax=vmax)

ax.set_ylabel('Frequency (Hz)', fontsize=12)
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_title(f'Wideband Spectrogram of Trill /r/ — {int(example["num_cycles"])} cycles',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, freq_max)

plt.colorbar(im, ax=ax, label='Power (dB)')
plt.tight_layout()
plt.savefig('paper_materials/figures/trill_spectrogram_only.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: paper_materials/figures/trill_spectrogram_only.png")
plt.close()

print("\nDone!")
