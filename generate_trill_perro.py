"""Generate publication-quality trill spectrogram using clear 'perro' example."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.ndimage import uniform_filter1d
import pandas as pd

# Load all acoustic measurements to find perro
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

# Find best "perro" example - intervocalic, good cycles, 100% voicing
perro = df[(df['word'].str.lower() == 'perro') &
           (df['context_label'] == 'intervocalic_rr') &
           (df['num_cycles'] >= 3) & (df['num_cycles'] <= 5) &
           (df['voicing_pct'] >= 95) &
           (df['duration_ms'] >= 50) & (df['duration_ms'] <= 120)]

if len(perro) == 0:
    # Try tierra as backup
    perro = df[(df['word'].str.lower() == 'tierra') &
               (df['context_label'] == 'intervocalic_rr') &
               (df['num_cycles'] >= 3) & (df['num_cycles'] <= 5) &
               (df['voicing_pct'] >= 95) &
               (df['duration_ms'] >= 50) & (df['duration_ms'] <= 120)]

print(f"Found {len(perro)} clear examples")
example = perro.iloc[0]
print(f"Using: '{example['word']}' from {example['dataset']}")
print(f"  Cycles: {int(example['num_cycles'])}, Duration: {example['duration_ms']:.1f}ms")
print(f"  Voicing: {example['voicing_pct']:.1f}%, Cycle rate: {example['cycle_rate_hz']:.1f} Hz")
print(f"  Audio: {example['audio_path']}")

# Load audio
audio_path = example['audio_path']
if audio_path.endswith('.mp3'):
    import subprocess
    import tempfile
    import os
    # Convert mp3 to wav
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

print(f"  Sample rate: {sr} Hz")

# Extract trill with context
start_sample = int((example['start_ms'] / 1000) * sr)
end_sample = int((example['end_ms'] / 1000) * sr)

# Add padding (30ms before and after)
pad_ms = 30
pad_samples = int(pad_ms / 1000 * sr)
start_pad = max(0, start_sample - pad_samples)
end_pad = min(len(audio), end_sample + pad_samples)

segment = audio[start_pad:end_pad]
trill_start_rel = (start_sample - start_pad) / sr * 1000
trill_end_rel = (end_sample - start_pad) / sr * 1000
time_ms = np.arange(len(segment)) / sr * 1000

# === Figure 1: Clean waveform + wideband spectrogram ===
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                          gridspec_kw={'height_ratios': [1, 1.3]})

# Waveform
ax1 = axes[0]
ax1.plot(time_ms, segment, 'k-', linewidth=0.5)
ax1.axvline(trill_start_rel, color='#E67E22', linewidth=2)
ax1.axvline(trill_end_rel, color='#E67E22', linewidth=2)
ax1.axvspan(trill_start_rel, trill_end_rel, alpha=0.15, color='#E67E22')
ax1.set_ylabel('Amplitude', fontsize=11)
ax1.set_title(f'Spanish Alveolar Trill /r/ in "{example["word"]}" — {int(example["num_cycles"])} cycles',
              fontsize=13, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0, time_ms[-1])

# Wideband spectrogram (2-3ms window for maximum time resolution)
ax2 = axes[1]
window_ms = 2.5
nperseg = int(window_ms / 1000 * sr)
nperseg = max(nperseg, 32)
noverlap = int(nperseg * 0.95)

f, t, Sxx = spectrogram(segment, sr, nperseg=nperseg, noverlap=noverlap, window='hann')
t_ms = t * 1000

# Focus on 0-5000 Hz
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
plt.savefig('paper_materials/figures/trill_perro_wideband.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\nCreated: paper_materials/figures/trill_perro_wideband.png")
plt.close()


# === Figure 2: With envelope and cycle markers ===
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                          gridspec_kw={'height_ratios': [1, 0.6, 1.3]})

# Waveform
ax1 = axes[0]
ax1.plot(time_ms, segment, 'k-', linewidth=0.5)
ax1.axvline(trill_start_rel, color='#E67E22', linewidth=2)
ax1.axvline(trill_end_rel, color='#E67E22', linewidth=2)
ax1.axvspan(trill_start_rel, trill_end_rel, alpha=0.15, color='#E67E22')
ax1.set_ylabel('Amplitude', fontsize=10)
ax1.set_title(f'Spanish Alveolar Trill /r/ in "{example["word"]}" — {int(example["num_cycles"])} cycles',
              fontsize=13, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Amplitude envelope with cycle markers
ax2 = axes[1]
trill_start_samp = int(trill_start_rel / 1000 * sr)
trill_end_samp = int(trill_end_rel / 1000 * sr)
trill_seg = segment[trill_start_samp:trill_end_samp]
trill_time = np.linspace(trill_start_rel, trill_end_rel, len(trill_seg))
envelope = np.abs(trill_seg)
envelope_smooth = uniform_filter1d(envelope, size=max(1, int(sr*0.003)))

ax2.fill_between(trill_time, 0, envelope_smooth, color='#3498DB', alpha=0.6)
ax2.plot(trill_time, envelope_smooth, 'k-', linewidth=0.8)

# Mark peaks (releases)
from scipy.signal import find_peaks
peaks, _ = find_peaks(envelope_smooth, distance=max(1, int(sr*0.012)),
                      prominence=0.02*np.max(envelope_smooth))

for i, pk in enumerate(peaks[:int(example['num_cycles'])+1]):
    if pk < len(trill_time):
        pk_time = trill_time[pk]
        ax2.axvline(pk_time, color='#E74C3C', linewidth=1.5, alpha=0.8)
        ax2.text(pk_time, np.max(envelope_smooth)*1.15, f'{i+1}', ha='center', fontsize=11,
                 fontweight='bold', color='#E74C3C')

ax2.set_ylabel('Envelope', fontsize=10)
ax2.set_xlim(trill_start_rel - 5, trill_end_rel + 5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Spectrogram
ax3 = axes[2]
im = ax3.pcolormesh(t_ms, f[freq_mask], Sxx_db, shading='gouraud',
                     cmap='inferno', vmin=vmin, vmax=vmax)
ax3.axvline(trill_start_rel, color='white', linewidth=2)
ax3.axvline(trill_end_rel, color='white', linewidth=2)

# Add cycle markers on spectrogram
for i, pk in enumerate(peaks[:int(example['num_cycles'])+1]):
    if pk < len(trill_time):
        pk_time = trill_time[pk]
        ax3.axvline(pk_time, color='white', linewidth=1, linestyle='--', alpha=0.7)

ax3.set_ylabel('Frequency (Hz)', fontsize=10)
ax3.set_xlabel('Time (ms)', fontsize=11)
ax3.set_ylim(0, freq_max)
ax3.set_xlim(0, time_ms[-1])

plt.tight_layout()
plt.savefig('paper_materials/figures/trill_perro_annotated.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: paper_materials/figures/trill_perro_annotated.png")
plt.close()


# === Figure 3: Zoomed on just the trill ===
trill_only = audio[start_sample:end_sample]
trill_time_only = np.arange(len(trill_only)) / sr * 1000

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                          gridspec_kw={'height_ratios': [1, 1.3]})

# Waveform
ax1 = axes[0]
ax1.plot(trill_time_only, trill_only, 'k-', linewidth=0.7)
ax1.set_ylabel('Amplitude', fontsize=11)
ax1.set_title(f'Trill /r/ in "{example["word"]}" — {int(example["num_cycles"])} cycles, {example["duration_ms"]:.0f} ms',
              fontsize=13, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Fine spectrogram
window_ms = 2.5
nperseg = int(window_ms / 1000 * sr)
nperseg = max(nperseg, 32)
noverlap = int(nperseg * 0.95)

f, t, Sxx = spectrogram(trill_only, sr, nperseg=nperseg, noverlap=noverlap, window='hann')
t_ms = t * 1000

freq_mask = f <= freq_max
Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-12)
vmin = np.percentile(Sxx_db, 3)
vmax = np.percentile(Sxx_db, 99)

ax2 = axes[1]
im = ax2.pcolormesh(t_ms, f[freq_mask], Sxx_db, shading='gouraud',
                     cmap='inferno', vmin=vmin, vmax=vmax)
ax2.set_ylabel('Frequency (Hz)', fontsize=11)
ax2.set_xlabel('Time (ms)', fontsize=11)
ax2.set_ylim(0, freq_max)

plt.colorbar(im, ax=ax2, label='Power (dB)', pad=0.02)
plt.tight_layout()
plt.savefig('paper_materials/figures/trill_perro_closeup.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: paper_materials/figures/trill_perro_closeup.png")
plt.close()


# === Figure 4: Grayscale version (for print) ===
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                          gridspec_kw={'height_ratios': [1, 1.3]})

# Waveform
ax1 = axes[0]
ax1.plot(time_ms, segment, 'k-', linewidth=0.5)
ax1.axvline(trill_start_rel, color='#333333', linewidth=2, linestyle='--')
ax1.axvline(trill_end_rel, color='#333333', linewidth=2, linestyle='--')
ax1.axvspan(trill_start_rel, trill_end_rel, alpha=0.1, color='gray')
ax1.set_ylabel('Amplitude', fontsize=11)
ax1.set_title(f'Spanish Alveolar Trill /r/ in "{example["word"]}" — {int(example["num_cycles"])} cycles',
              fontsize=13, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0, time_ms[-1])

# Grayscale spectrogram
window_ms = 2.5
nperseg = int(window_ms / 1000 * sr)
nperseg = max(nperseg, 32)
noverlap = int(nperseg * 0.95)

f, t, Sxx = spectrogram(segment, sr, nperseg=nperseg, noverlap=noverlap, window='hann')
t_ms = t * 1000

freq_mask = f <= freq_max
Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-12)
vmin = np.percentile(Sxx_db, 5)
vmax = np.percentile(Sxx_db, 99)

ax2 = axes[1]
im = ax2.pcolormesh(t_ms, f[freq_mask], Sxx_db, shading='gouraud',
                     cmap='Greys', vmin=vmin, vmax=vmax)

ax2.axvline(trill_start_rel, color='black', linewidth=2, linestyle='--')
ax2.axvline(trill_end_rel, color='black', linewidth=2, linestyle='--')
ax2.set_ylabel('Frequency (Hz)', fontsize=11)
ax2.set_xlabel('Time (ms)', fontsize=11)
ax2.set_ylim(0, freq_max)

plt.tight_layout()
plt.savefig('paper_materials/figures/trill_perro_grayscale.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: paper_materials/figures/trill_perro_grayscale.png")
plt.close()

print("\nDone!")
