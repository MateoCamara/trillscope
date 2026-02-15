"""Generate a clear didactic figure showing trill cycle definition."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
import pandas as pd

# Find a good trill example
df = pd.read_parquet('outputs/tables/acoustic_measurements_dimex100.parquet')

# Find a clear example with 3-4 cycles, intervocalic
good = df[(df['num_cycles'] >= 3) & (df['num_cycles'] <= 4) &
          (df['duration_ms'] >= 70) & (df['duration_ms'] <= 100) &
          (df['voicing_pct'] > 80)]

example = good.iloc[5]  # Try a different example
print(f"Using: {example['word']} - {int(example['num_cycles'])} cycles, {example['duration_ms']:.1f}ms")

# Load audio
sr, audio = wavfile.read(example['audio_path'])
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

# Extract trill segment
start_sample = int((example['start_ms'] / 1000) * sr)
end_sample = int((example['end_ms'] / 1000) * sr)
trill = audio[start_sample:end_sample]
time_ms = np.arange(len(trill)) / sr * 1000

# Smooth envelope for peak/valley detection
envelope = np.abs(trill)
envelope_smooth = uniform_filter1d(envelope, size=int(sr*0.003))

# Find peaks (releases) and valleys (closures)
from scipy.signal import find_peaks
peaks, _ = find_peaks(envelope_smooth, distance=int(sr*0.012), prominence=0.02*np.max(envelope_smooth))
valleys, _ = find_peaks(-envelope_smooth, distance=int(sr*0.012), prominence=0.01*np.max(envelope_smooth))

print(f"Detected {len(peaks)} peaks, {len(valleys)} valleys")

# === Create clear didactic figure ===
fig, ax = plt.subplots(figsize=(12, 5))

# Plot waveform
ax.plot(time_ms, trill, 'k-', linewidth=0.6, alpha=0.8)

# Mark cycles with brackets and labels
n_cycles = int(example['num_cycles'])
colors = ['#3498DB', '#E74C3C', '#27AE60', '#9B59B6', '#F39C12']

# Use detected peaks to mark cycle boundaries
if len(peaks) >= n_cycles:
    for i in range(min(n_cycles, len(peaks)-1)):
        # Cycle spans from one peak to the next
        if i < len(peaks) - 1:
            start_t = time_ms[peaks[i]]
            end_t = time_ms[peaks[i+1]] if i+1 < len(peaks) else time_ms[-1]
            mid_t = (start_t + end_t) / 2

            # Draw bracket at top
            bracket_y = 0.85 * np.max(np.abs(trill))
            ax.annotate('', xy=(start_t, bracket_y), xytext=(end_t, bracket_y),
                       arrowprops=dict(arrowstyle='<->', color=colors[i % len(colors)], lw=2))
            ax.text(mid_t, bracket_y + 0.08*np.max(np.abs(trill)), f'Cycle {i+1}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold', color=colors[i % len(colors)])

# Mark closure (valley) and release (peak) for first cycle
if len(peaks) >= 2 and len(valleys) >= 1:
    # Find valley between first two peaks
    first_valley = None
    for v in valleys:
        if peaks[0] < v < peaks[1]:
            first_valley = v
            break

    if first_valley is not None:
        # Mark release (peak)
        ax.annotate('Release\n(peak)', xy=(time_ms[peaks[0]], trill[peaks[0]]),
                   xytext=(time_ms[peaks[0]]-15, trill[peaks[0]]+0.15),
                   fontsize=10, ha='center',
                   arrowprops=dict(arrowstyle='->', color='#E67E22', lw=1.5),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E67E22'))

        # Mark closure (valley)
        ax.annotate('Closure\n(valley)', xy=(time_ms[first_valley], trill[first_valley]),
                   xytext=(time_ms[first_valley], trill[first_valley]-0.2),
                   fontsize=10, ha='center',
                   arrowprops=dict(arrowstyle='->', color='#E67E22', lw=1.5),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E67E22'))

# Add definition box
textstr = '1 cycle = 1 closure + 1 release\n(tongue tip contacts ridge, then releases)'
props = dict(boxstyle='round', facecolor='#FEF9E7', edgecolor='#E67E22', linewidth=2)
ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right', bbox=props)

ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_title(f'Spanish Alveolar Trill /r/ — {n_cycles} Cycles', fontsize=14, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-2, time_ms[-1]+2)

plt.tight_layout()
plt.savefig('paper_materials/figures/trill_cycle_definition.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\nCreated: paper_materials/figures/trill_cycle_definition.png")
plt.close()

# === Create combined waveform + spectrogram ===
from scipy.signal import spectrogram

# Add context
pad = int(0.03 * sr)
start_pad = max(0, start_sample - pad)
end_pad = min(len(audio), end_sample + pad)
segment = audio[start_pad:end_pad]
seg_time = np.arange(len(segment)) / sr * 1000
trill_start_rel = (start_sample - start_pad) / sr * 1000
trill_end_rel = (end_sample - start_pad) / sr * 1000

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                          gridspec_kw={'height_ratios': [1, 1.2]})

# Waveform
ax1 = axes[0]
ax1.plot(seg_time, segment, 'k-', linewidth=0.5)
ax1.axvline(trill_start_rel, color='#E67E22', linestyle='-', linewidth=2)
ax1.axvline(trill_end_rel, color='#E67E22', linestyle='-', linewidth=2)
ax1.axvspan(trill_start_rel, trill_end_rel, alpha=0.15, color='#E67E22')
ax1.set_ylabel('Amplitude', fontsize=11)
ax1.set_title(f'Trill /r/ in "{example["word"][:30]}..." — {n_cycles} cycles, {example["duration_ms"]:.0f} ms',
              fontsize=13, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Spectrogram
ax2 = axes[1]
nperseg = 256
f, t, Sxx = spectrogram(segment, sr, nperseg=nperseg, noverlap=nperseg*3//4)
t_ms = t * 1000

# Limit frequency
freq_max = 6000
freq_mask = f <= freq_max
im = ax2.pcolormesh(t_ms, f[freq_mask], 10*np.log10(Sxx[freq_mask] + 1e-10),
                     shading='gouraud', cmap='Greys')
ax2.axvline(trill_start_rel, color='#E67E22', linestyle='-', linewidth=2)
ax2.axvline(trill_end_rel, color='#E67E22', linestyle='-', linewidth=2)
ax2.set_ylabel('Frequency (Hz)', fontsize=11)
ax2.set_xlabel('Time (ms)', fontsize=11)
ax2.set_ylim(0, freq_max)

# Add F0 line estimate
ax2.axhline(example['mean_f0_hz'], color='#3498DB', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.text(seg_time[-1]-5, example['mean_f0_hz']+200, f"F0 ≈ {example['mean_f0_hz']:.0f} Hz",
         fontsize=9, color='#3498DB', ha='right')

plt.tight_layout()
plt.savefig('paper_materials/figures/trill_spectrogram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Created: paper_materials/figures/trill_spectrogram.png")
plt.close()

print("\nDone!")
