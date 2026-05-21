# erres-v2

Spanish trill /r/ closure detection and multi-corpus acoustic analysis. Rewrite of the analysis pipeline for the IberSpeech 2026 submission.

## What it does

Given a Spanish trill token (audio + segmental boundaries), counts the number of **occlusion–release events** (closures) rather than envelope peaks. The previous pipeline counted envelope peaks and likely reported ≈ 2× the number that the linguistic literature reports as "contacts/closures/oclusiones".

## Detector v2

Closure detection is anchored in the *energy minimum* of the trill, not its periodicity:

1. **Two-band RMS envelopes**: 0–500 Hz (voicing) and 500–3500 Hz (turbulence/burst).
2. **Closure candidates**: local minima where both envelopes drop below 30 % of their local max in a 60 ms window, with ≥ 6 dB prominence.
3. **Release verification**: each closure must be followed within 5–25 ms by a mid-band burst peak ≥ 60 % of the local max. Closures without a verified release are discarded.
4. **Periodicity refinement**: estimate the dominant inter-closure interval by autocorrelation; reject closures whose intervals deviate > 2 SD from the inferred period.
5. **Per-token confidence**: combines period regularity, closure depth uniformity, and voicing stability.

## Layout

```
config/        YAML configs (corpora, detector params)
erres/         Library code
  corpora/     One loader per corpus
  alignment/   MFA + native alignment readers
  extraction/  Phonotactic context extraction
  detector/    Detector v2
  features/    F0, HNR, voicing, duration
  stats/       Kruskal-Wallis, Dunn, epsilon-squared
  viz/         Paper figures
tests/         pytest, synthetic trills
annotations/   Manual annotation tooling (Phase A2)
notebooks/     paper_figures.ipynb
```

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
pytest
```
