# MFCC Notes

## What is MFCC?

**MFCC (Mel-Frequency Cepstral Coefficients)** is a widely used audio feature representation for speech processing. It captures the overall spectral characteristics of an audio signal using a scale (the Mel scale) that approximates how humans perceive pitch.

### Extraction pipeline

```text
Audio Signal
     ↓
Framing & Windowing
     ↓
Fourier Transform
     ↓
Mel Filter Bank
     ↓
Logarithmic Compression
     ↓
Discrete Cosine Transform (DCT)
     ↓
MFCC Features
```

MFCCs describe the general shape of the short-term spectrum rather than every fine spectral detail. Lower-order coefficients capture broad spectral shape, while higher-order coefficients capture progressively finer variation.

### Output shape (librosa)

In `librosa`, MFCCs are computed with:

```python
librosa.feature.mfcc()
```

The returned matrix has shape:

```text
(n_mfcc, T)
```

- `n_mfcc` — number of MFCC coefficients selected (e.g. 40)
- `T` — number of time frames, which depends on audio duration and analysis parameters (frame size, hop length)

Example: with `n_mfcc=40`, a 3-second clip might produce something like `(40, 130)`. The exact `T` will vary by audio length and parameters — this is expected.

## Why MFCCs for this project?

Speech carries characteristic spectral patterns from the human vocal tract and speech production process. MFCCs give a compact representation of these patterns and have a long track record in speech/audio classification, making them a reasonable choice for a **traditional ML baseline** before moving to deep learning.

Use case here:

```text
REAL HUMAN SPEECH   vs   AI-GENERATED / SPOOFED SPEECH
```

## Important limitation

MFCCs are **not guaranteed to capture every artifact of modern deepfake/TTS speech**. They are being used here as an initial baseline representation only. Richer representations (Mel-spectrograms, learned embeddings) can be compared later.

> Do not claim MFCC is the best feature for this task — we are testing it as a first baseline.
