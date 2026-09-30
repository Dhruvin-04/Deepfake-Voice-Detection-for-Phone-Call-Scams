"""
preprocess_audio.py
------------------
Week 1 standard audio preprocessing pipeline:

    Load -> Mono -> Resample -> Normalize

Denoising is intentionally NOT part of the default baseline. It remains
available as a separate experiment through denoise_audio.py.
"""

from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf

try:
    from .load_audio import load_audio
except ImportError:
    from load_audio import load_audio

TARGET_SAMPLE_RATE = 16000


def to_mono(signal: np.ndarray) -> np.ndarray:
    """Convert audio to mono. Mono input is returned unchanged."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        return signal
    if signal.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D audio, got shape {signal.shape}")
    return np.mean(signal, axis=1, dtype=np.float32)


def resample_audio(
    signal: np.ndarray,
    orig_sr: int,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Resample a mono signal to target_sr."""
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive integers.")
    if orig_sr == target_sr:
        return np.asarray(signal, dtype=np.float32)
    return librosa.resample(
        np.asarray(signal, dtype=np.float32),
        orig_sr=orig_sr,
        target_sr=target_sr,
    ).astype(np.float32)


def normalize_audio(signal: np.ndarray) -> np.ndarray:
    """Peak-normalize to approximately [-1, 1], safely handling silence."""
    signal = np.asarray(signal, dtype=np.float32)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak < 1e-8:
        return signal
    return (signal / peak).astype(np.float32)


def preprocess(
    file_path: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    denoise: bool = False,
) -> np.ndarray:
    """
    Standardize one audio file.

    Default baseline:
        Load -> Mono -> Resample -> Normalize

    Set denoise=True only for the optional denoising experiment.
    """
    signal, orig_sr = load_audio(file_path)
    signal = to_mono(signal)
    signal = resample_audio(signal, orig_sr, target_sr)

    if denoise:
        try:
            from .denoise_audio import denoise_audio
        except ImportError:
            from denoise_audio import denoise_audio
        signal = denoise_audio(signal, target_sr)

    return normalize_audio(signal)


def save_processed(
    signal: np.ndarray,
    out_path: str,
    sr: int = TARGET_SAMPLE_RATE,
) -> None:
    """Write processed mono audio as WAV."""
    sf.write(out_path, np.asarray(signal, dtype=np.float32), sr)


def describe(signal: np.ndarray, sr: int) -> dict:
    """Return basic signal properties for before/after comparisons."""
    signal = np.asarray(signal)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    return {
        "Channels": 1 if signal.ndim == 1 else signal.shape[1],
        "Sample rate (Hz)": sr,
        "Peak amplitude": round(peak, 3),
        "Duration (s)": round(len(signal) / sr, 2) if sr else 0.0,
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Run the standard Week 1 preprocessing pipeline on one file."
    )
    parser.add_argument("file", help="Input audio file path")
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Enable optional denoising experiment",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output WAV path (default: <input_stem>_clean.wav beside input)",
    )
    args = parser.parse_args()

    raw_signal, raw_sr = load_audio(args.file)
    processed = preprocess(args.file, denoise=args.denoise)

    input_path = Path(args.file)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(input_path.stem + "_clean.wav")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_processed(processed, output_path, TARGET_SAMPLE_RATE)

    before = describe(raw_signal, raw_sr)
    after = describe(processed, TARGET_SAMPLE_RATE)
    rows = [
        {"property": k, "raw": before[k], "processed": after[k]}
        for k in before
    ]

    print(f"\nRAW vs PROCESSED (denoise={'on' if args.denoise else 'off'})")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved audio -> {output_path.resolve()}")
