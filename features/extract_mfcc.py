"""MFCC extraction for the Week 1 baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
N_MFCC = 40


def validate_wav_file(audio_path: str | Path) -> Path:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Only WAV files are accepted: {path.name}")
    info = sf.info(path)
    if info.format.upper() != "WAV":
        raise ValueError(f"File is not a genuine WAV container: {path.name}")
    return path


def extract_mfcc(
    audio_path: str | Path,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
) -> np.ndarray:
    """Load standardized WAV audio and return an (n_mfcc, time) matrix."""
    path = validate_wav_file(audio_path)
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MFCC features from one WAV file.")
    parser.add_argument("audio_path")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--n-mfcc", type=int, default=N_MFCC)
    args = parser.parse_args()

    try:
        mfcc = extract_mfcc(args.audio_path, args.sample_rate, args.n_mfcc)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print("----------------------------------------")
    print("Audio file:", args.audio_path)
    print("Input format: WAV")
    print("MFCC shape:", mfcc.shape)
    print("MFCC dtype:", mfcc.dtype)
    print("----------------------------------------")


if __name__ == "__main__":
    main()
