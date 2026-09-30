"""Convert a variable-length MFCC matrix into a fixed 80-value vector."""

from __future__ import annotations

import argparse
import numpy as np

try:
    from .extract_mfcc import extract_mfcc, SAMPLE_RATE, N_MFCC
except ImportError:
    from extract_mfcc import extract_mfcc, SAMPLE_RATE, N_MFCC


def aggregate_mfcc(mfcc: np.ndarray) -> np.ndarray:
    """Concatenate per-coefficient mean and standard deviation."""
    if mfcc.ndim != 2:
        raise ValueError(f"Expected MFCC matrix with 2 dimensions, got {mfcc.shape}")
    means = np.mean(mfcc, axis=1)
    stds = np.std(mfcc, axis=1)
    return np.concatenate([means, stds]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MFCCs into fixed-length features.")
    parser.add_argument("audio_path")
    args = parser.parse_args()

    try:
        mfcc = extract_mfcc(args.audio_path, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC)
        features = aggregate_mfcc(mfcc)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print("----------------------------------------")
    print("Audio file:", args.audio_path)
    print("MFCC shape:", mfcc.shape)
    print("Feature vector shape:", features.shape)
    print("Number of features:", len(features))
    print("Feature dtype:", features.dtype)
    print("First 10 features:", features[:10])
    print("----------------------------------------")


if __name__ == "__main__":
    main()
