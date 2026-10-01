"""
build_phone_mfcc.py

Extract the same 80-dimensional MFCC features used by the clean baseline
from the phone-like 3-second test segments.

Pipeline:
    Phone-like 3-second WAV
        ->
    MFCC (40 coefficients)
        ->
    Mean + Standard Deviation
        ->
    80-dimensional feature vector
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .extract_mfcc import extract_mfcc, SAMPLE_RATE, N_MFCC
from .aggregate_mfcc import aggregate_mfcc


MANIFEST = Path("data/metadata/phone_segment_metadata.csv")
SEGMENT_DIR = Path("data/processed/phone_like_test")
OUTPUT_DIR = Path("D:/data/week2_controlled/features/phone_like_test")


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST)

    required = {
        "filename",
        "original_filename",
        "label",
        "class_name",
        "speaker_id",
        "condition",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    features = []
    labels = []
    metadata_rows = []

    print(f"Processing PHONE-LIKE TEST: {len(df)} segments")

    for index, row in enumerate(df.itertuples(index=False), start=1):
        audio_path = SEGMENT_DIR / row.filename

        if not audio_path.exists():
            raise FileNotFoundError(f"Segment file not found: {audio_path}")

        mfcc = extract_mfcc(
            audio_path,
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
        )

        vector = aggregate_mfcc(mfcc)

        if vector.shape != (80,):
            raise ValueError(
                f"Expected 80 features, got {vector.shape} "
                f"for {audio_path}"
            )

        features.append(vector)
        labels.append(int(row.label))

        metadata_rows.append({
            "segment_filename": row.filename,
            "original_filename": row.original_filename,
            "label": int(row.label),
            "class_name": row.class_name,
            "speaker_id": row.speaker_id,
            "condition": row.condition,
        })

        if index % 100 == 0:
            print(f"  processed {index}/{len(df)}")

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)

    pd.DataFrame(metadata_rows).to_csv(
        OUTPUT_DIR / "metadata.csv",
        index=False,
    )

    print()
    print("=" * 40)
    print("PHONE-LIKE MFCC FEATURES COMPLETE")
    print("=" * 40)
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"REAL    : {(y == 0).sum()}")
    print(f"FAKE    : {(y == 1).sum()}")
    print(f"Output  : {OUTPUT_DIR}")
    print("=" * 40)


if __name__ == "__main__":
    main()
