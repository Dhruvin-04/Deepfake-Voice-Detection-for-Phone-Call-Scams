"""
build_mfcc_dataset.py

Convert verified 3-second WAV segments into fixed-length MFCC features.

Pipeline:
    3-second WAV
        ->
    MFCC (40 coefficients)
        ->
    Mean + Standard Deviation
        ->
    80-dimensional feature vector

The segment manifest stores relative paths such as:

    processed/train/REAL/example.wav

The actual processed-data location is supplied at runtime.

Outputs:
    X.npy
    y.npy
    metadata.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .extract_mfcc import extract_mfcc, SAMPLE_RATE, N_MFCC
from .aggregate_mfcc import aggregate_mfcc


MANIFEST = Path("data/metadata/segment_metadata.csv")

SPLITS = ("train", "validation", "test")


def resolve_segment_path(segment_path: str, processed_root: Path) -> Path:
    path = Path(str(segment_path))

    if path.is_absolute():
        return path

    return processed_root / path


def build_split(
    df: pd.DataFrame,
    split: str,
    processed_root: Path,
    output_root: Path,
) -> None:

    split_df = df[df["split"] == split].copy().reset_index(drop=True)

    if split_df.empty:
        raise ValueError(f"No rows found for split: {split}")

    features = []
    metadata_rows = []

    print(f"\nProcessing {split.upper()}: {len(split_df)} segments")

    for index, row in enumerate(split_df.itertuples(index=False), start=1):

        audio_path = resolve_segment_path(
            row.segment_path,
            processed_root,
        )

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Segment file not found: {audio_path}"
            )

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

        metadata_rows.append(
            {
                "segment_path": str(row.segment_path),
                "source_filename": row.source_filename,
                "label": int(row.label),
                "class_name": row.class_name,
                "speaker_id": row.speaker_id,
                "split": row.split,
            }
        )

        if index % 100 == 0:
            print(f"  processed {index}/{len(split_df)}")

    X = np.stack(features).astype(np.float32)
    y = split_df["label"].to_numpy(dtype=np.int64)

    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)

    metadata_output = output_dir / "metadata.csv"

    pd.DataFrame(metadata_rows).to_csv(
        metadata_output,
        index=False,
    )

    print(f"Completed {split}:")
    print(f"  X shape = {X.shape}")
    print(f"  y shape = {y.shape}")
    print(f"  REAL    = {(y == 0).sum()}")
    print(f"  FAKE    = {(y == 1).sum()}")
    print(f"  X saved = {output_dir / 'X.npy'}")
    print(f"  y saved = {output_dir / 'y.npy'}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Build MFCC feature datasets from processed audio segments."
    )

    parser.add_argument(
        "--processed-root",
        required=True,
        help=(
            "Root directory containing the processed folder. "
            "Example: /path/to/processed-data-root"
        ),
    )

    parser.add_argument(
        "--output-root",
        default="artifacts/features",
        help="Directory where generated feature files are written.",
    )

    parser.add_argument(
        "--manifest",
        default=str(MANIFEST),
        help="Path to the segment metadata CSV.",
    )

    args = parser.parse_args()

    manifest = Path(args.manifest)
    processed_root = Path(args.processed_root)
    output_root = Path(args.output_root)

    if not manifest.exists():
        raise FileNotFoundError(
            f"Segment manifest not found: {manifest}"
        )

    if not processed_root.exists():
        raise FileNotFoundError(
            f"Processed-data root not found: {processed_root}"
        )

    df = pd.read_csv(manifest)

    required = {
        "segment_path",
        "source_filename",
        "label",
        "class_name",
        "speaker_id",
        "split",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing manifest columns: {sorted(missing)}"
        )

    for split in SPLITS:
        build_split(
            df,
            split,
            processed_root,
            output_root,
        )

    print("\nMFCC dataset creation complete.")


if __name__ == "__main__":
    main()
