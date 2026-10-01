"""
build_processed_dataset.py

Process the controlled Week 2 dataset.

For every labeled audio file:
    Load
    -> Mono
    -> Resample to 16 kHz
    -> Normalize
    -> Split into 3-second segments

The original train/validation/test split is preserved.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .segment_audio import process_file


BASE_DIR = Path("D:/data/ASVspoof2019/LA")

AUDIO_DIRS = {
    "train": BASE_DIR / "ASVspoof2019_LA_train" / "flac",
    "validation": BASE_DIR / "ASVspoof2019_LA_dev" / "flac",
    "test": BASE_DIR / "ASVspoof2019_LA_eval" / "flac",
}

SPLIT_FILES = {
    "train": Path("data/splits/train.csv"),
    "validation": Path("data/splits/validation.csv"),
    "test": Path("data/splits/test.csv"),
}

OUTPUT_ROOT = Path("D:/data/week2_controlled/processed")


def process_split(split: str) -> None:
    metadata_path = SPLIT_FILES[split]
    audio_dir = AUDIO_DIRS[split]
    output_dir = OUTPUT_ROOT / split

    df = pd.read_csv(metadata_path)

    print(f"\nProcessing {split.upper()}: {len(df)} source files")

    processed_sources = 0
    total_segments = 0

    for row in df.itertuples(index=False):
        input_path = audio_dir / row.filename
        label_dir = "REAL" if row.label == 0 else "FAKE"
        destination = output_dir / label_dir

        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        paths = process_file(
            input_file=input_path,
            output_dir=destination,
        )

        processed_sources += 1
        total_segments += len(paths)

        if processed_sources % 100 == 0:
            print(
                f"  processed {processed_sources}/{len(df)} "
                f"source files, {total_segments} segments"
            )

    print(f"Completed {split}:")
    print(f"  Source files = {processed_sources}")
    print(f"  Segments     = {total_segments}")


def main() -> None:
    for split in ("train", "validation", "test"):
        process_split(split)


if __name__ == "__main__":
    main()
