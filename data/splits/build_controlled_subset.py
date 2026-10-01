"""
build_controlled_subset.py

Create a reproducible, class-balanced subset for Experiment 001.

Source:
    data/metadata/dataset_metadata.csv

Output:
    data/splits/train.csv
    data/splits/validation.csv
    data/splits/test.csv

The original ASVspoof 2019 train/dev/eval boundaries are preserved.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


METADATA = Path("data/metadata/dataset_metadata.csv")
OUTPUT_DIR = Path("data/splits")

SEED = 42

SIZES = {
    "train": 500,
    "validation": 100,
    "test": 100,
}


def build_split(df: pd.DataFrame, split: str, per_class: int) -> pd.DataFrame:
    subset = df[df["split"] == split].copy()

    real = subset[subset["label"] == 0]
    fake = subset[subset["label"] == 1]

    if len(real) < per_class:
        raise ValueError(
            f"Not enough REAL samples for {split}: "
            f"need {per_class}, found {len(real)}"
        )

    if len(fake) < per_class:
        raise ValueError(
            f"Not enough FAKE samples for {split}: "
            f"need {per_class}, found {len(fake)}"
        )

    selected_real = real.sample(
        n=per_class,
        random_state=SEED,
    )

    selected_fake = fake.sample(
        n=per_class,
        random_state=SEED,
    )

    result = pd.concat(
        [selected_real, selected_fake],
        ignore_index=True,
    )

    return result.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)


def main() -> None:
    if not METADATA.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA}")

    df = pd.read_csv(METADATA)

    required_columns = {
        "filename",
        "label",
        "class_name",
        "speaker_id",
        "split",
        "dataset",
        "protocol",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, per_class in SIZES.items():
        subset = build_split(df, split, per_class)

        output_path = OUTPUT_DIR / f"{split}.csv"
        subset.to_csv(output_path, index=False)

        print(f"{split.upper()}:")
        print(f"  rows  = {len(subset)}")
        print(
            f"  REAL  = {(subset['label'] == 0).sum()}"
        )
        print(
            f"  FAKE  = {(subset['label'] == 1).sum()}"
        )
        print(f"  saved = {output_path.resolve()}")
        print()


if __name__ == "__main__":
    main()
