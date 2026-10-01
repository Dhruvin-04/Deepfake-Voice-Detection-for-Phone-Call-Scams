"""
build_segment_manifest.py

Create a manifest linking every processed 3-second segment
back to its original ASVspoof 2019 LA metadata.

Output:
    data/metadata/segment_metadata.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


METADATA_PATH = Path("data/metadata/dataset_metadata.csv")
PROCESSED_ROOT = Path("D:/data/week2_controlled/processed")
OUTPUT_PATH = Path("data/metadata/segment_metadata.csv")


def main() -> None:
    metadata = pd.read_csv(METADATA_PATH)

    lookup = {
        row.filename: row
        for row in metadata.itertuples(index=False)
    }

    rows = []

    for split in ("train", "validation", "test"):
        split_root = PROCESSED_ROOT / split

        for segment_path in sorted(split_root.rglob("*.wav")):
            stem = segment_path.stem

            marker = "_segment_"
            if marker not in stem:
                raise ValueError(
                    f"Unexpected segment filename: {segment_path.name}"
                )

            source_stem = stem.rsplit(marker, 1)[0]
            source_filename = source_stem + ".flac"

            if source_filename not in lookup:
                raise ValueError(
                    f"Original source not found in metadata: "
                    f"{source_filename}"
                )

            source = lookup[source_filename]

            if source.split != split:
                raise ValueError(
                    f"Split mismatch for {segment_path}: "
                    f"metadata={source.split}, folder={split}"
                )

            expected_class = "REAL" if source.label == 0 else "FAKE"

            # Check folder label matches metadata label.
            if segment_path.parent.name != expected_class:
                raise ValueError(
                    f"Label mismatch for {segment_path}: "
                    f"folder={segment_path.parent.name}, "
                    f"metadata={expected_class}"
                )

            rows.append(
                {
                    "segment_path": str(segment_path),
                    "source_filename": source_filename,
                    "label": int(source.label),
                    "class_name": source.class_name,
                    "speaker_id": source.speaker_id,
                    "split": source.split,
                    "dataset": source.dataset,
                    "protocol": source.protocol,
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError("No processed segments found.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Total segments: {len(result)}")
    print()
    print("Segments by split/class:")
    print(
        result.groupby(["split", "class_name"])
        .size()
        .to_string()
    )
    print()
    print(
        "Unique source files:",
        result["source_filename"].nunique(),
    )
    print(
        "Unique speakers:",
        result["speaker_id"].nunique(),
    )
    print(
        "Saved:",
        OUTPUT_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
