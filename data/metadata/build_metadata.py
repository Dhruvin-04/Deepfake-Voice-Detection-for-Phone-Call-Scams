"""
build_metadata.py

Build a labeled metadata CSV from the ASVspoof 2019 LA CM protocol files.

Only files listed in the official CM protocol files are included.
Therefore, the 142 unlabeled Dev files and 696 unlabeled Eval files
are automatically excluded.
"""

from __future__ import annotations

import csv
from pathlib import Path


BASE_DIR = Path("D:/data/ASVspoof2019/LA")

PROTOCOLS = {
    "train": BASE_DIR
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.train.trn.txt",
    "validation": BASE_DIR
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.dev.trl.txt",
    "test": BASE_DIR
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.eval.trl.txt",
}

AUDIO_DIRS = {
    "train": BASE_DIR / "ASVspoof2019_LA_train" / "flac",
    "validation": BASE_DIR / "ASVspoof2019_LA_dev" / "flac",
    "test": BASE_DIR / "ASVspoof2019_LA_eval" / "flac",
}

OUTPUT = Path("data/metadata/dataset_metadata.csv")


def parse_protocol(protocol_path: Path, split: str) -> list[dict]:
    rows = []

    with protocol_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) < 5:
                raise ValueError(
                    f"Invalid protocol line {line_number} in {protocol_path}: {line}"
                )

            speaker_id = parts[0]
            filename = parts[1]
            label_text = parts[4].lower()

            if label_text == "bonafide":
                label = 0
                class_name = "REAL"
            elif label_text == "spoof":
                label = 1
                class_name = "FAKE"
            else:
                raise ValueError(
                    f"Unknown label '{parts[4]}' at line {line_number}"
                )

            rows.append(
                {
                    "filename": filename + ".flac",
                    "label": label,
                    "class_name": class_name,
                    "speaker_id": speaker_id,
                    "split": split,
                    "dataset": "ASVspoof2019_LA",
                    "protocol": protocol_path.name,
                }
            )

    return rows


def main() -> None:
    all_rows = []

    for split, protocol_path in PROTOCOLS.items():
        if not protocol_path.exists():
            raise FileNotFoundError(
                f"Protocol file not found: {protocol_path}"
            )

        rows = parse_protocol(protocol_path, split)
        all_rows.extend(rows)

        real_count = sum(row["label"] == 0 for row in rows)
        fake_count = sum(row["label"] == 1 for row in rows)

        print(f"{split.upper()}:")
        print(f"  total = {len(rows)}")
        print(f"  REAL  = {real_count}")
        print(f"  FAKE  = {fake_count}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename",
        "label",
        "class_name",
        "speaker_id",
        "split",
        "dataset",
        "protocol",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print(f"Total metadata rows: {len(all_rows)}")
    print(f"Saved: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
