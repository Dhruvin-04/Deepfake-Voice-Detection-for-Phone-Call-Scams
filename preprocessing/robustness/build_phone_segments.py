import csv
from pathlib import Path

from preprocessing.segment_audio import process_file


INPUT_DIR = Path("data/robustness/phone_like")
INPUT_CSV = Path("data/metadata/phone_test.csv")

OUTPUT_DIR = Path("data/processed/phone_like_test")
OUTPUT_CSV = Path("data/metadata/phone_segment_metadata.csv")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    output_rows = []

    with INPUT_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, start=1):
            input_path = INPUT_DIR / row["filename"]

            if not input_path.exists():
                print(f"[WARNING] Missing phone-like file: {input_path}")
                continue

            segment_paths = process_file(
                input_file=input_path,
                output_dir=OUTPUT_DIR,
            )

            for segment_path in segment_paths:
                output_rows.append({
                    "filename": segment_path.name,
                    "original_filename": row["original_filename"],
                    "phone_filename": row["filename"],
                    "label": row["label"],
                    "class_name": row["class_name"],
                    "speaker_id": row["speaker_id"],
                    "split": row["split"],
                    "dataset": row["dataset"],
                    "condition": "phone_like",
                })

            if i % 25 == 0:
                print(f"Segmented {i} files...")

    fieldnames = [
        "filename",
        "original_filename",
        "phone_filename",
        "label",
        "class_name",
        "speaker_id",
        "split",
        "dataset",
        "condition",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print()
    print("=" * 40)
    print("PHONE-LIKE SEGMENTATION COMPLETE")
    print("=" * 40)
    print(f"Source files processed : {len(set(r['phone_filename'] for r in output_rows))}")
    print(f"Segments created       : {len(output_rows)}")
    print(f"Segment directory      : {OUTPUT_DIR}")
    print(f"Metadata               : {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()
