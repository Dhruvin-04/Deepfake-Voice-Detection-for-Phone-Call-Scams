import csv
from pathlib import Path

from preprocessing.robustness.simulate_phone_call import simulate_phone_call


TEST_CSV = Path("data/splits/test.csv")
INPUT_DIR = Path(r"D:\data\ASVspoof2019\LA\ASVspoof2019_LA_eval\flac")
OUTPUT_DIR = Path("data/robustness/phone_like")
OUTPUT_CSV = Path("data/metadata/phone_test.csv")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []

    with TEST_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, start=1):
            input_path = INPUT_DIR / row["filename"]
            output_name = Path(row["filename"]).stem + "_phone.wav"
            output_path = OUTPUT_DIR / output_name

            if not input_path.exists():
                print(f"[WARNING] Missing input: {input_path}")
                continue

            if not output_path.exists():
                simulate_phone_call(str(input_path), str(output_path))

            rows_out.append({
                "filename": output_name,
                "original_filename": row["filename"],
                "label": row["label"],
                "class_name": row["class_name"],
                "speaker_id": row["speaker_id"],
                "split": row["split"],
                "dataset": row["dataset"],
                "condition": "phone_like"
            })

            if i % 25 == 0:
                print(f"Processed {i} files...")

    fieldnames = [
        "filename",
        "original_filename",
        "label",
        "class_name",
        "speaker_id",
        "split",
        "dataset",
        "condition"
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print()
    print("=" * 40)
    print("PHONE-LIKE TEST SET COMPLETE")
    print("=" * 40)
    print(f"Files processed : {len(rows_out)}")
    print(f"Audio directory : {OUTPUT_DIR}")
    print(f"Metadata        : {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()
