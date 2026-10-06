from pathlib import Path
from collections import Counter
import argparse
import csv


def main():
    parser = argparse.ArgumentParser(
        description="Generate ASVspoof 2021 DF Part00 dataset statistics."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to ASVspoof2021_DF root directory."
    )
    parser.add_argument(
        "--output-csv",
        default="data/dataset_statistics.csv",
        help="Output CSV path."
    )

    args = parser.parse_args()

    base_dir = Path(args.dataset_root)
    audio_dir = (
        base_dir
        / "ASVspoof2021_DF_eval_part00"
        / "ASVspoof2021_DF_eval"
        / "flac"
    )

    metadata_file = (
        base_dir
        / "DF-keys-full"
        / "keys"
        / "DF"
        / "CM"
        / "trial_metadata.txt"
    )

    output_csv = Path(args.output_csv)

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    audio_ids = {
        file.stem
        for file in audio_dir.glob("*.flac")
    }

    print(f"Audio files found in part00: {len(audio_ids):,}")

    label_counts = Counter()
    matched = 0

    with metadata_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 6:
                continue

            trial_id = parts[1]
            label = parts[5].lower()

            if trial_id in audio_ids:
                matched += 1

                if label == "bonafide":
                    label_counts["bonafide"] += 1
                elif label == "spoof":
                    label_counts["spoof"] += 1

    real_count = label_counts["bonafide"]
    fake_count = label_counts["spoof"]
    total_labeled = real_count + fake_count

    print("\n==============================================")
    print("ASVspoof 2021 DF - Part00 Statistics")
    print("==============================================")

    print(f"Audio files in part00 : {len(audio_ids):,}")
    print(f"Matched metadata      : {matched:,}")
    print(f"Real / bonafide       : {real_count:,}")
    print(f"Fake / spoof          : {fake_count:,}")
    print(f"Total labeled         : {total_labeled:,}")

    if total_labeled > 0:
        print(
            f"Real percentage       : "
            f"{real_count / total_labeled * 100:.2f}%"
        )

        print(
            f"Fake percentage       : "
            f"{fake_count / total_labeled * 100:.2f}%"
        )

    print(f"Unmatched audio files : {len(audio_ids) - matched:,}")

    rows = [
        ["Statistic", "Value"],
        ["Part00 audio files", len(audio_ids)],
        ["Matched metadata", matched],
        ["Real / bonafide", real_count],
        ["Fake / spoof", fake_count],
        ["Total labeled", total_labeled],
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("\nStatistics saved to:")
    print(output_csv)


if __name__ == "__main__":
    main()
