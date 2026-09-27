from pathlib import Path
from collections import Counter
import argparse
import csv


def main():
    parser = argparse.ArgumentParser(
        description="Calculate ASVspoof 2021 DF Part00 label statistics."
    )

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the extracted ASVspoof2021_DF_eval_part00 folder",
    )

    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to trial_metadata.txt",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    metadata_file = Path(args.metadata)

    audio_dir = (
        dataset_dir
        / "ASVspoof2021_DF_eval"
        / "flac"
    )

    output_csv = Path(__file__).resolve().parent / "dataset_statistics.csv"

    if not audio_dir.exists():
        print(f"ERROR: Audio directory not found:\n{audio_dir}")
        return

    if not metadata_file.exists():
        print(f"ERROR: Metadata file not found:\n{metadata_file}")
        return

    audio_ids = {
        file.stem
        for file in audio_dir.glob("*.flac")
    }

    print(f"Audio files found in part00: {len(audio_ids):,}")

    label_counts = Counter()
    matched = 0
    matched_ids = set()

    with metadata_file.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

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
                matched_ids.add(trial_id)

                if label == "bonafide":
                    label_counts["bonafide"] += 1

                elif label == "spoof":
                    label_counts["spoof"] += 1

    real_count = label_counts["bonafide"]
    fake_count = label_counts["spoof"]
    total_labeled = real_count + fake_count
    unmatched = len(audio_ids - matched_ids)

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

    print(f"Unmatched audio files : {unmatched:,}")

    rows = [
        ["Statistic", "Value"],
        ["Part00 audio files", len(audio_ids)],
        ["Matched metadata", matched],
        ["Real / bonafide", real_count],
        ["Fake / spoof", fake_count],
        ["Total labeled", total_labeled],
        ["Real percentage", f"{real_count / total_labeled * 100:.2f}%"],
        ["Fake percentage", f"{fake_count / total_labeled * 100:.2f}%"],
        ["Unmatched audio files", unmatched],
    ]

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)
        writer.writerows(rows)

    print("\nStatistics saved to:")
    print(output_csv)


if __name__ == "__main__":
    main()
