from pathlib import Path
from collections import Counter
import csv

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(r"D:\data\ASVspoof2021_DF")

AUDIO_DIR = (
    BASE_DIR
    / "ASVspoof2021_DF_eval_part00"
    / "ASVspoof2021_DF_eval"
    / "flac"
)

METADATA_FILE = (
    BASE_DIR
    / "DF-keys-full"
    / "keys"
    / "DF"
    / "CM"
    / "trial_metadata.txt"
)

OUTPUT_CSV = BASE_DIR / "dataset_statistics.csv"


# --------------------------------------------------
# Step 1: Get audio IDs from part00
# --------------------------------------------------

audio_ids = {
    file.stem
    for file in AUDIO_DIR.glob("*.flac")
}

print(f"Audio files found in part00: {len(audio_ids):,}")


# --------------------------------------------------
# Step 2: Read metadata and match audio IDs
# --------------------------------------------------

label_counts = Counter()

matched = 0
unmatched = 0

with METADATA_FILE.open("r", encoding="utf-8", errors="ignore") as f:

    for line in f:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # We need at least:
        # 2nd column = trial ID
        # 6th column = key
        if len(parts) < 6:
            continue

        trial_id = parts[1]
        label = parts[5].lower()

        # Only process audio that exists in part00
        if trial_id in audio_ids:

            matched += 1

            if label == "bonafide":
                label_counts["bonafide"] += 1

            elif label == "spoof":
                label_counts["spoof"] += 1


# --------------------------------------------------
# Step 3: Calculate results
# --------------------------------------------------

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


# --------------------------------------------------
# Step 4: Save statistics to CSV
# --------------------------------------------------

rows = [
    ["Statistic", "Value"],
    ["Part00 audio files", len(audio_ids)],
    ["Matched metadata", matched],
    ["Real / bonafide", real_count],
    ["Fake / spoof", fake_count],
    ["Total labeled", total_labeled],
]

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)
    writer.writerows(rows)


print("\nStatistics saved to:")
print(OUTPUT_CSV)