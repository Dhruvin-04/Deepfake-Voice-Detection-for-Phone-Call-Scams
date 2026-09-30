from pathlib import Path
import soundfile as sf
import csv

# Path to the extracted ASVspoof 2021 DF audio folder
AUDIO_DIR = Path(
    r"D:\data\ASVspoof2021_DF"
    r"\ASVspoof2021_DF_eval_part00"
    r"\ASVspoof2021_DF_eval"
    r"\flac"
)

# Number of files to inspect
NUM_FILES = 20

# Output CSV file
OUTPUT_CSV = Path(r"D:\data\ASVspoof2021_DF\audio_inspection.csv")


def inspect_audio(file_path):
    """Return basic information about one audio file."""
    info = sf.info(file_path)

    return {
        "filename": file_path.name,
        "duration_seconds": round(info.duration, 3),
        "sampling_rate_hz": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
    }


def main():
    if not AUDIO_DIR.exists():
        print(f"ERROR: Audio directory not found:\n{AUDIO_DIR}")
        return

    audio_files = sorted(AUDIO_DIR.glob("*.flac"))[:NUM_FILES]

    if not audio_files:
        print("ERROR: No FLAC files found.")
        return

    results = []

    print("=" * 75)
    print("ASVspoof 2021 DF - Audio Inspection")
    print("=" * 75)

    for file_path in audio_files:
        try:
            data = inspect_audio(file_path)
            results.append(data)

            print(f"Filename       : {data['filename']}")
            print(f"Duration       : {data['duration_seconds']} sec")
            print(f"Sampling Rate  : {data['sampling_rate_hz']} Hz")
            print(f"Channels       : {data['channels']}")
            print(f"Format         : {data['format']}")
            print(f"Subtype        : {data['subtype']}")
            print("-" * 75)

        except Exception as e:
            print(f"ERROR reading {file_path.name}: {e}")

    # Save results to CSV
    if results:
        fieldnames = [
            "filename",
            "duration_seconds",
            "sampling_rate_hz",
            "channels",
            "format",
            "subtype",
        ]

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nInspection complete.")
        print(f"Results saved to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()