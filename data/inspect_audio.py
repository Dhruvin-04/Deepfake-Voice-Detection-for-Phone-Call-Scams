from pathlib import Path
import argparse
import soundfile as sf
import csv


def inspect_audio(file_path):
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
    parser = argparse.ArgumentParser(
        description="Inspect ASVspoof 2021 DF audio files."
    )

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the extracted ASVspoof2021_DF_eval_part00 folder",
    )

    parser.add_argument(
        "--num-files",
        type=int,
        default=20,
        help="Number of FLAC files to inspect (default: 20)",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    audio_dir = (
        dataset_dir
        / "ASVspoof2021_DF_eval"
        / "flac"
    )

    output_csv = Path(__file__).resolve().parent / "audio_inspection.csv"

    if not audio_dir.exists():
        print(f"ERROR: Audio directory not found:\n{audio_dir}")
        return

    audio_files = sorted(audio_dir.glob("*.flac"))[:args.num_files]

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

    fieldnames = [
        "filename",
        "duration_seconds",
        "sampling_rate_hz",
        "channels",
        "format",
        "subtype",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nInspection complete.")
    print(f"Results saved to:\n{output_csv}")


if __name__ == "__main__":
    main()
