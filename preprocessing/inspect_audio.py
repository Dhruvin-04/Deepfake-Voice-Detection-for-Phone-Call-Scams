"""
inspect_audio.py
-----------------
Batch inspection tool. Walks a folder of audio files and prints/reports:
    filename, duration, sampling rate, channels, format

This overlaps a bit with Member 1's dataset-inspection script by design --
you both need this info, but Member 1 is inspecting the RAW dataset to
write dataset_notes.md, while this version exists inside the preprocessing
module so the audio pipeline can sanity-check its own inputs at any time
(e.g. "did resampling actually work?").
"""

import os
import glob
import soundfile as sf
import pandas as pd


def inspect_file(file_path: str) -> dict:
    """Return a metadata dict for a single audio file."""
    info = sf.info(file_path)
    return {
        "filename": os.path.basename(file_path),
        "duration_sec": round(info.frames / info.samplerate, 3),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
        "frames": info.frames,
    }


def inspect_folder(folder_path: str, pattern: str = "*.wav") -> pd.DataFrame:
    """Inspect every matching file in a folder and return a summary DataFrame."""
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    rows = [inspect_file(f) for f in files]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/data/sample"
    df = inspect_folder(folder)

    if df.empty:
        print(f"No .wav files found in {folder}")
    else:
        print(df.to_string(index=False))
        print("\n--- Summary ---")
        print(f"Files inspected: {len(df)}")
        print(f"Sample rates found: {sorted(df['sample_rate'].unique().tolist())}")
        print(f"Channel counts found: {sorted(df['channels'].unique().tolist())}")
        print(f"Duration range: {df['duration_sec'].min():.2f}s - {df['duration_sec'].max():.2f}s")
