"""Run a simple before/after preprocessing demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .load_audio import load_audio
    from .preprocess_audio import preprocess, TARGET_SAMPLE_RATE
except ImportError:
    from load_audio import load_audio
    from preprocess_audio import preprocess, TARGET_SAMPLE_RATE


def describe(signal: np.ndarray, sr: int) -> dict:
    signal = np.asarray(signal)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    return {
        "sample_rate_hz": sr,
        "channels": 1 if signal.ndim == 1 else signal.shape[1],
        "duration_s": round(len(signal) / sr, 2),
        "peak": round(peak, 3),
    }


def run(files: list[str], out_dir: str, denoise: bool = False) -> None:
    rows = []
    for path in files:
        raw, raw_sr = load_audio(path)
        processed = preprocess(path, denoise=denoise)
        before = describe(raw, raw_sr)
        after = describe(processed, TARGET_SAMPLE_RATE)
        for key, label in [
            ("channels", "Channels"),
            ("sample_rate_hz", "Sample rate (Hz)"),
            ("peak", "Peak amplitude"),
            ("duration_s", "Duration (s)"),
        ]:
            rows.append({
                "file": Path(path).name,
                "property": label,
                "raw": before[key],
                "processed": after[key],
            })

    df = pd.DataFrame(rows)
    print("\nRAW vs PROCESSED" + (" (with denoise)" if denoise else ""))
    print(df.to_string(index=False))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "preprocessing_comparison.csv", index=False)

    lines = [
        "| file | property | raw | processed |",
        "|---|---|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(f"| {row.file} | {row.property} | {row.raw} | {row.processed} |")
    (out / "preprocessing_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved results -> {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="One or more audio files")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    run(args.files, args.out_dir, args.denoise)
