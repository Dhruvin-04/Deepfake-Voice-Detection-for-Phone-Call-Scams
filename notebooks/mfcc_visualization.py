"""Generate MFCC visualizations for one or more standardized WAV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow the script to be executed from the repository root or directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features.extract_mfcc import extract_mfcc, SAMPLE_RATE, N_MFCC


def plot_mfcc(audio_path: str, output_path: str) -> None:
    mfcc = extract_mfcc(audio_path, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC)
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(mfcc, x_axis="time", sr=SAMPLE_RATE, ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title(f"MFCC - {Path(audio_path).stem}")
    ax.set_ylabel("MFCC Coefficients")
    ax.set_xlabel("Time")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"{audio_path}: MFCC shape = {mfcc.shape}")
    print(f"Saved: {Path(output_path).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--fake", required=True)
    parser.add_argument("--output-dir", default="notebooks")
    args = parser.parse_args()

    plot_mfcc(args.real, str(Path(args.output_dir) / "real_mfcc.png"))
    plot_mfcc(args.fake, str(Path(args.output_dir) / "fake_mfcc.png"))
