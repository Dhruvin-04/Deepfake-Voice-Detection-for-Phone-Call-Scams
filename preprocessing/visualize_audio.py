"""Generate waveform and spectrogram visualizations for audio."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .load_audio import load_audio
    from .preprocess_audio import preprocess, TARGET_SAMPLE_RATE, to_mono
except ImportError:
    from load_audio import load_audio
    from preprocess_audio import preprocess, TARGET_SAMPLE_RATE, to_mono


def plot_single(file_path: str, out_dir: str = "plots") -> Path:
    signal, sr = load_audio(file_path)
    signal = to_mono(signal)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    librosa.display.waveshow(signal, sr=sr, ax=axes[0])
    axes[0].set_title(f"Waveform - {Path(file_path).name} ({sr} Hz)")
    axes[0].set_xlabel("Time (s)")

    D = librosa.amplitude_to_db(abs(librosa.stft(signal)), ref=max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=axes[1])
    axes[1].set_title("Spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    fig.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{Path(file_path).stem}_waveform_spectrogram.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path.resolve()


def plot_compare(file_path: str, out_dir: str = "plots") -> Path:
    raw_signal, orig_sr = load_audio(file_path)
    raw_mono = to_mono(raw_signal)
    processed_signal = preprocess(file_path, denoise=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    librosa.display.waveshow(raw_mono, sr=orig_sr, ax=axes[0, 0])
    axes[0, 0].set_title(f"Original waveform ({orig_sr} Hz)")
    librosa.display.waveshow(processed_signal, sr=TARGET_SAMPLE_RATE, ax=axes[0, 1])
    axes[0, 1].set_title(f"Standardized waveform ({TARGET_SAMPLE_RATE} Hz)")

    D_orig = librosa.amplitude_to_db(abs(librosa.stft(raw_mono)), ref=max)
    img1 = librosa.display.specshow(D_orig, sr=orig_sr, x_axis="time", y_axis="hz", ax=axes[1, 0])
    axes[1, 0].set_title("Original spectrogram")
    fig.colorbar(img1, ax=axes[1, 0], format="%+2.0f dB")

    D_proc = librosa.amplitude_to_db(abs(librosa.stft(processed_signal)), ref=max)
    img2 = librosa.display.specshow(D_proc, sr=TARGET_SAMPLE_RATE, x_axis="time", y_axis="hz", ax=axes[1, 1])
    axes[1, 1].set_title("Standardized spectrogram")
    fig.colorbar(img2, ax=axes[1, 1], format="%+2.0f dB")

    fig.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{Path(file_path).stem}_compare.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to audio file")
    parser.add_argument("--compare", action="store_true", help="Plot original vs standardized")
    parser.add_argument("--out-dir", default="plots")
    args = parser.parse_args()

    output = plot_compare(args.file, args.out_dir) if args.compare else plot_single(args.file, args.out_dir)
    print(f"Saved plot -> {output}")
