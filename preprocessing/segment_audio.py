"""
segment_audio.py
----------------
Week 2: Convert preprocessed audio into fixed-length 3-second segments.

Pipeline:
    Load -> Mono -> Resample -> Normalize -> 3-second segmentation

Rules:
    - Audio shorter than 3 seconds is zero-padded.
    - Audio longer than 3 seconds is split into non-overlapping chunks.
    - A final partial chunk is zero-padded to exactly 3 seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from .preprocess_audio import preprocess, TARGET_SAMPLE_RATE
except ImportError:
    from preprocess_audio import preprocess, TARGET_SAMPLE_RATE


TARGET_DURATION = 3.0
TARGET_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION)


def segment_signal(
    signal: np.ndarray,
    segment_samples: int = TARGET_SAMPLES,
) -> list[np.ndarray]:
    """Split a signal into fixed-length segments, zero-padding the last one."""
    signal = np.asarray(signal, dtype=np.float32).flatten()

    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive.")

    if signal.size == 0:
        return [np.zeros(segment_samples, dtype=np.float32)]

    segments = []

    for start in range(0, len(signal), segment_samples):
        chunk = signal[start:start + segment_samples]

        if len(chunk) < segment_samples:
            padded = np.zeros(segment_samples, dtype=np.float32)
            padded[:len(chunk)] = chunk
            chunk = padded

        segments.append(chunk.astype(np.float32))

    return segments


def save_segments(
    segments: list[np.ndarray],
    output_dir: str | Path,
    stem: str,
    sr: int = TARGET_SAMPLE_RATE,
) -> list[Path]:
    """Save all segments as WAV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []

    for index, segment in enumerate(segments, start=1):
        output_path = output_dir / f"{stem}_segment_{index:03d}.wav"
        sf.write(output_path, segment, sr)
        output_paths.append(output_path)

    return output_paths


def process_file(
    input_file: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Preprocess one file and save its fixed-length segments."""
    input_file = Path(input_file)

    processed = preprocess(
        str(input_file),
        target_sr=TARGET_SAMPLE_RATE,
        denoise=False,
    )

    segments = segment_signal(processed)

    return save_segments(
        segments,
        output_dir,
        input_file.stem,
        TARGET_SAMPLE_RATE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess an audio file and split it into 3-second segments."
    )
    parser.add_argument("file", help="Input audio file")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where segmented WAV files will be saved",
    )

    args = parser.parse_args()

    paths = process_file(args.file, args.output_dir)

    print(f"Input: {args.file}")
    print(f"Segments created: {len(paths)}")

    for path in paths:
        print(f"Saved: {path.resolve()}")

    print(
        f"Each segment: {TARGET_SAMPLES} samples "
        f"({TARGET_DURATION:.1f} seconds at {TARGET_SAMPLE_RATE} Hz)"
    )
