"""
load_audio.py
--------------
Step 1 of the preprocessing pipeline: Load

Responsible ONLY for reading an audio file off disk and handing back
the raw waveform + its native sampling rate. No conversion, no
resampling, no normalization happens here -- that's the next scripts'
job. Keeping this step "dumb" makes it easy to swap the backend
(soundfile vs librosa vs torchaudio) later without touching anything
downstream.
"""

import soundfile as sf
import numpy as np


def load_audio(file_path: str):
    """
    Load an audio file and return the raw signal exactly as stored.

    Parameters
    ----------
    file_path : str
        Path to a .wav/.flac/.ogg file (anything libsndfile supports).

    Returns
    -------
    signal : np.ndarray
        Shape (num_samples,) for mono or (num_samples, num_channels)
        for multi-channel audio. dtype float32, values roughly in
        [-1.0, 1.0].
    sample_rate : int
        The native sampling rate of the file (NOT resampled).
    """
    signal, sample_rate = sf.read(file_path, dtype="float32", always_2d=False)
    return signal, sample_rate


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/data/sample/real_001.wav"
    signal, sr = load_audio(path)

    print(f"File: {path}")
    print(f"Sample rate: {sr} Hz")
    print(f"Shape: {signal.shape}")
    print(f"Channels: {1 if signal.ndim == 1 else signal.shape[1]}")
    print(f"Duration: {len(signal) / sr:.2f} sec")
    print(f"dtype: {signal.dtype}")
    print(f"Min/Max amplitude: {signal.min():.3f} / {signal.max():.3f}")
