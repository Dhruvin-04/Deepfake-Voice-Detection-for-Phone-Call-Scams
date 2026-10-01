import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly


TARGET_SR = 16000
PHONE_SR = 8000

def simulate_phone_call(input_path: str, output_path: str):
    audio, sr = sf.read(input_path, always_2d=False)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # Resample to 8 kHz, representing a narrow-band telephone channel.
    if sr != PHONE_SR:
        audio = resample_poly(audio, PHONE_SR, sr).astype(np.float32)

    # Telephone-style bandwidth limitation.
    # Keep approximately 300 Hz to 3400 Hz.
    low = 300 / (PHONE_SR / 2)
    high = 3400 / (PHONE_SR / 2)

    sos = butter(
        6,
        [low, high],
        btype="bandpass",
        output="sos"
    )

    audio = sosfilt(sos, audio).astype(np.float32)

    # Return to the model's expected sampling rate.
    audio = resample_poly(audio, TARGET_SR, PHONE_SR).astype(np.float32)

    # Prevent clipping.
    peak = np.max(np.abs(audio))
    if peak > 0.99:
        audio = audio / peak * 0.99

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(output_path, audio, TARGET_SR)

    return TARGET_SR, len(audio)


def main():
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python -m preprocessing.robustness.simulate_phone_call "
            "<input.wav> <output.wav>"
        )
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    sr, samples = simulate_phone_call(input_path, output_path)

    print("Phone-call simulation complete")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Sample rate: {sr} Hz")
    print(f"Samples: {samples}")


if __name__ == "__main__":
    main()
