"""
predict_xgboost.py

Live inference demo for the MFCC + XGBoost baseline.

Pipeline:
    Input audio
        ->
    Standard preprocessing
        ->
    3-second segmentation
        ->
    MFCC
        ->
    Mean + Standard Deviation
        ->
    80 features
        ->
    XGBoost
        ->
    REAL / FAKE prediction

This is a demonstration/inference script for the baseline model.
It is not the final application or final research evaluation protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from preprocessing.preprocess_audio import preprocess, TARGET_SAMPLE_RATE
from preprocessing.segment_audio import segment_signal
from features.extract_mfcc import extract_mfcc, SAMPLE_RATE, N_MFCC
from features.aggregate_mfcc import aggregate_mfcc


REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    REPO_ROOT
    / "models"
    / "xgboost"
    / "xgboost_baseline.json"
)


def load_model() -> XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def predict_audio(
    audio_path: str | Path,
    model: XGBClassifier,
) -> None:

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print("\n========================================")
    print("Deepfake Voice Detection Demo")
    print("========================================")
    print(f"Input: {audio_path.resolve()}")
    print(f"Model: {MODEL_PATH}")

    # Standard preprocessing
    signal = preprocess(
        str(audio_path),
        target_sr=TARGET_SAMPLE_RATE,
        denoise=False,
    )

    # 3-second segmentation
    segments = segment_signal(signal)

    if not segments:
        raise ValueError("No audio segments were generated.")

    print(f"Segments: {len(segments)}")

    probabilities = []

    for index, segment in enumerate(segments, start=1):

        # Save the segment temporarily in memory is not directly supported
        # by extract_mfcc(), so MFCC is computed directly from the array.
        import librosa

        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
        )

        features = aggregate_mfcc(mfcc)
        X = features.reshape(1, -1)

        fake_probability = float(
            model.predict_proba(X)[0, 1]
        )

        prediction = (
            "FAKE"
            if fake_probability >= 0.5
            else "REAL"
        )

        probabilities.append(fake_probability)

        print(
            f"\nSegment {index}: "
            f"Prediction={prediction} | "
            f"FAKE probability={fake_probability:.2%}"
        )

    # Simple demo-level aggregation:
    # average fake probability across all segments.
    overall_fake_probability = float(
        np.mean(probabilities)
    )

    overall_prediction = (
        "FAKE"
        if overall_fake_probability >= 0.5
        else "REAL"
    )

    print("\n----------------------------------------")
    print("OVERALL DEMO RESULT")
    print("----------------------------------------")
    print(f"Prediction : {overall_prediction}")
    print(
        f"FAKE probability : "
        f"{overall_fake_probability:.2%}"
    )
    print(
        f"REAL probability : "
        f"{1.0 - overall_fake_probability:.2%}"
    )
    print("----------------------------------------")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MFCC + XGBoost deepfake voice demo."
    )

    parser.add_argument(
        "audio_path",
        help="Path to WAV or FLAC audio file",
    )

    args = parser.parse_args()

    try:
        model = load_model()
        predict_audio(args.audio_path, model)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
