"""
train_xgboost.py

Experiment 001: MFCC + XGBoost Baseline

Training data:
    <work-root>/features/train/

Validation data:
    <work-root>/features/validation/

Test data:
    <work-root>/features/test/

The test set is loaded only for final evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier


WORK_ROOT = Path(
    os.getenv("DEEPFAKE_WORK_ROOT", "artifacts")
)

FEATURE_ROOT = WORK_ROOT / "features"
MODEL_ROOT = WORK_ROOT / "models"
MODEL_PATH = MODEL_ROOT / "xgboost_baseline.json"


def load_split(split: str):
    split_dir = FEATURE_ROOT / split

    x_path = split_dir / "X.npy"
    y_path = split_dir / "y.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing features: {x_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Missing labels: {y_path}")

    X = np.load(x_path)
    y = np.load(y_path)

    if X.ndim != 2:
        raise ValueError(f"{split}: expected 2-D X, got {X.shape}")

    if y.ndim != 1:
        raise ValueError(f"{split}: expected 1-D y, got {y.shape}")

    if len(X) != len(y):
        raise ValueError(
            f"{split}: X/y mismatch: {len(X)} vs {len(y)}"
        )

    if X.shape[1] != 80:
        raise ValueError(
            f"{split}: expected 80 features, got {X.shape[1]}"
        )

    if not np.isfinite(X).all():
        raise ValueError(f"{split}: X contains NaN or infinite values")

    return X, y


def main() -> None:
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")

    print("Dataset:")
    print(f"  Train      : X={X_train.shape}, y={y_train.shape}")
    print(f"  Validation : X={X_val.shape}, y={y_val.shape}")
    print(f"  Test       : X={X_test.shape}, y={y_test.shape}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)

    print("\nTraining complete.")
    print(f"Model saved to: {MODEL_PATH}")

    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    val_probabilities = model.predict_proba(X_val)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    np.save(MODEL_ROOT / "validation_predictions.npy", val_predictions)
    np.save(MODEL_ROOT / "validation_probabilities.npy", val_probabilities)

    np.save(MODEL_ROOT / "test_predictions.npy", test_predictions)
    np.save(MODEL_ROOT / "test_probabilities.npy", test_probabilities)
    np.save(MODEL_ROOT / "test_labels.npy", y_test)

    print("\nPrediction files saved.")
    print("  Validation predictions")
    print("  Validation probabilities")
    print("  Test predictions")
    print("  Test probabilities")
    print("  Test labels")

    print("\nModel parameters:")
    print(model.get_params())


if __name__ == "__main__":
    main()
