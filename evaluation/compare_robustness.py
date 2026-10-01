"""
compare_robustness.py

Compare the existing XGBoost baseline on:
1. Clean ASVspoof 2019 LA test segments
2. Phone-like transformed versions of the same test source files
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from xgboost import XGBClassifier


MODEL_PATH = Path("D:/data/week2_controlled/models/xgboost_baseline.json")

CLEAN_X = Path("D:/data/week2_controlled/features/test/X.npy")
CLEAN_Y = Path("D:/data/week2_controlled/features/test/y.npy")

PHONE_X = Path("D:/data/week2_controlled/features/phone_like_test/X.npy")
PHONE_Y = Path("D:/data/week2_controlled/features/phone_like_test/y.npy")

OUTPUT_DIR = Path("experiments/robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr

    index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[index] + fnr[index]) / 2.0

    return float(eer)


def evaluate(model, X, y, condition):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    cm = confusion_matrix(y, predictions)

    metrics = {
        "condition": condition,
        "samples": len(y),
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "eer": calculate_eer(y, probabilities),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    np.save(
        OUTPUT_DIR / f"{condition}_probabilities.npy",
        probabilities,
    )

    np.save(
        OUTPUT_DIR / f"{condition}_predictions.npy",
        predictions,
    )

    return metrics


def main():
    print("=" * 50)
    print("XGBOOST ROBUSTNESS COMPARISON")
    print("=" * 50)

    print("\nLoading model...")
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    print("Model loaded.")

    print("\nLoading clean test set...")
    clean_X = np.load(CLEAN_X)
    clean_y = np.load(CLEAN_Y)

    print(f"Clean X shape: {clean_X.shape}")
    print(f"Clean y shape: {clean_y.shape}")

    print("\nLoading phone-like test set...")
    phone_X = np.load(PHONE_X)
    phone_y = np.load(PHONE_Y)

    print(f"Phone X shape: {phone_X.shape}")
    print(f"Phone y shape: {phone_y.shape}")

    clean_result = evaluate(
        model,
        clean_X,
        clean_y,
        "clean",
    )

    phone_result = evaluate(
        model,
        phone_X,
        phone_y,
        "phone_like",
    )

    results = pd.DataFrame([
        clean_result,
        phone_result,
    ])

    results.to_csv(
        OUTPUT_DIR / "robustness_results.csv",
        index=False,
    )

    print("\n")
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)

    for result in [clean_result, phone_result]:
        print(f"\nCondition: {result['condition']}")
        print(f"Samples    : {result['samples']}")
        print(f"Accuracy   : {result['accuracy']:.4f}")
        print(f"Precision  : {result['precision']:.4f}")
        print(f"Recall     : {result['recall']:.4f}")
        print(f"F1         : {result['f1']:.4f}")
        print(f"ROC-AUC    : {result['roc_auc']:.4f}")
        print(f"EER        : {result['eer']:.4f}")
        print(
            f"Confusion  : [[{result['tn']}, {result['fp']}], "
            f"[{result['fn']}, {result['tp']}]]"
        )

    print("\n")
    print("=" * 50)
    print("ROBUSTNESS EXPERIMENT COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {OUTPUT_DIR / 'robustness_results.csv'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
