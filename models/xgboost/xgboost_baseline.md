# XGBoost Baseline — Design & Architecture

## Why not feed the MFCC matrix directly?

`librosa.feature.mfcc()` returns a matrix of shape `(n_mfcc, T)` — for example `40 × T`. This is a **matrix**, not a fixed-length vector, and `T` varies with audio duration.

XGBoost (a tree-based gradient boosting library) expects a **tabular feature matrix**: every sample must have the same, fixed number of input features. So the raw MFCC matrix cannot be passed to XGBoost as-is.

## Feature aggregation

```text
MFCC Matrix (40 × T)
        ↓
 Feature Aggregation
        ↓
 Fixed-Length Vector
        ↓
     XGBoost
```

Initial idea (not finalized):

```text
40 MFCC coefficients
   Mean → 40 values
   Std  → 40 values
   ----------------
   Total → 80 features
```

Possible future extension:

```text
Mean, Std, Min, Max  → 40 × 4 = 160 features
```

**This aggregation strategy is not finalized for Week 1.** Documented statement to use:

> The MFCC matrix will be converted into a fixed-dimensional feature representation through statistical aggregation before being supplied to XGBoost. The exact aggregation strategy will be finalized during model-development experiments.

## Baseline architecture

```text
Audio
   ↓
Preprocessing
   ↓
MFCC Extraction
   ↓
MFCC Matrix
   ↓
Feature Aggregation
   ↓
Fixed-Length Feature Vector
   ↓
XGBoost Classifier
   ↓
REAL / FAKE
```

## Purpose of this baseline

> The XGBoost model will serve as the initial traditional machine-learning baseline for the project. Its purpose is to establish a reference performance level using MFCC-based acoustic features before evaluating more complex deep-learning architectures.

We are **not** claiming XGBoost is the final model — the goal is only to establish a first reference point.

## Not in scope for Week 1

- Building the CNN / CNN-LSTM models
- Training XGBoost on the full or a small random dataset
- Reporting accuracy numbers

Week 1 milestone:

> **Audio → MFCC → fixed feature representation → ready for XGBoost.**
