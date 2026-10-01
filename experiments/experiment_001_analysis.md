# Experiment 001 Analysis

## Experiment

MFCC + XGBoost Baseline

## Objective

Establish the first machine-learning baseline for deepfake voice detection using 3-second standardized audio, MFCC features, and XGBoost.

## Data

ASVspoof 2019 LA controlled subset.
Test segments: 313.

## Features

40 MFCC coefficients with mean and standard deviation aggregation, producing 80 features per segment.

## Model

XGBoost binary classifier.

## Results

Accuracy: 0.7636
Precision: 0.8226
Recall: 0.6623
F1-score: 0.7338
ROC-AUC: 0.8625

## Confusion Matrix

```text
[[137, 22],
 [ 52,102]]
```

Interpretation:
- True Negative (REAL correctly predicted as REAL): 137
- False Positive (REAL predicted as FAKE): 22
- False Negative (FAKE predicted as REAL): 52
- True Positive (FAKE correctly predicted as FAKE): 102

## Important Note

These are results from the controlled ASVspoof 2019 LA baseline experiment. They are not the final ASVspoof 2021 DF evaluation results.

## Next Analysis

The baseline will be used as the reference point for later model and robustness experiments.
