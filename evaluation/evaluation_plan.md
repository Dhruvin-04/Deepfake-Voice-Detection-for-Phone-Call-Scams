# Evaluation Plan

## Purpose

This document defines the evaluation strategy for the current deepfake voice detection development stage.

The current implementation uses an MFCC + XGBoost baseline trained and evaluated using the ASVspoof 2019 Logical Access (LA) dataset.

Additional robustness and cross-dataset experiments are planned for later stages.

---

## Primary Metrics

### Accuracy

The proportion of predictions that are correct overall.

Accuracy is useful as a general summary, but it should not be considered the only metric because the classes may be imbalanced.

### Precision

Among files or segments predicted as FAKE, the proportion that are actually FAKE.

### Recall

Among truly FAKE files or segments, the proportion correctly detected as FAKE.

### F1-score

The harmonic mean of precision and recall.

### ROC-AUC

Measures how well the model separates REAL and FAKE classes across different decision thresholds using prediction scores.

### Equal Error Rate (EER)

The operating point at which false acceptance and false rejection rates are approximately equal.

EER is included because it is commonly used for evaluating anti-spoofing systems.

### Confusion Matrix

Shows the counts of:

- True Positive: FAKE predicted as FAKE
- True Negative: REAL predicted as REAL
- False Positive: REAL predicted as FAKE
- False Negative: FAKE predicted as REAL

---

## Important Error Types

**False Positive:** a genuine/REAL recording is incorrectly classified as FAKE.

**False Negative:** a fake/SPOOF recording is incorrectly classified as REAL.

Both error types are tracked because they have different practical consequences.

---

## Evaluation Data Strategy

The current baseline keeps the ASVspoof 2019 LA official splits separate:

```text
ASVspoof 2019 LA Train        ->  Training
ASVspoof 2019 LA Development  ->  Validation
ASVspoof 2019 LA Evaluation   ->  Test
```

A controlled subset is currently used for efficient development:

| Split      | REAL source files | FAKE source files |
|------------|-------------------|-------------------|
| Training   | 500               | 500               |
| Validation | 100               | 100               |
| Test       | 100               | 100               |

The controlled test subset produces 313 three-second segments.

ASVspoof 2021 DF Part00 is kept separate from the current training, validation, and test data. It is reserved for future held-out / cross-dataset evaluation.

---

## Current Evaluation Procedure

The current baseline evaluation pipeline is:

```text
Audio
  ->
Preprocessing
  ->
3-second Segmentation
  ->
MFCC Extraction
  ->
Mean + Standard Deviation
  ->
80-dimensional Features
  ->
XGBoost
  ->
Prediction Scores
  ->
Evaluation Metrics
```

The current baseline evaluation is performed at the segment level.

---

## Current Baseline Results

The current controlled ASVspoof 2019 LA test experiment contains 313 test segments.

| Metric    | Result |
|-----------|--------|
| Accuracy  | 76.36% |
| Precision | 82.26% |
| Recall    | 66.23% |
| F1-score  | 73.38% |
| ROC-AUC   | 86.25% |
| EER       | 21.72% |

Confusion matrix (rows = actual, columns = predicted; order: REAL, FAKE):

```text
[[137,  22],
 [ 52, 102]]
```

These are preliminary development results and may change as the project progresses.

---

## Phone-like Robustness Evaluation

An initial robustness experiment has been completed using a simulated telephone-style condition.

The transformation includes:

```text
Resampling to 8 kHz
        +
Approximately 300-3400 Hz bandwidth limitation
        +
Resampling back to 16 kHz
```

The same trained XGBoost model is evaluated without retraining.

Current results:

| Condition  | Accuracy | Precision | Recall | F1-score | ROC-AUC | EER    |
|------------|----------|-----------|--------|----------|---------|--------|
| Clean      | 76.36%   | 82.26%    | 66.23% | 73.38%   | 86.25%  | 21.72% |
| Phone-like | 48.88%   | 49.04%    | 99.35% | 65.67%   | 61.05%  | 40.26% |

The initial result shows that the current baseline is affected by the simulated phone-call-like condition.

---

## Research Experiment Roadmap

### Completed

1. Clean audio baseline
2. Telephone-like degradation

### Planned

3. Background noise
4. Audio compression
5. Short audio clips
6. Unseen speakers
7. Cross-dataset testing

The experiment order may change as the project develops.

---

## Current Evaluation Status

```text
Metrics defined                         ✅
Baseline model evaluated                ✅
Confusion matrix generated              ✅
ROC-AUC calculated                      ✅
EER calculated                          ✅
Phone-like robustness experiment        ✅
Background noise experiment             🔜
Compression experiment                  🔜
Short-audio experiment                  🔜
Unseen-speaker evaluation               🔜
ASVspoof 2021 DF cross-dataset test     🔜
```

---

## Evaluation Notes

- The current results are from a controlled development subset rather than the complete ASVspoof 2019 LA dataset.
- The current 3-second segment duration and MFCC representation are initial experimental choices.
- Future experiments will investigate whether alternative segmentation, features, models, and audio conditions improve robustness and generalization.
- The current baseline is not considered the final detector.
