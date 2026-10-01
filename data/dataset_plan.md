# Week 2 Dataset Plan

## Purpose

This document defines the dataset sources, split strategy, preprocessing pipeline, and evaluation strategy for the current MFCC + XGBoost development experiments.

The current development stage uses ASVspoof 2019 Logical Access (LA) for training, validation, and testing. ASVspoof 2021 DF Part00 is kept separate for future held-out / cross-dataset evaluation.

---

## Dataset Sources

### Training

Dataset:

ASVspoof 2019 Logical Access (LA) Training

Location:

`ASVspoof2019_LA_train/flac/`

Audio files:

25,380

Labels:

- Bonafide = REAL
- Spoof = FAKE

Protocol:

`ASVspoof2019.LA.cm.train.trn.txt`

Verification:

25,380 audio files matched with 25,380 protocol entries.

No unmatched audio files.

No protocol entries without corresponding audio.

---

### Validation

Dataset:

ASVspoof 2019 Logical Access (LA) Development

Location:

`ASVspoof2019_LA_dev/flac/`

Labeled audio:

24,844

Labels:

- Bonafide = REAL
- Spoof = FAKE

Protocol:

`ASVspoof2019.LA.cm.dev.trl.txt`

Verification:

24,844 protocol entries correspond to audio files.

There are 142 additional audio files in the Dev folder that do not appear in the CM protocol. These files are excluded from the labeled ML dataset.

---

### Test

Dataset:

ASVspoof 2019 Logical Access (LA) Evaluation

Location:

`ASVspoof2019_LA_eval/flac/`

Labeled audio:

71,237

Labels:

- Bonafide = REAL
- Spoof = FAKE

Protocol:

`ASVspoof2019.LA.cm.eval.trl.txt`

Verification:

71,237 protocol entries correspond to audio files.

There are 696 additional audio files in the Eval folder that do not appear in the CM protocol. These files are excluded from the labeled ML dataset.

The current baseline experiment uses a controlled subset of this evaluation split for testing.

---

### Held-out / Cross-Dataset Evaluation

Dataset:

ASVspoof 2021 DF Evaluation Part00

Location:

`ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac/`

Audio files:

152,955

Labels:

- Bonafide = REAL
- Spoof = FAKE

This dataset is kept separate from the ASVspoof 2019 LA training, validation, and test data.

It is not used to train the current XGBoost baseline.

It is reserved for future held-out / cross-dataset evaluation.

---

## Label Mapping

```text
bonafide -> REAL -> 0

spoof    -> FAKE -> 1
```

The label mapping is derived from the dataset protocol/metadata information.

---

## Current Controlled Subset

For efficient development and testing, a controlled subset is used before scaling to the complete dataset.

### Training Subset

```text
500 REAL
500 FAKE

Total: 1,000 source files
```

### Validation Subset

```text
100 REAL
100 FAKE

Total: 200 source files
```

### Test Subset

```text
100 REAL
100 FAKE

Total: 200 source files
```

The controlled test subset produces 313 three-second segments.

The resulting test segments contain:

```text
REAL: 159
FAKE: 154
```

The training, validation, and test source files are kept separate.

---

## Current Experiment Strategy

The current baseline uses the following structure:

```text
ASVspoof 2019 LA Train
        |
        v
Training
        |
        v
MFCC + XGBoost


ASVspoof 2019 LA Dev
        |
        v
Validation


ASVspoof 2019 LA Eval
        |
        v
Test


ASVspoof 2021 DF Part00
        |
        v
Future held-out / cross-dataset evaluation
```

The 2021 DF evaluation data is not mixed with the current training, validation, or test sets.

---

## Audio Processing

The standard preprocessing pipeline is:

```text
Load
  ->
Mono
  ->
Resample to 16 kHz
  ->
Normalize
  ->
3-second segmentation
```

Audio shorter than 3 seconds is zero-padded.

Audio longer than 3 seconds is split into non-overlapping 3-second segments.

A final partial segment is zero-padded to the target duration.

Denoising support exists in the preprocessing module but is disabled for the current baseline experiment.

---

## Feature Extraction

The current feature representation is:

```text
3-second audio segment
        ->
40 MFCC coefficients
        ->
Mean + Standard Deviation
        ->
80-dimensional feature vector
```

Current configuration:

```text
Sampling rate: 16 kHz
MFCC coefficients: 40
Target segment duration: 3 seconds
Final feature size: 80
```

These are initial experimental settings and are not assumed to be optimal.

---

## Model

The current baseline model is:

```text
MFCC
  ->
XGBoost
  ->
REAL / FAKE
```

Current XGBoost configuration:

```text
n_estimators      = 200
max_depth         = 4
learning_rate     = 0.05
subsample         = 0.9
colsample_bytree  = 0.9
objective         = binary:logistic
eval_metric       = logloss
random_state      = 42
```

This is an initial baseline model and is not the final model of the project.

---

## Evaluation Metrics

The current evaluation reports:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Equal Error Rate (EER)
* Confusion Matrix

The current baseline is evaluated at the segment level.

---

## Phone-like Robustness Experiment

An initial robustness experiment has been implemented using a simulated telephone-style audio condition.

The transformation currently uses:

```text
Resampling to 8 kHz
        +
Approximately 300-3400 Hz bandwidth limitation
        +
Resampling back to 16 kHz
```

The same trained XGBoost model is evaluated on the transformed test data without retraining.

The phone-like test set is created from the same controlled test source files used for the clean baseline.

Current phone-like test set:

```text
Source files: 200
Segments: 313
REAL segments: 159
FAKE segments: 154
```

This experiment is intended as an initial robustness study. More realistic channel conditions may be investigated later.

---

## Current Baseline Results

The current controlled ASVspoof 2019 LA baseline produced:

```text
Accuracy  : 76.36%
Precision : 82.26%
Recall    : 66.23%
F1-score  : 73.38%
ROC-AUC   : 86.25%
EER       : 21.72%
```

Confusion matrix:

```text
[[137,  22],
 [ 52, 102]]
```

These are preliminary development results and may change in later experiments.

---

## Phone-like Robustness Results

The same trained XGBoost model produced the following results:

```text
Condition: Clean

Accuracy  : 76.36%
Precision : 82.26%
Recall    : 66.23%
F1-score  : 73.38%
ROC-AUC   : 86.25%
EER       : 21.72%


Condition: Phone-like

Accuracy  : 48.88%
Precision : 49.04%
Recall    : 99.35%
F1-score  : 65.67%
ROC-AUC   : 61.05%
EER       : 40.26%
```

The initial robustness experiment indicates that the current baseline is affected by the simulated phone-call-like condition.

---

## Methodology Notes

The following principles are used in the current development process:

1. Official ASVspoof 2019 LA train, development, and evaluation splits are kept separate.
2. Controlled subsets are selected from the official splits for efficient development.
3. The ASVspoof 2021 DF evaluation data is kept separate from current training and testing.
4. The same source files are not reused across the controlled training, validation, and test subsets.
5. The current 3-second segmentation duration is an initial experimental setting.
6. The current MFCC + XGBoost system is treated as a baseline rather than the final detector.
7. Dataset construction and experiment decisions are documented before scaling to larger experiments.
