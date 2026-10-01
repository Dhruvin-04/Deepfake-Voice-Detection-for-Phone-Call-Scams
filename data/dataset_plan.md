# Week 2 Dataset Plan

## Purpose

This document defines the dataset sources and evaluation strategy for the first MFCC + XGBoost experiment.

## Dataset Sources

### Training

Dataset:
ASVspoof 2019 Logical Access (LA) Training

Location:
ASVspoof2019_LA_train/flac/

Audio files:
25,380

Labels:
- Bonafide = REAL
- Spoof = FAKE

Protocol:
ASVspoof2019.LA.cm.train.trn.txt

Verification:
25,380 audio files matched with 25,380 protocol entries.
No unmatched audio files.
No protocol entries without corresponding audio.

### Validation

Dataset:
ASVspoof 2019 Logical Access (LA) Development

Location:
ASVspoof2019_LA_dev/flac/

Labeled audio:
24,844

Labels:
- Bonafide = REAL
- Spoof = FAKE

Protocol:
ASVspoof2019.LA.cm.dev.trl.txt

Verification:
24,844 protocol entries correspond to audio files.
There are 142 additional audio files in the Dev folder that do not appear in the CM protocol. These files will not be included in the labeled ML dataset.

### Held-out Evaluation

Dataset:
ASVspoof 2021 DF Evaluation Part00

Location:
ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac/

Audio files:
152,955

Labels:
- Bonafide = REAL
- Spoof = FAKE

This dataset will remain separate from training and validation.

It will not be randomly split into training and test subsets.

## Label Mapping

bonafide -> REAL -> 0

spoof -> FAKE -> 1

The label mapping is derived from the CM protocol files.

## Initial Experiment Strategy

For the first development experiment:

2019 LA Train
    ->
training

2019 LA Dev
    ->
validation

2021 DF Part00
    ->
held-out evaluation

The exact number of files processed initially will be limited to a controlled subset so that the complete pipeline can be tested efficiently before scaling.

## Audio Processing

The standard preprocessing pipeline is:

Load
-> Mono
-> Resample to 16 kHz
-> Normalize
-> 3-second segmentation

## Feature Extraction

Initial feature representation:

MFCC
-> Mean
-> Standard deviation
-> Fixed-length feature vector

Initial configuration:

Sampling rate: 16 kHz
MFCC coefficients: 40
Target segment duration: 3 seconds

These are initial experimental settings and are not assumed to be optimal.

## Model

First baseline:

MFCC
-> XGBoost
-> REAL / FAKE

## Evaluation Metrics

The first model evaluation will report:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Methodology Notes

The ASVspoof 2021 DF evaluation data is treated as held-out evaluation data.

No training samples will be taken from the 2021 DF evaluation set.

The 142 unlabeled ASVspoof 2019 LA Dev audio files will be excluded from the labeled ML dataset.

All dataset construction decisions will be recorded before large-scale processing.