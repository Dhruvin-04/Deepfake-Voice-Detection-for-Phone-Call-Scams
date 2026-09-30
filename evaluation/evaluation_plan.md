# Evaluation Plan

## Purpose

This document defines how the deepfake voice detection system will be evaluated once model training begins. Week 1 defines the metrics and research roadmap; it does not report final model performance.

## Primary Metrics

### Accuracy

The proportion of predictions that are correct overall. It is useful as a general summary, but class imbalance means it should not be used alone.

### Precision

Among files predicted as fake, the proportion that are actually fake.

### Recall

Among truly fake files, the proportion correctly detected as fake.

### F1-score

The harmonic mean of precision and recall. It summarizes the trade-off between the two.

### ROC-AUC

Measures how well the model separates the two classes across decision thresholds using prediction scores.

### Equal Error Rate (EER)

The operating point where the false acceptance/error rate and false rejection/error rate are equal. EER is an important anti-spoofing metric and will be reported for the appropriate scored evaluation.

### Confusion Matrix

Shows the counts of:

- True Positive: FAKE predicted as FAKE
- True Negative: REAL predicted as REAL
- False Positive: REAL predicted as FAKE
- False Negative: FAKE predicted as REAL

## Important Error Types

**False Positive:** a genuine/real recording is incorrectly flagged as fake.

**False Negative:** a fake/spoof recording is incorrectly accepted as real.

Both error types will be tracked because they have different practical consequences.

## Evaluation Data Strategy

The ASVspoof 2021 DF evaluation set should remain held out from model training and tuning. We will not randomly split the evaluation set into training and final test data.

The training and validation sources will be finalized in Week 2.

## Research Experiment Roadmap

1. Clean audio baseline
2. Background noise
3. Audio compression
4. Telephone-like degradation
5. Short audio clips
6. Unseen speakers
7. Cross-dataset testing

## Week 1 Status

Metrics defined ✅

Experiment roadmap defined ✅

Metric calculation will begin after the first trained baseline is available.
