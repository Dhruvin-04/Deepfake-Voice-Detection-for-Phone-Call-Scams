# Week 1 Completion Checklist

## Member 1 — Dataset Lead

- [x] Official ASVspoof 2021 DF Part00 obtained
- [x] Folder structure understood
- [x] Label/metadata structure understood
- [x] Audio inspection script implemented
- [x] 20 audio files inspected
- [x] Part00 Real/Fake statistics calculated
- [x] `dataset_notes.md` prepared

## Member 2 — Audio Processing Lead

- [x] Audio loading
- [x] Mono conversion
- [x] Resampling
- [x] Normalization
- [x] Real/fake preprocessing demonstration artifacts
- [x] Waveform visualization
- [x] Spectrogram visualization
- [x] Denoising separated as optional experiment

## Member 3 — Model/ML Lead

- [x] MFCC explanation
- [x] Initial MFCC configuration
- [x] `extract_mfcc.py`
- [x] MFCC visualization
- [x] Fixed-length feature aggregation
- [x] XGBoost baseline design

## Member 4 — Evaluation / PM Lead

- [x] Evaluation metric definitions
- [x] Experiment roadmap
- [x] Experiment tracking template
- [x] Central repository structure
- [x] Project README

## Integration / Methodology

- [x] One central GitHub repository identified as the source of truth
- [x] Large/raw audio excluded from GitHub
- [x] Python generated files excluded from Git
- [x] Standard preprocessing baseline excludes automatic denoising
- [x] Evaluation-set leakage avoided in the documented train/test strategy

## Week 1 boundary

Week 1 establishes the foundation. Full XGBoost training, final performance results, CNN training, robustness experiments, cross-dataset evaluation, and application deployment are later stages.
