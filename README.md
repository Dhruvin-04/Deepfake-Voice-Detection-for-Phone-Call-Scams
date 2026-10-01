# Deepfake Voice Detection for Phone Call Scams

## 📌 Project Overview

**Deepfake Voice Detection for Phone Call Scams** is an AI-based cybersecurity project focused on studying whether a voice is **genuine human speech or AI-generated/manipulated speech**.

With recent advances in voice cloning and speech synthesis, attackers can generate convincing copies of a person's voice and use them for impersonation and emergency scams. This project aims to develop an audio-based detection system that analyzes speech characteristics and identifies potentially synthetic or spoofed voices.

> **Project Status: In Development**

The current development milestone focuses on implementing and evaluating an initial **MFCC + XGBoost baseline detector**, together with an initial robustness experiment using simulated phone-call-like audio conditions. More advanced models and application-level integration are planned for later stages.

---

## 🎯 Problem Statement

AI-based voice cloning and speech synthesis technologies can generate highly realistic human-like speech. This creates a security risk when cloned voices are used in phone scams, such as impersonating family members, friends, or other trusted individuals.

Traditional spam and caller-identification systems primarily analyze the **phone number or caller metadata** and do not determine whether the actual voice is authentic.

Therefore, there is a need to study AI-based methods capable of analyzing voice audio and distinguishing between **genuine speech and potentially synthetic or spoofed speech**.

---

## 🎯 Objectives

### Current Objectives

The current development stage focuses on:

- Preparing and verifying suitable speech datasets.
- Building a standardized audio preprocessing pipeline.
- Segmenting speech into fixed-length audio clips.
- Extracting MFCC-based acoustic features.
- Training an initial XGBoost classifier.
- Evaluating the baseline using standard classification metrics.
- Studying the effect of a simulated phone-call-like condition.
- Demonstrating prediction on individual audio files.

### Future Objectives

Later stages of the project may investigate:

- Stronger deep-learning based architectures.
- Larger-scale experiments.
- Additional robustness conditions.
- Cross-dataset evaluation.
- Prototype application and future real-time integration.

---

## 🔄 Current Implemented Pipeline

The currently implemented baseline pipeline is:

```text
Audio Input
    |
    v
Mono Conversion
    |
    v
Resampling to 16 kHz
    |
    v
Normalization
    |
    v
3-second Segmentation
    |
    v
MFCC Extraction
    |
    v
40 MFCC Coefficients
    |
    v
Mean + Standard Deviation
    |
    v
80-dimensional Feature Vector
    |
    v
XGBoost Classifier
    |
    v
REAL / FAKE Prediction
```
Pipeline Stages

1. Audio Input

The system receives an audio file for analysis. Live phone-call integration is planned for a later stage.

2. Audio Preprocessing

The current preprocessing stage converts audio to mono, resamples it to 16 kHz, normalizes the signal, and prepares it for segmentation.

3. Audio Segmentation

Speech is divided into non-overlapping 3-second segments.

Audio shorter than 3 seconds is zero-padded. A final partial segment is also zero-padded.

The 3-second duration is an initial experimental setting and is not considered an optimal final value.

4. Feature Extraction

The current baseline extracts 40 MFCC coefficients from each 3-second segment and represents each segment using the mean and standard deviation of the MFCC features.

This produces an 80-dimensional feature vector.

5. Model

The current baseline classifier is XGBoost.

6. Prediction

The current inference pipeline produces:

REAL

or

FAKE

along with a probability score.

📊 Dataset
ASVspoof 2019 Logical Access

The current MFCC + XGBoost baseline uses the ASVspoof 2019 Logical Access (LA) dataset.

The official dataset splits are kept separate:

ASVspoof 2019 LA Train
        |
        v
Training

ASVspoof 2019 LA Development
        |
        v
Validation

ASVspoof 2019 LA Evaluation
        |
        v
Test
Label Mapping
bonafide -> REAL -> 0
spoof    -> FAKE -> 1
Controlled Development Subset

A controlled subset is currently used for development:

Training   : 500 REAL + 500 FAKE source files
Validation : 100 REAL + 100 FAKE source files
Test       : 100 REAL + 100 FAKE source files

The controlled test subset produces:

313 three-second test segments
159 REAL segments
154 FAKE segments

The same source files are not shared between the training, validation, and test subsets.

ASVspoof 2021 DF

ASVspoof 2021 DF Part00 has also been prepared as a separate dataset for later evaluation.

It is kept separate from the current ASVspoof 2019 LA baseline experiment.

The 2021 DF dataset is planned for future held-out / cross-dataset evaluation and is not used to train the current baseline model.

🎧 Audio Preprocessing

The current preprocessing pipeline is:

Load
  |
  v
Mono Conversion
  |
  v
Resample to 16 kHz
  |
  v
Normalize
  |
  v
3-second Segmentation

Audio shorter than 3 seconds is zero-padded.

Longer audio is divided into non-overlapping 3-second segments, with the final partial segment zero-padded when necessary.

Denoising support is available in the preprocessing module, but it is disabled for the current baseline experiment.

🎵 MFCC Feature Extraction

The current baseline uses MFCC acoustic features:

Sampling Rate : 16 kHz
MFCCs         : 40
Aggregation   : Mean + Standard Deviation
Final Features: 80

The resulting feature vector is used as the input to the XGBoost classifier.

🤖 Current Baseline Model

The current classifier is XGBoost.

The baseline configuration is:

n_estimators      = 200
max_depth         = 4
learning_rate     = 0.05
subsample         = 0.9
colsample_bytree  = 0.9
objective         = binary:logistic
eval_metric       = logloss
random_state      = 42

This is an initial baseline model, not the final model of the project.

📈 Current Baseline Results

The first controlled ASVspoof 2019 LA baseline experiment was evaluated on 313 test segments.

Results
Metric	Result
Accuracy	76.36%
Precision	82.26%
Recall	66.23%
F1-score	73.38%
ROC-AUC	86.25%
EER	21.72%
Confusion Matrix
[[137, 22],
 [ 52,102]]

These are preliminary development results from the current baseline experiment and may change as the project progresses.

📞 Phone-like Robustness Experiment

An initial robustness experiment has been implemented to study the effect of a simulated telephone-style audio condition.

The transformation currently uses:

Resampling to 8 kHz
        +
Approximately 300-3400 Hz bandwidth limitation
        +
Resampling back to 16 kHz

The same trained XGBoost model is evaluated on the transformed audio without retraining.

Current Comparison
Condition	Accuracy	Precision	Recall	F1-score	ROC-AUC	EER
Clean	76.36%	82.26%	66.23%	73.38%	86.25%	21.72%
Phone-like	48.88%	49.04%	99.35%	65.67%	61.05%	40.26%

The initial experiment indicates that the current baseline is affected by the simulated phone-call-like condition.

This is an initial robustness experiment. More realistic channel conditions and additional robustness experiments are planned for later stages.

🧪 Current Experiment Status
Experiment	Status
Clean audio baseline	✅ Completed
Background noise	🔜 Planned
Compression	🔜 Planned
Telephone-like degradation	✅ Completed
Short audio clips	🔜 Planned
Unseen speakers	🔜 Planned
Cross-dataset testing	🔜 Planned
💻 Live Prediction Demo

A command-line inference demo is currently implemented.

The current demo performs:

Input Audio
    |
    v
Preprocessing
    |
    v
3-second Segmentation
    |
    v
MFCC Extraction
    |
    v
XGBoost Prediction
    |
    v
Segment Predictions
    |
    v
Overall REAL / FAKE Result

Example command:

python -m models.xgboost.predict_xgboost "/path/to/audio.flac"

The demo reports the prediction and fake probability for each segment and also produces an overall prediction.

The current demo is intended for development and demonstration purposes and is not the final application.

📁 Current Project Structure
Deepfake-Voice-Detection/
|
├── data/
│   ├── metadata/
│   ├── splits/
│   └── robustness/
|
├── preprocessing/
│   ├── preprocess_audio.py
│   ├── segment_audio.py
│   └── robustness/
|
├── features/
│   ├── extract_mfcc.py
│   ├── aggregate_mfcc.py
│   ├── build_mfcc_dataset.py
│   └── build_phone_mfcc.py
|
├── models/
│   └── xgboost/
│       ├── train_xgboost.py
│       └── predict_xgboost.py
|
├── evaluation/
│   ├── evaluate_model.py
│   └── compare_robustness.py
|
├── experiments/
│   ├── experiment_log.csv
│   └── robustness/
|
├── requirements.txt
├── dataset_plan.md
└── README.md

Large raw audio datasets and generated model/feature artifacts are kept outside the Git repository.

📊 Evaluation Metrics

The current evaluation framework includes:

Accuracy
Precision
Recall
F1-score
ROC-AUC
Equal Error Rate (EER)
Confusion Matrix

Future experiments may also study robustness under:

Background noise
Audio compression
Different sampling rates
Unseen speakers
Unseen synthetic voice-generation methods
Different speaking conditions
Cross-dataset evaluation
📦 Current Deliverables

The current development milestone includes:

1. Dataset Preparation
Dataset organization and metadata preparation.
Dataset label verification.
Controlled train, validation, and test subsets.
2. Audio Preprocessing Module
Mono conversion.
Resampling.
Normalization.
3-second segmentation.
Zero-padding.
3. Feature Extraction Module
MFCC extraction.
Mean and standard deviation aggregation.
80-dimensional feature vectors.
4. Baseline Detection Model
XGBoost training.
Trained baseline model.
Prediction pipeline.
5. Model Evaluation
Accuracy.
Precision.
Recall.
F1-score.
ROC-AUC.
EER.
Confusion matrix.
6. Robustness Experiment
Simulated phone-call-like audio transformation.
Clean vs phone-like comparison.
7. Live Demonstration
Individual audio-file prediction using the trained baseline.

These represent the current development milestone, not the final project deliverables.

⚠️ Current Limitations

The current implementation has several limitations:

Controlled subset size is currently used for development.
The current detector uses MFCC + XGBoost as a baseline.
The 3-second segmentation duration is an initial experimental choice.
Only an initial phone-like robustness condition has been evaluated.
The phone-call condition is simulated rather than collected from real telephone networks.
ASVspoof 2021 DF cross-dataset evaluation is still pending.
Stronger deep-learning models have not yet been implemented.
A final real-time phone-call system has not yet been developed.
The current command-line demo is not a production application.
🚀 Planned Future Work

Future stages of the project may include:

Larger-scale experiments
        |
        v
Additional robustness conditions
        |
        v
ASVspoof 2021 DF cross-dataset evaluation
        |
        v
CNN / stronger deep-learning models
        |
        v
Model comparison and improvement
        |
        v
Prototype application
        |
        v
Future real-time integration

Potential advanced approaches may include CNN-based architectures, RawNet2, AASIST, and other suitable audio anti-spoofing methods.

These approaches are planned future work and are not represented as completed in the current milestone.

👥 Team Members
Sr. No.	Name	Current Responsibility
1	M1	Dataset preparation and metadata
2	M2	Audio preprocessing and segmentation
3	M3	MFCC feature extraction and XGBoost baseline
4	M4	Evaluation and experiment analysis

Replace M1, M2, M3, and M4 with the actual team member names before the final project submission.

🛠️ Technology Stack
Current Baseline
Python
NumPy
Pandas
SciPy
Librosa
SoundFile
Scikit-learn
XGBoost
Git
GitHub
Planned Future Technologies

Depending on later project requirements, the project may also use:

PyTorch
CNN-based architectures
RawNet2
AASIST
FastAPI
React / Next.js
GPU-based experimentation

These are listed as future/planned components unless implemented in a later project milestone.

🌐 Future Scope

Possible future extensions include:

Real-time call audio analysis.
Telecom or VoIP integration.
Lightweight on-device inference.
Detection under network compression and packet loss.
Speaker verification.
Scam-intent detection from call transcripts.
Multimodal scam detection.
Mobile application integration.
📚 Research Direction

The project is being developed in stages:

Baseline Detector
       ↓
Robustness Experiments
       ↓
Cross-Dataset Evaluation
       ↓
Advanced Model Comparison
       ↓
Model Improvement
       ↓
Prototype Application
       ↓
Future Real-Time System

The current repository represents an intermediate development milestone rather than the final research system.

⚠️ Disclaimer

This project is a research and development prototype for studying the detection of potentially synthetic or manipulated speech.

A model prediction should not be treated as definitive proof that a caller is fraudulent. Detection performance may vary depending on audio quality, compression, background noise, speaker characteristics, channel conditions, spoofing methods, and previously unseen voice-generation techniques.
