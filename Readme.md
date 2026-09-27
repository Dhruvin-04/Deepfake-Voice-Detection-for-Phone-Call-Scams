# Deepfake Voice Detection for Phone Call Scams

## 📌 Project Overview

**Deepfake Voice Detection for Phone Call Scams** is an AI-based cybersecurity project focused on detecting whether a voice in a phone call is **genuine human speech or AI-generated/manipulated speech**.

With recent advances in voice cloning and speech synthesis, attackers can generate convincing copies of a person's voice and use them in impersonation and emergency scams. This project aims to develop a detection system that analyzes audio characteristics and identifies potential deepfake voices.

---

## 🎯 Problem Statement

AI-based voice cloning technology has made it possible to generate highly realistic human-like speech using a small amount of voice data. This creates a security risk when cloned voices are used in phone scams, such as impersonating family members, friends, or other trusted individuals.

Traditional spam and caller-identification systems primarily analyze the **phone number or caller metadata** and do not determine whether the actual voice is authentic.

Therefore, there is a need for an AI-based system capable of analyzing voice audio and detecting whether it is likely to be **genuine or AI-generated**.

---

## 🎯 Objective

The main objective of this project is to develop a machine-learning/deep-learning based system for **deepfake voice detection in phone-call scenarios**.

### Specific objectives

* Detect AI-generated or manipulated speech.
* Preprocess and segment incoming audio for analysis.
* Extract meaningful acoustic features from speech.
* Experiment with suitable machine-learning/deep-learning architectures.
* Evaluate the models using standard classification metrics.
* Investigate robustness under realistic audio conditions such as noise and compression.
* Provide a confidence-based prediction indicating whether the analyzed voice is likely genuine or synthetic.
* Develop a prototype suitable for future real-time phone-call integration.

---

## 🔄 Proposed Pipeline

```text
                    Phone Call / Audio Input
                              │
                              ▼
                    ┌───────────────────┐
                    │ Audio Preprocessing│
                    │                   │
                    │ • Resampling      │
                    │ • Normalization   │
                    │ • Noise handling  │
                    │ • Segmentation    │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Feature Extraction│
                    │                   │
                    │ • MFCC            │
                    │ • Spectrogram     │
                    │ • Other features  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Deepfake Detection│
                    │      Model        │
                    │                   │
                    │ • CNN / RawNet2   │
                    │ • AASIST / other  │
                    │   suitable models │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │    Prediction     │
                    │                   │
                    │ Genuine / Deepfake│
                    │ Confidence Score  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ User Alert /      │
                    │ Detection Result  │
                    └───────────────────┘
```

### Pipeline Stages

**1. Audio Input**

The system receives a recorded audio clip or, in a later stage, audio from a live communication stream.

**2. Preprocessing**

The input audio is prepared for model analysis through operations such as resampling, normalization, segmentation, and noise handling.

**3. Feature Extraction**

Acoustic representations such as **MFCCs and spectrograms** are extracted from the audio.

**4. Model Training / Inference**

Machine-learning and deep-learning models are trained or evaluated to distinguish genuine speech from synthetic or manipulated speech.

Potential architectures include **CNN-based models, RawNet2, AASIST, and other suitable anti-spoofing architectures**.

**5. Prediction**

The system produces a classification such as:

```text
Genuine Voice
```

or

```text
Potential Deepfake Voice
```

along with a confidence/probability score.

**6. Alert**

For a potential phone scam scenario, the system can present a warning to the user when suspicious synthetic speech is detected.

---

## 👥 Team Members

| Sr. No. | Name | Role           |
| ------: | ---- | -------------- |
|       1 | TBD  | Project Member |
|       2 | TBD  | Project Member |
|       3 | TBD  | Project Member |
|       4 | TBD  | Project Member |

> Update the names and roles once the team is finalized.

---

## 🛠️ Technology Stack

### Programming Language

* **Python**

### Audio Processing

* **Librosa**
* **torchaudio**
* NumPy
* SciPy

### Machine Learning / Deep Learning

* **PyTorch**
* Scikit-learn
* CNN-based architectures
* RawNet2
* AASIST

### Dataset

* **ASVspoof**
* Additional suitable genuine and synthetic speech datasets where required

### Experimentation

* Jupyter Notebook
* Google Colab / GPU environment
* Python-based experiment scripts

### Backend / Application

* FastAPI
* REST API

### Frontend

* React / Next.js *(if required for the prototype)*

### Version Control

* Git
* GitHub

---

## 📁 Project Structure

```text
deepfake-voice-detection/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── preprocessing/
│   └── ...
│
├── features/
│   └── ...
│
├── models/
│   └── ...
│
├── evaluation/
│   └── ...
│
├── experiments/
│   └── ...
│
├── app/
│   ├── backend/
│   └── frontend/
│
├── notebooks/
│   └── ...
│
├── paper/
│   └── ...
│
└── README.md
```

### Directory Description

| Directory        | Purpose                                                    |
| ---------------- | ---------------------------------------------------------- |
| `data/`          | Dataset and processed audio information                    |
| `preprocessing/` | Audio cleaning, resampling, normalization and segmentation |
| `features/`      | MFCC, spectrogram and other feature extraction             |
| `models/`        | Model implementations, configurations and checkpoints      |
| `evaluation/`    | Evaluation scripts and metric calculations                 |
| `experiments/`   | Experimental configurations and results                    |
| `app/`           | Application/backend/frontend implementation                |
| `notebooks/`     | Exploratory analysis and experiments                       |
| `paper/`         | Research paper, literature review and related material     |
| `README.md`      | Project documentation                                      |

---

## 📊 Evaluation

The models will be evaluated using appropriate classification and anti-spoofing metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Equal Error Rate (EER)

Additional experiments may evaluate model robustness under:

* Background noise
* Audio compression
* Different sampling rates
* Unseen synthetic voice-generation methods
* Different speakers and speaking conditions

---

## 📦 Deliverables

The expected project deliverables include:

### 1. Dataset Preparation

* Collection/selection of suitable genuine and synthetic speech datasets.
* Dataset preprocessing and organization.

### 2. Audio Preprocessing Module

* Audio resampling
* Normalization
* Segmentation
* Noise handling

### 3. Feature Extraction Module

* MFCC extraction
* Spectrogram generation
* Investigation of additional acoustic representations

### 4. Deepfake Detection Model

* Implementation/integration of suitable deepfake voice detection models.
* Training and/or fine-tuning using appropriate datasets.

### 5. Model Evaluation

* Performance comparison between selected approaches.
* Evaluation using accuracy, precision, recall, F1-score, ROC-AUC and EER.

### 6. Prototype Application

A prototype capable of accepting audio and returning:

```text
Prediction: Potential Deepfake / Genuine
Confidence Score: XX%
```

### 7. Research Documentation

* Literature review
* Methodology
* Experimental results
* Discussion
* Limitations
* Future work
* Final research paper

---

## 🚀 Future Scope

Possible future extensions include:

* Real-time call audio analysis
* Telecom/VoIP integration
* Lightweight on-device inference
* Detection under network compression and packet loss
* Speaker verification
* Scam-intent detection from call transcripts
* Multimodal scam detection
* Mobile application integration

---

## ⚠️ Disclaimer

The system is intended as a **research prototype** for detecting potentially synthetic or manipulated speech. A deepfake prediction should not be treated as definitive proof that a caller is fraudulent. Detection performance may vary depending on audio quality, compression, background noise, speaker characteristics, and previously unseen voice-generation techniques.
