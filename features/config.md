# Initial Configuration — Week 1 Baseline

| Parameter | Initial Setting | Reason |
|---|---:|---|
| Sampling Rate | **16,000 Hz** | Standardized audio input for the ML pipeline |
| Number of MFCCs | **40** | Initial compact spectral representation |
| Segment Duration | **3 seconds** | Initial fixed-duration experimental target |
| Feature | **MFCC** | Initial acoustic feature representation |
| Input Audio Format | **WAV** | Standardized output from the audio preprocessing stage |
| Classifier | **XGBoost** | Initial traditional ML baseline |
| Task | **Binary Classification** | Real vs Fake |

## Audio Input Contract

The ML pipeline accepts **WAV files only**.

Expected team outputs:

```text
samples/
├── real/
│   └── real_01_clean.wav
│
└── fake/
    └── fake_01_clean.wav