# Dataset Usage

The raw ASVspoof audio and official key/metadata files are **not stored in GitHub** because of their size and because the project should reference the official distribution rather than duplicate it.

## Official resources

- ASVspoof 2021: https://www.asvspoof.org/index2021.html
- DF evaluation audio: https://zenodo.org/records/4835108
- DF keys and metadata: https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz

## Current implementation dataset

We currently use the official **ASVspoof 2021 DF Evaluation Part00** for initial development.

Expected local structure:

```text
ASVspoof2021_DF/
├── ASVspoof2021_DF_eval_part00/
│   └── ASVspoof2021_DF_eval/
│       ├── flac/
│       ├── ASVspoof2021.DF.cm.eval.trl.txt
│       ├── README.DF.txt
│       └── LICENSE.DF.txt
└── DF-keys-full/
    └── keys/
        └── DF/
            └── CM/
                └── trial_metadata.txt
```

Do not commit the raw audio archives, FLAC files, or the full key package to Git.
