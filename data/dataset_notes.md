\# ASVspoof 2021 DF Dataset Notes



\## 1. Dataset Name

ASVspoof 2021 Speech DeepFake (DF) Dataset



\## 2. Source

Official ASVspoof 2021 dataset.



\## 3. Dataset Subset Used

ASVspoof 2021 DF Evaluation Set - Part00



\## 4. Dataset Location

D:\\data\\ASVspoof2021\_DF



\## 5. Folder Structure



ASVspoof2021\_DF\_eval/

\- flac/ : audio files

\- ASVspoof2021.DF.cm.eval.trl.txt : evaluation trial list

\- README.DF.txt : dataset information

\- LICENSE.DF.txt : license information



Official DF keys/metadata:

DF-keys-full/keys/DF/CM/trial\_metadata.txt



\## 6. Audio Information

\- Format: FLAC

\- Sampling rate: 16 kHz

\- Bit depth: 16-bit

\- Channels observed: 1 (mono)



\## 7. Labels

The official metadata file `trial\_metadata.txt` contains the label for each trial.



\- bonafide = REAL

\- spoof = FAKE



\## 8. Part00 Statistics

\- Total audio files: 152,955

\- Real / bonafide: 5,535

\- Fake / spoof: 147,420

\- Real: 3.62%

\- Fake: 96.38%

\- Metadata matched: 152,955

\- Unmatched: 0



\## 9. Audio Inspection

20 audio files were inspected using `inspect\_audio.py`.



Observed:

\- Sampling rate: 16,000 Hz

\- Channels: 1

\- Format: FLAC

\- Subtype: PCM\_16



\## 10. Sample Duration

Based on the 20 inspected files:

\- Minimum: 1.408 seconds

\- Maximum: 4.712 seconds

\- Average: approximately 2.867 seconds



\## 11. Analysis Files

\- inspect\_audio.py

\- audio\_inspection.csv

\- dataset\_statistics.py

\- dataset\_statistics.csv



\## 12. Important Note

The statistics above are for Part00 only and should not be treated as statistics for the complete ASVspoof 2021 DF evaluation dataset.



\## 13. Dataset Lead Status

\- Dataset obtained

\- Folder structure understood

\- Label structure understood

\- Audio inspection completed

\- Part00 statistics calculated

\- Dataset documentation prepared

