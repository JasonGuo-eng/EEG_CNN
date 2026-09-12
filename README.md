# EEG_CNN: age prediction from EEG waveforms

A compact PyTorch CNN predicts chronological age from resting-state EEG. It learns
spatial and temporal filters, measures their log power, averages windows separately
for eyes-closed and eyes-open recordings, and predicts one age per person.

**Reference result: 7.67 +/- 0.31 years MAE**, using subject-separated five-fold
cross-validation on 608 adults aged 20-70. This uses one architecture trained with
three seeds (42, 43, 44), averaging their predictions. A single network's performance
has not been established by that result. See [RESULTS.md](RESULTS.md).

## 1. Install

Run commands from this folder. The tested interpreter is Python 3.13.14 on Windows,
with PyTorch 2.6.0+cu124 and an NVIDIA RTX 3080 Laptop GPU (8 GB). CPU is supported
but raw-waveform training is substantially slower.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The CUDA command follows the [official PyTorch 2.6 installation instructions](https://docs.pytorch.org/get-started/previous-versions/).
For CPU, replace `cu124` with `cpu`. On Linux/macOS use `.venv/bin/python` in place
of `.\.venv\Scripts\python.exe` and choose a supported PyTorch build.

## 2. Prepare data (choose one route)

### A. Reuse the original preprocessed data

If `data/processed/dataset.json` already exists, data preparation is complete; skip
to evaluation. The local project prepared for you already contains this import.

This route preserves the windows used for the reported result. Import runs once,
requires roughly 9.4 GB of additional disk space, and copies the waveform array.
The source needs X_raw.npy, y_raw.npy, groups_raw.npy, ch_names.npy, and the
on005385 session-1/pre-activity EEG JSON and channel TSV sidecars.

```powershell
.\.venv\Scripts\python.exe import_existing.py --source C:\eeg-brain-age
```

It verifies state boundaries and channel order, including the swapped Fp1/Fp2
channels for subject 075 and the shorter eyes-closed recording for subject 230.
After importing, this project no longer depends on the old folder.

### B. Start from original EEG recordings

Download the BIDS dataset and its actual EEG files from
[OpenNeuro ds005385 version 1.0.3](https://openneuro.org/datasets/ds005385/versions/1.0.3),
using the dataset page's download options. A metadata-only clone is insufficient.
The reference local data were the NEMAR on005385 mirror. See [DATASET.md](DATASET.md).
Place the downloaded BIDS root at data/bids, with participants.tsv immediately inside it.
Only session 1, acquisition pre, EyesClosed and EyesOpen are used.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-preprocess.txt
.\.venv\Scripts\python.exe -u preprocess.py --bids-root data/bids
```

Preprocessing selects EEG channels, applies common-average reference, fits 15-component
extended-Infomax/Picard ICA on 1-100 Hz data, removes components labeled as artifacts
by ICLabel, filters to 1-45 Hz, and resamples to 100 Hz. It creates five-second
windows with a one-second step, keeping up to 60 per recording state. Both states
are required. Failures are recorded in data/processed/preprocessing_log.json.
[MNE-ICALabel documentation](https://mne.tools/mne-icalabel/stable/api/index.html)
describes the automatic component labeling used here.

Fresh preprocessing now uses a stable subject seed. The older builder used Python's
process-randomized hash(), so the exact original window sample cannot be recovered
from a fresh EDF build. Reprocessed data must be evaluated again; 7.67 is a reference,
not a promised score. Do not combine the two preparation routes in one output folder.

## 3. Evaluate the CNN

```powershell
.\.venv\Scripts\python.exe -u train.py
```

This runs all five subject folds with the three reference seeds. It saves fold MAEs,
held-out predictions and checkpoints under outputs/cv. Read outputs/cv/summary.json
for the result. Each person's windows stay in one outer fold. A separate inner
subject holdout selects training duration; outer test ages never select checkpoints.

The command resumes completed folds. To retrain, use `--output outputs/cv_repeat`.
Use `--fold 1` through `--fold 5` to run folds separately into the same output folder.
A partial run is marked `complete_five_fold: false`; it is not a five-fold result.
`--seeds 42` runs faster, but does not reproduce the three-seed reference result.

## 4. Fit the final model on all available subjects

```powershell
.\.venv\Scripts\python.exe -u train.py --stage fit
```

This saves three CNN checkpoints and ensemble.json in outputs/final/model. Keep
those four files together. The final model uses all subjects after internal epoch
selection; its internal score is not an independent test score. Trained weights
are not included in the GitHub source package; this command generates them.

## 5. Predict a new recording

Prepare eyes-closed and eyes-open arrays with the same preprocessing, in **volts**,
shape **(windows, 64, 500)**, at **100 Hz**. Provide a JSON list of their channel names.
Prediction aligns the channels to training order. Both state arrays must use the
same input order. For a new recording stored in the supported BIDS layout:

```powershell
.\.venv\Scripts\python.exe preprocess.py --bids-root data/new_bids --subjects 999 --recording-only --output data/new_person
.\.venv\Scripts\python.exe predict.py --closed data/new_person/closed.npy --open data/new_person/open.npy --channels data/new_person/channels.json
```

Replace 999 with the actual subject ID. The recording-only path does not read age.
The prediction command uses outputs/final/model by default and prints predicted age
in years. It accepts any matching preprocessed arrays; BIDS conversion is optional
when these arrays already exist.

## Model in plain language

1. A 1x1 convolution learns 16 combinations of the 64 electrodes.
2. Temporal convolutions learn eight filters per combination (128 signals).
3. The model computes log mean-squared power of those signals and the 64 sensors.
4. It averages windows separately for eyes-closed and eyes-open states.
5. A small dense network maps the two state summaries to one age.

The model has **39,041 trainable parameters**. Temporal filters start as
frequency-selective filters and are then learned. Training preserves amplitude,
uses mild amplitude/noise augmentation, Huber loss, AdamW, and training-only age
normalization. Four windows per state are sampled during training, 12 during inner
validation, and 60 during final evaluation/inference. Deterministic sampling repeats
windows if fewer than 60 are available. There are no spectral-feature files, hybrid
branches, regression models or architecture search in this project.

## Project files

| File | Purpose |
|---|---|
| preprocess.py | Original BIDS EEG to clean windows |
| import_existing.py | Reuse the existing reference windows |
| data.py | Validate metadata and load EEG batches |
| model.py | The single raw-waveform CNN |
| train.py | Five-fold evaluation and final training |
| predict.py | Predict from saved models |
| tests/test_pipeline.py | Synthetic model/data/training checks |
| config/channels.json | Reference channel order |
| results/summary.json | Original raw-CNN aggregate metrics |
| results/verification.json | Standalone-code verification against the reference |
| results/reference_folds.csv | Original subject-to-fold assignments (1-5) |
| RESULTS.md, DATASET.md | Evidence, limits and data attribution |

Run checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Upload to GitHub

Upload the source package contents, including the dotfiles, config/, results/ and
tests/. The .gitignore excludes EEG data, environments, checkpoints and generated
outputs. A provided source ZIP contains only publishable project files; extract it
first for browser uploads. To include trained weights later, distribute them
separately with their model manifest and provenance. Do not drag data/processed into
GitHub's upload page: browser uploads do not apply .gitignore automatically.

No code license has been selected on your behalf. Add your preferred license before
inviting reuse. The dataset's CC0 license is separate from the code license.
