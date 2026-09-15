# EEG_CNN

This project uses a convolutional neural network (CNN) to predict a person's age from resting-state EEG recordings. The model is written in PyTorch and takes both eyes-closed and eyes-open recordings as input.

## Data

The project uses 64-channel EEG from 608 adults aged 20–70 in the OpenNeuro ds005385 dataset. It uses the recordings taken before cognitive activity in the first session.

Preprocessing removes artifacts with ICA, filters the signals to 1–45 Hz, and resamples them to 100 Hz. The recordings are then split into five-second windows, with up to 60 windows for each recording state. Dataset details and attribution are in [DATASET.md](DATASET.md).

## Model

The CNN learns combinations of EEG channels and temporal filters, then calculates the log power of the filtered signals and the original channels. It averages these features across windows separately for eyes-closed and eyes-open recordings. A small fully connected network combines the two summaries to predict one age per person.

The network has 39,041 trainable parameters. Training uses AdamW and Huber loss.

## Results

The reference experiment achieved a mean absolute error of **7.67 years**, with a standard deviation of **0.31 years** across five folds. All recordings from a person stay in the same fold, so test participants are separate from training participants. Each prediction averages three models trained with seeds 42, 43, and 44.

These results come from the original preprocessed data. Running preprocessing again can change the sampled windows and the resulting score. The model has only been evaluated within this dataset. Fold results and verification details are in [RESULTS.md](RESULTS.md).

## Code

- `preprocess.py` cleans the EEG recordings and creates input windows.
- `data.py` loads the windows and participant information.
- `model.py` defines the CNN.
- `train.py` runs five-fold evaluation or fits the model on all participants.
- `predict.py` predicts age from preprocessed eyes-closed and eyes-open recordings.

EEG data and trained weights are not included in the source package.
