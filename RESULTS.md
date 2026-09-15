# Raw-waveform CNN results

Reference five-fold subject-level MAE: **7.6694 +/- 0.3123 years**.
Pooled MAE: **7.6704 years**. Pooled R-squared: **0.5681**.

| Fold | Test subjects | MAE (years) |
|---|---:|---:|
| 1 | 122 | 7.6210 |
| 2 | 122 | 7.6927 |
| 3 | 122 | 8.2462 |
| 4 | 121 | 7.3998 |
| 5 | 121 | 7.3876 |

These are the completed raw_power experiments from the original eeg-brain-age
project, using three training seeds, 42/43/44. They are retained as reference
measurements, not presented as a new full rerun of this standalone packaging.
The model and training defaults are preserved. The standalone refactor removes
unnecessary spectral cache loading and alternative architectures.

Each outer fold holds out all EEG windows from its test subjects. A stratified
25% inner subject holdout selects the epoch count separately for each seed.
The CNN is then initialized again and trained on all outer training subjects
for that many epochs. Predictions from three seeds are averaged. Fold standard
deviation is descriptive variation, not a confidence interval.

Exact cached-data reproduction requires the original windows, channel alignment,
subject folds and compatible software/hardware. New preprocessing changes window
selection to a stable seed and therefore requires its own evaluation. No full
fresh-EDF five-fold result is claimed for this package.

This is chronological age prediction within one cohort already used for model
development. External validation has not been performed. Age 60-70 MAE is about
10.14 years; errors are not uniform across ages. These results do not establish
clinical biological-age accuracy.


## Verification

- All six synthetic pipeline tests passed, including fit/save/predict and channel alignment.
- All 608 subject fold assignments match the original reference.
- Fixed and random EEG batches match exactly, including subjects 075 and 230.
- Two CPU training epochs produced exactly the same predictions and checkpoint
  tensors as the original raw-CNN implementation.
- Inference with all 15 original fold/seed checkpoints reproduced all 608 saved
  test predictions exactly on the tested GPU; the MAE remains 7.6694 years.
- Original BIDS preprocessing completed for subject 230, producing 56 closed and
  60 open windows. Those arrays passed a saved-model prediction smoke check.

