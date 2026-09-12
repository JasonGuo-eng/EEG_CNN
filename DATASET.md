# Data source and scope

The reference data are **Resting-state EEG data before and after cognitive activity
across the adult lifespan and a 5-year follow-up**, by Edmund Wascher, Daniel
Schneider, Patrick D. Gajewski and Stephan Getzmann.

- [OpenNeuro ds005385, version 1.0.3](https://openneuro.org/datasets/ds005385/versions/1.0.3)
- Source DOI: https://doi.org/10.18112/openneuro.ds005385.v1.0.3
- Local NEMAR mirror DOI: https://doi.org/10.82901/nemar.on005385
- Dataset license in the local description: CC0.
- Study protocol: Gajewski et al. (2022), [Dortmund Vital Study](https://doi.org/10.2196/32352).

The reference uses 608 participants, ages 20-70, session 1, acquisition pre, both
EyesClosed and EyesOpen. It contains 72,956 preprocessed 64-channel, five-second
windows at 100 Hz. Participant 230 has 56 closed and 60 open windows; the other
607 participants have 60 per state. The channel-order correction is stored
explicitly by import_existing.py.

The data are not bundled with the GitHub source. Obtain them from the source and
retain the dataset authors' attribution. Keep sessions from a person together if
extending the dataset: this project uses only session 1 and pre-activity recordings.
