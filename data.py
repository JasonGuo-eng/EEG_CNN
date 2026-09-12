"""Explicit subject/state metadata and memory-mapped EEG; no spectral cache."""
import hashlib
import json
from pathlib import Path
import numpy as np

STATES = ("EyesClosed", "EyesOpen")
DATA_VERSION = 1


def local_path(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Data files must stay inside the dataset directory")
    return path


def load_data(root):
    root = Path(root).resolve()
    manifest_path = root / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["version"] != DATA_VERSION or manifest["sfreq"] != 100
            or manifest["samples_per_window"] != 500 or manifest["units"] != "V"):
        raise ValueError("Expected version-1 data in volts at 100 Hz, 500 samples/window")
    channels = manifest["channel_names"]
    if len(channels) != 64 or len(set(channels)) != 64:
        raise ValueError("Expected 64 unique canonical EEG channels")
    records = sorted(manifest["subjects"], key=lambda r: r["subject"])
    subjects = np.array([r["subject"] for r in records])
    ages = np.array([r["age"] for r in records], dtype=float)
    folds = np.array([r["fold"] for r in records], dtype=int)
    if (len(subjects) != len(set(subjects)) or not len(subjects)
            or not np.isfinite(ages).all() or np.any((folds < 0) | (folds > 4))):
        raise ValueError("Invalid subject IDs, ages, or five-fold assignments")
    files, claimed, lengths = {}, {}, {}
    for record in records:
        name = record["file"]
        path = local_path(root, name)
        if name not in files:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.float32 or array.ndim != 3 or array.shape[1:] != (64, 500):
                raise ValueError(f"Invalid waveform array: {name}")
            files[name] = [path.stat().st_size, path.stat().st_mtime_ns]
            lengths[name] = len(array)
            claimed[name] = set()
            del array
        if len(record["rows"]) != 2 or len(record["orders"]) != 2:
            raise ValueError("Every subject needs closed and open states")
        for rows, order in zip(record["rows"], record["orders"]):
            if not rows or sorted(order) != list(range(64)):
                raise ValueError("Empty state or invalid channel permutation")
            if any(not isinstance(i, int) or i < 0 or i >= lengths[name] for i in rows):
                raise ValueError("Invalid window index")
            if len(set(rows)) != len(rows) or claimed[name].intersection(rows):
                raise ValueError("Windows assigned to more than one subject/state")
            claimed[name].update(rows)
    source = dict(dataset_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), files=files)
    return dict(subjects=subjects, ages=ages, folds=folds, records=records,
                channel_names=np.array(channels), source=source)


class RawStore:
    def __init__(self, root, data):
        self.root = Path(root)
        self.records = data["records"]
        self.arrays = {}

    def batch(self, indices, k, rng=None):
        result = np.empty((len(indices), 2, k, 64, 500), dtype=np.float32)
        for i, idx in enumerate(indices):
            record = self.records[idx]
            name = record["file"]
            if name not in self.arrays:
                self.arrays[name] = np.load(local_path(self.root, name), mmap_mode="r")
            raw = self.arrays[name]
            for state in range(2):
                rows = np.asarray(record["rows"][state])
                select = (rng.choice(len(rows), size=k, replace=k > len(rows)) if rng is not None
                          else np.rint(np.linspace(0, len(rows)-1, k)).astype(int))
                result[i, state] = raw[rows[select]][:, record["orders"][state], :]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite raw EEG")
        result *= 1e5
        return np.clip(result, -20, 20)

    def close(self):
        self.arrays.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
