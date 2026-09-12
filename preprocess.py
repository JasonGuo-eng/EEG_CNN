"""Prepare session-1, pre-activity eyes-closed/open EEG from BIDS recordings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from data import STATES, load_data


def stable_seed(subject, seed):
    return int.from_bytes(hashlib.sha256(f"{seed}:{subject}".encode()).digest()[:4], "little")


def extract_subject(bids_root, subject, channels, seed=42):
    # These optional dependencies are not needed for training cached windows.
    from mne_bids import BIDSPath, read_raw_bids
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    rng = np.random.default_rng(stable_seed(subject,seed))
    states = []
    for task in STATES:
        path = BIDSPath(subject=subject,session="1",task=task,acquisition="pre",
                        datatype="eeg",root=bids_root)
        raw = read_raw_bids(path,verbose=False).load_data(verbose=False)
        raw.pick("eeg")
        if len(raw.ch_names)!=64 or set(raw.ch_names)!=set(channels):
            raise ValueError(f"Unexpected channels for {subject}/{task}")
        raw.set_montage("standard_1020",verbose=False)
        raw.set_eeg_reference("average",projection=False,verbose=False)
        broadband = raw.copy().filter(1.,100.,n_jobs=1,verbose=False)
        ica = ICA(n_components=15,method="picard",fit_params=dict(ortho=False,extended=True),
                  random_state=42,max_iter="auto")
        ica.fit(broadband,verbose=False)
        labels = label_components(broadband,ica,method="iclabel")
        ica.exclude = [i for i,label in enumerate(labels["labels"]) if label not in ("brain","other")]
        clean = raw.copy().filter(1.,45.,n_jobs=1,verbose=False)
        ica.apply(clean,verbose=False)
        clean.resample(100.,npad="auto",verbose=False)
        clean.reorder_channels(channels)
        values = clean.get_data().astype(np.float32)
        starts = np.arange(0,values.shape[1]-500+1,100)
        if not len(starts):
            raise ValueError(f"No complete windows for {subject}/{task}")
        if len(starts)>60:
            starts = starts[rng.choice(len(starts),60,replace=False)]
        windows = np.stack([values[:,start:start+500] for start in starts])
        if not np.isfinite(windows).all():
            raise ValueError(f"Nonfinite windows for {subject}/{task}")
        states.append(windows)
    return states


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bids-root",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("data/processed"))
    p.add_argument("--channels",type=Path,default=Path(__file__).parent/"config/channels.json")
    p.add_argument("--subjects",nargs="+",help="Optional IDs (e.g. 001 075)")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--recording-only",action="store_true",help="Export closed.npy/open.npy for one person, without needing age")
    args = p.parse_args()
    channels = json.loads(args.channels.read_text(encoding="utf-8"))
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output must be empty; existing data will not be overwritten")
    if args.recording_only and (not args.subjects or len(args.subjects)!=1):
        p.error("--recording-only requires exactly one --subjects ID")
    args.output.mkdir(parents=True,exist_ok=True)
    if args.recording_only:
        closed, opened = extract_subject(args.bids_root,args.subjects[0].removeprefix("sub-"),channels,args.seed)
        np.save(args.output/"closed.npy",closed)
        np.save(args.output/"open.npy",opened)
        (args.output/"channels.json").write_text(json.dumps(channels,indent=2),encoding="utf-8")
        return
    with (args.bids_root/"participants.tsv").open(newline="",encoding="utf-8-sig") as handle:
        participants = list(csv.DictReader(handle,delimiter="\t"))
    selected = None if args.subjects is None else {s.removeprefix("sub-") for s in args.subjects}
    records, failures = [], []
    for person in participants:
        sid = person["participant_id"].removeprefix("sub-")
        if selected is not None and sid not in selected:
            continue
        try:
            age = float(person["age"])
            if not np.isfinite(age):
                raise ValueError("Nonfinite age")
            closed, opened = extract_subject(args.bids_root,sid,channels,args.seed)
            name = f"sub-{sid}.npy"
            np.save(args.output/name,np.concatenate([closed,opened]))
            n = len(closed)
            records.append(dict(subject=sid,age=age,file=name,
                rows=[list(range(n)),list(range(n,n+len(opened)))],orders=[list(range(64)),list(range(64))]))
            print(f"sub-{sid}: {len(closed)} closed / {len(opened)} open",flush=True)
        except Exception as exc:
            failures.append(dict(subject=sid,error=str(exc)))
            print(f"sub-{sid}: skipped: {exc}",flush=True)
        (args.output/"preprocessing_log.json").write_text(json.dumps(dict(completed=len(records),failures=failures),indent=2),encoding="utf-8")
    if len(records)<5:
        raise ValueError("Fewer than five usable subjects; see preprocessing_log.json")
    records.sort(key=lambda r:r["subject"])
    groups = np.concatenate([np.repeat(r["subject"],sum(map(len,r["rows"]))) for r in records])
    fold_map = {}
    for fold,(_,test) in enumerate(GroupKFold(5).split(groups,groups=groups)):
        fold_map.update({sid:fold for sid in np.unique(groups[test])})
    for record in records:
        record["fold"] = fold_map[record["subject"]]
    manifest = dict(version=1,sfreq=100,samples_per_window=500,units="V",channel_names=channels,
        provenance="BIDS preprocessing with deterministic SHA-256 subject window seeds",seed=args.seed,subjects=records)
    (args.output/"dataset.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    load_data(args.output)
    print(f"Prepared {len(records)} subjects; skipped {len(failures)}. Inspect preprocessing_log.json.")


if __name__ == "__main__":
    main()
