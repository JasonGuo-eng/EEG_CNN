"""Evaluate or fit one raw-waveform CNN architecture, averaging three seeds."""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from data import load_data, RawStore
from model import CONFIG, build_model

CNN_VERSION = 1


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_normalization(data, train):
    ages = data["ages"][train]
    return dict(age_mean=float(ages.mean()), age_std=max(float(ages.std()), 1.))


@torch.no_grad()
def predict(model, indices, normal, device, raw, windows=60):
    model.eval()
    predictions = []
    for start in range(0, len(indices), 8):
        x = torch.as_tensor(raw.batch(indices[start:start+8], windows), device=device)
        predictions.append(model(x).cpu().numpy()*normal["age_std"] + normal["age_mean"])
    return np.concatenate(predictions)


def fit_one(data,train,val,seed,device,raw_store,epochs=None):
    name = "raw_power"
    cfg = copy.deepcopy(CONFIG)
    if epochs is not None:
        max_epochs = int(epochs)
    else:
        max_epochs = int(cfg["epochs"])
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    raw = raw_store
    normal = fit_normalization(data,train)
    targets = torch.as_tensor((data["ages"]-normal["age_mean"])/normal["age_std"],
                              dtype=torch.float32,device=device)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=cfg["epochs"],eta_min=cfg["lr"]*0.05)
    best_mae,best_epoch,best_state,best_predictions = float("inf"),0,None,None
    history = []
    t0 = time.perf_counter()
    evaluate_every = 5
    patience = 30
    for epoch in range(1,max_epochs+1):
        model.train()
        losses = []
        order = rng.permutation(train)
        for start in range(0,len(order),cfg["batch"]):
            ids = order[start:start+cfg["batch"]]
            if len(ids)<2:
                continue
            batch_targets = targets[ids]
            x = torch.as_tensor(raw.batch(ids,cfg["windows"],rng),device=device)
            # Mild perturbations; frequency/time warping is intentionally absent.
            x = x*(0.97+0.06*torch.rand((len(ids),2,1,1,1),device=device))
            if rng.random()<0.3:
                x = x + 0.005*torch.randn_like(x)
            output = model(x)
            loss = F.smooth_l1_loss(output,batch_targets,beta=0.5)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {name}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if val is not None and (epoch % evaluate_every==0 or epoch==1 or epoch==max_epochs):
            prediction = predict(model,val,normal,device,raw,windows=12)
            mae = float(np.abs(prediction-data["ages"][val]).mean())
            history.append(dict(epoch=epoch,mae=mae,training_loss=float(np.mean(losses))))
            if mae<best_mae:
                best_mae,best_epoch,best_predictions = mae,epoch,prediction
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch%25==0 or epoch==1:
                print(f"    {name} seed={seed} epoch={epoch}: inner MAE={mae:.3f}, "
                      f"best={best_mae:.3f} at {best_epoch}, {time.perf_counter()-t0:.1f}s",flush=True)
            if epoch>=40 and epoch-best_epoch>=patience:
                break
        elif val is None and (epoch%50==0 or epoch==max_epochs):
            print(f"    refit {name} seed={seed} epoch={epoch}/{max_epochs} "
                  f"{time.perf_counter()-t0:.1f}s",flush=True)
    if val is None:
        best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        best_epoch = max_epochs
        best_mae = None
    record = dict(name=name,config=cfg,seed=seed,epochs=best_epoch,
                  inner_mae=best_mae,history=history,seconds=time.perf_counter()-t0)
    bundle = dict(cnn_version=CNN_VERSION,data_version=1,record=record,
                  state_dict=best_state,normalization=normal,train_subjects=data["subjects"][train],
                  validation_subjects=[] if val is None else data["subjects"][val],
                  source=data["source"])
    del model,optimizer
    if device.type=="cuda":
        torch.cuda.empty_cache()
    return bundle,best_predictions



def load_bundle(path):
    # Load only your own or trusted checkpoints: these contain Python metadata.
    return torch.load(path, map_location="cpu", weights_only=False)


def split_inner(data, indices, seed=42):
    bins = np.minimum(data["ages"][indices] // 10, 6).astype(int)
    train, val = train_test_split(indices, test_size=0.25, random_state=seed, stratify=bins)
    return np.sort(train), np.sort(val)


def fit_ensemble(data, indices, seeds, split_seed, device, raw, output, test=None):
    output.mkdir(parents=True, exist_ok=True)
    selection_dir, model_dir = output/"selection", output/"model"
    selection_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    inner_train, val = split_inner(data, indices, split_seed)
    predictions, members = [], []
    for seed in seeds:
        path = selection_dir/f"raw_power_seed{seed}.pt"
        if path.exists():
            selected = load_bundle(path)
            if (selected["source"] != data["source"] or selected["record"]["config"] != CONFIG
                or not np.array_equal(selected["train_subjects"], data["subjects"][inner_train])
                or not np.array_equal(selected["validation_subjects"], data["subjects"][val])):
                raise ValueError("Incompatible checkpoint; choose a new output folder")
        else:
            selected, _ = fit_one(data, inner_train, val, seed, device, raw)
            torch.save(selected, path)
        bundle, _ = fit_one(data, indices, None, seed, device, raw,
                            epochs=selected["record"]["epochs"])
        bundle["channel_names"] = data["channel_names"]
        bundle["sfreq"] = 100.
        bundle["samples_per_window"] = 500
        name = f"raw_power_seed{seed}.pt"
        torch.save(bundle, model_dir/name)
        members.append(name)
        if test is not None:
            assert not set(bundle["train_subjects"]) & set(data["subjects"][test])
            model = build_model(bundle["record"]["config"]).to(device)
            model.load_state_dict(bundle["state_dict"])
            predictions.append(predict(model, test, bundle["normalization"], device, raw))
            del model
    (model_dir/"ensemble.json").write_text(json.dumps(dict(cnn_version=CNN_VERSION,
        members=members, training_subjects=data["subjects"][indices].tolist()), indent=2),encoding="utf-8")
    return np.mean(predictions, axis=0) if predictions else None


def summarize(output):
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(output.glob("fold_*/test_predictions.json"))]
    rows = [row for record in records for row in record["rows"]]
    if not rows:
        return
    if len({r["subject"] for r in rows}) != len(rows):
        raise ValueError("Duplicate test subjects")
    maes = [r["mae"] for r in records]
    ages, predicted = [r["age"] for r in rows], [r["predicted_age"] for r in rows]
    summary = dict(completed_folds=len(records), complete_five_fold=len(records)==5,
        n_subjects=len(rows), fold_maes=maes, fold_mean_mae=float(np.mean(maes)),
        fold_std_mae=float(np.std(maes)), pooled_mae=float(np.mean(np.abs(np.array(ages)-predicted))),
        r2=float(r2_score(ages, predicted)))
    (output/"summary.json").write_text(json.dumps(summary, indent=2),encoding="utf-8")
    with (output/"oof_predictions.csv").open("w", newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--stage", choices=["evaluate", "fit"], default="evaluate")
    p.add_argument("--output", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--fold", type=int, choices=range(1,6), help="Optionally run just one outer fold (1-5)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.threads < 1:
        p.error("Seeds must be unique and threads positive")
    if args.stage == "fit" and args.fold is not None:
        p.error("--fold applies only to evaluation")
    output = args.output or Path("outputs/cv" if args.stage=="evaluate" else "outputs/final")
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    with threadpool_limits(args.threads):
        data = load_data(args.data)
        raw = RawStore(args.data, data)
        output.mkdir(parents=True, exist_ok=True)
        run = dict(cnn_version=CNN_VERSION, config=CONFIG, seeds=args.seeds,
                   split_seed=args.split_seed, source=data["source"], stage=args.stage)
        manifest = output/"run.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != run:
            raise ValueError("Output belongs to a different run; use a new --output folder")
        manifest.write_text(json.dumps(run, indent=2),encoding="utf-8")
        print(f"Device={device}; subjects={len(data['subjects'])}; seeds={args.seeds}", flush=True)
        if args.stage == "fit":
            fit_ensemble(data, np.arange(len(data["subjects"])), args.seeds, args.split_seed,
                         device, raw, output)
            return
        if set(data["folds"]) != set(range(5)):
            raise ValueError("Five-fold evaluation requires all five nonempty subject folds")
        for fold in range(5):
            if args.fold is not None and args.fold != fold+1:
                continue
            directory = output/f"fold_{fold+1}"
            if (directory/"test_predictions.json").exists():
                print(f"Fold {fold+1} already completed", flush=True)
                continue
            train, test = np.flatnonzero(data["folds"] != fold), np.flatnonzero(data["folds"] == fold)
            print(f"OUTER FOLD {fold+1}: {len(train)} train / {len(test)} test", flush=True)
            prediction = fit_ensemble(data, train, args.seeds, args.split_seed, device, raw, directory, test)
            rows = [dict(subject=str(data["subjects"][i]), age=float(data["ages"][i]), fold=fold+1,
                         predicted_age=float(v), absolute_error=float(abs(v-data["ages"][i])))
                    for i,v in zip(test, prediction)]
            record = dict(mae=float(np.mean([r["absolute_error"] for r in rows])), rows=rows)
            (directory/"test_predictions.json").write_text(json.dumps(record, indent=2),encoding="utf-8")
        summarize(output)


if __name__ == "__main__":
    main()
