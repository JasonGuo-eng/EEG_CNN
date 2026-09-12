"""Predict chronological age from two preprocessed EEG state arrays."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model import build_model
from train import load_bundle, CNN_VERSION


def predict_recording(model_dir, closed, opened, channel_names, sfreq=100., device="cpu"):
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir/"ensemble.json").read_text(encoding="utf-8"))
    if manifest["cnn_version"] != CNN_VERSION or not manifest["members"]:
        raise ValueError("Invalid ensemble manifest")
    if sfreq != 100.:
        raise ValueError("Expected preprocessed EEG at 100 Hz")
    names = list(map(str, channel_names))
    device = torch.device(device)
    predictions = []
    expected = None
    with torch.no_grad():
        for name in manifest["members"]:
            path = (model_dir/name).resolve()
            if not path.is_relative_to(model_dir.resolve()):
                raise ValueError("Model member must be inside model directory")
            bundle = load_bundle(path)
            channels = list(map(str, bundle["channel_names"]))
            if bundle["cnn_version"] != CNN_VERSION or bundle["sfreq"] != sfreq:
                raise ValueError("Incompatible checkpoint")
            if expected is not None and channels != expected:
                raise ValueError("Inconsistent channel order across ensemble members")
            expected = channels
            if len(names)!=64 or len(set(names))!=64 or set(names)!=set(channels):
                raise ValueError("Expected the model's 64 unique EEG channel names")
            order = [names.index(n) for n in channels]
            samples = []
            for x in (closed, opened):
                x = np.asarray(x)
                if x.ndim!=3 or len(x)<1 or x.shape[1:]!=(64,500) or not np.isfinite(x).all():
                    raise ValueError("Each state must contain finite (windows,64,500) EEG in volts")
                indices = np.rint(np.linspace(0,len(x)-1,60)).astype(int)
                samples.append(np.clip(np.asarray(x[indices][:,order,:],dtype=np.float32)*1e5,-20,20))
            model = build_model(bundle["record"]["config"]).to(device).eval()
            model.load_state_dict(bundle["state_dict"])
            value = model(torch.as_tensor(np.stack(samples)[None], device=device))
            normal = bundle["normalization"]
            predictions.append(float(value.cpu()[0])*normal["age_std"]+normal["age_mean"])
    return float(np.mean(predictions))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,default=Path("outputs/final/model"))
    p.add_argument("--closed",type=Path,required=True)
    p.add_argument("--open",dest="opened",type=Path,required=True)
    p.add_argument("--channels",type=Path,required=True,help="JSON list of channel names")
    p.add_argument("--device",default="cpu")
    args = p.parse_args()
    torch.set_num_threads(4)
    value = predict_recording(args.model, np.load(args.closed), np.load(args.opened),
                              json.loads(args.channels.read_text(encoding="utf-8")), device=args.device)
    print(json.dumps(dict(predicted_age_years=value)))


if __name__ == "__main__":
    main()
