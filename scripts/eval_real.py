"""Phase C eval: old Metrica models vs new real-data models on the real TEST sets.

The key honest-generalization number: OLD model test AUC (Metrica-trained,
never saw real data) vs NEW model test AUC (trained on real train split),
per target, on the held-out real test matches.

V1 test: DFL_J03WR9 windows (shot only).
V2 test: DFL_J03WR9 + 4 SK graph windows (shot/box_entry/line_break;
        SK shot labels are -1 -> excluded from shot AUC).

Usage: python eval_real.py
Writes: data/real_runs/eval_comparison.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.value_lstm import ValueLSTM  # noqa: E402
from models.graph_value import GraphValue, TARGETS  # noqa: E402
from train_real import V1_SPLIT, V2_SPLIT, load_split, auc_report  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real_runs"
MODELS = ROOT / "models"

OLD = {"v1": "value_lstm.pt", "v2": "graph_value.pt"}
NEW = {"v1": "value_lstm_real.pt", "v2": "graph_value_real.pt"}


@torch.no_grad()
def predict(model, per_T, mu, sd, device, batch=4096):
    pv_all, y_all = [], []
    model.eval()
    for T, arrs in per_T.items():
        Xs = [torch.from_numpy(((X - mu) / sd).astype(np.float32)) for X, _ in arrs]
        ys = [torch.from_numpy(y.astype(np.float32)) for _, y in arrs]
        X = torch.cat(Xs)
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(device)
            pv_all.append(torch.sigmoid(model(xb)).cpu().numpy())
        y_all.append(torch.cat(ys).numpy())
    return np.concatenate(pv_all), np.concatenate(y_all)


def eval_one(model_key, ckpt_file):
    ckpt = torch.load(MODELS / ckpt_file, map_location="cpu", weights_only=False)
    split = V1_SPLIT if model_key == "v1" else V2_SPLIT
    per_T, info = load_split(model_key, split["test"])
    model = (ValueLSTM() if model_key == "v1" else GraphValue()).to(device)
    model.load_state_dict(ckpt["model"])
    mu, sd = ckpt["mu"], ckpt["sd"]
    pv, y = predict(model, per_T, mu, sd, device)
    targets = ["shot"] if model_key == "v1" else TARGETS
    aucs = auc_report(y, pv, targets)
    n = {t: int((y[:, i] >= 0).sum()) if y.ndim > 1 else int((y >= 0).sum())
         for i, t in enumerate(targets)}
    pos = {}
    for i, t in enumerate(targets):
        yt = y[:, i] if y.ndim > 1 else y
        known = yt >= 0
        pos[t] = round(float(yt[known].mean()), 4) if known.any() else None
    return {"aucs": aucs, "n_known": n, "pos_rate": pos,
            "n_windows": len(y), "test_games": split["test"]}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {}
    for key in ("v1", "v2"):
        old = eval_one(key, OLD[key])
        new = eval_one(key, NEW[key])
        delta = {t: (round(new["aucs"][t] - old["aucs"][t], 4)
                     if new["aucs"][t] is not None and old["aucs"][t] is not None
                     else None)
                 for t in old["aucs"]}
        report[key] = {"old_metrica_model": old, "new_real_model": new,
                       "delta_new_minus_old": delta}
        print(f"=== {key.upper()} (test games: {new['test_games']}) ===")
        for t in old["aucs"]:
            o, n_, d = old["aucs"][t], new["aucs"][t], delta[t]
            os = f"{o:.3f}" if o is not None else "n/a"
            ns = f"{n_:.3f}" if n_ is not None else "n/a"
            ds = f"{d:+.3f}" if d is not None else "n/a"
            print(f"  {t:10s} old={os}  new={ns}  delta={ds}  "
                  f"(n_known={new['n_known'][t]}, pos_rate={new['pos_rate'][t]})")
    with open(REAL / "eval_comparison.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {REAL / 'eval_comparison.json'}")
