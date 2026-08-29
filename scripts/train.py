"""Train the value model.

Windows from all games, split by possession_id (no leakage).
Loss: BCEWithLogitsLoss with pos_weight for the rare shot class.
Metric: ROC AUC on held-out possessions.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.value_lstm import ValueLSTM

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ["ball_x_att", "ball_y", "ball_speed", "ball_dist_goal",
                 "ball_vx_att", "n_att_ahead", "att_cx", "def_cx",
                 "compactness", "pressure", "n_att_in_box"]


def load(games):
    Xs, ys, ms = [], [], []
    for g in games:
        z = np.load(ROOT / "data" / "trajectories" / f"{g}_windows.npz",
                    allow_pickle=True)
        Xs.append(z["X"])
        ys.append(z["y"])
        ms.append([(g, pid, team, t) for pid, team, t in z["meta"]])
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    meta = [m for mm in ms for m in mm]
    # normalize with train stats later; standardize here on the fly per split
    return X, y, meta


def main(games, epochs=15, batch=256, lr=1e-3):
    X, y, meta = load(games)
    print(f"windows={len(y)}  shot_rate={y.mean():.4f}")
    # split by (game, possession_id)
    keys = np.array([f"{g}_{pid}" for g, pid, _, _ in meta])
    uniq = np.unique(keys)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    n_val = max(1, int(0.2 * len(uniq)))
    val_keys = set(uniq[:n_val])
    val_mask = np.array([k in val_keys for k in keys])
    Xtr, ytr = X[~val_mask], y[~val_mask]
    Xva, yva = X[val_mask], y[val_mask]
    mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                          batch_size=batch, shuffle=True)
    Xva_t, yva_t = torch.tensor(Xva).to(device), torch.tensor(yva).to(device)

    model = ValueLSTM().to(device)
    pos_weight = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva_t)).cpu().numpy()
        auc = roc_auc_score(yva, pv)
        print(f"ep {ep:2d}  train_loss={tot/n:.4f}  val_auc={auc:.3f}")

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd,
                "features": FEATURE_NAMES}, out / "value_lstm.pt")
    print(f"saved {out / 'value_lstm.pt'}  final val AUC={auc:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+",
                    default=["Sample_Game_1", "Sample_Game_2", "Sample_Game_3"])
    ap.add_argument("--epochs", type=int, default=15)
    main(**vars(ap.parse_args()))
