"""Train the graph-attention value model (V2).

Windows from all games, split by (game, possession_id) — no leakage.
Loss: masked BCEWithLogitsLoss with per-target pos_weight
      (labels of -1 = unknown horizon -> excluded from that target's loss).
Metric: ROC AUC per target on held-out possessions.
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
from models.graph_value import GraphValue, TARGETS

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ["x_att", "y", "vx_att", "vy", "speed",
                 "dist_to_ball", "dist_to_goal", "team"]


def load(games):
    Xs, ys, ms = [], [], []
    for g in games:
        z = np.load(ROOT / "data" / "trajectories" / f"{g}_graph_windows.npz",
                    allow_pickle=True)
        Xs.append(z["X"])
        ys.append(z["y"])
        ms.append([(g, pid, team, t) for pid, team, t in z["meta"]])
    return (np.concatenate(Xs), np.concatenate(ys),
            [m for mm in ms for m in mm])


def masked_bce(logits, y):
    """Mean BCE over known labels only, with per-target pos_weight."""
    mask = (y >= 0).float()
    pw = []
    for i in range(y.shape[1]):
        known = y[:, i] >= 0
        rate = y[known, i].mean() if known.any() else 0.5
        pw.append((1 - rate) / max(rate, 1e-6))
    pw = torch.tensor(pw, device=logits.device)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, y.clamp(0, 1), pos_weight=pw, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def main(games, epochs=15, batch=128, lr=1e-3):
    X, y, meta = load(games)
    print(f"windows={len(y)}  X{X.shape}")
    for i, t in enumerate(TARGETS):
        known = y[:, i] >= 0
        print(f"  {t}: known={known.mean():.2%}  rate={y[known, i].mean():.4f}")

    keys = np.array([f"{g}_{pid}" for g, pid, _, _ in meta])
    uniq = np.unique(keys)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    n_val = max(1, int(0.2 * len(uniq)))
    val_keys = set(uniq[:n_val])
    val_mask = np.array([k in val_keys for k in keys])
    Xtr, ytr = X[~val_mask], y[~val_mask]
    Xva, yva = X[val_mask], y[val_mask]
    mu, sd = Xtr.mean((0, 1, 2)), Xtr.std((0, 1, 2)) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # from_numpy shares memory (no copy); drop the big arrays right after
    train_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                          batch_size=batch, shuffle=True)
    Xva_t = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva).to(device)
    del X, y, Xtr, Xva, ytr  # keep yva (numpy) for the eval loop

    model = GraphValue().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}  device={device}")

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    best = {"shot": 0.0}
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = masked_bce(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva_t)).cpu().numpy()
        aucs = []
        auc_vals = {}
        for i, t in enumerate(TARGETS):
            known = yva[:, i] >= 0
            if known.sum() > 10 and yva[known, i].std() > 0:
                a = roc_auc_score(yva[known, i], pv[known, i])
                auc_vals[t] = a
                aucs.append(f"{t}={a:.3f}")
            else:
                aucs.append(f"{t}=n/a")
        print(f"ep {ep:2d}  train_loss={tot/n:.4f}  " + "  ".join(aucs), flush=True)
        # checkpoint every epoch so a killed run is resumable (VM replacements happen)
        ckpt = {"model": model.state_dict(), "opt": opt.state_dict(),
                "mu": mu, "sd": sd, "features": FEATURE_NAMES,
                "targets": TARGETS, "epoch": ep, "aucs": auc_vals}
        torch.save(ckpt, out / "graph_value_last.pt")
        if auc_vals.get("shot", 0) > best["shot"]:
            best = {"shot": auc_vals["shot"], "epoch": ep}
            torch.save(ckpt, out / "graph_value.pt")

    print(f"saved {out / 'graph_value.pt'} (best shot={best['shot']:.3f} @ ep {best['epoch']}) "
          f"+ {out / 'graph_value_last.pt'} (last epoch)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+",
                    default=["Sample_Game_1", "Sample_Game_2", "Sample_Game_3"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    main(**vars(ap.parse_args()))
