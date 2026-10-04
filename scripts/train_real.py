"""Phase C: retrain value models on real-data match-level splits (no leakage).

Mixed-T handling (additive design decision, 2026-10-04):
  DFL windows are T=75 (@25fps), SK windows are T=30 (@10fps). Instead of
  padding + pack_padded_sequence (which would need a `lengths` parameter
  threaded through both model forwards), we batch BY fps: one DataLoader
  per T-group, training batches alternate between groups. Both models are
  T-agnostic (final LSTM hidden state), and mu/sd normalization is
  per-feature so T-independent. No model code changed.

Splits (match level; every split contains DFL matches since only DFL has
shot labels):
  V1 (DFL only, shot target): train=5 (J03WMX,J03WN1,J03WOH,J03WOY,J03WPY),
      val=J03WQQ, test=J03WR9
  V2 (all 27): train=5 DFL + first 13 SK ids, val=J03WQQ + next 3 SK,
      test=J03WR9 + last 4 SK ids (SK ids sorted ascending)

V2 trains all 3 heads: shot head effectively DFL-only (SK shot=-1 masked via
masked_bce), box_entry/line_break heads on all 27 matches.

Outputs: models/value_lstm_real.pt / models/graph_value_real.pt (+ mu/sd,
split ids, per-epoch curves). NEVER touches value_lstm.pt / graph_value.pt.

Usage: python train_real.py --model v1|v2 [--epochs N] [--batch N]
"""
import argparse
import bisect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_graph import masked_bce  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.value_lstm import ValueLSTM  # noqa: E402
from models.graph_value import GraphValue, TARGETS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real_runs"

V1_SPLIT = {
    "train": ["DFL_J03WMX", "DFL_J03WN1", "DFL_J03WOH", "DFL_J03WOY", "DFL_J03WPY"],
    "val": ["DFL_J03WQQ"],
    "test": ["DFL_J03WR9"],
}
SK_ALL = ["SK_1874553", "SK_1886347", "SK_1899585", "SK_1925299", "SK_1927964",
          "SK_1953632", "SK_1959846", "SK_1986691", "SK_1996435", "SK_1996436",
          "SK_2006229", "SK_2006363", "SK_2007448", "SK_2007721", "SK_2010085",
          "SK_2011166", "SK_2013725", "SK_2015213", "SK_2016236", "SK_2017461"]
V2_SPLIT = {
    "train": V1_SPLIT["train"] + SK_ALL[:13],
    "val": V1_SPLIT["val"] + SK_ALL[13:16],
    "test": V1_SPLIT["test"] + SK_ALL[16:20],
}
SPLITS = {"v1": V1_SPLIT, "v2": V2_SPLIT}
WINDOWS_FILE = {"v1": "windows", "v2": "graph_windows"}


class GameListDataset(Dataset):
    """Indexes a list of (X, y) arrays without concatenating them.

    Normalization is applied per-item in __getitem__ (not pre-copied):
    pre-copying doubles ~2GB of window arrays and OOMs the 7GB VM.
    """

    def __init__(self, arrays, mu, sd):
        self.arrays = [(torch.from_numpy(X), torch.from_numpy(y))
                       for X, y in arrays]
        self.mu = torch.from_numpy(mu)
        self.sd = torch.from_numpy(sd)
        sizes = [len(X) for X, _ in self.arrays]
        self.cum = np.cumsum([0] + sizes)

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, i):
        g = bisect.bisect_right(self.cum, i) - 1
        j = i - self.cum[g]
        X, y = self.arrays[g]
        return (X[j] - self.mu) / self.sd, y[j]


def load_split(model, split_ids):
    """-> dict T -> (list of (X, y) arrays), plus per-game meta counts."""
    per_T, info = {}, {}
    for g in split_ids:
        z = np.load(REAL / f"{g}_{WINDOWS_FILE[model]}.npz", allow_pickle=True)
        X, y = z["X"].astype(np.float32), z["y"].astype(np.float32)
        T = X.shape[1]
        per_T.setdefault(T, []).append((X, y))
        info[g] = {"windows": len(y), "T": T}
    return per_T, info


def auc_report(y_true, pv, targets):
    out = {}
    for i, t in enumerate(targets):
        yt = y_true[:, i] if y_true.ndim > 1 else y_true
        pv_ = pv[:, i] if pv.ndim > 1 else pv
        known = yt >= 0
        if known.sum() > 10 and yt[known].std() > 0:
            out[t] = float(roc_auc_score(yt[known], pv_[known]))
        else:
            out[t] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v1", "v2"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    batch = args.batch or (256 if args.model == "v1" else 128)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    split = SPLITS[args.model]
    print(f"split: " + json.dumps({k: len(v) for k, v in split.items()}), flush=True)
    tr_T, tr_info = load_split(args.model, split["train"])
    va_T, va_info = load_split(args.model, split["val"])
    print("train games:", json.dumps(tr_info), flush=True)
    print("val games:", json.dumps(va_info), flush=True)
    n_tr = sum(len(y) for arrs in tr_T.values() for _, y in arrs)
    print(f"train windows={n_tr}  T-groups={sorted(tr_T)}", flush=True)

    # normalization stats over ALL train windows (per-feature, T-independent).
    # Mixed T means we can't concatenate: accumulate sum/sum-sq per group.
    # IMPORTANT: accumulate in float64 (float32 summation over ~1e8 elements
    # loses all precision), and divide by the total ELEMENT count per feature
    # (N windows x T frames x P players), not the window count.
    axes = (0, 1) if args.model == "v1" else (0, 1, 2)
    tot_elem, sum_x, sum_x2 = 0, 0.0, 0.0
    for arrs in tr_T.values():
        for X, _ in arrs:
            X64 = X.astype(np.float64)
            tot_elem += X64.shape[0] * int(np.prod(X64.shape[1:-1]))
            sum_x = sum_x + X64.sum(axis=axes)
            sum_x2 = sum_x2 + (X64 ** 2).sum(axis=axes)
    mu = (sum_x / tot_elem).astype(np.float32)
    var = sum_x2 / tot_elem - (sum_x / tot_elem) ** 2
    sd = (np.sqrt(np.maximum(var, 0)) + 1e-6).astype(np.float32)

    tr_loaders = {T: DataLoader(GameListDataset(arrs, mu, sd),
                               batch_size=(args.batch or (64 if T == 75 else 128)),
                               shuffle=True)
                  for T, arrs in tr_T.items()}
    # Val is evaluated streaming per game (arrays loaded, scored, discarded)
    # to keep peak RAM lean on the 7GB VM. Re-derive per-game T from the
    # already-loaded val arrays' game order.
    va_game_ids = split["val"]
    del va_T  # free the val arrays; reloaded per game in the eval loop
    import gc; gc.collect()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (ValueLSTM() if args.model == "v1" else GraphValue()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}  device={device}  batch={batch}", flush=True)

    if args.model == "v1":
        ytr = np.concatenate([y for arrs in tr_T.values() for _, y in arrs])
        pw = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)],
                          device=device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        loss_fn = lambda logits, yb: crit(logits, yb)  # noqa: E731
        targets = ["shot"]
    else:
        loss_fn = masked_bce
        targets = TARGETS

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    curves = []
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    ckpt_name = "value_lstm_real.pt" if args.model == "v1" else "graph_value_real.pt"

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        # alternate homogeneous-T batches (shuffle group order each epoch)
        order = sorted(tr_loaders)
        rng.shuffle(order)
        iters = {T: iter(dl) for T, dl in tr_loaders.items()}
        active = list(order)
        while active:
            for T in list(active):
                try:
                    xb, yb = next(iters[T])
                except StopIteration:
                    active.remove(T)
                    continue
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
                n += len(xb)
        model.eval()
        pv_all, y_all = [], []
        with torch.no_grad():
            # streaming val: one game at a time, small forward chunks
            for g in va_game_ids:
                z = np.load(REAL / f"{g}_{WINDOWS_FILE[args.model]}.npz",
                            allow_pickle=True)
                Xg = ((z["X"].astype(np.float32) - mu) / sd).astype(np.float32)
                yg = z["y"].astype(np.float32)
                del z
                Xt = torch.from_numpy(Xg)
                for i in range(0, len(Xt), 512):
                    pv_all.append(torch.sigmoid(
                        model(Xt[i:i + 512].to(device))).cpu().numpy())
                y_all.append(yg)
                del Xg, yg, Xt
        pv_all = np.concatenate(pv_all)
        y_all = np.concatenate(y_all)
        aucs = auc_report(y_all, pv_all, targets)
        ep_rec = {"epoch": ep, "train_loss": round(tot / max(n, 1), 4),
                  "val_auc": aucs, "s": round(time.time() - t0, 1)}
        curves.append(ep_rec)
        rss_mb = int(open("/proc/self/status").read().split("VmRSS:")[1]
                     .split()[0]) // 1024
        print(f"ep {ep:2d}  train_loss={ep_rec['train_loss']:.4f}  " +
              "  ".join(f"val_{k}={v:.3f}" if v else f"val_{k}=n/a"
                        for k, v in aucs.items()) +
              f"  ({ep_rec['s']}s, rss={rss_mb}MB)", flush=True)
        torch.save({"model": model.state_dict(), "mu": mu, "sd": sd,
                    "split": split, "curves": curves, "epoch": ep},
                   out / ckpt_name.replace(".pt", "_last.pt"))
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd,
                "split": split, "curves": curves, "epoch": args.epochs,
                "targets": targets}, out / ckpt_name)
    print(f"saved {out / ckpt_name}", flush=True)
    with open(REAL / f"train_{args.model}_curves.json", "w") as f:
        json.dump({"split": split, "curves": curves}, f, indent=1)


if __name__ == "__main__":
    main()
