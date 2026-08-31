"""Score every detected run: OffBallValue = V(after) - V(before).

For each run, build the same 75-frame feature windows the value model
was trained on: one ending at the run's start (before), one ending at
its end (after). Features are computed vectorized per possession.

Output: data/trajectories/<game>_runs_scored.parquet with
        value_before, value_after, value_gained columns.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from models.value_lstm import ValueLSTM
from build_windows import attack_dirs, BOX_X, BOX_Y

ROOT = Path(__file__).resolve().parents[1]
L, W = 105.0, 68.0
FPS = 25.0
WIN = 75


def possession_features(seg: pd.DataFrame, team: str, adir: int):
    """Vectorized 11-feature matrix (T, 11) for one possession segment."""
    opp = "away" if team == "home" else "home"
    frames = np.sort(seg["frame"].unique())
    T = len(frames)
    fidx = {f: i for i, f in enumerate(frames)}
    att_ids = sorted(seg[seg["team"] == team]["player_id"].unique())
    def_ids = sorted(seg[seg["team"] == opp]["player_id"].unique())
    na, nd = len(att_ids), len(def_ids)
    A = np.full((T, na, 2), np.nan)
    D = np.full((T, nd, 2), np.nan)
    B = np.full((T, 2), np.nan)
    aidx = {p: i for i, p in enumerate(att_ids)}
    didx = {p: i for i, p in enumerate(def_ids)}
    for _, r in seg.iterrows():
        i = fidx[r["frame"]]
        xy = np.array([r["x"], r["y"]])
        if r["team"] == team:
            A[i, aidx[r["player_id"]]] = xy
        elif r["team"] == opp:
            D[i, didx[r["player_id"]]] = xy
        else:
            B[i] = xy
    # attack-normalize x
    if adir == -1:
        A[:, :, 0] = 1 - A[:, :, 0]
        D[:, :, 0] = 1 - D[:, :, 0]
        B[:, 0] = 1 - B[:, 0]
    Am = A * [L, W]
    Dm = D * [L, W]
    Bm = B * [L, W]
    F = np.full((T, 11), np.nan)
    F[:, 0] = B[:, 0]
    F[:, 1] = B[:, 1]
    dt = 1 / FPS
    dB = np.diff(Bm, axis=0) / dt
    F[1:, 2] = np.linalg.norm(np.where(np.isnan(dB), 0, dB), axis=1)
    F[:, 2] = np.where(np.isnan(F[:, 2]), 0, F[:, 2])
    F[:, 3] = np.hypot((1 - B[:, 0]) * L, (B[:, 1] - 0.5) * W)
    F[1:, 4] = dB[:, 0]
    F[:, 4] = np.where(np.isnan(F[:, 4]), 0, F[:, 4])
    ax = A[:, :, 0]
    F[:, 5] = np.nansum(ax > B[:, 0:1], axis=1)
    F[:, 6] = np.nanmean(ax, axis=1)
    F[:, 7] = np.nanmean(D[:, :, 0], axis=1)
    # compactness + pressure via broadcast distances
    diff = Am[:, :, None, :] - Dm[:, None, :, :]
    dist = np.sqrt(np.nansum(diff ** 2, axis=3))
    dist = np.where(np.isnan(dist), np.inf, dist)
    with np.errstate(all="ignore"):
        F[:, 8] = np.nanmean(np.min(dist, axis=2), axis=1)
        F[:, 9] = np.min(np.sqrt(np.nansum((Dm - Bm[:, None, :]) ** 2, axis=2)), axis=1)
    F[:, 9] = np.where(np.isinf(F[:, 9]), np.nan, F[:, 9])
    F[:, 10] = np.nansum((ax > 1 - BOX_X) & (np.abs(A[:, :, 1] - 0.5) < BOX_Y / 2), axis=1)
    F[~np.isfinite(F[:, 0]), :] = np.nan  # no ball -> invalid
    return frames, F


def score_game(game, model, mu, sd, device):
    df = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}.parquet")
    poss = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_possessions.parquet")
    runs = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_runs.parquet")
    pmap = df[["frame", "period"]].drop_duplicates().set_index("frame")["period"]
    adirs = attack_dirs(df, poss)
    vb, va = [], []
    with torch.no_grad():
        for pid, g in runs.groupby("possession_id"):
            p = poss.loc[poss["possession_id"] == pid].iloc[0]
            team = p["team"]
            period = int(pmap.get(p["start_frame"], 1))
            adir = adirs.get((period, team), 1)
            seg = df[(df["frame"] >= p["start_frame"]) & (df["frame"] <= p["end_frame"])]
            frames, F = possession_features(seg, team, adir)
            fpos = {f: i for i, f in enumerate(frames)}
            for _, r in g.iterrows():
                vals = []
                for f in (r["start_frame"], r["end_frame"]):
                    i = fpos.get(f)
                    if i is None or i < WIN - 1:
                        vals.append(np.nan)
                        continue
                    win = F[i - WIN + 1:i + 1]
                    if np.isnan(win).any():
                        vals.append(np.nan)
                        continue
                    x = torch.tensor((win - mu) / sd, dtype=torch.float32).unsqueeze(0).to(device)
                    vals.append(float(torch.sigmoid(model(x)).item()))
                vb.append(vals[0])
                va.append(vals[1])
    runs["value_before"] = vb
    runs["value_after"] = va
    runs["value_gained"] = runs["value_after"] - runs["value_before"]
    out = ROOT / "data" / "trajectories" / f"{game}_runs_scored.parquet"
    runs.to_parquet(out, index=False)
    scored = runs.dropna(subset=["value_gained"])
    print(f"{game}: scored {len(scored)}/{len(runs)}  "
          f"mean_gain={scored['value_gained'].mean():.4f}")
    return scored


def main(games):
    ckpt = torch.load(ROOT / "models" / "value_lstm.pt", map_location="cpu",
                      weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ValueLSTM().to(device).eval()
    model.load_state_dict(ckpt["model"])
    mu, sd = ckpt["mu"], ckpt["sd"]
    all_runs = []
    for g in games:
        all_runs.append(score_game(g, model, mu, sd, device))
    runs = pd.concat(all_runs, ignore_index=True)
    top = runs.nlargest(5, "value_gained")
    print("\ntop 5 runs by value gained:")
    print(top[["runner", "team", "duration_s", "path_m",
               "defender_displacement_m", "value_before", "value_after",
               "value_gained", "shot_within_5s"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+",
                    default=["Sample_Game_1", "Sample_Game_2", "Sample_Game_3"])
    main(**vars(ap.parse_args()))
