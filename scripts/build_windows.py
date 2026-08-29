"""Build labeled training windows for the value model.

For each possession, slide a 3s window (75 frames @25fps, stride 1s).
Features per frame (attack-normalized: x_att=1 is the opponent goal):
  ball_x_att, ball_y, ball_speed, ball_dist_goal, ball_vx_att,
  n_attackers_ahead_of_ball, att_centroid_x, def_centroid_x,
  compactness (mean attacker min-dist to a defender),
  pressure (min defender dist to ball), n_attackers_in_box
Label: 1 if the possession team takes a SHOT within 5s after window end.

Output: data/trajectories/<game>_windows.npz (X [N,75,11], y [N], meta)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
L, W = 105.0, 68.0
FPS = 25.0
WIN, STRIDE = 75, 25
AFTER_S = 5.0
BOX_X = 16.5 / L
BOX_Y = 40.32 / W


def attack_dirs(df, poss):
    ball = df[df["team"] == "ball"].set_index("frame")["x"]
    pmap = df[["frame", "period"]].drop_duplicates().set_index("frame")["period"]
    dirs = {}
    for (period, team), g in poss.groupby([poss["start_frame"].map(pmap), "team"]):
        dx = (g["end_frame"].map(ball) - g["start_frame"].map(ball)).mean()
        dirs[(int(period), team)] = 1 if dx >= 0 else -1
    return dirs


def frame_features(fr, team, opp, adir):
    b = fr[fr["team"] == "ball"]
    if b.empty:
        return None
    bx, by = float(b["x"].iloc[0]), float(b["y"].iloc[0])
    bx_att = bx if adir == 1 else 1 - bx
    A = fr[fr["team"] == team][["x", "y"]].to_numpy()
    D = fr[fr["team"] == opp][["x", "y"]].to_numpy()
    if len(A) == 0 or len(D) == 0:
        return None
    ax_ = A[:, 0] if adir == 1 else 1 - A[:, 0]
    dx_ = D[:, 0] if adir == 1 else 1 - D[:, 0]
    Am = np.stack([ax_ * L, A[:, 1] * W], 1)
    Dm = np.stack([dx_ * L, D[:, 1] * W], 1)
    bm = np.array([bx_att * L, by * W])
    d_ab = np.linalg.norm(Am[:, None, :] - Dm[None, :, :], axis=2)
    return [
        bx_att, by,
        float(np.linalg.norm(bm)),  # placeholder replaced below
        float(np.hypot((1 - bx_att) * L, (by - 0.5) * W)),
        0.0,  # ball_vx_att filled by caller
        float((ax_ > bx_att).sum()),
        float(ax_.mean()), float(dx_.mean()),
        float(d_ab.min(1).mean()),
        float(np.linalg.norm(Dm - bm, axis=1).min()),
        float(((ax_ > 1 - BOX_X) & (np.abs(A[:, 1] - 0.5) < BOX_Y / 2)).sum()),
    ]


def build(game):
    df = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}.parquet")
    poss = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_possessions.parquet")
    ev = pd.read_csv(ROOT / "data" / "raw" / f"{game}_RawEventsData.csv")
    shots = ev[ev["Type"] == "SHOT"].copy()
    shots["team_norm"] = shots["Team"].str.lower()
    adirs = attack_dirs(df, poss)
    pmap = df[["frame", "period"]].drop_duplicates().set_index("frame")["period"]
    X, y, meta = [], [], []
    for _, p in poss.iterrows():
        team, opp = p["team"], ("away" if p["team"] == "home" else "home")
        period = int(pmap.get(p["start_frame"], 1))
        adir = adirs.get((period, team), 1)
        seg = df[(df["frame"] >= p["start_frame"]) & (df["frame"] <= p["end_frame"])]
        frames = sorted(seg["frame"].unique())
        fmap = {f: seg[seg["frame"] == f] for f in frames}
        # per-frame ball vx (attack dir)
        bxs = {f: fmap[f][fmap[f]["team"] == "ball"]["x"] for f in frames}
        for s in range(0, len(frames) - WIN + 1, STRIDE):
            wf = frames[s:s + WIN]
            feats = []
            ok = True
            for j, f in enumerate(wf):
                row = frame_features(fmap[f], team, opp, adir)
                if row is None:
                    ok = False
                    break
                # ball velocity toward goal
                if j > 0:
                    f0 = wf[j - 1]
                    x0 = bxs[f0].iloc[0] if len(bxs[f0]) else np.nan
                    x1 = bxs[f].iloc[0] if len(bxs[f]) else np.nan
                    dt = 1 / FPS
                    vx = ((x1 if adir == 1 else 1 - x1) - (x0 if adir == 1 else 1 - x0)) * L / dt \
                        if not (np.isnan(x0) or np.isnan(x1)) else 0.0
                    row[4] = vx
                    # ball speed
                    y0 = fmap[f0][fmap[f0]["team"] == "ball"]["y"]
                    y1 = fmap[f][fmap[f]["team"] == "ball"]["y"]
                    if len(y0) and len(y1):
                        row[2] = float(np.hypot((x1 - x0) * L, (float(y1.iloc[0]) - float(y0.iloc[0])) * W) / dt)
                feats.append(row)
            if not ok:
                continue
            end_time = float(seg[seg["frame"] == wf[-1]]["time"].iloc[0])
            label = int(((shots["team_norm"] == team) &
                         (shots["Start Time [s]"] > end_time) &
                         (shots["Start Time [s]"] <= end_time + AFTER_S)).any())
            X.append(feats)
            y.append(label)
            meta.append((int(p["possession_id"]), team, end_time))
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    out = ROOT / "data" / "trajectories" / f"{game}_windows.npz"
    np.savez_compressed(out, X=X, y=y, meta=np.array(meta, dtype=object))
    print(f"{game}: windows={len(y)}  shot_rate={y.mean():.3f} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    build(**vars(ap.parse_args()))
