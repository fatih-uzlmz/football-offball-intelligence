"""Score every detected run: OffBallValue = V(after) - V(before).

For each run, build the same 3s feature windows the value model
was trained on (75 frames @25fps / 30 @10fps — see scripts/fps.py):
one ending at the run's start (before), one ending at its end (after).
Features are computed vectorized per possession.

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
from fps import load_fps, win_frames

ROOT = Path(__file__).resolve().parents[1]
L, W = 105.0, 68.0
FPS = 25.0
WIN = 75


def possession_arrays(seg: pd.DataFrame, team: str, adir: int):
    """Raw per-player coordinate arrays for one possession segment.

    Returns (frames, A, D, B, Am, Dm, Bm, att_ids, def_ids):
      frames  : (T,) sorted frame numbers
      A/D     : (T, n_players, 2) normalized coords, attack-flipped if adir=-1
      B       : (T, 2) normalized ball coords, attack-flipped if adir=-1
      Am/Dm/Bm: same in meters (x*L, y*W)
      att_ids/def_ids: player_id ordering matching A's / D's second axis
    """
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
    return frames, A, D, B, Am, Dm, Bm, att_ids, def_ids


def features_from_arrays(frames, A, D, B, Am, Dm, Bm, fps=FPS):
    """Vectorized 11-feature matrix (T, 11) from possession_arrays output."""
    T = len(frames)
    F = np.full((T, 11), np.nan)
    F[:, 0] = B[:, 0]
    F[:, 1] = B[:, 1]
    dt = 1 / fps
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


def possession_features(seg: pd.DataFrame, team: str, adir: int, fps=FPS):
    """Vectorized 11-feature matrix (T, 11) for one possession segment."""
    frames, A, D, B, Am, Dm, Bm, _, _ = possession_arrays(seg, team, adir)
    return features_from_arrays(frames, A, D, B, Am, Dm, Bm, fps=fps)


def score_game(game, model, mu, sd, device, out_path=None):
    fps = load_fps(game)
    win = win_frames(fps)
    df = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}.parquet")
    poss = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_possessions.parquet")
    runs = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_runs.parquet")
    pmap = df[["frame", "period"]].drop_duplicates().set_index("frame")["period"]
    adirs = attack_dirs(df, poss)
    game_frames = np.sort(df["frame"].unique())
    vb, va, crosses = [], [], []
    with torch.no_grad():
        for pid, g in runs.groupby("possession_id"):
            p = poss.loc[poss["possession_id"] == pid].iloc[0]
            team = p["team"]
            period = int(pmap.get(p["start_frame"], 1))
            adir = adirs.get((period, team), 1)
            # Extend the segment backward by a full window so runs starting
            # near a possession boundary still get `win` observed frames. (This
            # fixes the "window padding" NaNs: previously the window was cut
            # at the possession edge.) Caveat: extended windows can include
            # the possession change itself, which training windows never
            # contained — noted in the reports.
            start_pos = int(np.searchsorted(game_frames, p["start_frame"]))
            lo = game_frames[max(0, start_pos - (win - 1))]
            seg = df[(df["frame"] >= lo) & (df["frame"] <= p["end_frame"])]
            frames, F = possession_features(seg, team, adir, fps=fps)
            fpos = {f: i for i, f in enumerate(frames)}
            for _, r in g.iterrows():
                vals = []
                # Flag runs whose before-window reaches into the previous
                # possession: value_before then mixes the turnover with the
                # run, so value_gained overstates the run's own contribution
                # (the V2 counterfactual gain is the cleaner number there).
                cross = False
                for k, f in enumerate((r["start_frame"], r["end_frame"])):
                    i = fpos.get(f)
                    if i is None or i < win - 1:
                        vals.append(np.nan)
                        continue
                    if k == 0 and frames[i - win + 1] < p["start_frame"]:
                        cross = True
                    win_ = F[i - win + 1:i + 1]
                    if np.isnan(win_).any():
                        vals.append(np.nan)
                        continue
                    x = torch.tensor((win_ - mu) / sd, dtype=torch.float32).unsqueeze(0).to(device)
                    vals.append(float(torch.sigmoid(model(x)).item()))
                vb.append(vals[0])
                va.append(vals[1])
                crosses.append(cross)
    runs["value_before"] = vb
    runs["value_after"] = va
    runs["value_gained"] = runs["value_after"] - runs["value_before"]
    runs["before_crosses_boundary"] = crosses
    out = Path(out_path) if out_path else ROOT / "data" / "trajectories" / f"{game}_runs_scored.parquet"
    runs.to_parquet(out, index=False)
    scored = runs.dropna(subset=["value_gained"])
    print(f"{game}: fps={fps:g} win={win} scored {len(scored)}/{len(runs)}  "
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
    cols = ["runner", "team", "duration_s", "path_m", "defender_displacement_m",
            "value_before", "value_after", "value_gained",
            "before_crosses_boundary", "shot_within_5s"]
    print("\ntop 5 runs by value gained (all):")
    top = runs.nlargest(5, "value_gained")
    print(top[cols].round(3).to_string(index=False))
    clean = runs[~runs["before_crosses_boundary"]]
    print("\ntop 5 runs by value gained (before-window inside possession):")
    topc = clean.nlargest(5, "value_gained")
    print(topc[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+",
                    default=["Sample_Game_1", "Sample_Game_2", "Sample_Game_3"])
    main(**vars(ap.parse_args()))
