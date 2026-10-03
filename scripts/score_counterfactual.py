"""Counterfactual run scoring: "what if he hadn't made the run?"

For each value-scored run, rebuild the possession's trajectory segment with
the runner replaced by a counterfactual baseline, recompute the 11
value-model features with the *same* pipeline as score_runs.py, and evaluate
the trained LSTM on the 75-frame window ending at the run's end:

    cf_gain = V(actual end) - V(counterfactual end)

This isolates the run's contribution: before/after scoring compares two
different moments in time; counterfactual scoring compares two versions of
the same moment.

Variants:
  frozen: runner stands still at his run-start position for
          (start_frame, end_frame]. "What if he never made the run?"
  drift:  runner continues with his mean pre-run velocity (1 s before
          start_frame), extrapolated and clipped to the pitch.
          "What if he kept jogging instead of sprinting?"

Output: data/trajectories/<game>_runs_counterfactual.parquet with
        value_cf_frozen, cf_gain_frozen, value_cf_drift, cf_gain_drift.

Caveats (read before citing numbers):
  - The value model's 11 features are pooled over all attackers, so one
    player's trajectory moves only attackers-ahead-of-ball, attacker
    centroid x, compactness, and attackers-in-box. cf_gain is the run's
    effect *through the model's pooled state* — a conservative lower bound,
    not the full causal effect.
  - Defenders do NOT react in the counterfactual world; they keep their
    observed trajectories. In reality a defender follows a runner, so the
    true value difference is likely larger, not smaller.
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
from build_windows import attack_dirs
from score_runs import features_from_arrays, possession_arrays

ROOT = Path(__file__).resolve().parents[1]
L, W = 105.0, 68.0
FPS = 25.0
WIN = 75
PRE_S = 1.0  # seconds of pre-run motion used for the drift baseline
VARIANTS = ("frozen", "drift")


def counterfactual_segment(seg: pd.DataFrame, runner: int, team: str,
                           start_frame: int, end_frame: int,
                           variant: str) -> pd.DataFrame:
    """Copy of `seg` with the runner's post-start trajectory replaced.

    Frames <= start_frame are never touched.
    """
    assert variant in VARIANTS, variant
    out = seg.copy()
    # player_id dtype differs between tables (str in trajectories, int in
    # runs) — compare as strings so the match can't silently miss.
    pid = out["player_id"].astype(str) == str(runner)
    r = out[(out["team"] == team) & pid].sort_values("frame")
    at = r[r["frame"] == start_frame]
    if at.empty:  # runner not observed exactly at start; take first frame after
        at = r[r["frame"] > start_frame].head(1)
        if at.empty:
            return out
        start_frame = int(at["frame"].iloc[0])
    x0, y0 = float(at["x"].iloc[0]), float(at["y"].iloc[0])
    t0 = float(at["time"].iloc[0])

    mask = ((out["team"] == team) & pid &
            (out["frame"] > start_frame) & (out["frame"] <= end_frame))
    if not mask.any():
        return out

    if variant == "frozen":
        out.loc[mask, "x"] = x0
        out.loc[mask, "y"] = y0
    else:
        pre = r[(r["frame"] >= start_frame - int(PRE_S * FPS)) &
                (r["frame"] <= start_frame)]
        vx = vy = 0.0
        if len(pre) >= 2:
            dt = float(pre["time"].iloc[-1] - pre["time"].iloc[0])
            if dt > 0:
                vx = float(pre["x"].iloc[-1] - pre["x"].iloc[0]) / dt
                vy = float(pre["y"].iloc[-1] - pre["y"].iloc[0]) / dt
        t = out.loc[mask, "time"].to_numpy()
        out.loc[mask, "x"] = np.clip(x0 + vx * (t - t0), 0, 1)
        out.loc[mask, "y"] = np.clip(y0 + vy * (t - t0), 0, 1)
    return out


def window_value(model, F: np.ndarray, i: int, mu, sd, device) -> float:
    if i < WIN - 1:
        return np.nan
    win = F[i - WIN + 1:i + 1]
    if np.isnan(win).any():
        return np.nan
    x = torch.tensor((win - mu) / sd, dtype=torch.float32).unsqueeze(0).to(device)
    return float(torch.sigmoid(model(x)).item())


def counterfactual_value(model, frames, A, D, B, Am, Dm, Bm, att_ids,
                          runner, team, seg, start_frame, end_frame,
                          variant, adir, mu, sd, device) -> float:
    """Value of the 75-frame window ending at end_frame under a counterfactual.

    The possession arrays are built once per possession; only the runner's
    column is patched with the counterfactual positions (attack-normalized
    the same way possession_arrays does).
    """
    aidx = {p: i for i, p in enumerate(att_ids)}
    ridx = aidx.get(str(runner), aidx.get(runner))
    if ridx is None:
        return np.nan
    cf = counterfactual_segment(seg, runner, team, start_frame, end_frame, variant)
    pid = cf["player_id"].astype(str) == str(runner)
    rows = cf[(cf["team"] == team) & pid &
              (cf["frame"] > start_frame) & (cf["frame"] <= end_frame)]
    if rows.empty:
        return np.nan
    fpos = {f: i for i, f in enumerate(frames)}
    A2 = A.copy()
    Am2 = Am.copy()
    for f, x, y in zip(rows["frame"], rows["x"], rows["y"]):
        i = fpos.get(f)
        if i is None:
            continue
        xn = 1 - x if adir == -1 else x
        A2[i, ridx, 0] = xn
        A2[i, ridx, 1] = y
        Am2[i, ridx, 0] = xn * L
        Am2[i, ridx, 1] = y * W
    _, F = features_from_arrays(frames, A2, D, B, Am2, Dm, Bm)
    return window_value(model, F, fpos.get(end_frame, -1), mu, sd, device)


def score_game(game, model, mu, sd, device, limit=None):
    df = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}.parquet")
    poss = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_possessions.parquet")
    runs = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_runs_scored.parquet")
    runs = runs.dropna(subset=["value_after"]).reset_index(drop=True)
    if limit:
        runs = runs.head(limit)
    pmap = df[["frame", "period"]].drop_duplicates().set_index("frame")["period"]
    adirs = attack_dirs(df, poss)
    cf = {v: [] for v in VARIANTS}
    with torch.no_grad():
        for pid, g in runs.groupby("possession_id"):
            p = poss.loc[poss["possession_id"] == pid].iloc[0]
            team = p["team"]
            period = int(pmap.get(p["start_frame"], 1))
            adir = adirs.get((period, team), 1)
            seg = df[(df["frame"] >= p["start_frame"]) & (df["frame"] <= p["end_frame"])]
            frames, A, D, B, Am, Dm, Bm, att_ids, _ = possession_arrays(seg, team, adir)
            for _, r in g.iterrows():
                for v in VARIANTS:
                    cf[v].append(counterfactual_value(
                        model, frames, A, D, B, Am, Dm, Bm, att_ids,
                        r["runner"], team, seg,
                        int(r["start_frame"]), int(r["end_frame"]),
                        v, adir, mu, sd, device))
    for v in VARIANTS:
        runs[f"value_cf_{v}"] = cf[v]
        runs[f"cf_gain_{v}"] = runs["value_after"] - runs[f"value_cf_{v}"]
    out = ROOT / "data" / "trajectories" / f"{game}_runs_counterfactual.parquet"
    runs.to_parquet(out, index=False)
    s = runs.dropna(subset=["cf_gain_frozen"])
    print(f"{game}: counterfactual-scored {len(s)}/{len(runs)}  "
          f"mean_cf_gain_frozen={s['cf_gain_frozen'].mean():.4f}  "
          f"mean_cf_gain_drift={s['cf_gain_drift'].mean():.4f}")
    return s


def main(games, limit):
    ckpt = torch.load(ROOT / "models" / "value_lstm.pt", map_location="cpu",
                      weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ValueLSTM().to(device).eval()
    model.load_state_dict(ckpt["model"])
    mu, sd = ckpt["mu"], ckpt["sd"]
    all_runs = [score_game(g, model, mu, sd, device, limit) for g in games]
    runs = pd.concat(all_runs, ignore_index=True)
    print("\ntop 5 runs by counterfactual gain (frozen):")
    top = runs.nlargest(5, "cf_gain_frozen")
    print(top[["runner", "team", "duration_s", "path_m", "defender_displacement_m",
               "value_after", "value_cf_frozen", "cf_gain_frozen",
               "value_gained", "shot_within_5s"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+",
                    default=["Sample_Game_1", "Sample_Game_2", "Sample_Game_3"])
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N runs per game (smoke test)")
    a = ap.parse_args()
    main(a.games, a.limit)
