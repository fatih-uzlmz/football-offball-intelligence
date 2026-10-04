"""Counterfactual run scoring with the graph-attention value model (V2).

Same "what if he hadn't made the run?" setup as score_counterfactual.py,
but the value function is the V2 graph model: per-frame self-attention over
23 per-player tokens instead of 11 pooled features. For each run:

    cf_gain = V2_shot(actual end) - V2_shot(counterfactual end)

where V2_shot is the graph model's shot probability on the 3s
per-player window ending at the run's end frame (75 frames @25fps,
30 @10fps — see scripts/fps.py).

Pipeline per run:
  1. patch the runner's (start_frame, end_frame] trajectory with the
     frozen/drift baseline (reuses counterfactual_segment)
  2. recompute velocities on the patched segment (add_velocities) — the
     frozen runner's vx/vy must be ~0, not his sprint velocities
  3. rebuild the [T, 23, 8] token tensor (possession_tokens)
  4. score the 3s window ending at end_frame with models/graph_value.pt

This fixes V1's core weakness: pooled features barely moved when one
player's trajectory changed, so mean cf_gain was ~0. Per-player attention
should make the run's contribution visible.

Output: data/trajectories/<game>_runs_counterfactual_v2.parquet with
  value_v2_{shot,box,line},
  value_cf_{frozen,drift}_{shot,box,line},
  cf_gain_{frozen,drift}  (on the shot target)

Same honest caveats as V1: defenders don't react in the counterfactual
world (conservative lower bound).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from models.graph_value import GraphValue
from build_windows import attack_dirs
from build_graph_windows import possession_tokens, add_velocities, TARGETS
from score_counterfactual import counterfactual_segment
from fps import load_fps, win_frames

ROOT = Path(__file__).resolve().parents[1]
WIN = 75
VARIANTS = ("frozen", "drift")


def window_from_tokens(toks, frames, good, end_frame, win=WIN):
    """3s window ending at end_frame, or None if unavailable."""
    if toks is None:
        return None
    fpos = {f: i for i, f in enumerate(frames)}
    i = fpos.get(end_frame)
    if i is None or i < win - 1 or not good[i - win + 1:i + 1].all():
        return None
    return toks[i - win + 1:i + 1]


def variant_window(seg, runner, team, opp, adir, start_frame, end_frame,
                   variant, fps=25.0):
    """Token window for one run under a counterfactual variant."""
    win = win_frames(fps)
    s = counterfactual_segment(seg, runner, team, start_frame, end_frame,
                               variant, fps=fps)
    s = add_velocities(s)
    toks, frames, good = possession_tokens(s, team, opp, adir)
    return window_from_tokens(toks, frames, good, end_frame, win=win)


def score_game(game, model, mu, sd, device, limit=None, out_path=None):
    fps = load_fps(game)
    win = win_frames(fps)
    df = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}.parquet")
    poss = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_possessions.parquet")
    runs = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_runs_scored.parquet")
    runs = runs.reset_index(drop=True)
    if limit:
        runs = runs.head(limit)
    print(f"{game}: scoring {len(runs)} runs (limit={limit})", flush=True)
    pmap = df[["frame", "period"]].drop_duplicates().set_index("frame")["period"]
    adirs = attack_dirs(df, poss)
    game_frames = np.sort(df["frame"].unique())

    # collect windows and score per possession (bounded memory: one
    # possession's windows at a time, not the whole game)
    vals = {}
    for pid, g in runs.groupby("possession_id"):
        p = poss.loc[poss["possession_id"] == pid].iloc[0]
        team = p["team"]
        opp = "away" if team == "home" else "home"
        period = int(pmap.get(p["start_frame"], 1))
        adir = adirs.get((period, team), 1)
        # Same backward extension as score_runs.py: windows may reach into
        # the previous possession so boundary runs become scorable.
        start_pos = int(np.searchsorted(game_frames, p["start_frame"]))
        lo = game_frames[max(0, start_pos - (win - 1))]
        seg = df[(df["frame"] >= lo) & (df["frame"] <= p["end_frame"])]
        # actual tokens built once per possession; variants rebuild per run
        toks_a, frames_a, good_a = possession_tokens(
            add_velocities(seg), team, opp, adir)
        recs = []
        for idx, r in g.iterrows():
            ef = int(r["end_frame"])
            w = {"actual": window_from_tokens(toks_a, frames_a, good_a, ef, win=win)}
            for v in VARIANTS:
                w[v] = variant_window(seg, r["runner"], team, opp, adir,
                                      int(r["start_frame"]), ef, v, fps=fps)
            for variant, w_ in w.items():
                if w_ is not None:
                    recs.append((idx, variant, w_))
        if not recs:
            continue
        X = np.stack([r[2] for r in recs]).astype(np.float32)
        X = (X - mu) / sd
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.from_numpy(X).to(device))).cpu().numpy()
        for (idx, variant, _), p in zip(recs, pv):
            vals[(idx, variant)] = p
        del X, pv, recs

    for v in ["actual", *VARIANTS]:
        for j, t in enumerate(TARGETS):
            col = f"value_{'v2' if v == 'actual' else 'cf_' + v}_{t}"
            runs[col] = [float(vals[(i, v)][j]) if (i, v) in vals else np.nan
                        for i in runs.index]
    for v in VARIANTS:
        runs[f"cf_gain_{v}"] = (runs["value_v2_shot"] -
                                runs[f"value_cf_{v}_shot"])
    out = Path(out_path) if out_path else ROOT / "data" / "trajectories" / f"{game}_runs_counterfactual_v2.parquet"
    runs.to_parquet(out, index=False)
    s = runs.dropna(subset=["cf_gain_frozen"])
    print(f"{game}: fps={fps:g} V2-scored {len(s)}/{len(runs)}  "
          f"mean_cf_gain_frozen={s['cf_gain_frozen'].mean():+.4f}  "
          f"mean_cf_gain_drift={s['cf_gain_drift'].mean():+.4f}")
    return s


def main(games, limit):
    ckpt = torch.load(ROOT / "models" / "graph_value.pt", map_location="cpu",
                      weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GraphValue().to(device).eval()
    model.load_state_dict(ckpt["model"])
    mu, sd = ckpt["mu"], ckpt["sd"]
    with torch.no_grad():
        all_runs = [score_game(g, model, mu, sd, device, limit) for g in games]
    runs = pd.concat(all_runs, ignore_index=True)

    # V1 vs V2 comparison
    v1 = pd.concat(
        [pd.read_parquet(ROOT / "data" / "trajectories" /
                         f"{g}_runs_counterfactual.parquet")
         for g in games], ignore_index=True)
    m = runs.merge(v1[["runner", "team", "start_frame", "end_frame",
                       "cf_gain_frozen"]].rename(
        columns={"cf_gain_frozen": "cf_gain_frozen_v1"}),
        on=["runner", "team", "start_frame", "end_frame"], how="left")
    mm = m.dropna(subset=["cf_gain_frozen", "cf_gain_frozen_v1"])
    print(f"\nV1 vs V2 cf_gain_frozen: n={len(mm)}  "
          f"mean_v1={mm['cf_gain_frozen_v1'].mean():+.4f}  "
          f"mean_v2={mm['cf_gain_frozen'].mean():+.4f}  "
          f"corr={mm['cf_gain_frozen'].corr(mm['cf_gain_frozen_v1']):.3f}  "
          f"std_v1={mm['cf_gain_frozen_v1'].std():.4f}  "
          f"std_v2={mm['cf_gain_frozen'].std():.4f}")
    print("\ntop 5 runs by V2 counterfactual gain (frozen):")
    top = runs.nlargest(5, "cf_gain_frozen")
    print(top[["runner", "team", "duration_s", "path_m",
               "defender_displacement_m", "value_v2_shot",
               "value_cf_frozen_shot", "cf_gain_frozen"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+",
                    default=["Sample_Game_1", "Sample_Game_2", "Sample_Game_3"])
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N runs per game (smoke test)")
    a = ap.parse_args()
    main(a.games, a.limit)
