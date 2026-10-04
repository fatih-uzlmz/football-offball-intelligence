"""Phase B driver: full pipeline on all 27 real matches -> data/real_runs/.

Per match, in order:
  build_possessions -> detect_runs -> build_windows (DFL only) ->
  build_graph_windows -> detect_presses -> score_runs -> score_counterfactual_v2

Notes (2026-10-04):
- score_runs / score_counterfactual_v2 read their INPUTS (df, possessions,
  runs, runs_scored) from hardcoded data/trajectories/ paths, so the small
  intermediates are copied there after being written to data/real_runs/.
  Nothing Sample_* or models/*.pt is touched.
- build_windows (V1, single shot target, no label masking in train.py) runs
  DFL-only. SkillCorner has no shot events anywhere (verified across all 20
  dynamic_events files: only off_ball_run / on_ball_engagement /
  passing_option / player_possession), so SK shot labels must be -1
  (masked), never 0 -> SK is skipped for V1 windows entirely.
- build_graph_windows uses shots_known=False for SK (shot=-1 everywhere;
  box_entry/line_break still come from trajectories).
- The scorers load the OLD Metrica models (models/value_lstm.pt,
  models/graph_value.pt) -- intended for Phase B.
- Resumable: steps whose outputs already exist are skipped.

Usage: python run_phase_b.py [--games DFL_J03WMX ...] [--steps possessions,runs,...]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_possessions import main as bp_main
from detect_runs import main as dr_main
from build_windows import build as bw_build
from build_graph_windows import build as bgw_build
from detect_presses import detect_game as dp_detect
from score_runs import score_game as sr_score
from score_counterfactual_v2 import score_game as v2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.value_lstm import ValueLSTM
from models.graph_value import GraphValue

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "data" / "trajectories"
REAL = ROOT / "data" / "real_runs"

DFL = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]


def _real_games():
    ids = [p.name[:-len(".meta.json")] for p in TRAJ.glob("*.meta.json")
           if not p.name.startswith("Sample_")]
    ids = [i for i in ids if i.startswith("DFL_") or i.startswith("SK_")]
    dfl = sorted(i for i in ids if i.startswith("DFL_"))
    sk = sorted(i for i in ids if i.startswith("SK_"))
    # canonical order: the 7 DFL matches first, then SK by match id
    assert [d[4:] for d in dfl] == DFL, f"DFL ids changed: {dfl}"
    assert len(sk) == 20, f"expected 20 SK matches, got {len(sk)}"
    return dfl + sk


GAMES = _real_games()


def is_dfl(game):
    return game.startswith("DFL_")


def run_match(game, model1, mu1, sd1, model2, mu2, sd2, device, steps, logf):
    stats = {"game": game, "t": {}}
    t_all = time.time()
    REAL.mkdir(exist_ok=True)

    def cp_to_traj(name):
        src = REAL / name
        if src.exists():
            shutil.copy2(src, TRAJ / name)

    if "possessions" in steps and not (REAL / f"{game}_possessions.parquet").exists():
        t = time.time()
        bp_main(game, out_dir=str(REAL))
        stats["t"]["possessions"] = round(time.time() - t, 1)
    for n in (f"{game}_carriers.parquet", f"{game}_possessions.parquet"):
        cp_to_traj(n)

    if "runs" in steps and not (REAL / f"{game}_runs.parquet").exists():
        t = time.time()
        dr_main(game, out_path=str(REAL / f"{game}_runs.parquet"))
        stats["t"]["runs"] = round(time.time() - t, 1)
    cp_to_traj(f"{game}_runs.parquet")

    if "windows" in steps and not (REAL / f"{game}_windows.npz").exists():
        if is_dfl(game):
            t = time.time()
            bw_build(game, out_path=str(REAL / f"{game}_windows.npz"))
            stats["t"]["windows"] = round(time.time() - t, 1)
        else:
            stats["windows"] = "skipped (no shot labels -> V1 DFL-only)"

    if "graph_windows" in steps and not (REAL / f"{game}_graph_windows.npz").exists():
        t = time.time()
        bgw_build(game, out_path=str(REAL / f"{game}_graph_windows.npz"),
                  shots_known=is_dfl(game))
        stats["t"]["graph_windows"] = round(time.time() - t, 1)

    if "presses" in steps and not (REAL / f"{game}_presses.parquet").exists():
        t = time.time()
        dp_detect(game, out_path=str(REAL / f"{game}_presses.parquet"))
        stats["t"]["presses"] = round(time.time() - t, 1)

    if "score" in steps and not (REAL / f"{game}_runs_scored.parquet").exists():
        t = time.time()
        sr_score(game, model1, mu1, sd1, device,
                 out_path=str(REAL / f"{game}_runs_scored.parquet"))
        stats["t"]["score"] = round(time.time() - t, 1)
    cp_to_traj(f"{game}_runs_scored.parquet")

    if "v2" in steps and not (REAL / f"{game}_runs_counterfactual_v2.parquet").exists():
        t = time.time()
        v2_score(game, model2, mu2, sd2, device,
                 out_path=str(REAL / f"{game}_runs_counterfactual_v2.parquet"))
        stats["t"]["v2"] = round(time.time() - t, 1)

    # ---- per-match summary stats ----
    try:
        runs = pd.read_parquet(REAL / f"{game}_runs.parquet")
        stats["n_runs"] = len(runs)
    except Exception:
        stats["n_runs"] = None
    try:
        sc = pd.read_parquet(REAL / f"{game}_runs_scored.parquet")
        stats["v1_scored"] = int(sc["value_gained"].notna().sum())
        stats["v1_boundary"] = int(sc["before_crosses_boundary"].fillna(False).sum())
        stats["v1_mean_gain"] = round(float(sc["value_gained"].mean()), 4)
    except Exception:
        stats["v1_scored"] = None
    try:
        v2 = pd.read_parquet(REAL / f"{game}_runs_counterfactual_v2.parquet")
        stats["v2_scored"] = int(v2["cf_gain_frozen"].notna().sum())
        stats["v2_mean_cf_gain"] = round(float(v2["cf_gain_frozen"].mean()), 4)
    except Exception:
        stats["v2_scored"] = None
    try:
        z = np.load(REAL / f"{game}_graph_windows.npz", allow_pickle=True)
        y = z["y"]
        stats["graph_windows"] = len(y)
        stats["graph_labels"] = {
            t: {"known": round(float((y[:, i] >= 0).mean()), 3),
                "rate": round(float(y[y[:, i] >= 0, i].mean()), 4)
                if (y[:, i] >= 0).any() else None}
            for i, t in enumerate(["shot", "box_entry", "line_break"])}
    except Exception:
        pass
    stats["total_s"] = round(time.time() - t_all, 1)
    logf.write(json.dumps(stats) + "\n")
    logf.flush()
    print(f"DONE {game}: " + json.dumps({k: v for k, v in stats.items()
                                        if k in ("n_runs", "v1_scored", "v2_scored",
                                                 "total_s")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=GAMES)
    ap.add_argument("--steps", default="possessions,runs,windows,graph_windows,presses,score,v2")
    args = ap.parse_args()
    steps = set(args.steps.split(","))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt1 = torch.load(ROOT / "models" / "value_lstm.pt", map_location="cpu",
                       weights_only=False)
    model1 = ValueLSTM().to(device).eval()
    model1.load_state_dict(ckpt1["model"])
    ckpt2 = torch.load(ROOT / "models" / "graph_value.pt", map_location="cpu",
                       weights_only=False)
    model2 = GraphValue().to(device).eval()
    model2.load_state_dict(ckpt2["model"])
    mu1, sd1 = ckpt1["mu"], ckpt1["sd"]
    mu2, sd2 = ckpt2["mu"], ckpt2["sd"]
    print(f"models loaded (device={device})", flush=True)

    REAL.mkdir(exist_ok=True)
    with open(REAL / "phase_b_stats.jsonl", "a") as logf:
        for game in args.games:
            run_match(game, model1, mu1, sd1, model2, mu2, sd2, device, steps, logf)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
