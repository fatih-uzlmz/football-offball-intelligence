"""Tactical search engine (Phase 6).

Query the scored runs with composable tactical filters, get a ranked table,
and optionally render each hit as a pitch visualization.

Examples:
  # runs that dragged a defender 20m+ out of position
  python scripts/search_runs.py --min-drag 20 --sort defender_displacement_m

  # runs that led to a shot, ranked by value gained
  python scripts/search_runs.py --shot-after --sort value_gained --plot 3

  # overlapping runs: behind the line with space to receive
  python scripts/search_runs.py --behind-line --min-space-after 4 --plot 3

  # most valuable runs in game 2
  python scripts/search_runs.py --game Sample_Game_2 --min-value 0.3
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_run import plot_row

ROOT = Path(__file__).resolve().parents[1]
GAMES = ["Sample_Game_1", "Sample_Game_2", "Sample_Game_3"]
COLS = ["game", "runner", "team", "duration_s", "path_m", "peak_speed",
        "nearest_def_id", "defender_displacement_m", "space_before_m",
        "space_after_m", "behind_line_after", "value_before", "value_after",
        "value_gained", "shot_within_5s"]


def load(games):
    frames = []
    for g in games:
        r = pd.read_parquet(ROOT / "data" / "trajectories" / f"{g}_runs_scored.parquet")
        r["game"] = g
        frames.append(r)
    return pd.concat(frames, ignore_index=True)


def main(a):
    runs = load(a.game)
    n0 = len(runs)
    q = runs
    if a.team:
        q = q[q["team"] == a.team]
    if a.min_drag:
        q = q[q["defender_displacement_m"] >= a.min_drag]
    if a.min_path:
        q = q[q["path_m"] >= a.min_path]
    if a.min_duration:
        q = q[q["duration_s"] >= a.min_duration]
    if a.behind_line:
        q = q[q["behind_line_after"]]
    if a.shot_after:
        q = q[q["shot_within_5s"]]
    if a.min_value:
        q = q[q["value_gained"] >= a.min_value]
    if a.min_space_after:
        q = q[q["space_after_m"] >= a.min_space_after]
    q = q.sort_values(a.sort, ascending=False).reset_index(drop=True)

    scored = q.dropna(subset=["value_gained"])
    print(f"{len(q)}/{n0} runs matched", end="")
    if len(scored):
        print(f"  |  avg value gained {scored['value_gained'].mean():+.3f}"
              f"  |  avg defender dragged {q['defender_displacement_m'].mean():.1f}m"
              f"  |  {int(q['shot_within_5s'].sum())} led to shots")
    else:
        print()
    if q.empty:
        return
    top = q.head(a.top)
    print(top[[c for c in COLS if c in top.columns]].round(3).to_string(index=False))

    if a.plot:
        trajs = {}
        for i, r in top.head(a.plot).iterrows():
            if r["game"] not in trajs:
                trajs[r["game"]] = pd.read_parquet(
                    ROOT / "data" / "trajectories" / f"{r['game']}.parquet")
            plot_row(r, trajs[r["game"]], f"reports/search/hit_{i:02d}.png",
                     rank_label=f"hit #{i}: ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", nargs="+", default=GAMES)
    ap.add_argument("--team", choices=["home", "away"])
    ap.add_argument("--min-drag", type=float, help="defender dragged >= N meters")
    ap.add_argument("--min-path", type=float, help="run length >= N meters")
    ap.add_argument("--min-duration", type=float, help="run duration >= N seconds")
    ap.add_argument("--behind-line", action="store_true")
    ap.add_argument("--shot-after", action="store_true")
    ap.add_argument("--min-value", type=float, help="value_gained >= X")
    ap.add_argument("--min-space-after", type=float, help="space after >= N meters")
    ap.add_argument("--sort", default="value_gained",
                    choices=["value_gained", "defender_displacement_m",
                             "path_m", "duration_s"])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--plot", type=int, default=0,
                    help="render top N hits to reports/search/")
    main(ap.parse_args())
