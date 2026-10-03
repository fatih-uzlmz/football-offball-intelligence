"""Visualize a counterfactual run: actual trajectory vs the 'frozen' baseline.

Left panel: the run as it happened (runner trail + dragged defender).
Right panel: the same moment with the runner frozen at his run-start spot.

Titles carry the value-model readouts: V_before -> V_after (temporal gain)
and V_cf (counterfactual), so the eye can compare "the run happened" vs
"the run never happened".
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from visualization.pitch import (AWAY_COLOR, HOME_COLOR, L, W, draw_pitch,
                                 style)
from score_counterfactual import counterfactual_segment

ROOT = Path(__file__).resolve().parents[1]


def _panel(ax, seg, r, team, opp, tcolor, frozen=False, title=""):
    draw_pitch(ax)
    seg = seg.copy()
    seg["xm"], seg["ym"] = seg["x"] * L, seg["y"] * W
    end = seg[seg["frame"] == r["end_frame"]]
    ctx = end[~((end["player_id"] == str(r["runner"])) & (end["team"] == team)) &
              ~((end["player_id"] == str(r["nearest_def_id"])) & (end["team"] == opp))]
    ax.scatter(ctx["xm"], ctx["ym"], s=80, c="white", alpha=0.35, zorder=3)

    rt = seg[(seg["player_id"] == str(r["runner"])) & (seg["team"] == team)].sort_values("frame")
    ax.plot(rt["xm"], rt["ym"], color=tcolor, lw=3, zorder=5)
    ax.scatter([rt["xm"].iloc[0]], [rt["ym"].iloc[0]], s=150, c=tcolor,
               edgecolors="white", linewidths=2, zorder=6)
    ax.scatter([rt["xm"].iloc[-1]], [rt["ym"].iloc[-1]], s=200, c=tcolor,
               edgecolors="white", linewidths=2, zorder=6)
    if not frozen:
        ax.annotate("", xy=(rt["xm"].iloc[-1], rt["ym"].iloc[-1]),
                    xytext=(rt["xm"].iloc[len(rt)//2], rt["ym"].iloc[len(rt)//2]),
                    arrowprops=dict(arrowstyle="->", color=tcolor, lw=2), zorder=6)
    ax.text(rt["xm"].iloc[-1] + 1.5, rt["ym"].iloc[-1] + 1.5, f"#{r['runner']}",
            color="white", fontsize=10, weight="bold", zorder=7)

    dt = seg[(seg["player_id"] == str(r["nearest_def_id"])) & (seg["team"] == opp)].sort_values("frame")
    dcolor = AWAY_COLOR if opp == "away" else HOME_COLOR
    if not dt.empty:
        ax.plot(dt["xm"], dt["ym"], color=dcolor, lw=3, ls="--", zorder=5)
        ax.scatter([dt["xm"].iloc[0], dt["xm"].iloc[-1]],
                   [dt["ym"].iloc[0], dt["ym"].iloc[-1]], s=150,
                   c=dcolor, edgecolors="white", linewidths=2, zorder=6)
    style(ax, title)


def main(game: str, rank: int, sort: str, out: str):
    runs = pd.read_parquet(
        ROOT / "data" / "trajectories" / f"{game}_runs_counterfactual.parquet")
    runs = runs.sort_values(sort, ascending=False).reset_index(drop=True)
    r = runs.iloc[rank]
    df = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}.parquet")
    team = r["team"]
    opp = "away" if team == "home" else "home"
    tcolor = HOME_COLOR if team == "home" else AWAY_COLOR

    seg = df[(df["frame"] >= r["start_frame"]) & (df["frame"] <= r["end_frame"])]
    cf = counterfactual_segment(seg, r["runner"], team,
                                int(r["start_frame"]), int(r["end_frame"]), "frozen")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 8))
    vg = (f"value {r['value_before']:.2f} -> {r['value_after']:.2f} "
          f"(+{r['value_gained']:.2f})"
          if pd.notna(r["value_gained"])
          else f"end-of-possession value {r['value_after']:.2f} "
               f"(no before-window: run starts < 3 s into possession)")
    t1 = (f"ACTUAL: #{r['runner']} ran {r['path_m']:.1f}m, dragged def "
          f"{r['defender_displacement_m']:.1f}m\n{vg}")
    t2 = (f"COUNTERFACTUAL (frozen): #{r['runner']} stays put\n"
          f"value {r['value_cf_frozen']:.2f}  |  "
          f"run's worth: +{r['cf_gain_frozen']:.2f}")
    _panel(a1, seg, r, team, opp, tcolor, title=t1)
    _panel(a2, cf, r, team, opp, tcolor, frozen=True, title=t2)
    fig.suptitle(f"{game} — run #{rank} by {sort}: what if he hadn't made the run?",
                 color="white", fontsize=14, weight="bold")
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {outp}")
    print(r[["runner", "team", "duration_s", "path_m", "defender_displacement_m",
             "value_before", "value_after", "value_gained",
             "value_cf_frozen", "cf_gain_frozen", "shot_within_5s"]].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Sample_Game_1")
    ap.add_argument("--rank", type=int, default=0,
                    help="rank by --sort metric (0 = top)")
    ap.add_argument("--sort", default="cf_gain_frozen")
    ap.add_argument("--out", default="reports/counterfactual/cf_sample.png")
    a = ap.parse_args()
    main(a.game, a.rank, a.sort, a.out)
