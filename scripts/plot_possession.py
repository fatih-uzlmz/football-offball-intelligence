"""Tidy possession plot: faint team trails, bold ball path with direction.

Design choices to kill the spaghetti:
  - possession team's trails at moderate alpha, opponent very faint
  - one trail per player, no markers along the way
  - ball path thick with arrowheads showing direction
  - start/end dots only
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visualization.pitch import (AWAY_COLOR, HOME_COLOR, L, W, draw_pitch,
                                 style)

ROOT = Path(__file__).resolve().parents[1]
FPS = 25.0


def plot_possession(game: str, possession_id: int, out: str):
    poss = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_possessions.parquet")
    p = poss.loc[poss["possession_id"] == possession_id].iloc[0]
    df = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}.parquet")
    seg = df[(df["frame"] >= p["start_frame"]) & (df["frame"] <= p["end_frame"])].copy()
    seg["xm"] = seg["x"] * L
    seg["ym"] = seg["y"] * W

    team, opp = p["team"], ("away" if p["team"] == "home" else "home")
    tcolor = HOME_COLOR if team == "home" else AWAY_COLOR
    ocolor = AWAY_COLOR if team == "home" else HOME_COLOR

    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)

    # opponent trails: very faint context
    for (_, _), g in seg[seg["team"] == opp].groupby(["player_id", "team"]):
        g = g.sort_values("frame")
        ax.plot(g["xm"], g["ym"], color=ocolor, lw=1, alpha=0.15, zorder=2)
    # possession team trails
    for (_, _), g in seg[seg["team"] == team].groupby(["player_id", "team"]):
        g = g.sort_values("frame")
        ax.plot(g["xm"], g["ym"], color=tcolor, lw=1.4, alpha=0.45, zorder=2)
    # end positions of possession team
    last = seg[seg["team"] == team].sort_values("frame").groupby("player_id").tail(1)
    ax.scatter(last["xm"], last["ym"], s=200, c=tcolor, edgecolors="white",
               linewidths=1.5, zorder=4)

    # ball path with direction arrows
    ball = seg[seg["team"] == "ball"].sort_values("frame")
    ax.plot(ball["xm"], ball["ym"], color="black", lw=2.5, alpha=0.9, zorder=5)
    step = max(1, len(ball) // 8)
    for i in range(0, len(ball) - step, step):
        x0, y0 = ball["xm"].iloc[i], ball["ym"].iloc[i]
        x1, y1 = ball["xm"].iloc[i + step], ball["ym"].iloc[i + step]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                    zorder=5)
    ax.scatter([ball["xm"].iloc[0]], [ball["ym"].iloc[0]], s=160, c="lime",
               edgecolors="black", linewidths=1.5, zorder=6)
    ax.scatter([ball["xm"].iloc[-1]], [ball["ym"].iloc[-1]], s=220, c="red",
               edgecolors="black", linewidths=1.5, zorder=6)

    style(ax, f"possession {possession_id} — {team}  ({p['duration_s']:.0f}s, "
               f"ball green→red)")
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    print(f"saved {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Sample_Game_1")
    ap.add_argument("--possession", type=int, default=184)
    ap.add_argument("--out", default="reports/possession_sample.png")
    a = ap.parse_args()
    plot_possession(a.game, a.possession, a.out)
