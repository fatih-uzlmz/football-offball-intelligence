"""Visualize one detected run: runner + dragged defender, before -> after."""
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


def plot_row(r, df, out, rank_label=""):
    """Render a single run row `r` against game trajectories `df`."""
    seg = df[(df["frame"] >= r["start_frame"]) & (df["frame"] <= r["end_frame"])].copy()
    seg["xm"], seg["ym"] = seg["x"] * L, seg["y"] * W

    team = r["team"]
    opp = "away" if team == "home" else "home"
    tcolor = HOME_COLOR if team == "home" else AWAY_COLOR

    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)

    # context: everyone else faint at end frame
    end = seg[seg["frame"] == r["end_frame"]]
    ctx = end[~((end["player_id"] == r["runner"]) & (end["team"] == team)) &
              ~((end["player_id"] == r["nearest_def_id"]) & (end["team"] == opp))]
    ax.scatter(ctx["xm"], ctx["ym"], s=120, c="white", alpha=0.35, zorder=3)

    # runner trail
    rt = seg[(seg["player_id"] == r["runner"]) & (seg["team"] == team)].sort_values("frame")
    ax.plot(rt["xm"], rt["ym"], color=tcolor, lw=3.5, zorder=5)
    ax.scatter([rt["xm"].iloc[0]], [rt["ym"].iloc[0]], s=200, c=tcolor,
               edgecolors="white", linewidths=2, zorder=6)
    ax.scatter([rt["xm"].iloc[-1]], [rt["ym"].iloc[-1]], s=260, c=tcolor,
               edgecolors="white", linewidths=2, zorder=6)
    ax.annotate("", xy=(rt["xm"].iloc[-1], rt["ym"].iloc[-1]),
                xytext=(rt["xm"].iloc[len(rt)//2], rt["ym"].iloc[len(rt)//2]),
                arrowprops=dict(arrowstyle="->", color=tcolor, lw=2.5), zorder=6)
    ax.text(rt["xm"].iloc[-1] + 1.5, rt["ym"].iloc[-1] + 1.5, f"#{r['runner']}",
            color="white", fontsize=11, weight="bold", zorder=7)

    # dragged defender trail
    dt = seg[(seg["player_id"] == r["nearest_def_id"]) & (seg["team"] == opp)].sort_values("frame")
    if not dt.empty:
        ax.plot(dt["xm"], dt["ym"], color=AWAY_COLOR if opp == "away" else HOME_COLOR,
                lw=3.5, ls="--", zorder=5)
        ax.scatter([dt["xm"].iloc[0], dt["xm"].iloc[-1]],
                   [dt["ym"].iloc[0], dt["ym"].iloc[-1]], s=200,
                   c=AWAY_COLOR if opp == "away" else HOME_COLOR,
                   edgecolors="white", linewidths=2, zorder=6)
        ax.text(dt["xm"].iloc[-1] + 1.5, dt["ym"].iloc[-1] - 2,
                f"def #{r['nearest_def_id']}", color="white", fontsize=11,
                weight="bold", zorder=7)

    title = (f"{rank_label}#{r['runner']} ({team}) dragged defender "
             f"{r['defender_displacement_m']:.1f}m  |  "
             f"space {r['space_before_m']:.1f}m → {r['space_after_m']:.1f}m  |  "
             f"{r['duration_s']:.1f}s @ {r['peak_speed']:.1f} m/s")
    if "value_gained" in r and pd.notna(r["value_gained"]):
        title += (f"  |  value {r['value_before']:.2f} → {r['value_after']:.2f} "
                  f"(+{r['value_gained']:.2f})")
    if r["shot_within_5s"]:
        title += "  |  SHOT followed"
    style(ax, title)
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {outp}")


def main(game: str, rank: int, out: str, sort: str, scored: bool):
    rp = (ROOT / "data" / "trajectories" / f"{game}_runs_scored.parquet"
          if scored else ROOT / "data" / "trajectories" / f"{game}_runs.parquet")
    runs = pd.read_parquet(rp)
    runs = runs.sort_values(sort, ascending=False).reset_index(drop=True)
    r = runs.iloc[rank]
    df = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}.parquet")
    plot_row(r, df, out, rank_label=f"run #{rank}: ")
    print(r[["runner", "team", "duration_s", "path_m", "peak_speed",
             "defender_displacement_m", "space_before_m", "space_after_m",
             "behind_line_after", "shot_within_5s"]].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Sample_Game_1")
    ap.add_argument("--rank", type=int, default=4,
                    help="rank by --sort metric (0 = top)")
    ap.add_argument("--sort", default="defender_displacement_m")
    ap.add_argument("--scored", action="store_true",
                    help="use the value-scored runs table")
    ap.add_argument("--out", default="reports/run_sample.png")
    a = ap.parse_args()
    main(a.game, a.rank, a.out, a.sort, a.scored)
