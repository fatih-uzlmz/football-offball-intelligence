"""Visualize one pressing event: pressers converging on the ball carrier."""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visualization.pitch import (AWAY_COLOR, HOME_COLOR, L, W, draw_pitch,
                                 style)

ROOT = Path(__file__).resolve().parents[1]


def main(game: str, rank: int, out: str):
    presses = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_presses.parquet")
    presses = presses.sort_values("n_pressers", ascending=False).reset_index(drop=True)
    r = presses.iloc[rank]
    df = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}.parquet")
    seg = df[(df["frame"] >= r["start_frame"]) & (df["frame"] <= r["end_frame"])].copy()
    seg["xm"], seg["ym"] = seg["x"] * L, seg["y"] * W

    team = r["team"]          # pressing team
    opp = "away" if team == "home" else "home"
    tcolor = HOME_COLOR if team == "home" else AWAY_COLOR
    ocolor = AWAY_COLOR if opp == "away" else HOME_COLOR
    pressers = [x for x in str(r["pressers"]).split(",") if x]

    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)

    end = seg[seg["frame"] == r["end_frame"]]
    ctx = end[~end["player_id"].isin(pressers) | (end["team"] != team)]
    ax.scatter(ctx["xm"], ctx["ym"], s=120, c="white", alpha=0.35, zorder=3)

    # ball carrier trail (white)
    car = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_carriers.parquet")
    cmap = dict(zip(car["frame"], car["carrier_id"]))
    frames = sorted(seg["frame"].unique())
    cpos = []
    for f in frames:
        cid = cmap.get(f)
        row = seg[(seg["frame"] == f) & (seg["player_id"] == cid) & (seg["team"] == opp)]
        if not row.empty:
            cpos.append((row["xm"].iloc[0], row["ym"].iloc[0]))
    if cpos:
        cx, cy = zip(*cpos)
        ax.plot(cx, cy, color="white", lw=3, zorder=5)
        ax.scatter([cx[-1]], [cy[-1]], s=260, c="white",
                   edgecolors=tcolor, linewidths=3, zorder=6)
        ax.text(cx[-1] + 1.5, cy[-1] + 1.5, "carrier", color="white",
                fontsize=11, weight="bold", zorder=7)

    # presser trails
    for pid in pressers[:4]:
        pt = seg[(seg["player_id"] == pid) & (seg["team"] == team)].sort_values("frame")
        if pt.empty:
            continue
        ax.plot(pt["xm"], pt["ym"], color=tcolor, lw=3.5, zorder=5)
        ax.scatter([pt["xm"].iloc[0]], [pt["ym"].iloc[0]], s=200, c=tcolor,
                   edgecolors="white", linewidths=2, zorder=6)
        ax.scatter([pt["xm"].iloc[-1]], [pt["ym"].iloc[-1]], s=260, c=tcolor,
                   edgecolors="white", linewidths=2, zorder=6)
        ax.annotate("", xy=(pt["xm"].iloc[-1], pt["ym"].iloc[-1]),
                    xytext=(pt["xm"].iloc[len(pt)//2], pt["ym"].iloc[len(pt)//2]),
                    arrowprops=dict(arrowstyle="->", color=tcolor, lw=2.5), zorder=6)
        ax.text(pt["xm"].iloc[-1] + 1.5, pt["ym"].iloc[-1] - 2, f"#{pid}",
                color="white", fontsize=11, weight="bold", zorder=7)

    title = (f"press #{rank}: {team} send {r['n_pressers']} "
             f"({'/'.join(map(str, pressers[:4]))})  |  "
             f"reaction {r['reaction_s']:.1f}s  |  "
             f"closest {r['min_carrier_dist_m']:.1f}m  |  "
             f"{r['duration_s']:.1f}s"
             + ("  |  TURNOVER FORCED" if r["forced_turnover"] else ""))
    style(ax, title)
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    print(f"saved {outp}")
    print(r.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Sample_Game_1")
    ap.add_argument("--rank", type=int, default=0,
                    help="rank by number of pressers (0 = top)")
    ap.add_argument("--out", default="reports/press_sample.png")
    main(**vars(ap.parse_args()))
