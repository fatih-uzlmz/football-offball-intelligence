"""Render one tracking frame on a styled 2D pitch (meters)."""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visualization.pitch import (AWAY_COLOR, HOME_COLOR, L, W, draw_pitch,
                                 style)

ROOT = Path(__file__).resolve().parents[1]


def main(game: str, frame: int, out: str):
    df = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}.parquet")
    f = df[df["frame"] == frame].copy()
    f["xm"] = f["x"] * L
    f["ym"] = f["y"] * W
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)
    for team, color in (("home", HOME_COLOR), ("away", AWAY_COLOR)):
        t = f[f["team"] == team]
        ax.scatter(t["xm"], t["ym"], s=340, c=color, edgecolors="white",
                   linewidths=2, zorder=3)
        for _, r in t.iterrows():
            ax.text(r["xm"], r["ym"], r["player_id"], ha="center", va="center",
                    color="white", fontsize=9, weight="bold", zorder=4)
    b = f[f["team"] == "ball"]
    if not b.empty:
        ax.scatter(b["xm"], b["ym"], s=130, c="black", edgecolors="white",
                   linewidths=2, zorder=5)
    style(ax, f"{game} — frame {frame}  (t={f['time'].iloc[0]:.1f}s)")
    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    print(f"saved {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Sample_Game_1")
    ap.add_argument("--frame", type=int, default=30000)
    ap.add_argument("--out", default="reports/frame_sample.png")
    main(**vars(ap.parse_args()))
