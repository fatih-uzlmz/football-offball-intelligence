"""Phase 2 demo: animated 2D pitch visualization of the top-valued real-data run.

Featured run: SK_2006363 (A-League 2024/25), away #27, period 2,
frames 30645-30672 (2.8 s, 12.6 m).
  V1 value_gained = +0.777   (P(shot) 0.001 -> 0.778)
  V2 cf_gain_drift = +0.764  (counterfactual: what if he hadn't run)

Outputs (demo/):
  offball_demo.mp4  (~11 s, 1280x720, 24 fps)
  offball_demo.gif  (640 px wide loop for chat/sharing)

Timeline: 1 s lead-in (1x) -> run at 0.5x slow-mo -> 2 s tail (1x) -> end card.
The P(shot) ticker interpolates the model's measured value_before/value_after
across the run window (flat outside it).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visualization.pitch import (AWAY_COLOR, HOME_COLOR, L, W, draw_pitch)  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo"
OUT.mkdir(exist_ok=True)

GAME = "SK_2006363"
RUNNER, RUN_TEAM, DEF_ID = "27", "away", "17"
F0, F1 = 30645, 30672          # run frames (10 fps)
LEAD, TAIL = 10, 20            # context frames
A0, A1 = F0 - LEAD, F1 + TAIL

VAL_BEFORE, VAL_AFTER = 0.001418, 0.778468
V1_GAIN, V2_GAIN = 0.777, 0.764

GOLD = "#ffd60a"
BG = "#141b16"


def load():
    df = pd.read_parquet(ROOT / "data" / "trajectories" / f"{GAME}.parquet")
    seg = df[(df["frame"] >= A0) & (df["frame"] <= A1)].copy()
    seg["xm"], seg["ym"] = seg["x"] * L, seg["y"] * W
    frames = sorted(seg["frame"].unique())
    by_frame = {f: g for f, g in seg.groupby("frame")}
    return frames, by_frame


def interp_pos(by_frame, frames, pid, team, t):
    """Linear-interpolated (xm, ym) for a player at fractional frame t."""
    f_lo = int(np.floor(t))
    f_hi = min(f_lo + 1, frames[-1])
    f_lo = max(f_lo, frames[0])
    out = []
    for f in (f_lo, f_hi):
        g = by_frame[f]
        row = g[(g["player_id"] == pid) & (g["team"] == team)]
        out.append((row["xm"].values[0], row["ym"].values[0]) if len(row) else (np.nan, np.nan))
    a = t - f_lo
    return ((1 - a) * out[0][0] + a * out[1][0],
            (1 - a) * out[0][1] + a * out[1][1])


def pshot(t):
    if t <= F0:
        return VAL_BEFORE
    if t >= F1:
        return VAL_AFTER
    a = (t - F0) / (F1 - F0)
    return VAL_BEFORE + a * (VAL_AFTER - VAL_BEFORE)


def draw_frame(ax, by_frame, frames, t, phase):
    draw_pitch(ax)
    # all players
    f_snap = int(np.clip(round(t), frames[0], frames[-1]))
    g = by_frame[f_snap]
    for _, r in g[g["team"] == "home"].iterrows():
        ax.scatter(r["xm"], r["ym"], s=90, c=HOME_COLOR, edgecolors="white",
                   linewidths=0.8, zorder=3)
    for _, r in g[g["team"] == "away"].iterrows():
        if str(r["player_id"]) == RUNNER:
            continue
        ax.scatter(r["xm"], r["ym"], s=90, c=AWAY_COLOR, edgecolors="white",
                   linewidths=0.8, zorder=3)
    ball = g[g["team"] == "ball"]
    if len(ball):
        ax.scatter(ball["xm"], ball["ym"], s=70, c="white", edgecolors="black",
                   linewidths=0.8, zorder=4)

    # runner trail (up to current t)
    t0 = frames[0]
    ts = np.linspace(max(t0, F0 - 4), t, max(2, int((min(t, F1 + 6) - max(t0, F0 - 4)) * 4)))
    trail = np.array([interp_pos(by_frame, frames, RUNNER, RUN_TEAM, tt) for tt in ts])
    ax.plot(trail[:, 0], trail[:, 1], color=GOLD, lw=3.5, zorder=5)
    rx, ry = interp_pos(by_frame, frames, RUNNER, RUN_TEAM, t)
    ax.scatter([rx], [ry], s=240, c=GOLD, edgecolors="white", linewidths=2, zorder=6)
    ax.text(rx + 1.6, ry + 1.6, "#27", color="white", fontsize=11,
            weight="bold", zorder=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.55))

    # dragged defender trail
    dts = np.linspace(max(t0, F0 - 4), t, max(2, int((min(t, F1 + 6) - max(t0, F0 - 4)) * 4)))
    dt = np.array([interp_pos(by_frame, frames, DEF_ID, "home", tt) for tt in dts])
    if np.isfinite(dt).all():
        ax.plot(dt[:, 0], dt[:, 1], color="white", lw=2.2, ls="--", alpha=0.85, zorder=5)

    # overlays
    ax.text(2, W + 5.2, "OFF-BALL INTELLIGENCE", color="white", fontsize=15,
            weight="bold", va="center", zorder=8)
    if phase == "tail":
        ax.text(L / 2, W + 1.2, f"VALUE GAINED  +{V1_GAIN:.2f}   ·   "
                f"COUNTERFACTUAL  +{V2_GAIN:.2f}", color=GOLD, fontsize=12,
                weight="bold", va="center", ha="center", zorder=8,
                bbox=dict(boxstyle="round,pad=0.4", fc="black", ec=GOLD, alpha=0.75))
    else:
        ax.text(2, W + 1.2, f"{GAME}  ·  2nd half  ·  Away #27  ·  12.6 m in 2.8 s",
                color="#cfe3d4", fontsize=10.5, va="center", zorder=8)
    if phase in ("run", "tail"):
        ax.text(L - 2, W + 5.2, "RUN DETECTED", color=GOLD, fontsize=13,
                weight="bold", va="center", ha="right", zorder=8)

    # P(shot) ticker (bottom)
    p = pshot(t)
    bx0, bx1, by = 30, 75, -6.2
    ax.add_patch(plt.Rectangle((bx0, by), bx1 - bx0, 2.6, facecolor="black",
                               alpha=0.55, edgecolor="none", zorder=8))
    ax.add_patch(plt.Rectangle((bx0, by), (bx1 - bx0) * min(p, 1.0), 2.6,
                               facecolor=GOLD, edgecolor="none", zorder=9))
    ax.text(bx0 - 1, by + 1.3, "P(shot)", color="white", fontsize=10,
            weight="bold", va="center", ha="right", zorder=9)
    ax.text(bx1 + 1, by + 1.3, f"{p:.2f}", color="white", fontsize=10,
            weight="bold", va="center", zorder=9)

    # value callouts once the run completes (drawn in the subtitle slot above)


def end_card(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 78, "OFF-BALL INTELLIGENCE", color="white", fontsize=22,
            weight="bold", ha="center")
    ax.text(50, 68, "Every run detected. Every run valued.", color="#cfe3d4",
            fontsize=13, ha="center")
    ax.text(50, 50, "+0.78", color=GOLD, fontsize=44, weight="bold", ha="center")
    ax.text(50, 41, "shot probability gained  (V1 value model)", color="white",
            fontsize=12, ha="center")
    ax.text(50, 30, "+0.76", color=GOLD, fontsize=44, weight="bold", ha="center")
    ax.text(50, 21, "counterfactual gain — \"what if he hadn't run\"  (V2)",
            color="white", fontsize=12, ha="center")
    ax.text(50, 8, f"{GAME}  ·  27 real matches  ·  research prototype",
            color="#8fa598", fontsize=10, ha="center")


def main():
    frames, by_frame = load()
    # timeline in game-frames (fractional ok): lead-in 1x, run 0.5x, tail 1x
    tl = []
    tl += list(np.arange(A0, F0, 10 / 24))          # 1 s lead-in @24fps out
    tl += list(np.arange(F0, F1 + 0.01, 5 / 24))    # run at 0.5x slow-mo
    tl += list(np.arange(F1, A1, 10 / 24))         # 2 s tail @24fps out
    phases = (["lead"] * sum(1 for _ in np.arange(A0, F0, 10 / 24)) +
              ["run"] * sum(1 for _ in np.arange(F0, F1 + 0.01, 5 / 24)) +
              ["tail"] * sum(1 for _ in np.arange(F1, A1, 10 / 24)))

    W_PX, H_PX = 1280, 800
    fig = plt.figure(figsize=(W_PX / 100, H_PX / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    frames_out = []
    for t, phase in zip(tl, phases):
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)
        draw_frame(ax, by_frame, frames, t, phase)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames_out.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy())
        fig.clf()
    # end card ~2.5 s
    for _ in range(60):
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)
        end_card(ax)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames_out.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy())
        fig.clf()
    plt.close(fig)
    print(f"rendered {len(frames_out)} frames", flush=True)

    import imageio.v2 as imageio
    mp4_path = OUT / "offball_demo.mp4"
    imageio.mimsave(mp4_path, frames_out, fps=24, codec="libx264",
                    quality=8, macro_block_size=16)
    print("wrote", mp4_path, f"{mp4_path.stat().st_size/1e6:.1f} MB", flush=True)

    # GIF: downscaled, 12 fps
    small = [f[::2, ::2] for f in frames_out[::2]]
    gif_path = OUT / "offball_demo.gif"
    imageio.mimsave(gif_path, small, fps=12, loop=0)
    print("wrote", gif_path, f"{gif_path.stat().st_size/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
