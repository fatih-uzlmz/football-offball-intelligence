"""Phase 2: architecture diagram of the off-ball intelligence pipeline.

Output: demo/architecture.png (+ PDF). Drawn with matplotlib, dark theme
matching the demo video styling. Source of truth is this script.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo"
OUT.mkdir(exist_ok=True)

BG = "#141b16"
BOX = "#1f2b24"
GOLD = "#ffd60a"
GREEN = "#3d8b4f"
TEXT = "white"
SUB = "#cfe3d4"

STAGES = [
    ("TRACKING\nINGESTION",
     "SkillCorner 10 fps\nDFL 25 Hz optical\nMetrica samples\n→ 27 matches · 43.1M rows"),
    ("RUN\nDETECTION",
     "velocity / acceleration\nprofiles\n→ off-ball runs"),
    ("POSSESSION\nSEGMENTATION",
     "possession chains\nno cross-turnover\nwindows"),
    ("FPS-AWARE\nWINDOWING",
     "3.0 s windows\n1.0 s stride\nper-dataset fps"),
    ("VALUE\nMODELS",
     "LSTM  P(shot | 5 s)\nGraph attention\nshot · box entry · line break"),
    ("V1 / V2\nSCORING",
     "value_gained\ncounterfactual\nfrozen + drift"),
    ("RANKED\nREPORT",
     "every run valued\ntop runs · demo\nclub-ready output"),
]


def main():
    fig, ax = plt.subplots(figsize=(17, 6.5), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 93, "OFF-BALL INTELLIGENCE  —  PIPELINE", color=TEXT,
            fontsize=17, weight="bold", ha="center")
    ax.text(50, 87.5, "from tracking data to valued runs", color=SUB,
            fontsize=11, ha="center")

    n = len(STAGES)
    bw, bh = 11.6, 34
    gap = (100 - n * bw) / (n + 1)
    y0 = 22
    for i, (title, body) in enumerate(STAGES):
        x0 = gap + i * (bw + gap)
        box = FancyBboxPatch((x0, y0), bw, bh, boxstyle="round,pad=0.6",
                             facecolor=BOX, edgecolor=GREEN, lw=1.6, zorder=2)
        ax.add_patch(box)
        ax.text(x0 + bw / 2, y0 + bh - 5.5, title, color=GOLD, fontsize=10.5,
                weight="bold", ha="center", va="top", zorder=3)
        ax.text(x0 + bw / 2, y0 + bh - 13.5, body, color=SUB, fontsize=8.2,
                ha="center", va="top", zorder=3, linespacing=1.5)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x0 + bw + 0.4, y0 + bh / 2), (x0 + bw + gap - 0.4, y0 + bh / 2),
                arrowstyle="-|>", mutation_scale=16, color=GOLD, lw=2, zorder=3))

    ax.text(50, 8, "27 real matches (20 SkillCorner A-League · 7 Bundesliga)  ·  "
            "match-level train/val/test splits  ·  research prototype, not affiliated with any club",
            color="#8fa598", fontsize=9, ha="center")
    for ext in ("png", "pdf"):
        p = OUT / f"architecture.{ext}"
        fig.savefig(p, facecolor=BG, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
