"""Pitch drawing in real meters (105 x 68) with clean broadcast styling."""
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

L, W = 105.0, 68.0
GRASS = "#3d8b4f"
GRASS_DARK = "#357a45"
LINE = "white"

HOME_COLOR = "#2f6df6"   # blue
AWAY_COLOR = "#e5484d"   # red


def draw_pitch(ax, linecolor=LINE, linewidth=2):
    ax.set_xlim(-2, L + 2)
    ax.set_ylim(-2, W + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    # mowed stripes
    for i in range(0, 14):
        x0 = i * L / 14
        ax.add_patch(Rectangle((x0, 0), L / 14, W,
                               facecolor=GRASS if i % 2 == 0 else GRASS_DARK,
                               edgecolor="none", zorder=0))
    kw = dict(edgecolor=linecolor, facecolor="none", lw=linewidth, zorder=2)
    ax.add_patch(Rectangle((0, 0), L, W, **kw))
    ax.plot([L / 2, L / 2], [0, W], color=linecolor, lw=linewidth, zorder=2)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, **kw))
    ax.add_patch(Circle((L / 2, W / 2), 0.4, facecolor=linecolor,
                        edgecolor="none", zorder=2))
    # boxes on both ends
    for x0, sgn in ((0, 1), (L, -1)):
        ax.add_patch(Rectangle(((x0 if sgn > 0 else x0 - 16.5), (W - 40.32) / 2),
                               16.5, 40.32, **kw))
        ax.add_patch(Rectangle(((x0 if sgn > 0 else x0 - 5.5), (W - 18.32) / 2),
                               5.5, 18.32, **kw))
        sx = x0 + sgn * 11
        ax.add_patch(Circle((sx, W / 2), 0.4, facecolor=linecolor,
                            edgecolor="none", zorder=2))
        ax.add_patch(Arc((sx, W / 2), 18.3, 18.3,
                         theta1=308 if sgn > 0 else 128,
                         theta2=52 if sgn > 0 else 232,
                         color=linecolor, lw=linewidth, zorder=2))
        for cy in (0, W):
            ax.add_patch(Arc((x0, cy), 2, 2,
                             theta1=0 if sgn > 0 else 90,
                             theta2=90 if sgn > 0 else 180,
                             color=linecolor, lw=linewidth, zorder=2))
    return ax


def style(ax, title: str = ""):
    ax.set_title(title, fontsize=13, weight="bold", color="white", pad=12)
    fig = ax.figure
    fig.patch.set_facecolor("#1d2b22")
    return fig
