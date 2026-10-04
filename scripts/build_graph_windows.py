"""Build per-player graph windows for the attention value model.

For each possession, slide a 3s window (75 frames @25fps / 30 @10fps,
stride 1s); frame counts derive from the dataset's fps (scripts/fps.py).
Each frame is a set of 23 tokens (11 attackers + 11 defenders + ball, fixed
order), each token carrying 8 attack-normalized features:
  x_att, y, vx_att (m/s), vy (m/s), speed (m/s),
  dist_to_ball (m), dist_to_goal (m), team (+1 attack / -1 defense / 0 ball)

Labels — 3 binary targets, event occurs within 5s AFTER window end:
  shot:      possession team takes a SHOT (from event feed)
  box_entry: ball is inside the penalty area (from trajectories)
  line_break: any attacker gets behind the 2nd-last defender (from trajectories)

Trajectory-derived labels use vectorized horizon queries over the game;
windows with < 1s of horizon get label -1 for those targets (masked in loss).

With 23 nodes, full self-attention over players == GAT on a complete graph.

Output: data/trajectories/<game>_graph_windows.npz
        (X [N,win,23,8], y [N,3], meta)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_windows import attack_dirs, BOX_X, BOX_Y
from fps import load_fps, win_frames, stride_frames

ROOT = Path(__file__).resolve().parents[1]
L, W = 105.0, 68.0
FPS = 25.0
WIN, STRIDE = 75, 25
AFTER_S = 5.0
HORIZON_FRAMES = int(AFTER_S * FPS)
MIN_HORIZON = int(1.0 * FPS)  # <1s of horizon -> label -1 (masked)
N_TOKENS = 23
N_FEAT = 8
TARGETS = ["shot", "box_entry", "line_break"]


def add_velocities(df: pd.DataFrame) -> pd.DataFrame:
    """Per-player vx/vy in m/s."""
    d = df.sort_values(["team", "player_id", "frame"]).copy()
    g = d.groupby(["team", "player_id"], observed=True)
    dt = g["time"].diff()
    d["vx"] = g["x"].diff() / dt * L
    d["vy"] = g["y"].diff() / dt * W
    d[["vx", "vy"]] = d[["vx", "vy"]].fillna(0.0)
    return d


def possession_tokens(seg: pd.DataFrame, team: str, opp: str, adir: int):
    """[T, 23, 8] token tensor for a possession segment (vectorized).

    Token order: 11 attackers (player_id sorted), 11 defenders, ball.
    Frames missing the ball or any player get NaN rows; callers must slice
    windows only from fully-observed stretches (see good_frames below).
    Returns (toks, frames, good) where good is a bool array per frame.
    """
    frames = np.sort(seg["frame"].unique())
    T = len(frames)
    cols = ["x", "y", "vx", "vy"]

    def block(tm):
        p = seg[seg["team"] == tm].pivot_table(
            index="frame", columns="player_id", values=cols, aggfunc="first")
        pids = sorted(p.columns.get_level_values(1).unique())
        if len(pids) != 11:
            return None
        arr = np.stack([p[(v, pid)].reindex(frames).to_numpy()
                        for pid in pids for v in cols], axis=-1)
        return arr.reshape(T, 11, 4)

    A = block(team)
    D = block(opp)
    if A is None or D is None:
        return None, None, None
    B = (seg[seg["team"] == "ball"].set_index("frame")[cols]
         .reindex(frames).to_numpy())  # [T, 4], NaN where missing

    good = ~(np.isnan(A).any((1, 2)) | np.isnan(D).any((1, 2)) |
             np.isnan(B).any(1))

    flip = adir == -1
    ax = 1 - A[:, :, 0] if flip else A[:, :, 0]
    avx = -A[:, :, 2] if flip else A[:, :, 2]
    dx = 1 - D[:, :, 0] if flip else D[:, :, 0]
    dvx = -D[:, :, 2] if flip else D[:, :, 2]
    bx = 1 - B[:, 0] if flip else B[:, 0]
    bvx = -B[:, 2] if flip else B[:, 2]

    Am = np.stack([ax * L, A[:, :, 1] * W], -1)
    Dm = np.stack([dx * L, D[:, :, 1] * W], -1)
    bm = np.stack([bx * L, B[:, 1] * W], -1)

    toks = np.full((T, N_TOKENS, N_FEAT), np.nan, dtype=np.float32)
    # attackers
    toks[:, :11, 0] = ax
    toks[:, :11, 1] = A[:, :, 1]
    toks[:, :11, 2] = avx
    toks[:, :11, 3] = A[:, :, 3]
    toks[:, :11, 4] = np.hypot(A[:, :, 2], A[:, :, 3])
    toks[:, :11, 5] = np.linalg.norm(Am - bm[:, None, :], axis=2)
    toks[:, :11, 6] = np.hypot((1 - ax) * L, (A[:, :, 1] - 0.5) * W)
    toks[:, :11, 7] = 1.0
    # defenders
    toks[:, 11:22, 0] = dx
    toks[:, 11:22, 1] = D[:, :, 1]
    toks[:, 11:22, 2] = dvx
    toks[:, 11:22, 3] = D[:, :, 3]
    toks[:, 11:22, 4] = np.hypot(D[:, :, 2], D[:, :, 3])
    toks[:, 11:22, 5] = np.linalg.norm(Dm - bm[:, None, :], axis=2)
    toks[:, 11:22, 6] = np.hypot((1 - dx) * L, (D[:, :, 1] - 0.5) * W)
    toks[:, 11:22, 7] = -1.0
    # ball
    toks[:, 22, 0] = bx
    toks[:, 22, 1] = B[:, 1]
    toks[:, 22, 2] = bvx
    toks[:, 22, 3] = B[:, 3]
    toks[:, 22, 4] = np.hypot(B[:, 2], B[:, 3])
    toks[:, 22, 5] = 0.0
    toks[:, 22, 6] = np.hypot((1 - bx) * L, (B[:, 1] - 0.5) * W)
    toks[:, 22, 7] = 0.0
    return toks, frames, good


def game_horizon_arrays(df: pd.DataFrame):
    """Per-frame arrays for vectorized horizon label queries.

    For each frame: ball x/y (raw), period, and per team-as-attacker the
    sorted attacker/defender raw x arrays (for the line-break test).
    """
    frames = np.sort(df["frame"].unique())
    fidx = {f: i for i, f in enumerate(frames)}
    n = len(frames)
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    period = np.zeros(n, dtype=np.int64)
    # per frame: sorted raw x for each team
    sx = {}
    for team in ("home", "away"):
        sx[team] = np.full((n, 11), np.nan)
    bdf = df[df["team"] == "ball"].sort_values("frame")
    bi = np.array([fidx[f] for f in bdf["frame"]])
    bx[bi] = bdf["x"].to_numpy()
    by[bi] = bdf["y"].to_numpy()
    pmap = df[["frame", "period"]].drop_duplicates().set_index("frame")["period"]
    for f, p in pmap.items():
        if f in fidx:
            period[fidx[f]] = int(p)
    for team in ("home", "away"):
        tdf = df[df["team"] == team].sort_values(["frame", "x"])
        # dense: 11 rows per frame -> reshape
        fr = tdf["frame"].to_numpy()
        # group counts per frame via searchsorted on unique frames
        u, counts = np.unique(fr, return_counts=True)
        vals = tdf["x"].to_numpy()
        pos = 0
        for f, c in zip(u, counts):
            if c == 11 and f in fidx:
                sx[team][fidx[f]] = np.sort(vals[pos:pos + c])
            pos += c
    return frames, bx, by, period, sx


def horizon_labels(frames, bx, by, period, sx, end_frame: int, end_period: int,
                   team: str, opp: str, adir: int,
                   horizon_frames: int = 125, min_horizon: int = 25):
    """box_entry / line_break from the 5s trajectory horizon (vectorized).

    horizon_frames/min_horizon default to the 25fps values (5s/1s); callers
    on other frame rates pass int(round(5.0*fps)) / int(round(1.0*fps)).
    """
    lo = np.searchsorted(frames, end_frame + 1)
    hi = np.searchsorted(frames, end_frame + horizon_frames + 1)
    sl = slice(lo, hi)
    same = period[sl] == end_period
    if same.sum() < min_horizon:
        return -1.0, -1.0
    slb = slice(lo, hi)
    bxa = bx[slb] if adir == 1 else 1 - bx[slb]
    in_box = (bxa > 1 - BOX_X) & (np.abs(by[slb] - 0.5) < BOX_Y / 2) & same
    box = 1.0 if np.any(in_box) else 0.0
    ax_s = sx[team][slb]   # ascending raw x
    dx_s = sx[opp][slb]
    valid = same & ~np.isnan(ax_s).any(1) & ~np.isnan(dx_s).any(1)
    if adir == 1:
        brk = (ax_s[valid][:, -1] > dx_s[valid][:, -2])
    else:
        # attack-normalized: max att x_att = 1 - min raw; 2nd-largest def x_att = 1 - 2nd-smallest raw
        brk = ((1 - ax_s[valid][:, 0]) > (1 - dx_s[valid][:, 1]))
    line = 1.0 if np.any(brk) else 0.0
    return box, line


def build(game: str, out_path=None, events_csv=None, shots_known: bool = True):
    """shots_known=False masks the shot target (-1 for every window); use for
    datasets with no shot events (e.g. SkillCorner). box_entry/line_break
    come from trajectories and are unaffected. (Additive 2026-10-04.)"""
    fps = load_fps(game)
    win, stride = win_frames(fps), stride_frames(fps)
    horizon_frames = int(round(AFTER_S * fps))
    min_horizon = int(round(1.0 * fps))  # <1s of horizon -> label -1 (masked)
    df = add_velocities(pd.read_parquet(
        ROOT / "data" / "trajectories" / f"{game}.parquet"))
    poss = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_possessions.parquet")
    if shots_known:
        ev = pd.read_csv(events_csv or ROOT / "data" / "raw" / f"{game}_RawEventsData.csv")
        shots = ev[ev["Type"] == "SHOT"].copy()
        shots["team_norm"] = shots["Team"].str.lower()
    else:
        shots = None
    adirs = attack_dirs(df, poss)
    pmap = df[["frame", "period"]].drop_duplicates().set_index("frame")["period"]
    frames_g, bx_g, by_g, period_g, sx_g = game_horizon_arrays(df)

    X, y, meta = [], [], []
    for _, p in poss.iterrows():
        team, opp = p["team"], ("away" if p["team"] == "home" else "home")
        period = int(pmap.get(p["start_frame"], 1))
        adir = adirs.get((period, team), 1)
        seg = df[(df["frame"] >= p["start_frame"]) & (df["frame"] <= p["end_frame"])]
        toks, frames, good = possession_tokens(seg, team, opp, adir)
        if toks is None:
            continue
        T = len(frames)
        tmap = dict(zip(frames, seg.sort_values("frame").drop_duplicates("frame")["time"].to_numpy()))
        for s in range(0, T - win + 1, stride):
            if not good[s:s + win].all():
                continue
            wf = frames[s:s + win]
            end_time = float(tmap[wf[-1]])
            if shots_known:
                shot = float(((shots["team_norm"] == team) &
                              (shots["Start Time [s]"] > end_time) &
                              (shots["Start Time [s]"] <= end_time + AFTER_S)).any())
            else:
                shot = -1.0  # unknown: masked in the loss, never 0
            box, brk = horizon_labels(frames_g, bx_g, by_g, period_g, sx_g,
                                      int(wf[-1]), period, team, opp, adir,
                                      horizon_frames, min_horizon)
            X.append(toks[s:s + win])
            y.append([shot, box, brk])
            meta.append((int(p["possession_id"]), team, end_time))
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    out = Path(out_path) if out_path else ROOT / "data" / "trajectories" / f"{game}_graph_windows.npz"
    np.savez_compressed(out, X=X, y=y, meta=np.array(meta, dtype=object))
    rates = []
    for i in range(3):
        known = y[:, i] >= 0
        rates.append(f"{TARGETS[i]}: known={known.mean():.1%} rate={y[known, i].mean():.3f}"
                     if known.any() else f"{TARGETS[i]}: none")
    print(f"{game}: fps={fps:g} win={win} windows={len(y)}  X{X.shape}  " + "  ".join(rates) + f" -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    build(**vars(ap.parse_args()))
