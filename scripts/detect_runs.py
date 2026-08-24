"""Phase 3 — heuristic off-ball run detector.

A candidate run = an attacker (not the ball carrier) sustaining
speed > SPEED_MIN for >= DUR_MIN, covering >= DIST_MIN meters.

Per run we record geometry before/after: nearest-defender distance,
that defender's displacement, whether the runner got behind the
defensive line, progress toward goal, and whether a shot followed
within 5s (from the event feed).

Output: data/trajectories/<game>_runs.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
L, W = 105.0, 68.0
FPS = 25.0

SPEED_MIN = 4.0    # m/s sustained
DUR_MIN = 1.0      # s
DIST_MIN = 5.0     # m path length
GAP_TOL = 12       # frames below threshold tolerated inside a run
SMOOTH = 13        # velocity smoothing window (~0.5s)
AFTER_S = 5.0      # outcome window


def attack_dir(team: str, period: int) -> int:
    # measured: p1 home +x / away -x, p2 swapped
    return 1 if (team == "home") == (period == 1) else -1


def add_velocity(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["team"].isin(("home", "away"))].copy()
    d["xm"], d["ym"] = d["x"] * L, d["y"] * W
    d = d.sort_values(["team", "player_id", "frame"])
    g = d.groupby(["team", "player_id"])
    d["vx"] = g["xm"].diff() / g["time"].diff()
    d["vy"] = g["ym"].diff() / g["time"].diff()
    d["speed"] = np.hypot(d["vx"], d["vy"])
    d["speed_sm"] = g["speed"].transform(
        lambda s: s.rolling(SMOOTH, center=True, min_periods=1).mean())
    return d


def runs_in_possession(seg: pd.DataFrame, carriers: pd.DataFrame,
                       possession_id: int, team: str, period: int) -> list:
    out = []
    cframes = set(carriers.loc[
        (carriers["frame"] >= seg["frame"].min()) &
        (carriers["frame"] <= seg["frame"].max()) &
        (carriers["carrier_team"] == team), "frame"])
    # carrier identity per frame
    cmap = carriers.set_index("frame")["carrier_id"]
    adir = attack_dir(team, period)
    opp = "away" if team == "home" else "home"
    for pid, g in seg[seg["team"] == team].groupby("player_id"):
        g = g.sort_values("frame").reset_index(drop=True)
        hot = (g["speed_sm"] > SPEED_MIN).to_numpy()
        # group with gap tolerance
        runs, cur = [], []
        gap = 0
        for i, h in enumerate(hot):
            if h:
                cur.append(i); gap = 0
            elif cur:
                gap += 1
                if gap > GAP_TOL:
                    runs.append(cur); cur = []; gap = 0
                else:
                    cur.append(i)
        if cur:
            runs.append(cur)
        for idx in runs:
            r = g.iloc[idx]
            dur = (r["frame"].iloc[-1] - r["frame"].iloc[0] + 1) / FPS
            if dur < DUR_MIN:
                continue
            path = np.hypot(r["xm"].diff(), r["ym"].diff()).sum()
            if path < DIST_MIN:
                continue
            # carrier check: drop if carrier for most of the run
            is_carrier = sum(cmap.get(f) == pid for f in r["frame"])
            if is_carrier > 0.5 * len(r):
                continue
            f0, f1 = int(r["frame"].iloc[0]), int(r["frame"].iloc[-1])
            p0 = r.iloc[0][["xm", "ym"]].to_numpy()
            p1 = r.iloc[-1][["xm", "ym"]].to_numpy()
            disp = float(np.linalg.norm(p1 - p0))
            prog = float((p1[0] - p0[0]) * adir)  # toward opponent goal

            fr = seg[seg["frame"] == f0]
            fr1 = seg[seg["frame"] == f1]
            defs0 = fr[fr["team"] == opp][["player_id", "xm", "ym"]]
            defs1 = fr1[fr1["team"] == opp][["player_id", "xm", "ym"]]
            if defs0.empty or defs1.empty:
                continue
            d0 = np.hypot(defs0["xm"] - p0[0], defs0["ym"] - p0[1])
            d1 = np.hypot(defs1["xm"] - p1[0], defs1["ym"] - p1[1])
            nd_id = defs0.iloc[int(np.argmin(d0))]["player_id"]
            nd0 = defs0[defs0["player_id"] == nd_id][["xm", "ym"]].iloc[0].to_numpy()
            nd1row = defs1[defs1["player_id"] == nd_id][["xm", "ym"]]
            nd1 = nd1row.iloc[0].to_numpy() if not nd1row.empty else nd0
            nd_disp = float(np.linalg.norm(nd1 - nd0))

            # behind the defensive line? (attack-x beyond 2nd-last opponent)
            ax = lambda df_: np.sort((df_["xm"] * adir).to_numpy())
            line0 = ax(defs0)[-2] if len(defs0) >= 2 else ax(defs0)[-1]
            line1 = ax(defs1)[-2] if len(defs1) >= 2 else ax(defs1)[-1]
            behind_before = bool(p0[0] * adir > line0)
            behind_after = bool(p1[0] * adir > line1)

            out.append(dict(
                possession_id=possession_id, team=team, period=period,
                runner=pid, start_frame=f0, end_frame=f1,
                start_time=float(r["time"].iloc[0]),
                end_time=float(r["time"].iloc[-1]),
                duration_s=round(dur, 2), path_m=round(float(path), 1),
                displacement_m=round(disp, 1),
                progress_to_goal_m=round(prog, 1),
                peak_speed=round(float(r["speed_sm"].max()), 1),
                nearest_def_id=nd_id,
                space_before_m=round(float(d0.min()), 1),
                space_after_m=round(float(d1.min()), 1),
                defender_displacement_m=round(nd_disp, 1),
                behind_line_before=behind_before,
                behind_line_after=behind_after,
            ))
    return out


def add_outcomes(runs: pd.DataFrame, game: str) -> pd.DataFrame:
    ev = pd.read_csv(ROOT / "data" / "raw" / f"{game}_RawEventsData.csv")
    shots = ev[ev["Type"] == "SHOT"].copy()
    shots["team_norm"] = shots["Team"].str.lower()
    runs["shot_within_5s"] = False
    for i, r in runs.iterrows():
        s = shots[(shots["team_norm"] == r["team"]) &
                  (shots["Start Time [s]"] > r["end_time"]) &
                  (shots["Start Time [s]"] <= r["end_time"] + AFTER_S)]
        runs.at[i, "shot_within_5s"] = len(s) > 0
    return runs


def main(game: str):
    df = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}.parquet")
    poss = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_possessions.parquet")
    carriers = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_carriers.parquet")
    vel = add_velocity(df)
    pmap = df[["frame", "period"]].drop_duplicates().set_index("frame")["period"]

    all_runs = []
    for _, p in poss.iterrows():
        seg = vel[(vel["frame"] >= p["start_frame"]) & (vel["frame"] <= p["end_frame"])]
        period = int(pmap.get(p["start_frame"], 1))
        all_runs += runs_in_possession(seg, carriers, int(p["possession_id"]),
                                       p["team"], period)
    runs = pd.DataFrame(all_runs)
    if not runs.empty:
        runs = add_outcomes(runs, game)
    out = ROOT / "data" / "trajectories" / f"{game}_runs.parquet"
    runs.to_parquet(out, index=False)
    print(f"runs detected: {len(runs)} -> {out}")
    if not runs.empty:
        print(runs[["duration_s", "path_m", "peak_speed", "defender_displacement_m",
                    "space_before_m", "space_after_m"]].describe().round(1))
        print("\nshot within 5s:", runs["shot_within_5s"].mean().round(3))
        print("\ntop 5 by defender displacement:")
        print(runs.nlargest(5, "defender_displacement_m")[
            ["runner", "team", "duration_s", "path_m", "peak_speed",
             "defender_displacement_m", "space_before_m", "space_after_m",
             "behind_line_after", "shot_within_5s"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Sample_Game_1")
    main(**vars(ap.parse_args()))
