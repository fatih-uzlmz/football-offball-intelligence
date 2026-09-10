"""Pressing module (Phase 7).

Detect pressing events from tracking data: a defending player sprinting
(>4.5 m/s) *toward* the ball carrier (closing >2 m/s), sustained >=0.75s.
Overlapping presser intervals merge into one press event.

Per press: pressers, reaction time (possession start -> first press),
closest approach to carrier, and outcome (forced turnover within 5s?).

Output: data/trajectories/<game>_presses.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
L, W = 105.0, 68.0
FPS = 25.0
SPEED_T = 4.5      # m/s
CLOSE_T = 2.0      # m/s closing speed toward carrier
MIN_LEN = 19       # 0.75s sustained
GAP = 6            # gap tolerance
TURNOVER_S = 5.0


def segments(mask: np.ndarray, min_len: int, gap: int):
    """Contiguous True runs with gap tolerance, min length."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    runs, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 > gap:
            runs.append((s, prev))
            s = i
        prev = i
    runs.append((s, prev))
    return [(a, b) for a, b in runs if b - a + 1 >= min_len]


def detect_game(game: str):
    df = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}.parquet")
    carriers = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_carriers.parquet")
    poss = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}_possessions.parquet")
    poss = poss.sort_values("start_frame").reset_index(drop=True)
    cmap = dict(zip(carriers["frame"], carriers["carrier_id"]))

    presses = []
    for _, p in poss.iterrows():
        T = p["team"]
        D = "away" if T == "home" else "home"
        seg = df[(df["frame"] >= p["start_frame"]) & (df["frame"] <= p["end_frame"])]
        frames = np.sort(seg["frame"].unique())
        n = len(frames)
        if n < MIN_LEN:
            continue
        # defender positions (T, nd, 2)
        dsub = seg[seg["team"] == D]
        dids = sorted(dsub["player_id"].unique())
        piv = dsub.pivot_table(index="frame", columns="player_id", values=["x", "y"])
        Dpos = np.full((n, len(dids), 2), np.nan)
        for j, pid in enumerate(dids):
            for k, col in enumerate(["x", "y"]):
                if (col, pid) in piv.columns:
                    Dpos[:, j, k] = piv[(col, pid)].reindex(frames).to_numpy()
        Dpos_m = Dpos * [L, W]
        # carrier positions (T, 2)
        carr = pd.DataFrame({"frame": frames,
                             "carrier_id": [cmap.get(f) for f in frames]})
        csub = seg[seg["team"] == T][["frame", "player_id", "x", "y"]]
        m = csub.merge(carr, left_on=["frame", "player_id"],
                       right_on=["frame", "carrier_id"], how="inner")
        Cpos = np.full((n, 2), np.nan)
        if not m.empty:
            mp = m.drop_duplicates("frame").set_index("frame")
            Cpos = mp.reindex(frames)[["x", "y"]].to_numpy()
        Cpos_m = Cpos * [L, W]

        dist = np.linalg.norm(Dpos_m - Cpos_m[:, None, :], axis=2)
        vel = np.full_like(Dpos_m, np.nan)
        vel[1:] = np.diff(Dpos_m, axis=0) * FPS
        speed = np.linalg.norm(np.where(np.isnan(vel), 0, vel), axis=2)
        speed[np.isnan(vel[:, :, 0])] = np.nan
        closing = np.full_like(dist, np.nan)
        closing[1:] = -np.diff(dist, axis=0) * FPS
        active = (speed > SPEED_T) & (closing > CLOSE_T) & np.isfinite(dist)

        # per-player press intervals -> merged press events
        ivals = []
        for j, pid in enumerate(dids):
            for a, b in segments(active[:, j], MIN_LEN, GAP):
                ivals.append((a, b, pid))
        ivals.sort()
        events = []
        for a, b, pid in ivals:
            if events and a - events[-1][1] <= 12:
                pa, pb, pp = events[-1]
                events[-1] = (pa, max(pb, b), pp | {pid})
            else:
                events.append((a, b, {pid}))
        # outcome: next possession
        nxt = poss[poss["start_frame"] > p["end_frame"]]
        for a, b, pp in events:
            end_f = frames[b]
            forced = False
            if not nxt.empty:
                q = nxt.iloc[0]
                forced = (q["team"] == D and
                          q["start_frame"] - end_f <= TURNOVER_S * FPS)
            presses.append({
                "possession_id": int(p["possession_id"]),
                "team": D,  # pressing team
                "start_frame": int(frames[a]),
                "end_frame": int(end_f),
                "duration_s": round((b - a + 1) / FPS, 2),
                "n_pressers": len(pp),
                "pressers": ",".join(map(str, sorted(pp))),
                "reaction_s": round((frames[a] - p["start_frame"]) / FPS, 2),
                "min_carrier_dist_m": round(float(np.nanmin(dist[a:b + 1])), 2),
                "forced_turnover": bool(forced),
            })
    presses = pd.DataFrame(presses)
    out = ROOT / "data" / "trajectories" / f"{game}_presses.parquet"
    presses.to_parquet(out, index=False)
    return presses


def report(game, presses):
    mins = 96  # ~per game
    print(f"\n{game}: {len(presses)} presses")
    for team, g in presses.groupby("team"):
        print(f"  {team}: {len(g)} presses "
              f"({len(g)/mins*90:.1f}/90)  "
              f"reaction {g['reaction_s'].mean():.1f}s  "
              f"turnover rate {g['forced_turnover'].mean():.0%}  "
              f"avg pressers {g['n_pressers'].mean():.1f}")


def main(games):
    for g in games:
        report(g, detect_game(g))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+",
                    default=["Sample_Game_1", "Sample_Game_2", "Sample_Game_3"])
    main(**vars(ap.parse_args()))
