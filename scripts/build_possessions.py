"""Possession segmentation + ball-carrier estimation from tracking data.

Method (MVP heuristic):
  1. Per frame, ball carrier = closest outfield player to the ball
     (must be within CARRIER_DIST_M meters).
  2. Smooth the carrier team with a 1-second majority vote to kill flicker.
  3. Possessions = contiguous same-team stretches; drop < MIN_DUR_S,
     merge gaps < MERGE_GAP_S.

Outputs:
  data/trajectories/<game>_carriers.parquet  (frame, carrier_id, carrier_team)
  data/trajectories/<game>_possessions.parquet
      (possession_id, team, start_frame, end_frame, start_time, end_time,
       duration_s, n_frames)

Limitation: pure nearest-player heuristic — no event fusion yet, so
contested/loose-ball phases can flip. Good enough for the run-detector MVP.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PITCH_L, PITCH_W = 105.0, 68.0  # meters
CARRIER_DIST_M = 3.0
SMOOTH_WIN = 25      # frames @25fps = 1s
MIN_DUR_S = 3.0
MERGE_GAP_S = 1.0


def estimate_carriers(df: pd.DataFrame) -> pd.DataFrame:
    players = df[df["team"].isin(("home", "away"))].copy()
    ball = df[df["team"] == "ball"][["frame", "x", "y"]].rename(
        columns={"x": "bx", "y": "by"})
    m = players.merge(ball, on="frame", how="left")
    m["dx"] = (m["x"] - m["bx"]) * PITCH_L
    m["dy"] = (m["y"] - m["by"]) * PITCH_W
    m["dist"] = np.hypot(m["dx"], m["dy"])
    idx = m.dropna(subset=["dist"]).groupby("frame")["dist"].idxmin()
    carriers = m.loc[idx, ["frame", "player_id", "team", "dist"]].rename(
        columns={"player_id": "carrier_id", "team": "carrier_team"})
    # frames with no ball data -> no carrier
    all_frames = pd.DataFrame({"frame": sorted(df["frame"].unique())})
    carriers = all_frames.merge(carriers, on="frame", how="left")
    carriers.loc[carriers["dist"] > CARRIER_DIST_M, ["carrier_id", "carrier_team"]] = [None, None]
    carriers = carriers.drop(columns="dist").sort_values("frame").reset_index(drop=True)
    # smooth team with majority vote (encode as numeric; rolling doesn't do strings)
    enc = carriers["carrier_team"].map({"home": 1.0, "away": -1.0})
    sm = enc.rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
    carriers["carrier_team"] = sm.map(lambda v: "home" if v > 0 else ("away" if v < 0 else np.nan))
    return carriers


def segment_possessions(carriers: pd.DataFrame, fps: float = 25.0) -> pd.DataFrame:
    c = carriers.dropna(subset=["carrier_team"]).copy()
    c["grp"] = (c["carrier_team"] != c["carrier_team"].shift()).cumsum()
    segs = c.groupby(["grp", "carrier_team"], as_index=False).agg(
        start_frame=("frame", "min"), end_frame=("frame", "max"))
    segs["duration_s"] = (segs["end_frame"] - segs["start_frame"] + 1) / fps
    segs = segs[segs["duration_s"] >= MIN_DUR_S].reset_index(drop=True)
    # merge same-team segments separated by a short gap
    merged = []
    for _, r in segs.iterrows():
        if (merged and merged[-1]["team"] == r["carrier_team"]
                and (r["start_frame"] - merged[-1]["end_frame"]) / fps <= MERGE_GAP_S):
            merged[-1]["end_frame"] = r["end_frame"]
        else:
            merged.append({"team": r["carrier_team"],
                           "start_frame": r["start_frame"],
                           "end_frame": r["end_frame"]})
    poss = pd.DataFrame(merged)
    poss["duration_s"] = (poss["end_frame"] - poss["start_frame"] + 1) / fps
    poss["n_frames"] = poss["end_frame"] - poss["start_frame"] + 1
    poss.insert(0, "possession_id", range(len(poss)))
    return poss


def main(game: str):
    df = pd.read_parquet(ROOT / "data" / "trajectories" / f"{game}.parquet")
    carriers = estimate_carriers(df)
    possessions = segment_possessions(carriers)
    # attach wall-clock times
    tmap = df[["frame", "time"]].drop_duplicates().set_index("frame")["time"]
    for col in ("start_frame", "end_frame"):
        possessions[col.replace("frame", "time")] = possessions[col].map(tmap)
    carriers.to_parquet(ROOT / "data" / "trajectories" / f"{game}_carriers.parquet", index=False)
    possessions.to_parquet(ROOT / "data" / "trajectories" / f"{game}_possessions.parquet", index=False)
    print(f"possessions: {len(possessions)}")
    print(possessions.groupby("team")["duration_s"].agg(["count", "mean", "max"]).round(1))
    print("\nlongest 5:")
    print(possessions.nlargest(5, "duration_s")[
        ["possession_id", "team", "duration_s", "start_time", "end_time"]].round(1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Sample_Game_1")
    main(**vars(ap.parse_args()))
