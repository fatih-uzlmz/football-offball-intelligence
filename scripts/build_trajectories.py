"""Convert Metrica Sports raw tracking CSVs into long-format trajectory tables.

Metrica format (per team file):
    row 0: team labels
    row 1: player numbers
    row 2: Period | Frame | Time [s] | Player11 | '' | Player1 | '' | ... | Ball | ''
    rows 3+: data, each player contributes an (x, y) pair; Ball contributes (x, y).

Coordinates are normalized 0-1 (x = pitch length, y = pitch width).

Output: data/trajectories/<game>.parquet with columns:
    period, frame, time, player_id, team, x, y
Ball rows have player_id='ball', team='ball'.
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "trajectories"


def parse_team_file(path: Path, team: str) -> pd.DataFrame:
    with open(path) as f:
        rows = list(csv.reader(f))
    header = rows[2]
    # columns: Period, Frame, Time [s], then (player, '', player, '', ...) ending with (Ball, '')
    entities = []  # (player_id, x_col)
    for i in range(3, len(header), 2):
        name = header[i].strip()
        entities.append((name.replace("Player", ""), i))
    records = []
    for row in rows[3:]:
        if len(row) < 4 or not row[1]:
            continue
        period = int(float(row[0]))
        frame = int(float(row[1]))
        t = float(row[2])
        for pid, col in entities:
            try:
                x = float(row[col])
                y = float(row[col + 1])
            except (ValueError, IndexError):
                continue
            import math
            if math.isnan(x) or math.isnan(y):
                continue
            records.append((period, frame, t, pid, team, x, y))
    return pd.DataFrame(
        records, columns=["period", "frame", "time", "player_id", "team", "x", "y"]
    )


def main(game: str):
    home = parse_team_file(RAW / f"{game}_RawTrackingData_Home_Team.csv", "home")
    away = parse_team_file(RAW / f"{game}_RawTrackingData_Away_Team.csv", "away")
    # Ball appears in both files; keep one copy (from home)
    for df_ in (home, away):
        df_["player_id"] = df_["player_id"].replace({"Ball": "ball"})
    home.loc[home["player_id"] == "ball", "team"] = "ball"
    away = away[away["player_id"] != "ball"]
    df = pd.concat([home, away], ignore_index=True).sort_values(["frame", "team", "player_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{game}.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}  rows={len(df):,}  frames={df['frame'].nunique():,}  "
          f"duration={df['time'].max():.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Sample_Game_1")
    args = ap.parse_args()
    main(args.game)
