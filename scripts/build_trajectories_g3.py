"""Convert Metrica Sample Game 3 (legacy FIFA EPTS format) to our trajectory schema.

tracking.txt: one line per frame:
    frame:x1,y1;x2,y2;...;x22,y22:ball_x,ball_y
  - first 11 pairs  = home XI (Team A), in metadata player order
  - next 11 pairs   = away XI (Team B), in metadata player order
  - NaN,NaN when untracked
Periods from metadata: frames 1-69661 = 1st half, 69662-143761 = 2nd.
Team A -> home, Team B -> away. player_id = shirt number.

Also normalizes events.json -> CSV in the Sample_Game_1/2 event format so
the run detector's outcome join works unchanged.

Output: data/trajectories/Sample_Game_3.parquet
        data/raw/Sample_Game_3_RawEventsData.csv
"""
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "trajectories"
META = RAW / "Sample_Game_3_metadata.xml"
HALF1_END = 69661
FPS = 25.0


def player_order():
    root = ET.parse(META).getroot()
    players = root.findall(".//{*}Players/{*}Player")
    home, away = [], []
    for p in players:
        entry = (p.find("{*}ShirtNumber").text, p.get("teamId"))
        (home if p.get("teamId") == "FIFATMA" else away).append(entry)
    # starting XI = first 11 listed per team (matches tracking column order)
    return [("home", n) for n, _ in home[:11]] + [("away", n) for n, _ in away[:11]]


def convert_tracking(order):
    recs = []
    with open(RAW / "Sample_Game_3_tracking.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"(\d+):(.+):(.+)$", line)
            frame = int(m.group(1))
            period = 1 if frame <= HALF1_END else 2
            t = frame / FPS
            pairs = m.group(2).split(";")
            for (team, num), xy in zip(order, pairs):
                x_s, y_s = xy.split(",")
                try:
                    x, y = float(x_s), float(y_s)
                except ValueError:
                    continue
                import math
                if math.isnan(x) or math.isnan(y):
                    continue
                recs.append((period, frame, t, num, team, x, y))
            bx_s, by_s = m.group(3).split(",")
            try:
                bx, by = float(bx_s), float(by_s)
                import math
                if not (math.isnan(bx) or math.isnan(by)):
                    recs.append((period, frame, t, "ball", "ball", bx, by))
            except ValueError:
                pass
    df = pd.DataFrame(recs, columns=["period", "frame", "time", "player_id", "team", "x", "y"])
    df = df.sort_values(["frame", "team", "player_id"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "Sample_Game_3.parquet", index=False)
    print(f"tracking: rows={len(df):,} frames={df['frame'].nunique():,}")


def convert_events():
    ev = json.load(open(RAW / "Sample_Game_3_events.json"))["data"]
    rows = []
    for e in ev:
        st = e.get("subtypes") or []
        subtype = st[0].get("name", "") if isinstance(st, list) and st else ""
        team = "Home" if e["team"]["name"] == "Team A" else "Away"
        rows.append({
            "Team": team,
            "Type": e["type"]["name"],
            "Subtype": subtype,
            "Period": e.get("period", ""),
            "Start Frame": e["start"]["frame"],
            "Start Time [s]": e["start"]["time"],
            "End Frame": e["end"]["frame"],
            "End Time [s]": e["end"]["time"],
            "From": (e.get("from") or {}).get("name", ""),
            "To": (e.get("to") or {}).get("name", ""),
            "Start X": e["start"]["x"] if e["start"]["x"] is not None else "NaN",
            "Start Y": e["start"]["y"] if e["start"]["y"] is not None else "NaN",
            "End X": e["end"]["x"] if e["end"]["x"] is not None else "NaN",
            "End Y": e["end"]["y"] if e["end"]["y"] is not None else "NaN",
        })
    pd.DataFrame(rows).to_csv(RAW / "Sample_Game_3_RawEventsData.csv", index=False)
    types = pd.DataFrame(rows)["Type"].value_counts()
    print("event types:\n", types.head(8).to_string())


if __name__ == "__main__":
    convert_tracking(player_order())
    convert_events()
