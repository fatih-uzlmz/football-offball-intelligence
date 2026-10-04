"""Convert DFL/IDSSE (Bassek et al. 2025) Sportec XML tracking into trajectory tables.

Input (data/raw/dfl/):
    DFL_02_01_matchinformation_<comp>_DFL-MAT-<match_id>.xml  teams, pitch, shirts
    DFL_04_03_positions_raw_observed_<comp>_DFL-MAT-<match_id>.xml  25 Hz tracking
    DFL_03_02_events_raw_<comp>_DFL-MAT-<match_id>.xml  (kept raw; not converted)

Output (data/trajectories/):
    DFL_<match_id>.parquet  columns: period, frame, time, player_id, team, x, y
    DFL_<match_id>.meta.json  source metadata

Tracking XML: <FrameSet GameSection=firstHalf|secondHalf TeamId=... PersonId=...>
              <Frame N=.. T=.. X=.. Y=.. .../>, meters, pitch center origin.
TeamId 'BALL' -> ball rows; TeamId 'referee' skipped. PersonId -> shirt number
from the matchinformation XML. x,y normalized 0-1 (x/PitchX+0.5, y/PitchY+0.5).
time = seconds from the match's first frame timestamp (continuous, Metrica-style).

License: CC-BY 4.0 (Bassek, M., Rein, R., Weber, H. et al. Sci Data 12, 195
(2025). https://doi.org/10.1038/s41597-025-04505-y). Commercial-safe.
"""
import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "dfl"
OUT = Path(__file__).resolve().parents[1] / "data" / "trajectories"

COMPETITION = {
    "J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002", "J03WOY": "DFL-COM-000002",
    "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}
MATCH_NAMES = {
    "J03WMX": ("1. FC Köln", "FC Bayern München"),
    "J03WN1": ("VfL Bochum 1848", "Bayer 04 Leverkusen"),
    "J03WPY": ("Fortuna Düsseldorf", "1. FC Nürnberg"),
    "J03WOH": ("Fortuna Düsseldorf", "SSV Jahn Regensburg"),
    "J03WQQ": ("Fortuna Düsseldorf", "FC St. Pauli"),
    "J03WOY": ("Fortuna Düsseldorf", "F.C. Hansa Rostock"),
    "J03WR9": ("Fortuna Düsseldorf", "1. FC Kaiserslautern"),
}
LICENSE_NOTE = (
    "CC-BY 4.0. Bassek, M., Rein, R., Weber, H. et al. 'An integrated dataset "
    "of synchronized spatiotemporal and event data in elite soccer.' Sci Data "
    "12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y. "
    "Via HuggingFace pysport/idsse-data."
)


def parse_meta(match_id: str):
    comp = COMPETITION[match_id]
    path = RAW / f"DFL_02_01_matchinformation_{comp}_DFL-MAT-{match_id}.xml"
    root = ET.parse(path).getroot()
    gen = root.find(".//General")
    env = root.find(".//Environment")
    home_id = gen.get("HomeTeamId")
    pmap = {}
    for team in root.findall(".//Team"):
        tid = team.get("TeamId")
        side = "home" if tid == home_id else "away"
        for pl in team.findall(".//Player"):
            pmap[pl.get("PersonId")] = (pl.get("ShirtNumber"), side)
    return {
        "pitch": (float(env.get("PitchX")), float(env.get("PitchY"))),
        "home": gen.get("HomeTeamName"), "away": gen.get("GuestTeamName"),
        "result": gen.get("Result"), "date": gen.get("KickoffTime"),
        "competition": gen.get("CompetitionName"), "season": gen.get("Season"),
        "pmap": pmap,
    }


def main(match_id: str):
    comp = COMPETITION[match_id]
    meta = parse_meta(match_id)
    L, W = meta["pitch"]
    pmap = meta["pmap"]
    tpath = RAW / f"DFL_04_03_positions_raw_observed_{comp}_DFL-MAT-{match_id}.xml"

    records = []
    t0 = None
    n_framesets = 0
    sec = {"firstHalf": 1, "secondHalf": 2}
    ctx = ET.iterparse(tpath, events=("start", "end"))
    cur = None  # (period, kind, pid, team)
    for ev, el in ctx:
        if ev == "start" and el.tag == "FrameSet":
            tid = el.get("TeamId")
            if tid == "referee":
                cur = None
            elif tid == "BALL":
                cur = (sec[el.get("GameSection")], "ball", "ball", "ball")
            else:
                info = pmap.get(el.get("PersonId"))
                cur = (sec[el.get("GameSection")], "player", info[0], info[1]) \
                    if info else None
            n_framesets += 1
        elif ev == "end" and el.tag == "Frame":
            if cur is not None:
                period, kind, pid, team = cur
                t = datetime.fromisoformat(el.get("T"))
                if t0 is None:
                    t0 = t
                ts = (t - t0).total_seconds()
                try:
                    x = float(el.get("X")); y = float(el.get("Y"))
                except (TypeError, ValueError):
                    x = y = None
                if x is not None:
                    records.append((period, int(el.get("N")), ts, pid, team,
                                    x / L + 0.5, y / W + 0.5))
        elif ev == "end" and el.tag == "FrameSet":
            el.clear()
            cur = None

    df = pd.DataFrame(
        records, columns=["period", "frame", "time", "player_id", "team", "x", "y"]
    ).sort_values(["frame", "team", "player_id"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"DFL_{match_id}.parquet"
    df.to_parquet(out, index=False)

    home, away = MATCH_NAMES[match_id]
    meta_out = {
        "source": "dfl-idsse", "match_id": match_id, "home": home, "away": away,
        "result": meta["result"], "date": meta["date"],
        "competition": f"{meta['competition']} {meta['season']}",
        "fps": 25, "pitch_m": [L, W],
        "periods": sorted(df["period"].unique().tolist()),
        "frames": int(df["frame"].nunique()), "rows": len(df),
        "framesets": n_framesets,
        "coordinate_note": "native meters, pitch center origin; normalized x/L+0.5, y/W+0.5",
        "events_file": f"DFL_03_02_events_raw_{comp}_DFL-MAT-{match_id}.xml (raw, corner-origin 0-105/0-68)",
        "license": LICENSE_NOTE,
    }
    with open(OUT / f"DFL_{match_id}.meta.json", "w") as f:
        json.dump(meta_out, f, indent=1)
    print(f"wrote {out}  rows={len(df):,}  frames={meta_out['frames']:,}  "
          f"duration={df['time'].max()/60:.1f}min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    args = ap.parse_args()
    main(args.match)
