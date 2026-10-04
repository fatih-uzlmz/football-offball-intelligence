"""Convert SkillCorner open-data broadcast tracking into pipeline trajectory tables.

Input (data/raw/skillcorner/):
    <match_id>_match.json                  lineup, pitch size, teams
    <match_id>_tracking_extrapolated.jsonl  per-frame tracking (meters, pitch
                                          center at (0,0), 10 fps)

Output (data/trajectories/):
    SK_<match_id>.parquet  columns: period, frame, time, player_id, team, x, y
                           (x, y normalized 0-1; ball rows: player_id='ball')
    SK_<match_id>.meta.json  source metadata: fps, pitch, teams, license, counts

Conventions match scripts/build_trajectories.py (Metrica loader):
player_id is the shirt number as a string; team is 'home'/'away'/'ball'.
Orientation is the provider's native frame (pitch center origin); the pipeline
derives attack direction per (period, team) from ball displacement itself, so no
flipping is applied here.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "skillcorner"
OUT = Path(__file__).resolve().parents[1] / "data" / "trajectories"

LICENSE_NOTE = (
    "SkillCorner Open Data (https://github.com/SkillCorner/opendata). "
    "Repository LICENSE is MIT (2020 SkillCorner); README asks users to credit "
    "SkillCorner. No separate data-license file was found as of 2026-10-03."
)


def parse_ts(ts):
    if not ts:
        return None
    h, m, rest = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def main(match_id: str):
    meta = json.load(open(RAW / f"{match_id}_match.json"))
    L = float(meta["pitch_length"])
    W = float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    teams = {home_id: "home", meta["away_team"]["id"]: "away"}
    pmap = {}  # tracking player_id -> (shirt number str, team str)
    for p in meta["players"]:
        num = p.get("number")
        pid = str(num) if num is not None else f"p{p['id']}"
        pmap[p["id"]] = (pid, teams[p["team_id"]])

    records = []
    n_frames = n_periods = 0
    n_detected = n_total_pts = 0
    periods_seen = set()
    with open(RAW / f"{match_id}_tracking_extrapolated.jsonl") as f:
        for line in f:
            d = json.loads(line)
            frame = d["frame"]
            period = d["period"] or 0
            periods_seen.add(period)
            t = parse_ts(d["timestamp"])
            if t is None:
                t = frame / 10.0
            n_frames += 1
            b = d["ball_data"] or {}
            bx, by = b.get("x"), b.get("y")
            if bx is not None and by is not None and not (
                math.isnan(bx) or math.isnan(by)
            ):
                records.append((period, frame, t, "ball", "ball",
                                bx / L + 0.5, by / W + 0.5))
            for p in d["player_data"]:
                x, y = p.get("x"), p.get("y")
                if x is None or y is None or math.isnan(x) or math.isnan(y):
                    continue
                info = pmap.get(p["player_id"])
                if info is None:
                    continue
                pid, team = info
                n_total_pts += 1
                n_detected += 1 if p.get("is_detected") else 0
                records.append((period, frame, t, pid, team,
                                x / L + 0.5, y / W + 0.5))

    df = pd.DataFrame(
        records, columns=["period", "frame", "time", "player_id", "team", "x", "y"]
    ).sort_values(["frame", "team", "player_id"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"SK_{match_id}.parquet"
    df.to_parquet(out, index=False)

    meta_out = {
        "source": "skillcorner-opendata",
        "match_id": match_id,
        "home": meta["home_team"]["name"],
        "away": meta["away_team"]["name"],
        "score": f"{meta['home_team_score']}-{meta['away_team_score']}",
        "date": meta["date_time"],
        "competition": meta["competition_edition"]["name"],
        "fps": 10,
        "pitch_m": [L, W],
        "periods": sorted(periods_seen),
        "frames": n_frames,
        "rows": len(df),
        "player_points": n_total_pts,
        "detected_fraction": round(n_detected / max(n_total_pts, 1), 4),
        "coordinate_note": "native meters, pitch center origin; normalized x/L+0.5, y/W+0.5",
        "license": LICENSE_NOTE,
    }
    with open(OUT / f"SK_{match_id}.meta.json", "w") as f:
        json.dump(meta_out, f, indent=1)
    print(f"wrote {out}  rows={len(df):,}  frames={n_frames:,}  "
          f"detected={meta_out['detected_fraction']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    args = ap.parse_args()
    main(args.match)
