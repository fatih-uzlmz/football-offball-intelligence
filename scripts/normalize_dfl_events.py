"""Normalize DFL/IDSSE event XML into Metrica-style shot event CSVs.

Input (data/raw/dfl/):
    DFL_02_01_matchinformation_<comp>_DFL-MAT-<match>.xml  (team id -> home/away)
    DFL_03_02_events_raw_<comp>_DFL-MAT-<match>.xml        (events)
Output: data/raw/DFL_<match>_RawEventsData.csv
    columns: Type, Team, Start Time [s]   (Type='SHOT', Team='home'/'away')

Shot tag inventory (checked 2026-10-04 across all 7 raw event files): the
only shot-like child tag present is <ShotAtGoal>. (The task's candidate list
ShotWide/ShotWoodWork/SavedShot/BlockedShot/SuccessfulShot does not occur in
this dataset; ChanceWithoutShot is a chance with NO shot and is excluded.)

Time alignment: the events XML carries per-event CalculatedTimestamp, in the
same wall-clock base as the tracking Frame T attributes. The trajectory
parquet `time` = seconds from the first tracking frame T0, so
    game_seconds = (CalculatedTimestamp - T0).total_seconds()
Validation: kickoff maps near the period's first frame time; every shot lands
inside a period's [t_min, t_max]; shot counts are sane vs the match score.

Additive: touches nothing existing; only writes data/raw/DFL_*_RawEventsData.csv.
"""
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "dfl"

COMPETITION = {
    "J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002", "J03WOY": "DFL-COM-000002",
    "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}
SHOT_TAGS = {"ShotAtGoal"}  # inventory-verified 2026-10-04


def team_sides(match: str) -> dict:
    comp = COMPETITION[match]
    root = ET.parse(
        RAW / f"DFL_02_01_matchinformation_{comp}_DFL-MAT-{match}.xml").getroot()
    gen = root.find(".//General")
    home_id, guest_id = gen.get("HomeTeamId"), gen.get("GuestTeamId")
    return {home_id: "home", guest_id: "away"}


def first_frame_wallclock(match: str) -> datetime:
    """Wall clock of the first tracking frame (T0 of trajectory `time`)."""
    comp = COMPETITION[match]
    path = RAW / f"DFL_04_03_positions_raw_observed_{comp}_DFL-MAT-{match}.xml"
    for ev, el in ET.iterparse(path, events=("end",)):
        if el.tag == "Frame" and el.get("T"):
            t0 = datetime.fromisoformat(el.get("T"))
            el.clear()
            return t0
    raise RuntimeError(f"no Frame T found in {path}")


def parse_shots(match: str, sides: dict, t0: datetime):
    comp = COMPETITION[match]
    root = ET.parse(
        RAW / f"DFL_03_02_events_raw_{comp}_DFL-MAT-{match}.xml").getroot()
    shots, kickoffs = [], {}
    for ev in root.iter("Event"):
        kids = [ch for ch in ev if ch.tag in SHOT_TAGS]
        ts_raw = ev.get("CalculatedTimestamp") or ev.get("EventTime")
        if ts_raw is None:
            continue
        ts = datetime.fromisoformat(ts_raw)
        game_s = (ts - t0).total_seconds()
        for ch in ev:
            if ch.tag == "KickOff":
                sec = ch.get("GameSection", "")
                if "first" in sec:
                    kickoffs["first"] = game_s
                elif "second" in sec:
                    kickoffs["second"] = game_s
        for ch in kids:
            tid = ch.get("Team")
            side = sides.get(tid)
            if side is None:
                print(f"  WARNING: unknown team id {tid} on a ShotAtGoal; skipped")
                continue
            shots.append({"Type": "SHOT", "Team": side,
                          "Start Time [s]": round(game_s, 2),
                          "event_id": ev.get("EventId")})
    # dedupe: one row per EventId (no multi-shot events found, but be safe)
    seen, rows = set(), []
    for s in shots:
        if s["event_id"] in seen:
            continue
        seen.add(s["event_id"])
        rows.append({k: s[k] for k in ("Type", "Team", "Start Time [s]")})
    return rows, kickoffs


def main(match: str):
    sides = team_sides(match)
    t0 = first_frame_wallclock(match)
    shots, kickoffs = parse_shots(match, sides, t0)
    df = pd.DataFrame(shots, columns=["Type", "Team", "Start Time [s]"])
    df = df.sort_values("Start Time [s]").reset_index(drop=True)
    out = ROOT / "data" / "raw" / f"DFL_{match}_RawEventsData.csv"
    df.to_csv(out, index=False)

    # ---- validation ----
    traj = pd.read_parquet(ROOT / "data" / "trajectories" / f"DFL_{match}.parquet")
    ranges = traj.groupby("period")["time"].agg(["min", "max"])
    print(f"{match}: t0={t0.isoformat()}  shots={len(df)} -> {out}")
    print(f"  kickoff game_s: first={kickoffs.get('first', float('nan')):.1f}s "
          f"second={kickoffs.get('second', float('nan')):.1f}s")
    ok = True
    for _, r in df.iterrows():
        t = r["Start Time [s]"]
        if not any(lo <= t <= hi for lo, hi in
                   zip(ranges["min"], ranges["max"])):
            ok = False
            print(f"  BAD: shot at {t:.1f}s outside all periods")
    per_team = df["Team"].value_counts().to_dict()
    print(f"  shots home/away: {per_team}  all-inside-periods={ok}")
    if not ok:
        raise SystemExit("time alignment failed validation")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    main(ap.parse_args().match)
