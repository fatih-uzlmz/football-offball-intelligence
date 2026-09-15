"""Video -> trajectories: detect + track + team assignment (+ calibration).

  python scripts/video_to_trajectories.py --video clip.mp4 --preview reports/video/track_preview.jpg

With --calib correspondences.json, foot points are projected to pitch
meters and the result is written as a parquet in the project's trajectory
schema (period, frame, time, player_id, team, x, y), ready for the whole
downstream pipeline (possessions, runs, value model, search, pressing).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from video.detection import Tracker
from video.teams import TeamAssigner, jersey_color
from video.calibration import Homography

ROOT = Path(__file__).resolve().parents[1]


def annotate(frame, dets, teams):
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        tid = d["track_id"]
        if d["cls"] == "ball":
            color, label = (0, 255, 255), "ball"
        else:
            t = teams.get(tid)
            color = (255, 0, 0) if t == "home" else ((0, 0, 255) if t == "away" else (200, 200, 200))
            label = f"{t or '?'} #{tid}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
    return out


def main(a):
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = Tracker(model=a.model, device="cpu")
    assigner = TeamAssigner()

    frames, all_dets, colors = [], [], []
    tracks = {}
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_frames and n >= a.max_frames):
            break
        frames.append(frame)
        dets = tracker.track_frame(frame)
        all_dets.append(dets)
        for d in dets:
            if d["cls"] == "person" and d["track_id"] >= 0:
                tracks.setdefault(d["track_id"], []).append((n, d["xyxy"]))
                if n % 10 == 0:
                    c = jersey_color(frame, d["xyxy"])
                    if c is not None:
                        colors.append(c)
        n += 1
    cap.release()
    print(f"frames={n}  person-tracks={len(tracks)}  "
          f"avg persons/frame={np.mean([sum(1 for d in ds if d['cls']=='person') for ds in all_dets]):.1f}")

    assigner.fit(colors)
    teams = assigner.assign_tracks(tracks, frames)
    n_home = sum(1 for t in teams.values() if t == "home")
    n_away = sum(1 for t in teams.values() if t == "away")
    print(f"teams: home={n_home} away={n_away}")

    # preview: frame with most detections
    bi = int(np.argmax([len(ds) for ds in all_dets]))
    prev = annotate(frames[bi], all_dets[bi], teams)
    outp = ROOT / a.preview
    outp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outp), prev)
    print(f"preview -> {outp} (frame {bi})")

    if a.calib:
        H = Homography.from_json(a.calib)
        rows = []
        for fidx, dets in enumerate(all_dets):
            for d in dets:
                if d["cls"] == "ball":
                    team, pid = "ball", "ball"
                else:
                    team, pid = teams.get(d["track_id"], "away"), str(d["track_id"])
                xm, ym = H.project([Tracker.foot_point(d["xyxy"])])[0]
                rows.append({"period": 1, "frame": fidx + 1,
                             "time": (fidx + 1) / fps, "player_id": pid,
                             "team": team, "x": xm / 105.0, "y": ym / 68.0})
        df = pd.DataFrame(rows)
        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"trajectories -> {out} ({len(df)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--calib", default=None,
                    help="JSON with image/pitch point correspondences")
    ap.add_argument("--out", default="data/trajectories/video_clip.parquet")
    ap.add_argument("--preview", default="reports/video/track_preview.jpg")
    main(ap.parse_args())
