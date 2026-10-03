"""Video -> trajectories: detect + track + team assignment (+ calibration).

  python scripts/video_to_trajectories.py --video clip.mp4 --preview reports/video/track_preview.jpg

With --calib correspondences.json, foot points are projected to pitch
meters (manual V1 calibration). With --auto-calib, the pitch homography
is estimated automatically per sampled frame with the pretrained
PnLCalib keypoint/line detectors + temporal smoothing (V2, no manual
points needed). Either way the result is written as a parquet in the
project's trajectory schema (period, frame, time, player_id, team, x, y),
ready for the whole downstream pipeline.
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
from calibration.pnlcalib_wrapper import Calibrator, apply_homography
from calibration.smooth import HomographySmoother

ROOT = Path(__file__).resolve().parents[1]
CAL_W, CAL_H = 960, 540


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

    if a.calib or a.auto_calib:
        if a.calib:
            H = Homography.from_json(a.calib)
            project = lambda pts: H.project(pts)
            frame_H = None
        else:
            cal = Calibrator(a.weights_kp, a.weights_line, device="cpu")
            smoother = HomographySmoother(alpha=0.35)
            h0, w0 = frames[0].shape[:2]
            # scale: detection pixels -> 960x540 calibration space
            S = np.array([[CAL_W / w0, 0, 0], [0, CAL_H / h0, 0], [0, 0, 1.0]])
            frame_H = []
            for fidx, frame in enumerate(frames):
                if fidx % a.calib_stride == 0:
                    Hc, n_kp = cal.calibrate(frame)
                    Hs = smoother.update(Hc)
                    print(f"  calib frame {fidx}: "
                          f"{'ok' if Hc is not None else 'FAIL'} n_kp={n_kp}", flush=True)
                else:
                    Hs = smoother.H
                frame_H.append(None if Hs is None else Hs @ S)
            n_ok = sum(1 for h in frame_H if h is not None)
            print(f"auto-calib: {n_ok}/{len(frames)} frames have a homography")
            project = None  # per-frame below
        rows = []
        for fidx, dets in enumerate(all_dets):
            Hf = frame_H[fidx] if a.auto_calib else None
            if a.auto_calib and Hf is None:
                continue  # no calibration for this frame: skip, don't fabricate
            for d in dets:
                if d["cls"] == "ball":
                    team, pid = "ball", "ball"
                else:
                    team, pid = teams.get(d["track_id"], "away"), str(d["track_id"])
                if a.auto_calib:
                    xm, ym = apply_homography(Hf, [Tracker.foot_point(d["xyxy"])])[0]
                else:
                    xm, ym = project([Tracker.foot_point(d["xyxy"])])[0]
                if np.isnan(xm):
                    continue
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
                    help="JSON with image/pitch point correspondences (manual V1)")
    ap.add_argument("--auto-calib", action="store_true",
                    help="estimate the homography automatically (PnLCalib V2)")
    ap.add_argument("--calib-stride", type=int, default=15,
                    help="run the detector every N frames for --auto-calib")
    ap.add_argument("--weights-kp", default=str(Path.home() / "workspace/pnlcalib/weights/SV_kp"))
    ap.add_argument("--weights-line", default=str(Path.home() / "workspace/pnlcalib/weights/SV_lines"))
    ap.add_argument("--out", default="data/trajectories/video_clip.parquet")
    ap.add_argument("--preview", default="reports/video/track_preview.jpg")
    main(ap.parse_args())
