"""Validate auto calibration on the SoccerNet calibration-2023 test set.

Metric: for each annotated pitch-marking point (ground-plane markings only;
goal posts/crossbar are 3D and excluded), map the image point to pitch meters
with our estimated homography and measure the distance to the true marking
geometry. Reports mean/median error in meters + success rate. This measures
exactly what our pipeline needs: image pixels -> trustworthy pitch coords.

Usage:
    python scripts/validate_calibration.py --inspect   # print annotation format
    python scripts/validate_calibration.py --n 60 --out reports/calibration_validation.json
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "src")
from calibration.pnlcalib_wrapper import Calibrator, apply_homography
from calibration.smooth import _ORIENTATIONS

L, W = 105.0, 68.0
IMG_W, IMG_H = 960, 540

# label -> ("seg", (x1,y1,x2,y2)) or ("circle", (cx,cy,r)); corner-origin meters.
# Convention verified visually: image-right goal = x=105 ("right"),
# far touchline (image top) = y=68 ("top").
GEOM = {
    "Side line top": ("seg", (0, 68, 105, 68)),
    "Side line bottom": ("seg", (0, 0, 105, 0)),
    "Side line left": ("seg", (0, 0, 0, 68)),
    "Side line right": ("seg", (105, 0, 105, 68)),
    "Middle line": ("seg", (52.5, 0, 52.5, 68)),
    "Big rect. left main": ("seg", (16.5, 13.84, 16.5, 54.16)),
    "Big rect. left top": ("seg", (0, 54.16, 16.5, 54.16)),
    "Big rect. left bottom": ("seg", (0, 13.84, 16.5, 13.84)),
    "Big rect. right main": ("seg", (88.5, 13.84, 88.5, 54.16)),
    "Big rect. right top": ("seg", (88.5, 54.16, 105, 54.16)),
    "Big rect. right bottom": ("seg", (88.5, 13.84, 105, 13.84)),
    "Small rect. left main": ("seg", (5.5, 24.84, 5.5, 43.16)),
    "Small rect. left top": ("seg", (0, 43.16, 5.5, 43.16)),
    "Small rect. left bottom": ("seg", (0, 24.84, 5.5, 24.84)),
    "Small rect. right main": ("seg", (99.5, 24.84, 99.5, 43.16)),
    "Small rect. right top": ("seg", (99.5, 43.16, 105, 43.16)),
    "Small rect. right bottom": ("seg", (99.5, 24.84, 105, 24.84)),
    "Circle central": ("circle", (52.5, 34, 9.15)),
    "Circle left": ("circle", (11, 34, 9.15)),
    "Circle right": ("circle", (94, 34, 9.15)),
}
SKIP = {"Line unknown"}  # plus anything starting with "Goal " (3D, not ground plane)


def point_to_geom(px, py, kind, g):
    if kind == "seg":
        x1, y1, x2, y2 = g
        dx, dy = x2 - x1, y2 - y1
        t = np.clip(((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy + 1e-9), 0, 1)
        return np.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
    cx, cy, r = g
    return abs(np.hypot(px - cx, py - cy) - r)


def evaluate_image(cal, jpg_path):
    ann = json.load(open(jpg_path.with_suffix(".json")))
    img = cv2.imread(str(jpg_path))
    t0 = time.time()
    H, n_kp = cal.calibrate(img)
    dt = time.time() - t0
    if H is None:
        return {"ok": False, "n_kp": n_kp, "secs": dt}
    # A single frame is genuinely ambiguous under pitch symmetries; score the
    # best of the 4 orientations (the geometry is what matters for our use).
    best = None
    for name, F in _ORIENTATIONS.items():
        Ho = F @ H
        errs = []
        for label, pts in ann.items():
            if label in SKIP or label.startswith("Goal ") or label not in GEOM:
                continue
            kind, g = GEOM[label]
            uv = np.array([[p["x"] * IMG_W, p["y"] * IMG_H] for p in pts])
            xy = apply_homography(Ho, uv)
            for (x, y) in xy:
                if np.isnan(x):
                    continue
                errs.append(point_to_geom(x, y, kind, g))
        if errs and (best is None or np.mean(errs) < np.mean(best[1])):
            best = (name, errs)
    if best is None:
        return {"ok": False, "n_kp": n_kp, "secs": dt, "note": "no ground-plane annotations"}
    name, errs = best
    return {"ok": True, "n_kp": n_kp, "secs": dt, "orientation": name,
            "mean": float(np.mean(errs)), "median": float(np.median(errs)),
            "n_pts": len(errs), "errs": errs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/soccernet/test")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="reports/calibration_validation.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cal = Calibrator("/home/hatch/workspace/pnlcalib/weights/SV_kp",
                     "/home/hatch/workspace/pnlcalib/weights/SV_lines", device="cpu")
    jpgs = sorted(Path(args.data).glob("*.jpg"))
    rng = np.random.default_rng(args.seed)
    sel = [jpgs[i] for i in rng.choice(len(jpgs), min(args.n, len(jpgs)), replace=False)]

    results = []
    for j in sel:
        r = evaluate_image(cal, j)
        r["img"] = j.name
        results.append(r)
        flag = f"mean={r['mean']:.2f}m" if r["ok"] else "FAIL"
        print(f"{j.name}: {flag} (n_kp={r['n_kp']}, {r['secs']:.1f}s)", flush=True)

    ok = [r for r in results if r["ok"]]
    all_errs = [e for r in ok for e in r["errs"]]
    summary = {
        "n_images": len(results),
        "success_rate": len(ok) / len(results),
        "mean_secs": float(np.mean([r["secs"] for r in results])),
        "point_error_m": {
            "mean": float(np.mean(all_errs)) if all_errs else None,
            "median": float(np.median(all_errs)) if all_errs else None,
            "p90": float(np.percentile(all_errs, 90)) if all_errs else None,
            "frac_lt_1m": float(np.mean(np.array(all_errs) < 1.0)) if all_errs else None,
            "frac_lt_2m": float(np.mean(np.array(all_errs) < 2.0)) if all_errs else None,
            "n_points": len(all_errs),
        },
    }
    print(json.dumps(summary, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary,
               "results": [{k: v for k, v in r.items() if k != "errs"} for r in results]},
              open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
