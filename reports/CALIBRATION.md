# Auto Calibration (V2)

No more manual point correspondences. The pitch homography is estimated
automatically from each frame with pretrained deep keypoint/line detectors.

## Pipeline

1. **Detection** — two HRNetV2-W48 heatmap models (pretrained PnLCalib
   weights, zero training on our side) detect pitch keypoints (57) and
   field-line extremities per frame (`src/calibration/pnlcalib_wrapper.py`).
2. **Geometric fitting** — PnLCalib's `FramebyFrameCalib` fits the full
   camera via heuristic voting over DLT+RANSAC variants. A quality gate
   rejects bad fits: fewer than 6 keypoints or reprojection error > 5px
   returns `None` instead of a wrong homography.
3. **Homography** — camera params → 3×4 projection → homography of the
   ground plane → inverted to image→pitch (105×68, corner origin).
4. **Temporal smoothing** (`src/calibration/smooth.py`) — homographies are
   re-estimated fresh on sampled frames and EMA-smoothed. Homographies are
   **never chained** frame-to-frame (unbounded drift). A 3m disagreement at
   the on-pitch agreement grid triggers re-anchoring (shot cut / fast pan); short
   detection failures hold the last good H.

## The mirror problem

A single broadcast frame of one penalty box is genuinely ambiguous under
pitch symmetries — the detector sometimes returns a geometrically perfect
but left-right mirrored solution (low reprojection error, 20m+ pitch
error). In video this cannot happen between consecutive frames, so the
smoother snaps each new estimate to the orientation (of the 4 dihedral
symmetries) closest to history before comparing. For single images the
orientation is unknowable without extra context.

## Validation (SoccerNet calibration-2023 test set)

`scripts/validate_calibration.py` — for each annotated pitch marking
(ground-plane only; goal posts/crossbar are 3D), maps the annotated image
points to pitch meters and measures distance to the true marking geometry.
Orientation ambiguity scored as best-of-4 (documented above).

| metric | value |
|---|---|
| images | 60 (random SoccerNet test subset) |
| success rate | **67%** (40/60; failures are honest rejections — zoomed-in frames with <6 detectable keypoints) |
| point error, successful frames | **median 0.16m**, mean 0.42m, p90 0.52m over 1,122 annotated marking points |
| points within 1m / 2m | **97.7%** / 99.5% |
| speed | ~15.6 s/frame on 2 vCPUs (detector-bound) |
| orientation | scored best-of-4 dihedral symmetries (single-frame mirror ambiguity, § above) |

## Use

```bash
# video pipeline with automatic calibration (was: --calib correspondences.json)
python scripts/video_to_trajectories.py --video clip.mp4 --auto-calib \
    --calib-stride 15 --out data/trajectories/video_clip.parquet
```

## Honest limits

- Needs visible pitch markings: heavy zoom, night games, shadows, and
  occlusion break detection (returns `None`; the pipeline skips those frames).
- ~15 s/frame on CPU — fine for sampled frames, not real-time. GPU or a
  lighter backbone (ViT-tiny variant in the literature) is the speedup path.
- Single-image left/right/180° orientation is ambiguous; video mode resolves
  it via temporal consistency.
- Homography assumes a flat pitch — ball height is unrecoverable.
- Weights are GPL-2.0 (PnLCalib); fine for research/portfolio, check before
  commercial use.
