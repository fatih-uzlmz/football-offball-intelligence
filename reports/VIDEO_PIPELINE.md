# Video Pipeline — Status & Roadmap

**Goal:** raw match video → trajectories in the project's schema
`(period, frame, time, player_id, team, x, y)` → the entire downstream
pipeline (possessions, run detector, value model, search, pressing) works
unchanged.

## What exists (V1, working)

| Stage | Module | Status |
|-------|--------|--------|
| Detection | `src/video/detection.py` (YOLOv8 person + ball) | ✅ validated on real footage |
| Tracking | ByteTrack via `model.track(persist=True)` | ✅ IDs persist across frames |
| Team assignment | `src/video/teams.py` (jersey-color KMeans, k=2) | ✅ separates kits |
| Calibration | `src/video/calibration.py` (homography from JSON correspondences) | ✅ implemented, needs wide-shot footage to validate end-to-end |
| Export | `scripts/video_to_trajectories.py` | ✅ writes project-schema parquet |

**Validated 2026-09-16** on 10 s of FIFA Beach Soccer World Cup footage
(Switzerland v Senegal): YOLO detected players, ByteTrack held IDs, and the
color clustering separated the red and white kits. Preview at
`reports/video/track_preview.jpg`.

## What's next (V2)

1. **Wide-shot broadcast footage** — the test clip was penalty close-ups.
   Need a full-pitch view to validate calibration end-to-end.
2. **Automatic calibration** — V1 uses manual point correspondences
   (`--calib correspondences.json`). The real solution is a learned
   pitch-keypoint model (SoccerNet camera-calibration challenge style)
   that outputs the homography per frame, handling camera motion.
3. **Hardening** — soccer-specific detector fine-tune (players are small in
   wide shots; YOLOv8n will miss distant players), track stitching across
   shot cuts, goalkeeper/referee as separate classes, ball-track smoothing
   (ball is tiny and often missed).
4. **Scale** — GPU inference, batched frames; a 90-min match at 25 fps is
   135k frames.

## Honest assessment

Detection + tracking + team ID are solved problems with off-the-shelf
tools — that part works today. **Calibration is the hard part**: mapping
broadcast pixels to pitch coordinates under camera motion is an active
research area. The manual-correspondence path unblocks development now;
the learned keypoint model is the real milestone.
