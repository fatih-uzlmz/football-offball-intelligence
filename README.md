# Football Off-Ball Intelligence

Quantifying off-ball football movement from tracking data with PyTorch:
did an off-ball run create space, move a defender, open a passing lane,
or increase the team's attacking threat?

Research prototype — not affiliated with any club.

## Pipeline

1. **Trajectories** — Metrica Sports sample tracking data converted to
   long-format `(period, frame, time, player_id, team, x, y)` tables.
2. **Run detection** (next) — heuristic off-ball run detector:
   acceleration + distance + defensive-geometry change.
3. **Value model** (next) — PyTorch sequence model estimating attacking
   state before/after a run: `OffBallValue = V(after) - V(before)`.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# fetch free Metrica Sports sample tracking data (~64MB)
mkdir -p data/raw && cd data/raw
for f in Sample_Game_1_RawTrackingData_Home_Team.csv \
         Sample_Game_1_RawTrackingData_Away_Team.csv \
         Sample_Game_1_RawEventsData.csv; do
  curl -sL -o "$f" "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/$f"
done && cd ../..
.venv/bin/python scripts/build_trajectories.py --game Sample_Game_1
.venv/bin/python scripts/build_possessions.py --game Sample_Game_1
.venv/bin/python scripts/detect_runs.py --game Sample_Game_1
.venv/bin/python scripts/plot_run.py --rank 4
```

## Status

- [x] Repo skeleton + env
- [x] Metrica Sample Game 1 tracking data (145k frames, ~96 min) -> parquet
- [x] 2D pitch frame visualization
- [ ] Possession segmentation + ball-carrier estimation
- [x] Heuristic run detector (Phase 3: 960 runs, geometry before/after)
- [ ] Defender-displacement metric (Phase 4 MVP)
- [x] PyTorch value model (Phase 5: LSTM, val AUC 0.944)
- [x] Run scoring: OffBallValue = V(after) - V(before) for ~1,400 runs
- [x] Tactical search engine (Phase 6: `scripts/search_runs.py` — filter,
  rank, and render runs by tactical criteria)
- [x] Pressing module (Phase 7: `scripts/detect_presses.py` — 883 presses,
  reaction time, pressers, forced-turnover rate; `scripts/plot_press.py`)
- [x] Video pipeline V1 (detection + ByteTrack + team colors validated on
  real footage; `src/video/`, `scripts/video_to_trajectories.py`)
- [ ] Auto calibration (learned pitch-keypoint model) + wide-shot validation
