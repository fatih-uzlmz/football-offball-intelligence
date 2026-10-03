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
- [x] Possession segmentation + ball-carrier estimation (748 possessions)
- [x] Heuristic run detector (Phase 3: 3,001 runs, geometry before/after)
- [x] Defender-displacement metric (Phase 4 MVP: per-run defender displacement)
- [x] PyTorch value model (Phase 5: LSTM, val AUC 0.944)
- [x] Run scoring: OffBallValue = V(after) - V(before) for ~1,400 runs
- [x] Counterfactual run scoring (`scripts/score_counterfactual.py` — "what if he
  hadn't made the run?": frozen + drift baselines, 2,259 runs, top run +0.514)
- [x] Tactical search engine (Phase 6: `scripts/search_runs.py` — filter,
  rank, and render runs by tactical criteria)
- [x] Pressing module (Phase 7: `scripts/detect_presses.py` — 883 presses,
  reaction time, pressers, forced-turnover rate; `scripts/plot_press.py`)
- [x] Video pipeline V1 (detection + ByteTrack + team colors validated on
  real footage; `src/video/`, `scripts/video_to_trajectories.py`)
- [x] Graph-attention value model V2 (`src/models/graph_value.py`, `models/graph_value.pt` —
  per-frame player self-attention + 3 targets: shot 0.978 / box entry 0.962 /
  line break 0.687 val AUC; see `reports/GRAPH_MODEL.md`)
- [ ] Auto calibration (learned pitch-keypoint model) + wide-shot validation
