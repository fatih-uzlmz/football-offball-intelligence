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
.venv/bin/python scripts/build_trajectories.py --game Sample_Game_1
.venv/bin/python scripts/plot_frame.py --game Sample_Game_1 --frame 30000
```

## Status

- [x] Repo skeleton + env
- [x] Metrica Sample Game 1 tracking data (145k frames, ~96 min) -> parquet
- [x] 2D pitch frame visualization
- [ ] Possession segmentation + ball-carrier estimation
- [x] Heuristic run detector (Phase 3: 960 runs, geometry before/after)
- [ ] Defender-displacement metric (Phase 4 MVP)
- [ ] PyTorch value model (Phase 5)
