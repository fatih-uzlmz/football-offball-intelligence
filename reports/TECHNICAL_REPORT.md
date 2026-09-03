# Football Off-Ball Intelligence — Technical Report

**Goal:** quantify how a player's movement *without the ball* contributes to
attacking opportunities, using real tracking data and PyTorch.

**Core question:** did an off-ball run create space, move a defender, open a
passing lane, or increase the team's attacking threat?

**Status:** end-to-end pipeline working on 3 full matches. All code at
[github.com/fatih-uzlmz/football-offball-intelligence](https://github.com/fatih-uzlmz/football-offball-intelligence).

---

## 1. Data

**Source:** Metrica Sports free sample datasets (public GitHub repo),
released for education. Teams and players are **anonymized** ("home"/"away",
shirt numbers only) — these are *not* Chelsea matches; they are generic
demo games used to build and prove the system.

| Game | Format | Frames | Duration |
|------|--------|--------|----------|
| Sample_Game_1 | CSV (tracking + events) | 145,006 | ~96 min |
| Sample_Game_2 | CSV (tracking + events) | 141,156 | ~94 min |
| Sample_Game_3 | Legacy FIFA EPTS (`tracking.txt` + `events.json` + `metadata.xml`) | 143,761 | ~96 min |

- Tracking: 25 fps, normalized 0–1 pitch coordinates for all 22 players + ball.
- Game 3 needed a custom parser (`scripts/build_trajectories_g3.py`): the
  `.txt` format stores `frame:x,y;…:ball_x,ball_y` with column order taken
  from the metadata player list; events JSON was normalized into the same
  CSV schema as Games 1–2.
- All games converted to one schema: `data/trajectories/<game>.parquet`
  with `(period, frame, time, player_id, team, x, y)` — **~9.7M rows total**.

**Key decision:** we went *tracking-data first* instead of building the
video pipeline (detection + broadcast camera calibration) up front. Same
trajectory schema either way, so the CV pipeline can bolt on later without
rewriting anything.

## 2. Possession segmentation + ball-carrier estimation

`scripts/build_possessions.py`

- Ball carrier per frame = closest outfield player within **3 m** of the ball.
- Carrier *team* smoothed with a 1-second majority vote (kills flicker during
  duels/loose balls).
- Possessions = contiguous same-team stretches; dropped if < 3 s; merged
  across gaps < 1 s.
- **Result: 748 possessions** (Game 1: 237, Game 2: 232, Game 3: 279),
  averaging ~14–21 s per team.

## 3. Heuristic run detector (Phase 3)

`scripts/detect_runs.py`

Before any ML, a rule-based detector flags candidate off-ball runs. A run is
an attacker (not the ball carrier) sustaining **> 4 m/s for ≥ 1 s**,
covering **≥ 5 m** (speed smoothed over ~0.5 s, 12-frame gap tolerance).

Per run we record:
- `duration_s`, `path_m`, `displacement_m`, `peak_speed`
- `progress_to_goal_m` (attack direction **measured empirically per
  game/period** from mean ball displacement — no hardcoded assumptions)
- `nearest_def_id`, `space_before_m` / `space_after_m` (nearest-defender distance)
- `defender_displacement_m` (how far *that defender* moved during the run)
- `behind_line_before` / `behind_line_after` (runner vs. second-last defender,
  offside-line style)
- `shot_within_5s` (joined from the event feed)

**Result: 3,001 runs** (960 / 883 / 1,158). Median run: ~2.7 s, ~12 m,
peak ~6–8 m/s. ~5% are followed by a shot within 5 s.

## 4. Value model (Phase 5) — PyTorch

**The learning problem:** `OffBallValue(run) = V(after) − V(before)`, where
`V(state)` = probability the possession team shoots within the next 5 s.

`scripts/build_windows.py` → `src/models/value_lstm.py` → `scripts/train.py`

- **Windows:** 3 s (75 frames @25 fps), stride 1 s, within possessions →
  **6,696 windows** (shot rate ~2%).
- **Features (11, attack-normalized** so x=1 is always the opponent goal):
  ball x/y, ball speed, ball distance-to-goal, ball velocity toward goal,
  attackers ahead of ball, attacker/defender centroid x, compactness (mean
  attacker min-distance to a defender), pressure (closest defender to ball),
  attackers in box.
- **Model V1:** 2-layer LSTM (hidden 64, dropout 0.2) → MLP head → sigmoid.
  `[B, 75, 11] → P(shot)`.
- **Training:** split **by possession** (no leakage), BCEWithLogitsLoss with
  `pos_weight` for the rare shot class, Adam 1e-3, 15 epochs, CPU.
- **Result: validation AUC 0.944.** The model reliably separates dangerous
  from safe attacking states.
- Saved to `models/value_lstm.pt` (with normalization stats).

## 5. Run scoring — the payoff

`scripts/score_runs.py`

Every run gets the model's 75-frame window ending at run start (`V_before`)
and at run end (`V_after`); `value_gained = V_after − V_before`. Features
computed vectorized per possession (NumPy broadcasting, not per-frame
Python loops).

- **1,400 runs scored** (the rest sit too close to a possession edge for full
  75-frame windows — fixable with padding later).
- Mean value gained per game: **+0.18 / +0.11 / +0.08**.
- **Best run in the dataset:** Game 2, away #24 — 10.6 s, 64.7 m at
  7.6 m/s, dragged defender #3 **35.7 m**, value **0.00 → 0.93 (+0.93)**.
  The run single-handedly took the attack from nothing to a 93% shot
  probability. (`reports/run_value_top.png`)

## 6. Visualization

`src/visualization/pitch.py` + `scripts/plot_{frame,possession,run}.py`

- Real 105×68 m pitch, mowed stripes, correct boxes/spots/arcs, broadcast styling.
- Frame view (all 22 + ball), possession view (faint trails, bold ball path
  with direction arrows), run view (runner + dragged-defender trails,
  before/after metrics and value in the title).

## 7. Tech stack

Python · pandas · pyarrow · NumPy · matplotlib · **PyTorch (CPU)** ·
scikit-learn (AUC) · GitHub API (repo publishing).

## 8. Limitations (honest)

- Tracking data is anonymized demo matches, not Chelsea — the system is
  proven, the clubs are not identified.
- Ball-carrier estimation is a nearest-player heuristic; no event fusion yet.
- Only ~47% of runs scored (window-edge effects).
- One target only (shot probability); no passing-lane or xT modeling yet.
- No video pipeline yet — trajectory schema is ready for it.

---

## 9. Next steps

1. **Tactical search engine (Phase 6):** query interface — "show runs that
   dragged a center-back out," "runs before shots," "overlaps creating a free
   man" — returning clips, pitch viz, model score, geometric explanation.
2. **Pressing module (Phase 7):** same tracking infra → press initiation,
   reaction time, passing options removed, turnovers forced.
3. **Score the remaining runs** (window padding at possession edges).
4. **Video pipeline:** detector + ByteTrack-style tracking + pitch
   calibration → feed the same trajectory schema from raw match video.
5. **Model upgrades:** graph/attention over players instead of pooled
   features; more targets (penalty-area entry, xT); counterfactual runs
   ("what if he hadn't made the run?") as the research centerpiece.
6. **ML-systems extension:** `torch.compile`, mixed precision, PyTorch
   Profiler, DataLoader tuning, DDP — turn it into an ML-systems portfolio
   piece too.
