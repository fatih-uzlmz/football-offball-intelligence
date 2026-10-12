# Off-Ball Intelligence: Quantifying Football's Invisible Work

**TL;DR:** I built an end-to-end system that watches tracking data and scores
*off-ball runs* — not goals or assists, the runs that create them. Across 27
real matches (43.1M tracking rows), a detector found 27,113 runs; two PyTorch
models score how each run changed the attack's danger; a counterfactual module
asks "what if he hadn't made the run?" Everything is open source:
[github.com/fatih-uzlmz/football-offball-intelligence](https://github.com/fatih-uzlmz/football-offball-intelligence).

*Research prototype — not affiliated with any club. The name says Chelsea
because that's where the obsession started; the data is real matches from
public datasets, no club is identified.*

---

## The question nobody prices

Football analytics is brilliant at the ball. Expected goals, expected threat,
on-ball value — all of it prices what the ball-carrier does. But the sport's
most expensive resource is space, and space is created by the 20 players who
*don't* have the ball. A winger's 60-metre sprint that drags two defenders
wide and opens the cut-back never touches the ball. On the stats sheet, it
doesn't exist.

**Off-Ball Intelligence** is my attempt to price it: detect every off-ball run
in a match, measure what it did to the defence, and score how much attacking
danger it created — with the receipts to prove it.

## The pipeline

Seven stages, one data schema end to end:

1. **Trajectories** — raw tracking data → long-format
   `(period, frame, time, player_id, team, x, y)` parquet tables.
2. **Possession segmentation** — ball-carrier estimation (nearest outfield
   player within 3 m, team smoothed with a 1-second majority vote) →
   contiguous team possessions, 3-second minimum.
3. **Run detection** — a heuristic detector flags candidate runs: an
   attacker (not the carrier) sustaining > 4 m/s for ≥ 1 s, covering ≥ 5 m.
   Per run it records geometry: nearest defender, space before/after,
   **defender displacement** (how far the nearest defender moved during the
   run), whether the runner got behind the offside line, and whether a shot
   followed within 5 seconds.
4. **Windowing** — 3-second (fps-aware) windows, 1-second stride, inside
   possessions → model inputs.
5. **Value models** — two PyTorch models score attacking danger of any state.
6. **Run scoring** — `OffBallValue(run) = V(after) − V(before)`.
7. **Counterfactuals** — re-score each run against "what if he hadn't made
   it?" baselines: *frozen* (runner stands still) and *drift* (runner keeps
   his pre-run velocity).

## The data: from 3 sample games to 27 real matches

The system was first proven on 3 anonymized Metrica sample games. Then it
grew up: **27 real matches** — 20 SkillCorner A-League games (10 fps
broadcast tracking) plus 7 DFL/Bassek Bundesliga matches (25 Hz full optical
tracking) — totaling **43.1M rows across 1.88M frames**. A second's worth of
engineering made every pipeline stage fps-aware, so windows are defined in
seconds, not frames, and a regression test proved the refactored code
reproduces the original artifacts bit-identically.

On the real data: **27,113 runs detected**, 99.8% scored by V1, 94.2% by the
V2 model.

## The models: pooled features → per-player attention

**V1 — the baseline.** A 2-layer LSTM over 11 attack-normalized features
(ball position/speed/distance-to-goal, attacker/defender centroids,
compactness, pressure, attackers in the box), outputting P(shot within 5 s).
Trained on the sample games it reached 0.944 validation AUC. But trained on
three demo games, would it survive real football? Yes: evaluated on held-out
real matches, the retrained V1 hit **0.945 test AUC** (the old sample-trained
model still managed 0.926 — it generalized surprisingly well).

**V2 — the graph model.** Per-player self-attention over all 23 entities
(position, velocity, speed, team, ball flag, 8 features each), a 2-layer
LSTM, and a **3-target head**: shot, penalty-area entry, line break.
Retrained on 18 matches, tested on 5 held-out:

| target     | old (sample-trained) | new (real-data) | delta |
|------------|---------------------:|----------------:|------:|
| shot       | 0.943 | **0.953** | +0.010 |
| box entry  | 0.931 | **0.946** | +0.015 |
| line break | 0.677 | **0.812** | **+0.135** |

The headline: real data barely moved shot/box AUC (the sample models already
generalized), but **line-break prediction jumped +0.135** — exactly where the
old model was weakest, and exactly the target that lives in off-ball
movement.

## The counterfactual: what if he hadn't made the run?

`value_gained = V(after) − V(before)` tells you the run coincided with rising
danger. The counterfactual tells you the run *caused* it: `cf_gain =
V(actual end) − V(counterfactual end)`, where the counterfactual replays the
play with the runner frozen or drifting.

The average run barely registers (mean cf_gain ≈ 0) — as it should; most runs
are filler. But the distribution has teeth: **~9% of runs move the needle by
+0.05 or more**, and the top run in the dataset gained **+0.784**: a run that
took the attack from 1% to 79% shot probability, and would have left it at 1%
if the runner had stayed put.

One subtle result: V1 and V2 counterfactual gains are **uncorrelated
(r = 0.003)**. V1's pooled features can barely see the run; V2's per-player
attention can. They measure different things — V1 grades the situation, V2
grades the run.

## The proof, animated

The demo reel ([video](demo/offball_demo.mp4)) animates the top real-data
run — A-League, away #27, 12.6 m in 2.8 s: value gained +0.78, counterfactual
+0.76, shot-probability ticker climbing 0.00 → 0.78, the runner's trail and
the dragged defender drawn on the pitch.

## The other modules

- **Tactical search engine** (`scripts/search_runs.py`): query runs by
  tactical criteria — "show me runs that dragged a centre-back out", "runs
  before shots", "overlaps creating a free man" — returning pitch visuals,
  model scores, and geometric explanations.
- **Pressing module** (883 presses detected): press initiation, reaction
  time, pressers, forced-turnover rate.
- **Video pipeline V1**: YOLOv8 detection + ByteTrack + jersey-colour team
  assignment, validated on real footage; an auto-calibration module derives
  per-frame pitch homographies from pretrained keypoint/line detectors —
  validated on 60 SoccerNet frames at **67% success, 0.16 m median error**.
  All of it emits the same trajectory schema, so raw match video feeds the
  identical downstream pipeline.

## Honest limitations

- The data is real matches but not *any one club's* data; club-specific
  conclusions would need club-specific data.
- Two matches have frames with fewer than 11 tracked players — sparse graph
  windows there, an honest data gap, not a code bug.
- The SkillCorner matches have no shot events, so shot labels are masked
  there (V1 is DFL-only).
- Ball-carrier estimation is a nearest-player heuristic — no event fusion
  yet.
- The project name says Chelsea; no Chelsea data was used. Kept clean on
  purpose.

## What's next

The research engine is done. The remaining work is the write-up (this post),
sharing with the analytics community, and then club pilots — the offer being
free: *"give me 5 matches of tracking data, I'll send back an off-ball
report."* The pilot is also the data strategy, because club partnerships are
how real tracking data scales.

If you work in football analytics — tracking-data pipelines, run detection,
counterfactual evaluation — I'd love feedback. The code is private, but the
results are open for discussion: open an issue here or reach out. Come argue
with the counterfactuals.

*— Fatih Uzulmez, UC Merced · built with PyTorch, pandas, NumPy, matplotlib*
