# Counterfactual Runs V2 — scored with the graph-attention value model

Re-runs the "what if he hadn't made the run?" analysis from
`reports/COUNTERFACTUAL.md`, but the value function is now the V2 graph
model (per-player attention, `models/graph_value.pt`) instead of the V1
pooled-feature LSTM.

`cf_gain = V2_shot(actual end) − V2_shot(counterfactual end)`,
for 2,224 runs (frozen + drift baselines, `scripts/score_counterfactual_v2.py`).

## Headline

V2's counterfactual signal is dramatically stronger and genuinely different:

|                        | V1 (pooled LSTM) | V2 (graph attention) |
|------------------------|------------------|----------------------|
| mean cf_gain (frozen)  | +0.001           | **+0.015**           |
| share of runs > +0.05  | ~3%              | **~10%**             |
| share of runs > +0.10  | —                | **~5%**              |
| corr(V1 cf_gain, V2 cf_gain) | —           | **0.003**            |

The correlation of ~0 is the key number: the two models are not measuring
the same thing. V1's pooled features barely registered one player's
movement, so its cf_gain was mostly noise around zero. V2's per-player
attention sees the run, so its cf_gain is a real signal.

Top run: Game 3 away #21, **+0.784** — 3.2 s, 14 m down the wing, dragged a
defender 17.5 m; shot probability 0.86 → 0.08 frozen. See
`reports/counterfactual_v2/cf_v2_game3_top1.png`.

## What V2 rewards (and doesn't)

- cf_gain barely correlates with run length (r=0.04) or defender
  displacement (r=0.02). It rewards runs that *create shots*, not long runs
  or defender-dragging per se — the attention model reads the game state.
- Weak correlation with V1's temporal value_gained (r=0.15): still a
  different signal, as intended.

## Honest disagreement: the V1 headline run

V1's headline (Game 3 home #8, +0.514 — a 39 m run dragging a defender
41.6 m) scores only **+0.007** under V2, which rates the absolute situation
as low-value (shot prob 0.085 vs V1's 0.525). The two models genuinely
disagree here: V1's pooled features overreacted to the defender
displacement; V2's attention judges the shooting situation as poor
regardless. Without video of the moment, neither can be declared right —
this is exactly the kind of case to eyeball in the demo.

## Method notes

- Per run: patch the runner's (start, end] trajectory (frozen/drift),
  recompute velocities on the patched segment, rebuild the [75, 23, 8]
  token tensor, score the 75-frame window ending at run end. Windows need
  75 fully-observed frames (ball tracking is sparse, ~61% coverage), so
  777 of 3,001 runs can't be scored — same limitation as V1, slightly
  stricter.
- Scoring is per-possession batched (an early version accumulated all
  windows and got OOM-killed).
- Same caveats as V1: defenders don't react in the counterfactual world,
  so cf_gain remains a conservative lower bound.
