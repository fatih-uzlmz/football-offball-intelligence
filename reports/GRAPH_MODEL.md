# Graph-Attention Value Model (V2) — results

Replaces the V1 pooled-feature LSTM (`models/value_lstm.pt`) with a
per-player attention model trained on three possession-outcome targets.

## Architecture (`src/models/graph_value.py`)

Per frame, multi-head self-attention over 23 tokens (22 players + ball) =
a graph attention network on the complete graph. Each token carries 8
attack-normalized features (position, velocity, speed, team, ball flag,
goal distance/angle). Learned attention pooling collapses the 23 tokens to
one frame vector → 2-layer LSTM → 3-target sigmoid head. The model also
returns per-frame player attention weights (`return_attn=True`), which is
what makes it useful for run analysis: the value function can now "see"
individual players instead of pooled team averages.

Deliberate design choice: no transformer feed-forward block after attention.
Benchmarked ~9x faster per batch on this CPU box (9 s → 1 s). Attention is
doing the relational work; the FFN was paying rent for nothing at 23 tokens.

86,147 params. Trains on CPU in ~25 min.

## Windows (`scripts/build_graph_windows.py`)

`[N, 75, 23, 8]` — 75-frame (3 s) windows, 23 entities, 8 features.
6,488 windows across the 3 games. Split by (game, possession_id) to avoid
leakage. Discovery during build: ball tracking is sparse (~61% frame
coverage), so windows slice only fully-observed stretches.

## Targets (all 100% labeled — no masked labels needed in practice)

| target     | base rate | val AUC (best) |
|------------|-----------|----------------|
| shot       | 2.0%      | **0.978**      |
| box_entry  | 8.8%      | 0.962          |
| line_break | 39.7%     | 0.687          |

V1 baseline: shot AUC 0.944 (single target, pooled features).

Training: masked multi-task BCE with per-target pos_weight, Adam 1e-3,
15 epochs, batch 64 (`scripts/train_graph.py`). Checkpoints:
`models/graph_value.pt` (best shot AUC @ epoch 13) and
`models/graph_value_last.pt` (final epoch). Per-epoch checkpointing was
added after a VM replacement killed a run at epoch ~11 with no checkpoint —
never again.

## Reading the numbers honestly

- **shot 0.978 vs 0.944**: a real improvement, not a rounding error. The
  per-player attention earns its keep where V1's pooled features blurred
  individual movement.
- **box_entry 0.962**: strong. New capability, no V1 baseline.
- **line_break 0.687**: the honest weak spot. 40% base rate, and "breaking
  a defensive line" is genuinely the hardest of the three to read from
  tracking alone. Above random, below useful. Candidate fixes: longer
  windows, defender-relative features, or accepting it's a weak auxiliary
  target that still helps the shared representation.

## Why this matters for the project

The counterfactual module (V1 value model) found that `cf_gain` averaged
~0 because pooled features barely register one player's run. This model
fixes exactly that bottleneck: with per-player attention, re-running the
counterfactuals against V2 should produce sharper, more discriminative
run valuations. That's the next build.
