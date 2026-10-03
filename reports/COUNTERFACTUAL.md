# Counterfactual Runs — "What if he hadn't made the run?"

**Method:** for each of the 2,259 value-scored runs, rebuild the possession with
the runner replaced by a counterfactual baseline, recompute the 11 value-model
features with the identical pipeline (`score_runs.py`), and evaluate the trained
LSTM on the 75-frame window ending at the run's end:

```
cf_gain = V(actual end) − V(counterfactual end)
```

Unlike temporal scoring (`V_after − V_before`, which compares two different
moments), this compares two versions of the *same* moment — isolating the run.

**Baselines** (`scripts/score_counterfactual.py`):
- `frozen`: runner stands still at his run-start position. "What if he never made the run?"
- `drift`: runner continues with his mean pre-run (1 s) velocity, extrapolated and
  clipped to the pitch. "What if he kept jogging instead of sprinting?"

**Results** (2,259 runs, all 3 games):

| metric | mean | median | p90 | max | > +0.05 | < −0.05 |
|---|---|---|---|---|---|---|
| cf_gain_frozen | +0.0010 | 0.0000 | +0.0152 | +0.5139 | 3.0% | 1.7% |
| cf_gain_drift | +0.0004 | −0.0000 | +0.0073 | +0.2722 | 1.5% | 1.2% |

**Top run:** Game 3, home #8 — 9.2 s, 39.3 m, dragged a defender **41.6 m**:
V 0.525 → 0.011 frozen, **cf_gain +0.514**. Without the run, the attack was
worthless; the run single-handedly created a 52% shot probability.

**Key findings:**
1. **The average run barely registers** (mean ≈ 0) — expected: the model's 11
   features are pooled over all attackers, so one player's trajectory moves only
   attackers-ahead-of-ball, attacker centroid x, compactness, and attackers-in-box.
2. **But the runs that matter, matter a lot** — 3% of runs gain > 0.05; the top
   decile of cf_gain captures the long defender-dragging runs the project set out
   to find.
3. **cf_gain is a different signal, not a louder one**: correlation with
   defender displacement is only 0.05, with temporal `value_gained` only 0.21 —
   and **579 of 1,400 runs disagree in sign** between the two. "Before vs after"
   conflates the run with the possession evolving; "run vs no run" isolates it.
4. **Bonus coverage:** counterfactual scoring needs only the end window, so it
   scores runs too close to possession edges for temporal scoring (the top-5
   cf_gain runs all had `value_gained = NaN`).

**Honest limits** (do not cite cf_gain as full causality):
- Defenders do NOT react in the counterfactual world — they keep observed
  trajectories. In reality a defender follows the runner, so the true effect is
  likely *larger*, not smaller: cf_gain is a conservative lower bound.
- The pooled-feature bottleneck (finding 1) means subtle runs are invisible to
  this metric. A per-player attention/graph model would sharpen it — see
  "Next steps" in the technical report.
- `frozen` is a harsh baseline (a real player would at least jog); `drift` is
  the fairer comparison and gives smaller, more conservative numbers.

**Files:** `scripts/score_counterfactual.py`, `scripts/plot_counterfactual.py`,
`tests/test_counterfactual.py` (8 tests), `data/trajectories/*_runs_counterfactual.parquet`,
`reports/counterfactual/`.
