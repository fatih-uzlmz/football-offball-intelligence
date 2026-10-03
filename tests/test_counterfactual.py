"""Tests for counterfactual run scoring (scripts/score_counterfactual.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from score_counterfactual import (WIN, counterfactual_segment,
                                  window_value)
from score_runs import possession_features

FPS = 25.0
TEAM, OPP = "home", "away"


def make_seg(n_frames=100, runner_speed=6.0):
    """Synthetic possession segment: runner 1 sprints, everyone else still."""
    rows = []
    for f in range(n_frames):
        t = f / FPS
        rows.append(dict(period=1, frame=f, time=t, player_id=1, team=TEAM,
                         x=0.30 + (runner_speed / 105.0) * max(0.0, t - 1.0), y=0.5))
        rows.append(dict(period=1, frame=f, time=t, player_id=2, team=TEAM,
                         x=0.45, y=0.4))
        rows.append(dict(period=1, frame=f, time=t, player_id=3, team=OPP,
                         x=0.60, y=0.5))
        rows.append(dict(period=1, frame=f, time=t, player_id=4, team=OPP,
                         x=0.65, y=0.6))
        rows.append(dict(period=1, frame=f, time=t, player_id=0, team="ball",
                         x=0.45 + 0.0005 * f, y=0.42))
    return pd.DataFrame(rows)


def test_frozen_does_not_touch_frames_before_start():
    seg = make_seg()
    cf = counterfactual_segment(seg, runner=1, team=TEAM,
                                start_frame=25, end_frame=75, variant="frozen")
    before = seg[seg["frame"] <= 25].sort_values(["team", "player_id", "frame"])
    before_cf = cf[cf["frame"] <= 25].sort_values(["team", "player_id", "frame"])
    pd.testing.assert_frame_equal(before.reset_index(drop=True),
                                  before_cf.reset_index(drop=True))


def test_frozen_runner_holds_start_position():
    seg = make_seg()
    cf = counterfactual_segment(seg, runner=1, team=TEAM,
                                start_frame=25, end_frame=75, variant="frozen")
    x0 = seg[(seg["team"] == TEAM) & (seg["player_id"] == 1) &
             (seg["frame"] == 25)]["x"].iloc[0]
    moved = cf[(cf["team"] == TEAM) & (cf["player_id"] == 1) &
               (cf["frame"] > 25) & (cf["frame"] <= 75)]
    assert (moved["x"] == x0).all(), "frozen runner must hold start x"


def test_frozen_leaves_ball_features_identical():
    seg = make_seg()
    cf = counterfactual_segment(seg, runner=1, team=TEAM,
                                start_frame=25, end_frame=90, variant="frozen")
    _, F = possession_features(seg, TEAM, 1)
    _, Fc = possession_features(cf, TEAM, 1)
    # ball x/y, ball speed, ball distance-to-goal, ball velocity: untouched
    np.testing.assert_allclose(F[:, :5], Fc[:, :5], equal_nan=True)
    # pooled attacker features DO change after the run starts
    assert not np.allclose(F[90, 5:11], Fc[90, 5:11], equal_nan=True), \
        "counterfactual should move pooled attacker features"
    # but nothing changes at/before the run start
    np.testing.assert_allclose(F[:26], Fc[:26], equal_nan=True)


def test_drift_matches_frozen_when_runner_was_still():
    seg = make_seg(runner_speed=0.0)  # runner never moves pre-start
    frozen = counterfactual_segment(seg, 1, TEAM, 25, 75, "frozen")
    drift = counterfactual_segment(seg, 1, TEAM, 25, 75, "drift")
    fm = frozen[(frozen["team"] == TEAM) & (frozen["player_id"] == 1)]
    dm = drift[(drift["team"] == TEAM) & (drift["player_id"] == 1)]
    pd.testing.assert_series_equal(fm["x"].reset_index(drop=True),
                                   dm["x"].reset_index(drop=True),
                                   check_names=False)


def test_drift_extrapolates_pre_run_velocity():
    seg = make_seg(runner_speed=6.0)
    cf = counterfactual_segment(seg, 1, TEAM, 25, 75, "drift")
    # pre-run velocity: runner accelerates from t=1.0s (frame 25)
    # at frames < 25 he is still -> drift should be frozen-like early on,
    # but the runner was still pre-start, so drift == frozen here too.
    # Use a start where he IS moving: start at frame 50.
    cf2 = counterfactual_segment(seg, 1, TEAM, 50, 90, "drift")
    r0 = seg[(seg["team"] == TEAM) & (seg["player_id"] == 1)]
    x0 = r0[r0["frame"] == 50]["x"].iloc[0]
    last = cf2[(cf2["team"] == TEAM) & (cf2["player_id"] == 1) &
               (cf2["frame"] == 90)]["x"].iloc[0]
    assert last > x0, "drift must continue forward with pre-run velocity"


def test_runner_id_type_mismatch_still_matches():
    # trajectories store player_id as str while runs store runner as int
    seg = make_seg()
    seg["player_id"] = seg["player_id"].astype(str)
    cf = counterfactual_segment(seg, runner=1, team=TEAM,
                                start_frame=25, end_frame=75, variant="frozen")
    x0 = seg[(seg["team"] == TEAM) & (seg["player_id"] == "1") &
             (seg["frame"] == 25)]["x"].iloc[0]
    moved = cf[(cf["team"] == TEAM) & (cf["player_id"] == "1") &
               (cf["frame"] > 25) & (cf["frame"] <= 75)]
    assert (moved["x"] == x0).all(), "int runner must match str player_id"


def test_refactor_matches_original_implementation():
    # /tmp/fob is a pristine clone of the same commit: the original
    # possession_features must produce identical output to the refactor.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "orig_score_runs", "/tmp/fob/scripts/score_runs.py")
    orig = importlib.util.module_from_spec(spec)
    sys.path.insert(0, "/tmp/fob/scripts")
    sys.path.insert(0, "/tmp/fob/src")
    try:
        spec.loader.exec_module(orig)
    finally:
        sys.path.remove("/tmp/fob/scripts")
        sys.path.remove("/tmp/fob/src")
    seg = make_seg(n_frames=120)
    from score_runs import possession_features  # refactored
    f1, F1 = orig.possession_features(seg, TEAM, 1)
    f2, F2 = possession_features(seg, TEAM, 1)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_allclose(F1, F2, equal_nan=True)
    f1, F1 = orig.possession_features(seg, TEAM, -1)
    f2, F2 = possession_features(seg, TEAM, -1)
    np.testing.assert_allclose(F1, F2, equal_nan=True)


def test_window_value_in_unit_interval_on_synthetic():
    import torch
    from models.value_lstm import ValueLSTM
    ckpt = torch.load(ROOT / "models" / "value_lstm.pt", map_location="cpu",
                      weights_only=False)
    model = ValueLSTM().eval()
    model.load_state_dict(ckpt["model"])
    _, F = possession_features(make_seg(n_frames=100), TEAM, 1)
    v = window_value(model, F, WIN - 1, ckpt["mu"], ckpt["sd"], "cpu")
    assert 0.0 <= v <= 1.0, v
    assert np.isnan(window_value(model, F, 10, ckpt["mu"], ckpt["sd"], "cpu")), \
        "window before frame 74 must be NaN"
