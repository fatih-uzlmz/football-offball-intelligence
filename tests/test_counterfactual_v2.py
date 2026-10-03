"""Tests for the V2 (graph-model) counterfactual scorer."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_graph_windows import possession_tokens, add_velocities, TARGETS
from build_windows import attack_dirs
from score_counterfactual_v2 import window_from_tokens, variant_window, WIN


@pytest.fixture(scope="module")
def game_seg():
    df = pd.read_parquet(ROOT / "data" / "trajectories" / "Sample_Game_1.parquet")
    poss = pd.read_parquet(ROOT / "data" / "trajectories" / "Sample_Game_1_possessions.parquet")
    runs = pd.read_parquet(ROOT / "data" / "trajectories" / "Sample_Game_1_runs_scored.parquet")
    p = poss.iloc[0]
    team = p["team"]
    seg = df[(df["frame"] >= p["start_frame"]) & (df["frame"] <= p["end_frame"])]
    return seg, team, runs


def test_window_from_tokens_shape(game_seg):
    seg, team, _ = game_seg
    opp = "away" if team == "home" else "home"
    adirs = attack_dirs(
        pd.read_parquet(ROOT / "data" / "trajectories" / "Sample_Game_1.parquet"),
        pd.read_parquet(ROOT / "data" / "trajectories" / "Sample_Game_1_possessions.parquet"))
    pmap = seg[["frame"]].drop_duplicates()
    adir = 1
    toks, frames, good = possession_tokens(add_velocities(seg), team, opp, adir)
    assert toks is not None
    # a fully-good stretch should yield a valid [75, 23, 8] window
    idx = np.where(good)[0]
    assert len(idx) >= WIN
    w = window_from_tokens(toks, frames, good, int(frames[idx[WIN - 1]]))
    assert w is not None and w.shape == (WIN, 23, 8)
    assert not np.isnan(w).any()
    # end_frame before the window start -> None
    assert window_from_tokens(toks, frames, good, int(frames[0])) is None


def test_variant_window_differs_from_actual(game_seg):
    """Frozen baseline must change the window (runner stops moving)."""
    seg, team, runs = game_seg
    opp = "away" if team == "home" else "home"
    toks_a, frames_a, good_a = possession_tokens(add_velocities(seg), team, opp, 1)
    pids = sorted(seg[seg["team"] == team]["player_id"].unique())
    for _, r in runs.iterrows():
        w_actual = window_from_tokens(toks_a, frames_a, good_a, int(r["end_frame"]))
        w_frozen = variant_window(seg, r["runner"], team, opp, 1,
                                  int(r["start_frame"]), int(r["end_frame"]), "frozen")
        if w_actual is None or w_frozen is None:
            continue
        assert not np.array_equal(w_actual, w_frozen, equal_nan=True)
        # frozen runner's speed token (feature 4) should be ~0 over the
        # frozen span (frames after start_frame); the window's head may
        # still hold pre-run motion, so only check the frozen span.
        fpos = {f: i for i, f in enumerate(frames_a)}
        i = fpos[int(r["end_frame"])]
        wframes = frames_a[i - WIN + 1:i + 1]
        fz = wframes > int(r["start_frame"])
        assert fz.sum() > 5
        ridx = pids.index(str(r["runner"]))
        assert w_frozen[fz, ridx, 4].max() < 0.1
        return
    pytest.skip("no run in the first possession has a fully-observed window")


def test_output_column_contract():
    """Full-run output (once scored) carries the expected columns."""
    out = pd.read_parquet(
        ROOT / "data" / "trajectories" / "Sample_Game_1_runs_counterfactual_v2.parquet")
    if len(out) <= 10:
        pytest.skip("full V2 scoring not finished yet")
    for t in TARGETS:
        assert f"value_v2_{t}" in out.columns
        assert f"value_cf_frozen_{t}" in out.columns
        assert f"value_cf_drift_{t}" in out.columns
    assert "cf_gain_frozen" in out.columns and "cf_gain_drift" in out.columns
    s = out.dropna(subset=["cf_gain_frozen"])
    assert len(s) > 100  # most runs score; ball gaps explain the rest
    # V2 must be more sensitive to single runs than V1's pooled features
    assert s["cf_gain_frozen"].std() > 0.001
