"""Tests for the graph-attention value model (V2)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from models.graph_value import GraphValue, TARGETS
from build_graph_windows import (N_FEAT, N_TOKENS, WIN, add_velocities,
                                 horizon_labels, possession_tokens)
from train_graph import masked_bce


def _mini_seg():
    rows = []
    for f in range(100):
        t = f / 25.0
        for pid in range(1, 12):
            rows.append(dict(period=1, frame=f, time=t, player_id=str(pid),
                             team="home", x=0.3 + 0.001 * pid, y=0.4,
                             vx=0.5, vy=0.0))
        for pid in range(12, 23):
            rows.append(dict(period=1, frame=f, time=t, player_id=str(pid),
                             team="away", x=0.6, y=0.5, vx=0.0, vy=0.0))
        rows.append(dict(period=1, frame=f, time=t, player_id="ball",
                         team="ball", x=0.35, y=0.42, vx=1.0, vy=0.5))
    return pd.DataFrame(rows)


def test_possession_tokens_shape_and_conventions():
    toks, frames, good = possession_tokens(_mini_seg(), "home", "away", 1)
    assert toks.shape == (100, N_TOKENS, N_FEAT)
    assert good.all()
    assert (toks[:, :11, 7] == 1.0).all()    # attackers
    assert (toks[:, 11:22, 7] == -1.0).all()  # defenders
    assert (toks[:, 22, 7] == 0.0).all()      # ball last, team 0
    assert (toks[:, 22, 5] == 0.0).all()      # ball dist_to_ball 0
    assert not np.isnan(toks).any()


def test_attack_normalization_flips_x_and_vx():
    seg = _mini_seg()
    t1, _, _ = possession_tokens(seg, "home", "away", 1)
    t2, _, _ = possession_tokens(seg, "home", "away", -1)
    np.testing.assert_allclose(t2[:, :11, 0], 1 - t1[:, :11, 0], rtol=1e-5)
    np.testing.assert_allclose(t2[:, :11, 2], -t1[:, :11, 2], rtol=1e-5)
    # y untouched
    np.testing.assert_allclose(t2[:, :11, 1], t1[:, :11, 1], rtol=1e-5)


def test_missing_ball_marks_frame_bad():
    seg = _mini_seg()
    seg = seg[~((seg["team"] == "ball") & (seg["frame"] >= 50))]
    toks, frames, good = possession_tokens(seg, "home", "away", 1)
    assert not good[50:].any()
    assert good[:50].all()


def test_model_forward_shapes():
    m = GraphValue().eval()
    x = torch.randn(2, WIN, N_TOKENS, N_FEAT)
    with torch.no_grad():
        logits = m(x)
        assert logits.shape == (2, len(TARGETS))
        logits2, w = m(x, return_attn=True)
        assert w.shape == (2, WIN, N_TOKENS)
        np.testing.assert_allclose(w.sum(-1).numpy(),
                                   np.ones((2, WIN)), rtol=1e-5)


def test_masked_bce_ignores_unknown_labels():
    import torch.nn.functional as F
    torch.manual_seed(0)
    logits = torch.randn(16, 3)
    y = torch.randint(0, 2, (16, 3)).float()
    y[::2, 1] = -1  # half of target-1 labels unknown
    got = masked_bce(logits, y).item()
    # manual: mean BCE over known entries only, with per-target pos_weight
    tot, n = 0.0, 0
    for i in range(3):
        known = y[:, i] >= 0
        rate = y[known, i].mean().item()
        pw = torch.tensor((1 - rate) / max(rate, 1e-6))
        tot += F.binary_cross_entropy_with_logits(
            logits[known, i], y[known, i], pos_weight=pw,
            reduction="sum").item()
        n += known.sum().item()
    assert abs(got - tot / n) < 1e-5


def test_horizon_labels_no_horizon():
    frames = np.arange(100)
    bx = np.full(100, 0.9)
    by = np.full(100, 0.5)
    period = np.ones(100, dtype=np.int64)
    sx = {"home": np.full((100, 11), 0.5), "away": np.full((100, 11), 0.6)}
    b, l = horizon_labels(frames, bx, by, period, sx, 95, 1, "home", "away", 1)
    assert (b, l) == (-1.0, -1.0)


def test_horizon_labels_box_entry():
    frames = np.arange(200)
    bx = np.full(200, 0.5)
    by = np.full(200, 0.5)
    bx[110:120] = 0.95  # ball deep in the box (attack-normalized, adir=1)
    period = np.ones(200, dtype=np.int64)
    sx = {"home": np.full((200, 11), 0.5), "away": np.full((200, 11), 0.6)}
    b, l = horizon_labels(frames, bx, by, period, sx, 100, 1, "home", "away", 1)
    assert b == 1.0
