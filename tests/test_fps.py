"""FPS-aware windowing (scripts/fps.py).

Windows are defined in seconds; frame counts derive from each dataset's fps.
On 25 fps data the derived counts must equal the old hardcoded constants
(WIN=75, STRIDE=25), so legacy behavior is bit-identical.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fps import load_fps, win_frames, stride_frames, DEFAULT_FPS


def test_window_counts_at_25fps_match_old_hardcode():
    assert win_frames(25.0) == 75
    assert stride_frames(25.0) == 25


def test_window_counts_at_10fps():
    assert win_frames(10.0) == 30
    assert stride_frames(10.0) == 10


def test_load_fps_reads_sidecars():
    assert load_fps("SK_1874553") == 10.0
    assert load_fps("DFL_J03WMX") == 25.0


def test_load_fps_falls_back_for_legacy_games():
    assert load_fps("Sample_Game_1") == DEFAULT_FPS == 25.0
    assert load_fps("no_such_game") == DEFAULT_FPS


def test_module_default_constants_preserved():
    # Existing tests import these names and assume 25fps semantics.
    import score_runs
    import score_counterfactual
    import score_counterfactual_v2
    import build_graph_windows
    assert score_runs.WIN == 75 and score_runs.FPS == 25.0
    assert score_counterfactual.WIN == 75 and score_counterfactual.FPS == 25.0
    assert score_counterfactual_v2.WIN == 75
    assert build_graph_windows.WIN == 75


def _synthetic_seg(fps):
    """One attacker, one defender, ball moving at constant 10.5 m/s in x."""
    rows = []
    n = int(fps) + 1  # 1 second of frames
    for f in range(n):
        t = f / fps
        x = 0.30 + 0.10 * t  # 0.1 normalized x per second = 10.5 m/s
        rows.append(dict(period=1, frame=f, time=t, player_id=1,
                         team="home", x=0.45, y=0.5))
        rows.append(dict(period=1, frame=f, time=t, player_id=2,
                         team="away", x=0.60, y=0.5))
        rows.append(dict(period=1, frame=f, time=t, player_id=0,
                         team="ball", x=x, y=0.5))
    return pd.DataFrame(rows)


def test_velocity_features_are_fps_invariant():
    """Same physical motion at 25fps and 10fps must give the same m/s."""
    from score_runs import possession_features
    _, F25 = possession_features(_synthetic_seg(25.0), "home", 1, fps=25.0)
    _, F10 = possession_features(_synthetic_seg(10.0), "home", 1, fps=10.0)
    # ball speed (m/s) and attack-normalized ball vx (m/s), skipping frame 0
    np.testing.assert_allclose(F25[1:, 2], 10.5, rtol=1e-6)
    np.testing.assert_allclose(F10[1:, 2], 10.5, rtol=1e-6)
    np.testing.assert_allclose(F25[1:, 4], 10.5, rtol=1e-6)
    np.testing.assert_allclose(F10[1:, 4], 10.5, rtol=1e-6)


def test_window_covers_same_physical_duration():
    """A full window at either fps spans 3s of the same physical motion."""
    from score_runs import possession_features

    def seg_at(fps, seconds=3.0):
        rows = []
        n = int(seconds * fps) + 1
        for f in range(n):
            t = f / fps
            x = 0.30 + 0.10 * t  # 0.1 normalized x per second
            rows.append(dict(period=1, frame=f, time=t, player_id=1,
                             team="home", x=0.45, y=0.5))
            rows.append(dict(period=1, frame=f, time=t, player_id=2,
                             team="away", x=0.60, y=0.5))
            rows.append(dict(period=1, frame=f, time=t, player_id=0,
                             team="ball", x=x, y=0.5))
        return pd.DataFrame(rows)

    _, F25 = possession_features(seg_at(25.0), "home", 1, fps=25.0)
    _, F10 = possession_features(seg_at(10.0), "home", 1, fps=10.0)
    w25, w10 = win_frames(25.0), win_frames(10.0)
    assert F25.shape[0] == w25 + 1 and F10.shape[0] == w10 + 1
    # ball x displacement over the window: 0.3 normalized at both fps
    np.testing.assert_allclose(F25[w25, 0] - F25[0, 0], 0.3, rtol=1e-6)
    np.testing.assert_allclose(F10[w10, 0] - F10[0, 0], 0.3, rtol=1e-6)
