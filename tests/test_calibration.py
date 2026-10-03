"""Tests for auto calibration: homography math + temporal smoother.

The heavyweight Calibrator (HRNet inference) is covered by
scripts/validate_calibration.py on the SoccerNet test set, not here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibration.pnlcalib_wrapper import homography_from_cam_params, apply_homography
from calibration.smooth import HomographySmoother, _ORIENTATIONS


def fake_cam_params():
    # overhead-ish camera: focal 800px, center principal point,
    # 40m up, 30m back, looking at pitch center.
    R = np.eye(3)
    return {"x_focal_length": 800.0, "y_focal_length": 800.0,
            "principal_point": np.array([480.0, 270.0]),
            "position_meters": np.array([0.0, -30.0, 40.0]),
            "rotation_matrix": R}


def test_homography_roundtrip():
    H = homography_from_cam_params(fake_cam_params())
    assert H.shape == (3, 3)
    # pitch center maps somewhere sane in the image, and back again
    xy = np.array([[52.5, 34.0], [0.0, 0.0], [105.0, 68.0]])
    Hinv = np.linalg.inv(H)
    uv = apply_homography(Hinv, xy)
    assert not np.isnan(uv).any()
    back = apply_homography(H, uv)
    np.testing.assert_allclose(back, xy, atol=1e-6)


def test_apply_homography_nan_safe():
    H = homography_from_cam_params(fake_cam_params())
    out = apply_homography(H, np.array([[100.0, 100.0], [np.nan, np.nan]]))
    assert not np.isnan(out[0]).any()
    assert np.isnan(out[1]).all()


def test_smoother_holds_through_failures():
    s = HomographySmoother(alpha=0.5, max_hold=3)
    H = np.eye(3)
    assert s.update(H) is not None
    assert s.update(None) is not None  # hold 1
    assert s.update(None) is not None  # hold 2
    assert s.update(None) is not None  # hold 3
    assert s.update(None) is None      # exceeded: give up


def test_smoother_snaps_mirrored_estimate():
    s = HomographySmoother(alpha=1.0)  # alpha=1: no blending, pure snap test
    H = homography_from_cam_params(fake_cam_params())
    s.update(H)
    # feed the left-right mirrored variant: smoother must snap it back
    H_flip = _ORIENTATIONS["flip_x"] @ H
    out = s.update(H_flip)
    c = np.array([[0., 0.], [105., 0.], [105., 68.], [0., 68.]])
    np.testing.assert_allclose(apply_homography(out, c),
                               apply_homography(H, c), atol=1e-6)


def test_smoother_reanchors_on_shot_cut():
    s = HomographySmoother(alpha=0.35, reanchor_thresh=3.0)
    H = homography_from_cam_params(fake_cam_params())
    s.update(H)
    # a wildly different (but valid) homography -> treated as shot cut
    H2 = H @ np.array([[1., 0, 500.], [0, 1., 0], [0, 0, 1.]])
    out = s.update(H2)
    c = np.array([[0., 0.], [105., 0.], [105., 68.], [0., 68.]])
    np.testing.assert_allclose(apply_homography(out, c),
                               apply_homography(H2, c), atol=1e-6)
