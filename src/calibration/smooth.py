"""Temporal smoothing of per-frame calibration homographies for video.

Design (per the research consensus): NEVER chain homographies frame to
frame -- they drift unboundedly. Instead:
  - estimate H fresh on each sampled frame (every `stride` frames),
  - smooth the 8 homography parameters with an exponential moving average,
  - re-anchor hard whenever a fresh estimate disagrees with the smoothed
    state by more than `reanchor_thresh` meters at the on-pitch agreement grid
    (shot cut or fast pan -> trust the fresh estimate),
  - hold the last good H through short detection failures (up to
    `max_hold` sampled frames), then report failure.

H is normalized so H[2,2] == 1 before smoothing.
"""
import numpy as np

# Agreement between two homographies is measured at a grid of points over the
# CENTRAL region of the frame (middle 60%). The image corners often map to
# far-away extrapolated pitch areas where tiny homography jitter explodes
# into 10m+ disagreements -- using the corners as the re-anchor metric caused
# perpetual false re-anchors on smoothly panning video. The central grid maps
# to the visible pitch where the fit is actually constrained.
_gx = np.linspace(960 * 0.2, 960 * 0.8, 5)
_gy = np.linspace(540 * 0.2, 540 * 0.8, 5)
_AGREE_GRID = np.array([[x, y] for y in _gy for x in _gx])

# Dihedral orientation fixes in pitch space (applied as F @ H_new).
# A single broadcast frame of one penalty box is genuinely ambiguous under
# these symmetries; the detector sometimes picks the mirrored solution with
# low reprojection error. In video the orientation cannot flip between
# frames, so we snap each new estimate to the orientation closest to state.
_ORIENTATIONS = {
    "identity": np.eye(3),
    "flip_x": np.array([[-1., 0, 105.], [0, 1., 0], [0, 0, 1.]]),
    "flip_y": np.array([[1., 0, 0], [0, -1., 68.], [0, 0, 1.]]),
    "flip_180": np.array([[-1., 0, 105.], [0, -1., 68.], [0, 0, 1.]]),
}


def _orient_to_state(H_state, H_new):
    """Best orientation-fixed variant of H_new vs H_state, plus its error."""
    best, best_err = H_new, np.inf
    for F in _ORIENTATIONS.values():
        cand = F @ H_new
        e = _corner_error(H_state, cand)
        if e < best_err:
            best, best_err = cand, e
    return best, best_err


def _norm(H):
    return H / H[2, 2]


def _corner_error(H1, H2):
    """Max pitch-meter disagreement of two homographies at the agreement grid."""
    from .pnlcalib_wrapper import apply_homography
    p1 = apply_homography(H1, _AGREE_GRID)
    p2 = apply_homography(H2, _AGREE_GRID)
    if np.isnan(p1).any() or np.isnan(p2).any():
        return np.inf
    return np.abs(p1 - p2).max()


class HomographySmoother:
    def __init__(self, alpha=0.35, reanchor_thresh=3.0, max_hold=5):
        self.alpha = alpha
        self.reanchor_thresh = reanchor_thresh
        self.max_hold = max_hold
        self.H = None
        self.held = 0

    def update(self, H_new):
        """Feed a fresh estimate (or None on failure). Returns smoothed H or None."""
        if H_new is not None:
            H_new = _norm(H_new)
            if self.H is None:
                self.H, self.held = H_new, 0
                return self.H.copy()
            # snap to the orientation consistent with history; but if NO
            # orientation agrees (shot cut), trust the raw new estimate --
            # snapping a genuinely new view to history picks a wrong mirror.
            snapped, err = _orient_to_state(self.H, H_new)
            if err > self.reanchor_thresh:
                self.H, self.held = H_new, 0  # shot cut / fast pan: re-anchor
            else:
                self.H = _norm(self.alpha * snapped + (1 - self.alpha) * self.H)
                self.held = 0
        else:
            self.held += 1
            if self.held > self.max_hold:
                self.H, self.held = None, 0
        return None if self.H is None else self.H.copy()
