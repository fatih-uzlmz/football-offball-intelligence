"""Pitch calibration: image pixels -> pitch meters.

V1: homography from manually supplied point correspondences
(e.g. the 4 corners of the penalty box). Stored/loaded as JSON:
  {"image": [[px, py], ...], "pitch": [[x_m, y_m], ...]}

V2 (future): automatic calibration from a learned pitch-keypoint model
(SoccerNet calibration challenge style). The interface stays the same:
a Homography that maps foot points to (x, y) meters.
"""
import cv2
import numpy as np


class Homography:
    def __init__(self, H):
        self.H = H

    @classmethod
    def from_correspondences(cls, image_pts, pitch_pts):
        H, _ = cv2.findHomography(np.array(image_pts, dtype=np.float32),
                                  np.array(pitch_pts, dtype=np.float32))
        return cls(H)

    @classmethod
    def from_json(cls, path):
        import json
        d = json.load(open(path))
        return cls.from_correspondences(d["image"], d["pitch"])

    def project(self, pts):
        """(N,2) image points -> (N,2) pitch meters."""
        p = np.concatenate([np.asarray(pts, dtype=np.float32),
                            np.ones((len(pts), 1))], axis=1)
        q = (self.H @ p.T).T
        return q[:, :2] / q[:, 2:3]
