"""Automatic pitch calibration via pretrained PnLCalib keypoint/line detectors.

Pipeline per frame:
  1. HRNet-W48 heatmap models detect pitch keypoints + line extremities
     (pretrained weights, zero training on our side).
  2. PnLCalib's FramebyFrameCalib fits the camera (heuristic voting over
     RANSAC variants) -> full camera params.
  3. Camera params -> 3x4 projection P -> homography for the z=0 plane ->
     inverted to an image->pitch homography H (3x3).

H maps image pixels (960x540 space) to pitch meters in our 105x68
corner-origin frame (PnLCalib's world frame is pitch-centered, so we add
the [52.5, 34] offset).

If calibration fails (too few keypoints), calibrate() returns None --
callers must have a fallback (previous frame's H, or skip).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as f
import yaml
from PIL import Image

PNLCALIB = Path.home() / "workspace" / "pnlcalib"
sys.path.insert(0, str(PNLCALIB))
from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import (get_keypoints_from_heatmap_batch_maxpool,
                                 get_keypoints_from_heatmap_batch_maxpool_l,
                                 complete_keypoints, coords_to_dict)

L, W = 105.0, 68.0
CAL_W, CAL_H = 960, 540  # detector input size; H is expressed in this space


class Calibrator:
    def __init__(self, weights_kp, weights_line, device="cpu",
                 kp_threshold=0.3434, line_threshold=0.7867,
                 pnl_refine=False, min_keypoints=6, max_rep_err=5.0):
        self.device = device
        self.kp_threshold = kp_threshold
        self.line_threshold = line_threshold
        self.pnl_refine = pnl_refine
        self.min_keypoints = min_keypoints
        self.max_rep_err = max_rep_err
        self.resize = T.Resize((CAL_H, CAL_W))

        cfg = yaml.safe_load(open(PNLCALIB / "config" / "hrnetv2_w48.yaml"))
        cfg_l = yaml.safe_load(open(PNLCALIB / "config" / "hrnetv2_w48_l.yaml"))
        self.model = get_cls_net(cfg)
        self.model.load_state_dict(
            torch.load(weights_kp, map_location=device, weights_only=False))
        self.model.to(device).eval()
        self.model_l = get_cls_net_l(cfg_l)
        self.model_l.load_state_dict(
            torch.load(weights_line, map_location=device, weights_only=False))
        self.model_l.to(device).eval()

    @torch.no_grad()
    def calibrate(self, frame_bgr):
        """frame_bgr: HxWx3 BGR image. Returns (H, n_keypoints) or (None, 0).

        H: 3x3 homography mapping image pixels (in the *resized 960x540*
        space) to pitch meters (105x68, corner origin).
        """
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        t = f.to_tensor(frame).float().unsqueeze(0)
        t = t if t.size()[-1] == CAL_W else self.resize(t)
        t = t.to(self.device)
        _, _, h, w = t.size()

        heatmaps = self.model(t)
        heatmaps_l = self.model_l(t)
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
        kp_dict = coords_to_dict(kp_coords, threshold=self.kp_threshold)
        lines_dict = coords_to_dict(line_coords, threshold=self.line_threshold)
        kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0],
                                                 w=w, h=h, normalize=True)
        n_kp = len(kp_dict)
        if n_kp < self.min_keypoints:
            return None, n_kp

        cam = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)
        cam.update(kp_dict, lines_dict)
        res = cam.heuristic_voting(refine_lines=self.pnl_refine)
        # Quality gate: heuristic_voting returns the lowest-reprojection-error
        # fit; a high rep_err means the keypoints were inconsistent (wrong or
        # underconstrained solution). Reject rather than return a bad H.
        if res is None or res["rep_err"] > self.max_rep_err:
            return None, n_kp
        return homography_from_cam_params(res["cam_params"]), n_kp


def homography_from_cam_params(cam_params):
    """3x3 homography: image pixels (960x540) -> pitch meters (corner origin)."""
    fx = cam_params["x_focal_length"]
    fy = cam_params["y_focal_length"]
    cx, cy = cam_params["principal_point"]
    pos = np.array(cam_params["position_meters"])
    R = np.array(cam_params["rotation_matrix"])
    Q = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    Rt = np.eye(4)[:-1]
    Rt[:, -1] = -pos
    P = Q @ (R @ Rt)          # 3x4: centered pitch meters -> image
    H_centered_from_img = np.linalg.inv(P[:, [0, 1, 3]])
    # PnLCalib world frame is pitch-centered; shift to corner origin
    T = np.array([[1, 0, L / 2], [0, 1, W / 2], [0, 0, 1]])
    return T @ H_centered_from_img


def apply_homography(H, pts):
    """pts: [N,2] image pixels -> [N,2] pitch meters. NaN-safe."""
    pts = np.asarray(pts, dtype=float)
    out = np.full_like(pts, np.nan)
    valid = ~np.isnan(pts).any(axis=1)
    if not valid.any():
        return out
    h = np.concatenate([pts[valid], np.ones((valid.sum(), 1))], axis=1) @ H.T
    out[valid] = h[:, :2] / h[:, 2:3]
    return out
