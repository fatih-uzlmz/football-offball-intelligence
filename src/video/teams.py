"""Team assignment from jersey colors.

Samples player crops across the video, takes the jersey region
(top-center of each box), clusters mean HSV colors with KMeans(k=2).
The two clusters are the two outfield teams; each track is assigned by
majority vote of its detections. Goalkeepers/referees fall into the
nearest cluster (V1 limitation, documented).
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans


def jersey_color(frame, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return None
    # jersey region: top 15-45% vertically, center 60% horizontally
    crop = frame[y1 + int(0.15 * h):y1 + int(0.45 * h),
                 x1 + int(0.2 * w):x1 + int(0.8 * w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).mean(axis=0)


class TeamAssigner:
    def __init__(self, n_teams=2):
        self.n_teams = n_teams
        self.km = None

    def fit(self, colors):
        X = np.array([c for c in colors if c is not None])
        self.km = KMeans(n_clusters=self.n_teams, n_init=10,
                         random_state=0).fit(X)
        return self

    def predict(self, color):
        if color is None or self.km is None:
            return None
        return "home" if self.km.predict([color])[0] == 0 else "away"

    def assign_tracks(self, tracks_by_id, frames):
        """Majority vote per track_id -> team label dict."""
        votes = {}
        for tid, dets in tracks_by_id.items():
            for fidx, xyxy in dets[:20]:  # sample up to 20 detections
                c = jersey_color(frames[fidx], xyxy)
                t = self.predict(c)
                if t:
                    votes.setdefault(tid, []).append(t)
        return {tid: max(set(v), key=v.count) for tid, v in votes.items() if v}
