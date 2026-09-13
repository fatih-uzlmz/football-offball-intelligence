"""Player/ball detection + tracking with Ultralytics YOLO.

Uses the built-in ByteTrack tracker (model.track with persist=True),
which returns stable track IDs across frames out of the box.
"""
import numpy as np


class Tracker:
    def __init__(self, model="yolov8n.pt", conf=0.35, device="cpu"):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.conf = conf
        self.device = device
        # COCO classes: 0 = person, 32 = sports ball
        self.person_cls, self.ball_cls = 0, 32

    def track_frame(self, frame):
        """Return list of dicts: track_id, cls, xyxy, conf for one frame."""
        res = self.model.track(frame, persist=True, conf=self.conf,
                               device=self.device, verbose=False,
                               tracker="bytetrack.yaml")[0]
        out = []
        if res.boxes is None:
            return out
        ids = res.boxes.id
        for i, b in enumerate(res.boxes):
            cls = int(b.cls.item())
            if cls not in (self.person_cls, self.ball_cls):
                continue
            out.append({
                "track_id": int(ids[i].item()) if ids is not None else -1,
                "cls": "person" if cls == self.person_cls else "ball",
                "xyxy": b.xyxy.cpu().numpy()[0],
                "conf": float(b.conf.item()),
            })
        return out

    @staticmethod
    def foot_point(xyxy):
        """Bottom-center of box -> the player's pitch contact point."""
        x1, y1, x2, y2 = xyxy
        return np.array([(x1 + x2) / 2, y2])
