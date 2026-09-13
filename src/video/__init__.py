"""Video pipeline: raw match video -> tracking trajectories.

Stages:
  detection  (YOLO person/ball boxes)
  tracking   (ByteTrack track IDs via ultralytics)
  teams      (jersey-color clustering -> home/away)
  calibration(homography image -> pitch meters; manual correspondences V1)
  export     (parquet in the project's trajectory schema)

V1 validates detection+tracking+teams on real footage. Automatic pitch
calibration (learned keypoint model) is the documented next step.
"""
