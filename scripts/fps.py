"""FPS-aware windowing helpers.

Windows are defined in SECONDS; frame counts derive from each dataset's fps,
read from the data/trajectories/<game>.meta.json sidecar. The legacy Metrica
sample games have no sidecar and fall back to 25.0 fps (the old hardcode), so
behavior on 25 fps data is bit-identical to the previous constants.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FPS = 25.0

WIN_S = 3.0     # value-model window length, seconds
STRIDE_S = 1.0  # window stride, seconds


def load_fps(game: str) -> float:
    """Frames per second for a game, from its .meta.json sidecar."""
    meta = ROOT / "data" / "trajectories" / f"{game}.meta.json"
    if meta.exists():
        try:
            fps = json.loads(meta.read_text()).get("fps")
            if fps:
                return float(fps)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return DEFAULT_FPS


def win_frames(fps: float = DEFAULT_FPS) -> int:
    """Frames in a WIN_S-second window (75 @25fps, 30 @10fps)."""
    return int(round(WIN_S * fps))


def stride_frames(fps: float = DEFAULT_FPS) -> int:
    """Frames in a STRIDE_S-second stride (25 @25fps, 10 @10fps)."""
    return int(round(STRIDE_S * fps))
