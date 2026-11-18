"""Where the sensor-expert code, data and weights live.

The unified detector reuses the experts' audited splits and trained weights. In the
original single-folder project the expert folders (Optical/, IRdata/, SAR/) sit next
to this repository; when the two parts are cloned as separate repositories, clone
them side by side or point AIRCRAFT_EXPERTS_ROOT at the experts repository.
"""

from __future__ import annotations

import os
from pathlib import Path

UNIFIED_ROOT = Path(__file__).resolve().parents[1]
EXPERTS_REPO_NAME = "aircraft-detection-yolo-experts"


def _experts_root() -> Path:
    env = os.environ.get("AIRCRAFT_EXPERTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in (UNIFIED_ROOT.parent, UNIFIED_ROOT.parent / EXPERTS_REPO_NAME):
        if (candidate / "Optical").is_dir():
            return candidate
    return UNIFIED_ROOT.parent / EXPERTS_REPO_NAME


EXPERTS_ROOT = _experts_root()

# Aircraft-free out-of-domain tiles for the refusal test (not distributed with the code).
MAN_TILES_DIR = Path(os.environ.get("MAN_TILES_DIR", EXPERTS_ROOT / "outputs" / "figures" / "man_patches"))


def pretrained_weights(name: str = "yolov8s.pt") -> str:
    """A local copy of the COCO checkpoint if the experts keep one, else let Ultralytics fetch it."""
    local = EXPERTS_ROOT / "IRdata" / name
    return str(local) if local.exists() else name
