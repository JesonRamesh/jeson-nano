"""Smoke fog for the drone camera feeds shown on the dashboard (P). UI only.

Same look as the firefighter view (three.js FogExp2, colour #202326): per pixel
    f = 1 - exp(-(density * depth)^2),   out = rgb * (1 - f) + fog * f
using the camera's depth buffer. It is applied to the copy that goes to the dashboard; the detector and
Gemma always get the clean frame, so perception is unaffected.
"""

from __future__ import annotations

import os

import numpy as np

# 0.2: corridors fade into smoke past ~6-8 m, a fire 5 m away stays clear. The firefighter's view uses 0.42
# (~4 m), which would hide detections on the feeds. SGR_UI_FOG=0 turns it off.
UI_FOG_DENSITY = float(os.environ.get("SGR_UI_FOG", "0.2"))
FOG_RGB = np.array([0x20, 0x23, 0x26], np.float32)  # the firefighter view's fog colour


def apply_fog(rgb: np.ndarray, depth: np.ndarray, density: float, colour: np.ndarray = FOG_RGB) -> np.ndarray:
    """A new uint8 image with distance fog; rgb (H, W, 3) uint8, depth (H, W) metres."""
    if density <= 0:
        return rgb
    f = 1.0 - np.exp(-np.square(density * depth.astype(np.float32)))
    out = rgb.astype(np.float32) * (1.0 - f[..., None]) + colour * f[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)
