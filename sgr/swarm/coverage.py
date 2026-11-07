"""Camera coverage (spec §8.2): a cell is searched when it is inside a drone camera's 90° frustum,
within 5 m, with line of sight on the belief grid.

LOS is a vectorised ray fan marched at quarter-cell steps over `belief.opaque()` (DDA-equivalent).
"""

from __future__ import annotations

import math

import numpy as np

from sgr.contracts import Pose

from .belief import BeliefGrid

FOV_DEG = 90.0
RANGE_M = 5.0
_N_RAYS = 91  # 1 deg apart: 0.09 m arc spacing at 5 m, well under one 0.2 m cell
_OFFSETS = np.radians(np.linspace(-FOV_DEG / 2, FOV_DEG / 2, _N_RAYS))


def camera_footprint(belief: BeliefGrid, pose: Pose, opaque: np.ndarray | None = None) -> np.ndarray:
    """Bool mask [j, i] of the cells this drone's camera sees right now (walls it hits included)."""
    mask = np.zeros((belief.h, belief.w), bool)
    ii, jj, vis, _ = belief.march(pose.x, pose.y, pose.yaw + _OFFSETS, RANGE_M, opaque)
    mask[jj[vis], ii[vis]] = True
    i0, j0 = belief.world_to_cell(pose.x, pose.y)
    if belief.inside(i0, j0):
        mask[j0, i0] = True
    return mask


def in_frustum(pose: Pose, x: float, y: float) -> bool:
    """Is (x, y) inside the 2D camera wedge (ignores occlusion)?"""
    dx, dy = x - pose.x, y - pose.y
    if math.hypot(dx, dy) > RANGE_M:
        return False
    b = (math.atan2(dy, dx) - pose.yaw + math.pi) % (2 * math.pi) - math.pi
    return abs(b) <= math.radians(FOV_DEG / 2)
