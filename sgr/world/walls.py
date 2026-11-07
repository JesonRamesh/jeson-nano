"""grid.png -> merged wall boxes (spec §7.1) and grid/world coordinate helpers (§3).

Grid convention: 0 = wall, 255 = free. Resolution 0.1 m/px. The bottom-left
pixel is world (0, 0), so image rows are flipped.
"""

from __future__ import annotations

import numpy as np

Rect = tuple[int, int, int, int]  # (r0, c0, r1, c1) in image pixels, end-exclusive


def grid_to_boxes(wall: np.ndarray) -> list[Rect]:
    """Greedy merge of wall cells into axis-aligned rectangles.

    For each unused wall cell (row-major), extend right as far as possible,
    then extend down while the whole span is still unused wall.
    """
    wall = np.asarray(wall, dtype=bool)
    h, w = wall.shape
    used = np.zeros_like(wall)
    boxes: list[Rect] = []
    for r in range(h):
        row, urow = wall[r], used[r]
        c = 0
        while c < w:
            if not row[c] or urow[c]:
                c += 1
                continue
            c1 = c
            while c1 < w and row[c1] and not urow[c1]:
                c1 += 1
            r1 = r + 1
            while r1 < h and wall[r1, c:c1].all() and not used[r1, c:c1].any():
                r1 += 1
            used[r:r1, c:c1] = True
            boxes.append((r, c, r1, c1))
            c = c1
    return boxes


def rect_to_world(rect: Rect, grid_h: int, res: float, origin=(0.0, 0.0)):
    """Pixel rect -> (cx, cy, half_x, half_y) in metres."""
    r0, c0, r1, c1 = rect
    x0, x1 = origin[0] + c0 * res, origin[0] + c1 * res
    y0, y1 = origin[1] + (grid_h - r1) * res, origin[1] + (grid_h - r0) * res
    return (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2


def world_to_cell(x: float, y: float, grid_h: int, res: float, origin=(0.0, 0.0)) -> tuple[int, int]:
    """World metres -> (row, col) image indices (may be out of bounds)."""
    c = int(np.floor((x - origin[0]) / res))
    r = grid_h - 1 - int(np.floor((y - origin[1]) / res))
    return r, c


def nearest_wall_normal(free: np.ndarray, x: float, y: float, res: float, origin=(0.0, 0.0), max_m: float = 3.0):
    """Nearest wall along +-x / +-y from (x, y).

    Returns (distance_m, face_x, face_y, normal_yaw). The face point is the
    wall surface. normal_yaw points from the wall back into free space.
    """
    h, w = free.shape
    best = None
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        steps = int(max_m / res)
        for k in range(1, steps):
            px, py = x + dx * k * res, y + dy * k * res
            r, c = world_to_cell(px, py, h, res, origin)
            if not (0 <= r < h and 0 <= c < w) or not free[r, c]:
                # wall face = boundary of this cell facing the start point
                if dx:
                    face = origin[0] + (c + (0 if dx > 0 else 1)) * res
                    d = abs(face - x)
                    cand = (d, face, y, float(np.arctan2(0, -dx)))
                else:
                    face = origin[1] + (h - 1 - r + (0 if dy > 0 else 1)) * res
                    d = abs(face - y)
                    cand = (d, x, face, float(np.arctan2(-dy, 0)))
                if best is None or cand[0] < best[0]:
                    best = cand
                break
    return best
