"""Planner (spec §8.3): A* (heapq) on the 0.2 m belief grid.

- Obstacles (walls, blocked_new) are inflated by 0.3 m; fire plus a 1.0 m margin is impassable.
- Cost per cell = 1 + 5·heat_norm + 3·near_wall.
- `distance_field` is a fast Dijkstra over the same costs (exploration's nearest-target search).
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.graph import MCP_Geometric

from .belief import BLOCKED, FIRE, WALL, BeliefGrid

INFLATE_M = 0.3
FIRE_MARGIN_M = 1.0
NEAR_WALL_M = 0.8  # near_wall ramps 1 -> 0 between INFLATE_M and this
SQRT2 = math.sqrt(2.0)

Point = tuple[float, float]
Cell = tuple[int, int]  # (i, j)


class CostMap:
    """Traversable mask + per-cell cost for one belief version (cached on `belief.version`)."""

    def __init__(self, belief: BeliefGrid) -> None:
        self.version = belief.version
        res = belief.res
        obstacle = (belief.state == WALL) | (belief.state == BLOCKED)
        d_obs = distance_transform_edt(~obstacle) * res  # distance from cell centre to the nearest obstacle centre
        fire = belief.state == FIRE
        d_fire = distance_transform_edt(~fire) * res if fire.any() else np.full(obstacle.shape, np.inf)
        # an obstacle cell centre at distance d means its face is ~d - res/2 away
        self.traversable = (d_obs - res / 2 >= INFLATE_M - 1e-6) & (d_fire - res / 2 > FIRE_MARGIN_M)
        self.traversable &= ~obstacle & ~fire
        near_wall = np.clip((NEAR_WALL_M - (d_obs - res / 2)) / (NEAR_WALL_M - INFLATE_M), 0.0, 1.0)
        heat = np.clip(belief.heat, 0.0, 1.0)
        self.cost = 1.0 + 5.0 * heat + 3.0 * near_wall
        self.mcp_cost = np.where(self.traversable, self.cost, np.inf)


def cost_map(belief: BeliefGrid) -> CostMap:
    cm = getattr(belief, "_costmap", None)
    if cm is None or cm.version != belief.version:
        cm = CostMap(belief)
        belief._costmap = cm  # type: ignore[attr-defined]
    return cm


def snap(belief: BeliefGrid, cm: CostMap, x: float, y: float, max_r: int = 10) -> Cell | None:
    """Nearest traversable cell to (x, y) within max_r cells (a drone hugging a wall is still plannable)."""
    i0, j0 = belief.world_to_cell(x, y)
    if belief.inside(i0, j0) and cm.traversable[j0, i0]:
        return i0, j0
    best, bd = None, math.inf
    for dj in range(-max_r, max_r + 1):
        for di in range(-max_r, max_r + 1):
            i, j = i0 + di, j0 + dj
            if belief.inside(i, j) and cm.traversable[j, i]:
                d = di * di + dj * dj
                if d < bd:
                    best, bd = (i, j), d
    return best


_NEIGH = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


def astar(belief: BeliefGrid, start: Point, goal: Point) -> list[Point] | None:
    """A* from start to goal (world xy). Returns waypoints (cell centres, start..goal) or None if no route."""
    cm = cost_map(belief)
    s, g = snap(belief, cm, *start), snap(belief, cm, *goal)
    if s is None or g is None:
        return None
    trav, cost, w = cm.traversable, cm.cost, belief.w

    def h(i: int, j: int) -> float:  # octile, admissible since cost >= 1
        dx, dy = abs(i - g[0]), abs(j - g[1])
        return (dx + dy) + (SQRT2 - 2) * min(dx, dy)

    open_: list[tuple[float, float, int, int]] = [(h(*s), 0.0, *s)]
    gbest = {s[1] * w + s[0]: 0.0}
    parent: dict[int, int] = {}
    while open_:
        _, gc, i, j = heapq.heappop(open_)
        k = j * w + i
        if gc > gbest.get(k, math.inf):
            continue
        if (i, j) == g:
            cells = [(i, j)]
            while k in parent:
                k = parent[k]
                cells.append((k % w, k // w))
            return [belief.cell_center(a, b) for a, b in reversed(cells)]
        for di, dj, step in _NEIGH:
            a, b = i + di, j + dj
            if not (0 <= a < w and 0 <= b < belief.h) or not trav[b, a]:
                continue
            if di and dj and not (trav[j, a] and trav[b, i]):
                continue  # no corner cutting
            ng = gc + step * 0.5 * (cost[j, i] + cost[b, a])
            nk = b * w + a
            if ng < gbest.get(nk, math.inf):
                gbest[nk] = ng
                parent[nk] = k
                heapq.heappush(open_, (ng + h(a, b), ng, a, b))
    return None


class DistanceField:
    """Dijkstra from one start over the cost map (skimage MCP, C speed)."""

    def __init__(self, belief: BeliefGrid, start: Point) -> None:
        self.belief = belief
        cm = cost_map(belief)
        self.start = snap(belief, cm, *start)
        if self.start is None:
            self.cost = np.full((belief.h, belief.w), np.inf)
            self._mcp = None
            return
        self._mcp = MCP_Geometric(cm.mcp_cost, fully_connected=True)
        self.cost, _ = self._mcp.find_costs([(self.start[1], self.start[0])])
        self.cost = np.where(cm.traversable, self.cost, np.inf)

    def nearest(self, mask: np.ndarray) -> Cell | None:
        c = np.where(mask, self.cost, np.inf)
        k = int(np.argmin(c))
        j, i = divmod(k, self.belief.w)
        return (i, j) if np.isfinite(c[j, i]) else None

    def path_to(self, cell: Cell) -> list[Point] | None:
        if self._mcp is None or not np.isfinite(self.cost[cell[1], cell[0]]):
            return None
        return [self.belief.cell_center(i, j) for j, i in self._mcp.traceback((cell[1], cell[0]))]


def path_length(points: list[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def path_hits(path: list[Point], cells: list[Cell], belief: BeliefGrid, margin: float = 0.5) -> bool:
    """True if any of `cells` lies within `margin` m of the path (re-plan trigger, §8.3)."""
    if not path or not cells:
        return False
    pts = np.asarray(path)
    for i, j in cells:
        cx, cy = belief.cell_center(i, j)
        if np.min(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)) <= margin + belief.res / 2:
            return True
    return False


def rdp(points: list[Point], eps: float = 0.15) -> list[Point]:
    """Ramer-Douglas-Peucker simplification."""
    if len(points) < 3:
        return list(points)
    (x0, y0), (x1, y1) = points[0], points[-1]
    dx, dy = x1 - x0, y1 - y0
    n = math.hypot(dx, dy)
    best, bk = -1.0, 0
    for k in range(1, len(points) - 1):
        px, py = points[k]
        d = math.hypot(px - x0, py - y0) if n < 1e-9 else abs(dy * (px - x0) - dx * (py - y0)) / n
        if d > best:
            best, bk = d, k
    if best <= eps:
        return [points[0], points[-1]]
    return rdp(points[: bk + 1], eps)[:-1] + rdp(points[bk:], eps)


def turn_points(points: list[Point], min_deg: float = 35.0) -> list[tuple[Point, float]]:
    """Interior vertices of a simplified path whose heading change exceeds min_deg: [(xy, turn_deg)]."""
    out = []
    for a, b, c in zip(points, points[1:], points[2:]):
        h1 = math.atan2(b[1] - a[1], b[0] - a[0])
        h2 = math.atan2(c[1] - b[1], c[0] - b[0])
        turn = abs(math.degrees((h2 - h1 + math.pi) % (2 * math.pi) - math.pi))
        if turn > min_deg:
            out.append((b, turn))
    return out
