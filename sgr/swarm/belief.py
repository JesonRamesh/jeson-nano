"""Belief grid (spec §8.1) at 0.2 m, built from the PriorMap, plus LiDAR change detection (§8.3).

Indexing: arrays are [j, i] with i = column (x) and j = row counted from the bottom (y),
the same as map_delta cells. Cell (i, j) covers x in [i*res, (i+1)*res), y in [j*res, (j+1)*res).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.ndimage import label

from sgr import config
from sgr.contracts import CellState, Pose, PriorMap

STATES: tuple[CellState, ...] = ("wall_prior", "free", "blocked_new", "fire", "danger", "unsafe")
CODE = {s: k for k, s in enumerate(STATES)}
WALL, FREE, BLOCKED, FIRE, DANGER, UNSAFE = range(len(STATES))

RAY_STEP = 0.05  # m, quarter cell
BLOCK_MARGIN = 0.3  # measured < expected - margin -> something new is there (0.2 m closed rubble gaps in the map)
CONSECUTIVE = 2  # scans needed to mark / clear
CLEAR_BEYOND = 0.4  # m: a ray must reach this far past a blocked cell to count as "passed through"
_LIDAR_ANG = np.radians(np.arange(config.LIDAR_RAYS) * 360.0 / config.LIDAR_RAYS)


class BeliefGrid:
    def __init__(self, wall: np.ndarray, res: float, origin: tuple[float, float], entrance_xy: tuple[float, float]):
        self.res = float(res)
        self.origin = (float(origin[0]), float(origin[1]))
        self.h, self.w = wall.shape
        self.state = np.where(wall, WALL, FREE).astype(np.uint8)
        self.searched = np.zeros(wall.shape, bool)
        self.heat = np.zeros(wall.shape, np.float32)  # 0..1, filled by later blocks
        self.version = 0  # bumped on every state change (planner caches key on it)
        self.now = 0.0  # sim time (set by the swarm each tick; dead-end marks are stamped with it)
        self.open_version = 0  # bumped only when the map OPENS up (debris / fire gone): dead ends get a re-try
        self.dead = np.zeros(wall.shape, bool)  # shared: cells a drone looked for and could not see (dead ends)
        self.dead_t = np.full(wall.shape, -np.inf, np.float32)  # when each was marked (they expire: see swarm)
        self._delta: dict[tuple[int, int], CellState] = {}
        self._searched_delta: list[tuple[int, int]] = []
        self._hits: dict[tuple[str, int, int], int] = {}  # (drone, i, j) -> consecutive "shorter" scans
        self._clears: dict[tuple[str, int, int], int] = {}
        self.reachable = self._component(entrance_xy)  # prior: free cells connected to the entrance
        self.n_reachable = int(self.reachable.sum())
        jj, ii = np.nonzero(self.reachable)
        e = self.world_to_cell(*entrance_xy)
        k = int(np.argmin((ii - e[0]) ** 2 + (jj - e[1]) ** 2)) if len(ii) else 0
        self._entrance_cell = (int(ii[k]), int(jj[k])) if len(ii) else e
        self._reach_version = -1
        self._reach = self.reachable

    @classmethod
    def from_prior(cls, prior: PriorMap, res: float = config.BELIEF_RES) -> BeliefGrid:
        free = prior.free  # [iy, ix] at prior.resolution, iy = 0 at y = 0
        k = max(1, int(round(res / prior.resolution)))
        h, w = free.shape[0] // k, free.shape[1] // k
        blocks = free[: h * k, : w * k].reshape(h, k, w, k)
        wall = ~blocks.all(axis=(1, 3))  # any wall sub-pixel -> wall (conservative)
        return cls(wall, k * prior.resolution, prior.origin, (prior.entrance.x, prior.entrance.y))

    # ------------------------------------------------------------ geometry
    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor((x - self.origin[0]) / self.res)), int(math.floor((y - self.origin[1]) / self.res))

    def cell_center(self, i: int, j: int) -> tuple[float, float]:
        return self.origin[0] + (i + 0.5) * self.res, self.origin[1] + (j + 0.5) * self.res

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.w and 0 <= j < self.h

    def opaque(self) -> np.ndarray:
        """What blocks LiDAR / camera line of sight in the belief: walls + new blockages."""
        return (self.state == WALL) | (self.state == BLOCKED)

    def free_mask(self) -> np.ndarray:
        return self.state != WALL

    # ------------------------------------------------------------ state + deltas
    def set_state(self, i: int, j: int, s: CellState) -> bool:
        c = CODE[s]
        if self.state[j, i] == c:
            return False
        if self.state[j, i] in (BLOCKED, FIRE) and c not in (BLOCKED, FIRE):
            self.open_version += 1  # an obstacle went away (Undo): what was cut off may be reachable now
        self.state[j, i] = c
        self._delta[(i, j)] = s
        self.version += 1
        return True

    def mark_searched(self, mask: np.ndarray) -> int:
        new = mask & ~self.searched
        if not new.any():
            return 0
        self.searched |= new
        jj, ii = np.nonzero(new)
        self._searched_delta.extend(zip(ii.tolist(), jj.tolist()))
        return int(new.sum())

    def pop_map_delta(self) -> list[tuple[int, int, CellState]]:
        out = [(i, j, s) for (i, j), s in self._delta.items()]
        self._delta = {}
        return out

    def pop_searched_delta(self) -> list[tuple[int, int]]:
        out, self._searched_delta = self._searched_delta, []
        return out

    def coverage(self) -> float:
        """Searched fraction of the cells still reachable from the entrance on the current belief.

        Uses only what the swarm sensed: a room cut off by debris (blocked_new) drops out of the total,
        so a tree-shaped building with a blocked branch can still reach 100 %.
        """
        reach = self.reachable_now()
        return float((self.searched & reach).sum()) / max(1, int(reach.sum()))

    def reachable_now(self) -> np.ndarray:
        if self._reach_version != self.version:
            open_ = (self.state != WALL) & (self.state != BLOCKED)
            lab, _ = label(open_)
            k = lab[self._entrance_cell[1], self._entrance_cell[0]]
            self._reach = (lab == k) & self.reachable if k > 0 else self.reachable
            self._reach_version = self.version
        return self._reach

    # ------------------------------------------------------------ ray-marching
    def march(self, x: float, y: float, angles: np.ndarray, max_r: float, opaque: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """March rays from (x, y). Returns (ii, jj, visible) of shape (rays, steps) and the hit range per ray.

        visible[r, s] is True for samples before (and at) the first opaque sample along ray r.
        """
        opq = self.opaque() if opaque is None else opaque
        n = int(max_r / RAY_STEP)
        r = (np.arange(1, n + 1) * RAY_STEP)[None, :]
        px = x + np.cos(angles)[:, None] * r
        py = y + np.sin(angles)[:, None] * r
        ii = np.floor((px - self.origin[0]) / self.res).astype(np.int32)
        jj = np.floor((py - self.origin[1]) / self.res).astype(np.int32)
        out = (ii < 0) | (jj < 0) | (ii >= self.w) | (jj >= self.h)
        hit = out.copy()
        ok = ~out
        hit[ok] = opq[jj[ok], ii[ok]]
        before = np.logical_or.accumulate(hit, axis=1)
        visible = ~np.concatenate([np.zeros((hit.shape[0], 1), bool), before[:, :-1]], axis=1)  # incl. 1st hit
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), n)
        rng = np.where(first < n, (first + 1) * RAY_STEP, max_r)
        return np.clip(ii, 0, self.w - 1), np.clip(jj, 0, self.h - 1), visible & ~out, rng

    def expected_ranges(self, x: float, y: float, angles: np.ndarray, max_r: float = config.LIDAR_MAX) -> np.ndarray:
        return self.march(x, y, angles, max_r)[3]

    # ------------------------------------------------------------ change detection (§8.3)
    def update_lidar(self, drone: str, pose: Pose, ranges: list[float] | np.ndarray
                     ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Compare a scan with the belief. Returns (newly blocked cells, newly cleared cells).

        Measured < expected - 0.3 m on 2 consecutive scans -> the hit cell becomes blocked_new.
        A blocked_new cell the rays now pass through, on 2 scans -> cleared back to free.
        """
        meas = np.asarray(ranges, float)
        ang = pose.yaw + _LIDAR_ANG
        ii, jj, vis, exp = self.march(pose.x, pose.y, ang, config.LIDAR_MAX)
        seen_hit: set[tuple[int, int]] = set()  # unexpected hits on cells the belief thinks are free
        hit_now: set[tuple[int, int]] = set()  # every cell a ray ends in this scan
        blocked: list[tuple[int, int]] = []
        for k in range(len(meas)):
            m = meas[k]
            if m >= config.LIDAR_MAX - 0.05:
                continue
            hx, hy = pose.x + math.cos(ang[k]) * (m + 0.05), pose.y + math.sin(ang[k]) * (m + 0.05)
            c = self.world_to_cell(hx, hy)
            if not self.inside(*c):
                continue
            hit_now.add(c)
            if m < exp[k] - BLOCK_MARGIN and self.state[c[1], c[0]] not in (WALL, BLOCKED):
                seen_hit.add(c)
        for c in seen_hit:
            key = (drone, *c)
            self._hits[key] = self._hits.get(key, 0) + 1
            if self._hits[key] >= CONSECUTIVE and self.set_state(c[0], c[1], "blocked_new"):
                blocked.append(c)
        for key in [k for k in self._hits if k[0] == drone and (k[1], k[2]) not in seen_hit]:
            del self._hits[key]
        # clearing: blocked cells that rays pass well beyond (and none ends in) are now known free
        passed: set[tuple[int, int]] = set()
        nsteps = ii.shape[1]
        for k in range(len(meas)):
            s_end = min(nsteps, int((meas[k] - CLEAR_BEYOND) / RAY_STEP))
            if s_end <= 0:
                continue
            ci, cj = ii[k, :s_end], jj[k, :s_end]
            sel = self.state[cj, ci] == BLOCKED
            passed.update(zip(ci[sel].tolist(), cj[sel].tolist()))
        passed -= hit_now
        cleared: list[tuple[int, int]] = []
        for c in passed:
            key = (drone, *c)
            self._clears[key] = self._clears.get(key, 0) + 1
            if self._clears[key] >= CONSECUTIVE and self.set_state(c[0], c[1], "free"):
                cleared.append(c)
        for key in [k for k in self._clears if k[0] == drone and (k[1], k[2]) not in passed]:
            del self._clears[key]
        return blocked, cleared

    # ------------------------------------------------------------ helpers
    def _component(self, xy: tuple[float, float]) -> np.ndarray:
        free = self.state != WALL
        seen = np.zeros_like(free)
        i0, j0 = self.world_to_cell(*xy)
        if not (self.inside(i0, j0) and free[j0, i0]):
            fj, fi = np.nonzero(free)
            if len(fi) == 0:
                return seen
            k = int(np.argmin((fi - i0) ** 2 + (fj - j0) ** 2))
            i0, j0 = int(fi[k]), int(fj[k])
        q = deque([(i0, j0)])
        seen[j0, i0] = True
        while q:
            i, j = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < self.w and 0 <= b < self.h and free[b, a] and not seen[b, a]:
                    seen[b, a] = True
                    q.append((a, b))
        return seen
