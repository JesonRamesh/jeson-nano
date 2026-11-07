"""One drone's controller: a tactical layer (where to go, what to look at) and a reflex layer (every tick).

Tactical (§8.2): go to the nearest reachable unsearched spot in the claimed area, yaw-sweep 360° on
arrival, retarget when the spot gets searched, the path gets blocked, or progress stalls.
Reflex (§8.2): 2 m separation, LiDAR wall repulsion under 0.6 m, 1.2 m/s clamp, smoothing.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_dilation

from sgr import config
from sgr.contracts import Pose, Readings, VelCmd

from .areas import Area
from .belief import BeliefGrid
from .planner import DistanceField, cost_map

# tactical
GOAL_NEAR_CELLS = 3  # a goal is a traversable cell within 0.6 m of an unsearched cell
LOOKAHEAD = 3  # waypoints (0.6 m)
ARRIVE_M = 0.35
RETARGET_EVERY_S = 3.0
STALL_S = 3.0
STALL_MIN_PROGRESS = 0.3  # m of path-distance gained per STALL_S window
SWEEP_WZ = 2.0  # rad/s
# reflex
SEP_M = 2.0
SEP_GAIN = 0.8
WALL_M = 0.6
WALL_GAIN = 1.5
SMOOTH = 0.5  # EMA weight of the new command
YAW_GAIN = 2.5
WZ_MAX = 2.5
ALT_GAIN = 1.0

_LIDAR_ANG = np.radians(np.arange(config.LIDAR_RAYS) * 360.0 / config.LIDAR_RAYS)
_STRUCT = np.ones((2 * GOAL_NEAR_CELLS + 1, 2 * GOAL_NEAR_CELLS + 1), bool)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Agent:
    def __init__(self, drone: str, altitude: float) -> None:
        self.id = drone
        self.altitude = altitude
        self.mode = "IDLE"  # IDLE | TRANSIT | EXPLORING | SWEEP | LOST
        self.area: str | None = None
        self.path: list[tuple[float, float]] | None = None
        self.goal: tuple[int, int] | None = None
        self._goal_t = 0.0
        self._stall_t = 0.0
        self._stall_left = math.inf
        self._sweep_left = 0.0
        self._v = np.zeros(2)
        self.given_up: np.ndarray | None = None  # cells this drone looked for and could not see (bool [j, i])

    # ---------------------------------------------------------------- tactical
    def set_area(self, area: str | None) -> None:
        if area != self.area:
            self.area = area
            self.invalidate()

    def invalidate(self) -> None:
        """Drop the current path (blocked / new area); a new one is planned next tick."""
        self.path, self.goal = None, None
        if self.mode == "SWEEP":
            self._sweep_left = 0.0

    @property
    def task(self) -> str:
        return f"area {self.area}" if self.area else ""

    def _unsearched(self, belief: BeliefGrid, area: Area) -> np.ndarray:
        uns = area.unsearched(belief)
        if self.given_up is not None:
            uns &= ~self.given_up
        return uns

    def _give_up_goal(self, belief: BeliefGrid, area: Area) -> None:
        """Swept / stalled at this goal and the cells near it are still unseen: stop chasing them."""
        if self.goal is None:
            return
        if self.given_up is None:
            self.given_up = np.zeros((belief.h, belief.w), bool)
        i, j = self.goal
        r = GOAL_NEAR_CELLS
        sl = (slice(max(0, j - r), j + r + 1), slice(max(0, i - r), i + r + 1))
        self.given_up[sl] |= area.unsearched(belief)[sl]

    def forget_given_up(self) -> None:
        """The map changed (e.g. debris cleared): cells given up on are worth another try."""
        self.given_up = None

    def _goal_mask(self, belief: BeliefGrid, area: Area) -> np.ndarray:
        cm = cost_map(belief)
        return binary_dilation(self._unsearched(belief, area), _STRUCT) & cm.traversable

    def _plan(self, t: float, pose: Pose, belief: BeliefGrid, area: Area) -> bool:
        """Pick the nearest reachable goal near unsearched cells of the area. False = nothing left here."""
        df = DistanceField(belief, (pose.x, pose.y))
        goal = df.nearest(self._goal_mask(belief, area))
        if goal is None:
            self.path, self.goal = None, None
            return False
        self.goal = goal
        self.path = df.path_to(goal)
        self._goal_t = self._stall_t = t
        self._stall_left = self._remaining(pose)
        return True

    def _remaining(self, pose: Pose) -> float:
        if not self.path:
            return 0.0
        pts = np.asarray(self.path)
        k = int(np.argmin(np.hypot(pts[:, 0] - pose.x, pts[:, 1] - pose.y)))
        seg = np.hypot(np.diff(pts[k:, 0]), np.diff(pts[k:, 1])).sum()
        return float(math.hypot(pts[k, 0] - pose.x, pts[k, 1] - pose.y) + seg)

    def _goal_still_useful(self, belief: BeliefGrid, area: Area) -> bool:
        if self.goal is None:
            return False
        i, j = self.goal
        r = GOAL_NEAR_CELLS
        sl = (slice(max(0, j - r), j + r + 1), slice(max(0, i - r), i + r + 1))
        return bool(self._unsearched(belief, area)[sl].any())

    def step(self, t: float, r: Readings, belief: BeliefGrid, area: Area | None,
             others: list[Pose]) -> tuple[VelCmd, bool]:
        """One control tick. Returns (command, area_done)."""
        pose = r.pose
        want = np.zeros(2)
        yaw_des: float | None = None
        wz_ff = 0.0
        done = False

        if area is None:
            self.mode = "IDLE"
            wz_ff = 0.6  # slow look-around while waiting
        elif self.mode == "SWEEP" and self._sweep_left > 0:
            self._sweep_left -= SWEEP_WZ * config.CONTROL_DT
            wz_ff = SWEEP_WZ
            if self._sweep_left <= 0:
                self.mode = "EXPLORING"
                if area is not None and self._goal_still_useful(belief, area):
                    self._give_up_goal(belief, area)
                self.invalidate()
        else:
            need = (
                self.path is None
                or not self._goal_still_useful(belief, area)
                or t - self._goal_t > RETARGET_EVERY_S
            )
            if t - self._stall_t > STALL_S:  # no progress -> drop this goal
                left = self._remaining(pose)
                if self._stall_left - left < STALL_MIN_PROGRESS and self.goal is not None:
                    self._give_up_goal(belief, area)
                    need = True
                self._stall_t, self._stall_left = t, left
            if need and not self._plan(t, pose, belief, area):
                done = True
                self.mode = "IDLE"
            elif self.path:
                i, j = belief.world_to_cell(pose.x, pose.y)
                inside = belief.inside(i, j) and bool(area.mask[j, i])
                self.mode = "EXPLORING" if inside else "TRANSIT"
                gx, gy = self.path[-1]
                dist_goal = math.hypot(gx - pose.x, gy - pose.y)
                if dist_goal < ARRIVE_M:
                    if self._goal_still_useful(belief, area):
                        self.mode = "SWEEP"
                        self._sweep_left = 2 * math.pi
                    else:
                        self.invalidate()
                else:
                    pts = np.asarray(self.path)
                    k = int(np.argmin(np.hypot(pts[:, 0] - pose.x, pts[:, 1] - pose.y)))
                    cx, cy = pts[min(k + LOOKAHEAD, len(pts) - 1)]
                    d = np.array([cx - pose.x, cy - pose.y])
                    n = float(np.linalg.norm(d))
                    speed = min(config.MAX_SPEED, 0.4 + 1.2 * dist_goal)
                    if n > 1e-6:
                        want = d / n * speed
                        yaw_des = math.atan2(d[1], d[0])

        # ------------------------------------------------------------ reflex
        v = want + self._separation(pose, others) + self._walls(pose, r.lidar)
        sp = float(np.linalg.norm(v))
        if sp > config.MAX_SPEED:
            v *= config.MAX_SPEED / sp
        self._v = (1 - SMOOTH) * self._v + SMOOTH * v
        if yaw_des is not None:
            wz = float(np.clip(YAW_GAIN * _wrap(yaw_des - pose.yaw), -WZ_MAX, WZ_MAX))
        else:
            wz = wz_ff
        vz = ALT_GAIN * (self.altitude - pose.z) if pose.z > 0.05 else 0.0
        return VelCmd(vx=float(self._v[0]), vy=float(self._v[1]), vz=vz, wz=wz), done

    # ---------------------------------------------------------------- reflex terms
    def _separation(self, pose: Pose, others: list[Pose]) -> np.ndarray:
        out = np.zeros(2)
        for o in others:
            d = np.array([pose.x - o.x, pose.y - o.y])
            n = float(np.linalg.norm(d))
            if n >= SEP_M:
                continue
            if n < 1e-3:  # coincident: split by id
                d, n = np.array([1.0, 0.0]) * (1 if self.id > "D" else -1), 1.0
            away = d / n
            side = np.array([-away[1], away[0]])  # slide past instead of a head-on standoff
            out += SEP_GAIN * (SEP_M - n) / SEP_M * (away + 0.3 * side)
        return out

    @staticmethod
    def _walls(pose: Pose, lidar: list[float]) -> np.ndarray:
        rng = np.asarray(lidar, float)
        close = rng < WALL_M
        if not close.any():
            return np.zeros(2)
        a = pose.yaw + _LIDAR_ANG[close]
        push = WALL_GAIN * (WALL_M - rng[close]) / WALL_M
        return -np.array([np.sum(np.cos(a) * push), np.sum(np.sin(a) * push)])
