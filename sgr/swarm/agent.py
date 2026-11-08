"""One drone's controller: a tactical layer (where to go, what to look at) and a reflex layer (every tick).

Tactical (§8.2): go to the nearest reachable unsearched spot in the claimed area, yaw-sweep 360° on
arrival, retarget when the spot gets searched, the path gets blocked, or progress stalls.
Reflex (§8.2): 2 m separation, LiDAR wall repulsion under 0.6 m, 1.2 m/s clamp, smoothing.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.ndimage import binary_dilation

from sgr import config
from sgr.contracts import Pose, Readings, VelCmd

from .areas import Area
from .belief import BeliefGrid
from .planner import NO_STAY, DistanceField, cost_map, plan_path

# tactical
GOAL_NEAR_CELLS = 3  # a goal is a traversable cell within 0.6 m of an unsearched cell
LOOKAHEAD = 3  # waypoints (0.6 m)
ARRIVE_M = 0.35
RETARGET_EVERY_S = 3.0
STALL_S = 3.0
STALL_MIN_PROGRESS = 0.3  # m of path-distance gained per STALL_S window
SWEEP_WZ = 2.0  # rad/s
THERMO_LEN = 40  # thermometer history: the last 4 s of (t, x, y, temp), to locate an unseen heat source
SWEEP_EVERY_S = 15.0  # at most one 360° sweep per this many seconds ...
SWEEP_APART_M = 3.0  # ... unless it is this far from the last one (no spinning on the spot)
HEAT_SLOW_NORM, HEAT_SLOW_MS = 0.6, 0.6  # known heat >= ~50 °C: fly at most 0.6 m/s
HEATING_S, HEATING_C = 0.5, 15.0  # retreating but this much hotter in this long: wrong way, reverse
CROSS_MIN_SPEED = 0.3  # m/s: slower than this is hovering, not crossing
SLOW_MS = 0.15  # below this speed, don't chase the travel direction (it flips around: spinning)
SCAN_AMPL = math.radians(70)  # idle / sentinel camera scan: ±70° ...
SCAN_RATE = 0.5  # ... at this angular frequency (rad/s), never a continuous spin
# inspect / retreat (§8.4)
IN_POSITION_M = 0.4
FACING_RAD = math.radians(15)
INSPECT_SPEED = 0.8
RETREAT_M = 3.0
RETREAT_SPEED = 1.0
RETREAT_TIMEOUT_S = 8.0
# patrol (Act 1 sentinel sweeping the route with LiDAR)
PATROL_SPEED = 1.2
PATROL_WP_M = 0.6  # waypoint reached
PATROL_STALL_S = 3.0  # no progress for this long (new debris on the route): turn round
PATROL_DIRECT_M = 2.0  # next waypoint further than this: fly there along a planned path first
# reflex
SEP_M = 2.0
SEP_GAIN = 0.8
WALL_M = 0.45  # reflex only for real near-misses: the planner already keeps 0.3 m (a 0.6 m reflex jammed
WALL_GAIN = 1.2  # drones at the mouth of 1 m debris gaps, both sides pushing at once)
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
        self.mode = "IDLE"  # IDLE | TRANSIT | EXPLORING | SWEEP | INSPECTING | RETREAT | CHAIN | WATCHING | LOST
        self.area: str | None = None
        self.path: list[tuple[float, float]] | None = None
        self.goal: tuple[int, int] | None = None
        self._goal_t = 0.0
        self._stall_t = 0.0
        self._stall_left = math.inf
        self._sweep_left = 0.0
        self._v = np.zeros(2)
        # inspect: face `_focus`; if `_standoff` is set, fly there first (§8.4)
        self._focus: tuple[float, float] | None = None
        self._standoff: tuple[float, float] | None = None
        self._ipath: list[tuple[float, float]] | None = None
        self.in_position = False
        # retreat
        self._retreat_dir: tuple[float, float] | None = None
        self._retreat_path: list[tuple[float, float]] | None = None
        self._retreat_left = 0.0
        self._retreated = False
        # station (chain position / fire watch): fly to `_station`, hold; face `_station_face` or sweep
        self._station: tuple[float, float] | None = None
        self._station_face: tuple[float, float] | None = None
        self._station_sweep = False
        self._station_mode = "CHAIN"
        self._station_task = ""
        self._spath: list[tuple[float, float]] | None = None
        self._scan_base: float | None = None
        self._last_sweep: tuple[float, float, float] | None = None  # (t, x, y) of the last 360° sweep
        self.thermo: deque[tuple[float, float, float, float]] = deque(maxlen=THERMO_LEN)  # (t, x, y, temp)
        # patrol: back and forth along `_patrol` (route waypoints); replaces holding the station
        self._patrol: list[tuple[float, float]] | None = None
        self._patrol_k = 0
        self._patrol_dir = 1
        self._patrol_best = math.inf
        self._patrol_t = 0.0

    # ---------------------------------------------------------------- orders from the swarm
    def hold(self, focus: tuple[float, float]) -> None:
        """Stop and face `focus` (a detection waiting for Gemma)."""
        self._focus, self._standoff, self._ipath = focus, None, None
        self.in_position = False

    def approach(self, standoff: tuple[float, float], focus: tuple[float, float]) -> None:
        """Fly to `standoff` (~2 m in front of the target) and face `focus`, for a closer look."""
        self._focus, self._standoff, self._ipath = focus, standoff, None
        self.in_position = False

    def release(self) -> None:
        """Inspection over: back to exploring."""
        self._focus = self._standoff = self._ipath = None
        self.in_position = False
        if self.mode == "INSPECTING":
            self.mode = "IDLE"
        self.invalidate()

    def retreat(self, path: list[tuple[float, float]] | None, away: tuple[float, float]) -> None:
        """Too hot: follow `path` out of the heat (or back off RETREAT_M along `away`), then drop the goal."""
        self._retreat_path, self._retreat_dir = path, away
        self._retreat_left = RETREAT_M if not path else RETREAT_TIMEOUT_S
        self.mode = "RETREAT"
        self._retreated = True
        self._focus = self._standoff = self._ipath = None
        self.in_position = False
        self.invalidate()

    def station(self, xy: tuple[float, float], face: tuple[float, float] | None = None, sweep: bool = False,
                mode: str = "CHAIN", task: str = "") -> None:
        """Hold a fixed spot (chain checkpoint / fire watch) until `unstation()`; replaces exploring."""
        if self._station != xy:
            self._spath = None
        self._station, self._station_face, self._station_sweep = xy, face, sweep
        self._station_mode, self._station_task = mode, task

    def patrol(self, waypoints: list[tuple[float, float]] | None) -> None:
        """Fly back and forth along `waypoints` (None = stop patrolling, hold the station again)."""
        if waypoints is None or len(waypoints) < 2:
            self._patrol = None
            self._spath = None
            return
        if self._patrol == waypoints:
            return
        self._patrol, self._patrol_dir, self._patrol_best = list(waypoints), 1, math.inf
        self._patrol_k = -1  # join at the nearest waypoint (picked on the next step, when the pose is known)

    @property
    def patrolling(self) -> bool:
        return self._patrol is not None

    def unstation(self) -> None:
        self._patrol = None
        self._station = self._station_face = self._spath = None
        self._station_sweep = False
        if self.mode in ("CHAIN", "WATCHING"):
            self.mode = "IDLE"
        self.invalidate()

    def _heating_up(self, t: float) -> bool:
        """Temperature rose by more than HEATING_C over the last HEATING_S."""
        old = [tc for ts, _, _, tc in self.thermo if t - HEATING_S - 0.15 <= ts <= t - HEATING_S + 0.05]
        return bool(old) and self.thermo[-1][3] - old[0] > HEATING_C

    @property
    def crossing(self) -> bool:
        """Moving along a planned path (not hovering): may pass briefly through heat between 70 and 150 °C."""
        moving = self.mode in ("TRANSIT", "EXPLORING", "CHAIN", "WATCHING", "INSPECTING") or self._patrol is not None
        return moving and float(np.linalg.norm(self._v)) > CROSS_MIN_SPEED

    @property
    def stationed(self) -> bool:
        return self._station is not None

    @property
    def inspecting(self) -> bool:
        return self._focus is not None

    # ---------------------------------------------------------------- tactical
    def set_area(self, area: str | None) -> None:
        if area != self.area:
            self.area = area
            self.invalidate()

    def invalidate(self) -> None:
        """Drop the current path (blocked / new area); a new one is planned next tick."""
        self.path, self.goal = None, None
        self._spath = None
        if self.mode == "SWEEP":
            self._sweep_left = 0.0

    @property
    def task(self) -> str:
        if self.mode == "RETREAT":
            return "backing off the heat"
        if self._focus is not None:
            return "closer look" if self._standoff else "waiting for Gemma"
        if self._station is not None:
            return "patrolling the route" if self._patrol is not None else self._station_task
        return f"area {self.area}" if self.area else ""

    def _unsearched(self, belief: BeliefGrid, area: Area) -> np.ndarray:
        return area.unsearched(belief)  # dead ends any drone found are already out (belief.dead)

    def _give_up_goal(self, belief: BeliefGrid, area: Area) -> None:
        """Swept / stalled at this goal and the cells near it are still unseen: a dead end. Shared with the
        whole swarm (belief.dead), so no other drone is sent to look for them either."""
        if self.goal is None:
            return
        i, j = self.goal
        r = GOAL_NEAR_CELLS
        sl = (slice(max(0, j - r), j + r + 1), slice(max(0, i - r), i + r + 1))
        newly = area.unsearched(belief)[sl]
        belief.dead[sl] |= newly
        belief.dead_t[sl][newly] = belief.now

    def _goal_mask(self, belief: BeliefGrid, area: Area) -> np.ndarray:
        cm = cost_map(belief)
        return binary_dilation(self._unsearched(belief, area), _STRUCT) & cm.traversable & (belief.heat < NO_STAY)

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
        self.thermo.append((t, pose.x, pose.y, r.temperature))  # where it was, how hot it was (last few s)
        want = np.zeros(2)
        yaw_des: float | None = None
        self._yaw_from_travel = True  # cleared below where yaw_des is a deliberate facing (target, scan)
        wz_ff = 0.0
        done = False

        if self.mode == "RETREAT":
            if self._heating_up(t):  # hotter while backing off: a fire nobody knows about is ahead -> turn round
                v = self._v if float(np.linalg.norm(self._v)) > 0.05 else np.array([math.cos(pose.yaw), math.sin(pose.yaw)])
                self._retreat_dir = tuple(-v / float(np.linalg.norm(v)))
                self._retreat_path, self._retreat_left = None, RETREAT_M
                self.thermo.clear()
            if self._retreat_path:
                gx, gy = self._retreat_path[-1]
                self._retreat_left -= config.CONTROL_DT
                if math.hypot(gx - pose.x, gy - pose.y) > ARRIVE_M and self._retreat_left > 0:
                    want, _ = self._follow(pose, self._retreat_path, RETREAT_SPEED)
                else:
                    self.mode, self._retreat_path = "IDLE", None
                    if self._station is not None:
                        si, sj = belief.world_to_cell(*self._station)
                        if belief.inside(si, sj) and belief.heat[sj, si] >= NO_STAY:
                            self._station = (pose.x, pose.y)  # the post itself is too hot: hold here instead
                        self._spath = None  # otherwise back to the post (the lead must still reach the entrance)
            elif self._retreat_left > 0 and self._retreat_dir is not None:
                want = np.array(self._retreat_dir) * RETREAT_SPEED
                self._retreat_left -= RETREAT_SPEED * config.CONTROL_DT
            else:
                self.mode = "IDLE"
        elif self._focus is not None:
            self.mode = "INSPECTING"
            fx, fy = self._focus
            face = math.atan2(fy - pose.y, fx - pose.x)
            close = True
            if self._standoff is not None:
                sx, sy = self._standoff
                dist = math.hypot(sx - pose.x, sy - pose.y)
                close = dist < IN_POSITION_M
                if not close:
                    if self._ipath is None:
                        self._ipath = plan_path(belief, (pose.x, pose.y), self._standoff) or [(pose.x, pose.y)]
                    want, travel = self._follow(pose, self._ipath, INSPECT_SPEED)
                    # face the target for the last metre so the camera is already on it
                    yaw_des = face if dist < 1.0 or travel is None else travel
            if close:
                yaw_des = face
                self._yaw_from_travel = False
            self.in_position = close and abs(_wrap(face - pose.yaw)) < FACING_RAD
        elif self._station is not None and self._patrol is not None:
            self.mode = self._station_mode
            pts = self._patrol
            if self._patrol_k < 0:  # first step: join the route at the nearest waypoint, head for the far end
                k = int(np.argmin([math.hypot(x - pose.x, y - pose.y) for x, y in pts]))
                self._patrol_k, self._patrol_dir = k, (1 if k < len(pts) / 2 else -1)
                self._patrol_t, self._patrol_best = t, math.inf
            gx, gy = pts[self._patrol_k]
            dist = math.hypot(gx - pose.x, gy - pose.y)
            self._patrol_best = min(self._patrol_best, dist)
            stalled = t - self._patrol_t > PATROL_STALL_S
            if dist < PATROL_WP_M or stalled:
                if stalled and dist >= PATROL_WP_M:  # blocked (e.g. new debris): turn round
                    self._patrol_dir = -self._patrol_dir
                nk = self._patrol_k + self._patrol_dir
                if not 0 <= nk < len(pts):  # end of the stretch: turn round
                    self._patrol_dir = -self._patrol_dir
                    nk = self._patrol_k + self._patrol_dir
                self._patrol_k = max(0, min(len(pts) - 1, nk))
                self._patrol_t, self._patrol_best = t, math.inf
                gx, gy = pts[self._patrol_k]
            if math.hypot(gx - pose.x, gy - pose.y) > PATROL_DIRECT_M:  # far (joining the stretch): plan, no walls
                if self._spath is None or self._spath[-1] != (gx, gy):
                    self._spath = plan_path(belief, (pose.x, pose.y), (gx, gy)) or [(pose.x, pose.y)]
                    self._spath[-1] = (gx, gy)
                want, yaw_des = self._follow(pose, self._spath, PATROL_SPEED)
                self._patrol_t = t  # progress along a planned path is not a stall
            else:
                self._spath = None
                d = np.array([gx - pose.x, gy - pose.y])
                n = float(np.linalg.norm(d))
                if n > 1e-6:
                    want = d / n * min(PATROL_SPEED, 0.3 + n)
                    yaw_des = math.atan2(d[1], d[0])  # camera looks along the route too
        elif self._station is not None:
            self.mode = self._station_mode
            if self._spath is None and math.hypot(self._station[0] - pose.x, self._station[1] - pose.y) > IN_POSITION_M:
                self._spath = plan_path(belief, (pose.x, pose.y), self._station) or [(pose.x, pose.y)]
            # arrive where the plan ends: if the post itself can't be reached (in a fire's hot zone, cut off),
            # that is the closest safe point, so hold there instead of nudging at it forever
            sx, sy = self._spath[-1] if self._spath else self._station
            dist = math.hypot(sx - pose.x, sy - pose.y)
            if dist > IN_POSITION_M:
                if self._spath is None:
                    self._spath = plan_path(belief, (pose.x, pose.y), self._station) or [(pose.x, pose.y)]
                want, yaw_des = self._follow(pose, self._spath, config.MAX_SPEED)
                if dist < 1.0 and self._station_face is not None:
                    fx, fy = self._station_face
                    yaw_des = math.atan2(fy - pose.y, fx - pose.x)
            elif self._station_sweep:  # sentinel: scan its stretch of route (LiDAR is 360° anyway)
                yaw_des = self._scan_yaw(t, self._station_face)
                self._yaw_from_travel = False
            elif self._station_face is not None:
                fx, fy = self._station_face
                yaw_des = math.atan2(fy - pose.y, fx - pose.x)
                self._yaw_from_travel = False
        elif area is None:
            self.mode = "IDLE"
            yaw_des = self._scan_yaw(t, None, pose.yaw)  # look around without spinning on the spot
            self._yaw_from_travel = False
        elif self.mode == "SWEEP" and self._sweep_left > 0:
            self._sweep_left -= SWEEP_WZ * config.CONTROL_DT
            wz_ff = SWEEP_WZ
            if self._sweep_left <= 0:
                self.mode = "EXPLORING"
                if area is not None and self._goal_still_useful(belief, area):
                    self._give_up_goal(belief, area)
                self.invalidate()
        else:
            if self._retreated:  # don't fly straight back into the heat
                self._retreated = False
                self._give_up_goal(belief, area)
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
                    recent = (self._last_sweep is not None and t - self._last_sweep[0] < SWEEP_EVERY_S
                              and math.hypot(pose.x - self._last_sweep[1], pose.y - self._last_sweep[2]) < SWEEP_APART_M)
                    if self._goal_still_useful(belief, area) and not recent:
                        self.mode = "SWEEP"
                        self._sweep_left = 2 * math.pi
                        self._last_sweep = (t, pose.x, pose.y)
                    else:
                        if recent and self._goal_still_useful(belief, area):
                            self._give_up_goal(belief, area)  # just swept near here: don't spin again
                        self.invalidate()
                else:
                    want, yaw_des = self._follow(pose, self.path, config.MAX_SPEED)

        # ------------------------------------------------------------ reflex
        hi, hj = belief.world_to_cell(pose.x, pose.y)  # slow down in known heat: a retreat then starts before
        if belief.inside(hi, hj) and belief.heat[hj, hi] >= HEAT_SLOW_NORM:  # half a metre of overshoot is +100 °C
            n = float(np.linalg.norm(want))
            if n > HEAT_SLOW_MS:
                want = want * (HEAT_SLOW_MS / n)
        sep, wall = self._separation(pose, others), self._walls(pose, r.lidar)
        self.debug = (want, sep, wall)  # last command's parts (diagnostics)
        v = want + sep + wall
        sp = float(np.linalg.norm(v))
        if sp > config.MAX_SPEED:
            v *= config.MAX_SPEED / sp
        self._v = (1 - SMOOTH) * self._v + SMOOTH * v
        if yaw_des is not None and self._yaw_from_travel and float(np.linalg.norm(want)) < SLOW_MS:
            yaw_des = None  # nearly stopped: the "travel direction" is noise, hold the heading
            wz_ff = 0.0
        if yaw_des is not None:
            wz = float(np.clip(YAW_GAIN * _wrap(yaw_des - pose.yaw), -WZ_MAX, WZ_MAX))
        else:
            wz = wz_ff
        vz = ALT_GAIN * (self.altitude - pose.z) if pose.z > 0.05 else 0.0
        return VelCmd(vx=float(self._v[0]), vy=float(self._v[1]), vz=vz, wz=wz), done

    def _scan_yaw(self, t: float, face: tuple[float, float] | None, around: float | None = None) -> float:
        """Slow back-and-forth camera scan (±70°) instead of spinning in circles."""
        if self._scan_base is None or face is not None:
            self._scan_base = around if around is not None else (self._scan_base or 0.0)
        return _wrap(self._scan_base + SCAN_AMPL * math.sin(SCAN_RATE * t))

    @staticmethod
    def _follow(pose: Pose, path: list[tuple[float, float]], vmax: float) -> tuple[np.ndarray, float | None]:
        """Carrot-follow `path`: velocity toward a waypoint LOOKAHEAD ahead, slowing near the end."""
        pts = np.asarray(path)
        k = int(np.argmin(np.hypot(pts[:, 0] - pose.x, pts[:, 1] - pose.y)))
        cx, cy = pts[min(k + LOOKAHEAD, len(pts) - 1)]
        gx, gy = pts[-1]
        d = np.array([cx - pose.x, cy - pose.y])
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return np.zeros(2), None
        speed = min(vmax, 0.4 + 1.2 * math.hypot(gx - pose.x, gy - pose.y))
        return d / n * speed, math.atan2(d[1], d[0])

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
