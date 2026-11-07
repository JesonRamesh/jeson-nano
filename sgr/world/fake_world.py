"""FakeWorld: a kinematic stand-in for the MuJoCo World, with the same §4 facade.

Written by S in the scaffold so nobody waits for MuJoCo. P's real World lives in
sgr/world/world.py. Keep this file working: S develops the swarm against it.

- Drones and the firefighter move kinematically on grid.png (axis-sliding collision).
- LiDAR is ray-marched on the grid (walls + debris), 36 rays, 6 m max.
- Heat follows §7.4. Cameras return a flat placeholder frame.
- set_drone_cmd velocities are WORLD frame (x east, y north).
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np

from sgr import config
from sgr.contracts import (
    XY,
    CamInfo,
    Fire,
    GroundTruth,
    Handle,
    Pose,
    PriorMap,
    Scenario,
)

_RAY_STEP = 0.05
_DEBRIS_POOL = 40
_FIRE_POOL = 30
_CLICK_CLEARANCE = 1.0  # m from drones / firefighter / survivor (§10.2)


class FakeWorld:
    def __init__(self, map_dir: Path, scenario: Scenario):
        self.map_dir = Path(map_dir)
        self.scenario = scenario
        self.res = scenario.grid.resolution
        self._walls = ~PriorMap.load_free_grid(self.map_dir / scenario.grid.file)
        self.h, self.w = self._walls.shape
        self._t = 0.0
        self._next_id = 0

        s = scenario
        self._drones: dict[str, dict] = {}
        for i, d in enumerate(s.drones.ids):
            x, y = s.drones.start[i] if i < len(s.drones.start) else (s.entrance.x, s.entrance.y)
            self._drones[d] = {
                "pose": [x, y, s.drones.altitude, s.entrance.yaw],
                "cmd": [0.0, 0.0, 0.0, 0.0],
                "alive": True,
            }
        self._ff = {"pose": [s.entrance.x, s.entrance.y, 0.0, s.entrance.yaw], "input": [0.0, 0.0]}
        self._debris: dict[str, Handle] = {}
        self._fires: dict[str, Handle] = {}
        self._static_debris = self._rasterize_compromised()
        self._rebuild_blocked()

    # ---------------------------------------------------------------- physics
    def step(self, n: int) -> None:
        dt = n * config.PHYSICS_DT
        for d in self._drones.values():
            if not d["alive"]:
                continue
            p, (vx, vy, vz, wz) = d["pose"], d["cmd"]
            p[3] = _wrap(p[3] + wz * dt)
            p[2] = min(2.5, max(0.3, p[2] + vz * dt))
            self._slide(p, vx * dt, vy * dt, config.DRONE_RADIUS)
        fp = self._ff["pose"]
        fwd, strafe = self._ff["input"]
        c, s_ = math.cos(fp[3]), math.sin(fp[3])
        vx = config.FF_SPEED * (fwd * c + strafe * s_)  # strafe + = right
        vy = config.FF_SPEED * (fwd * s_ - strafe * c)
        self._slide(fp, vx * dt, vy * dt, 0.25)
        self._t += dt

    def set_drone_cmd(self, drone: str, vx: float, vy: float, vz: float, wz: float) -> None:
        d = self._drones[drone]
        if d["alive"]:
            d["cmd"] = [vx, vy, vz, wz]

    def drone_pose(self, drone: str) -> Pose:
        x, y, z, yaw = self._drones[drone]["pose"]
        return Pose(x=x, y=y, z=z, yaw=yaw)

    # ---------------------------------------------------------------- sensors
    def lidar(self, drone: str) -> np.ndarray:
        x, y, _, yaw = self._drones[drone]["pose"]
        angles = yaw + np.arange(config.LIDAR_RAYS) * (2 * math.pi / config.LIDAR_RAYS)
        return self._raycast(x, y, angles)

    def temperature(self, drone: str) -> float:
        x, y = self._drones[drone]["pose"][:2]
        return self._heat_at(x, y)

    def render_drone(self, drone: str) -> tuple[np.ndarray, CamInfo]:
        frame = np.full((config.CAM_H, config.CAM_W, 3), 45, dtype=np.uint8)
        frame[: config.CAM_H // 2] = 30  # "ceiling" darker than "floor"
        return frame, self._cam_info(drone)

    def pixel_to_world(self, drone: str, u: int, v: int) -> XY | None:
        x, y, _, yaw = self._drones[drone]["pose"]
        cam = self._cam_info(drone)
        a = yaw + math.atan2(cam.cx - u, cam.fx)
        r = float(self._raycast(x, y, np.array([a]))[0])
        if r >= config.LIDAR_MAX:
            return None
        r = max(0.0, r - _RAY_STEP)
        return XY(x=x + r * math.cos(a), y=y + r * math.sin(a))

    # ---------------------------------------------------------------- change tool
    def place_debris(self, x: float, y: float, force: bool = False) -> Handle | None:
        if len(self._debris) >= _DEBRIS_POOL or not (force or self._placeable(x, y)):
            return None
        h = Handle(id=self._new_id("debris"), kind="debris", x=x, y=y)
        self._debris[h.id] = h
        self._rebuild_blocked()
        return h

    def place_fire(self, x: float, y: float, force: bool = False) -> Handle | None:
        """force=True skips the click-safety checks (flare-up puts fire on its drone, §7.4)."""
        if len(self._fires) >= _FIRE_POOL or not (force or self._placeable(x, y)):
            return None
        h = Handle(id=self._new_id("fire"), kind="fire", x=x, y=y)
        self._fires[h.id] = h
        return h

    def remove(self, h: Handle) -> None:
        if self._debris.pop(h.id, None) is not None:
            self._rebuild_blocked()
        self._fires.pop(h.id, None)

    def kill_drone(self, drone: str) -> None:
        d = self._drones[drone]
        d["alive"] = False
        d["cmd"] = [0.0, 0.0, 0.0, 0.0]
        d["pose"][2] = 0.0

    # ---------------------------------------------------------------- firefighter
    def set_ff_input(self, fwd: float, strafe: float, yaw: float) -> None:
        self._ff["input"] = [max(-1.0, min(1.0, fwd)), max(-1.0, min(1.0, strafe))]
        self._ff["pose"][3] = _wrap(yaw)

    def ff_pose(self) -> Pose:
        x, y, z, yaw = self._ff["pose"]
        return Pose(x=x, y=y, z=z, yaw=yaw)

    def push_ff_back(self) -> None:
        p = self._ff["pose"]
        self._slide(p, -math.cos(p[3]) * 1.0, -math.sin(p[3]) * 1.0, 0.25)

    # ---------------------------------------------------------------- ground truth / snapshots
    def ground_truth(self) -> GroundTruth:
        s = self.scenario
        fires = list(s.fires) + [Fire(x=h.x, y=h.y, size=1.5) for h in self._fires.values()]
        return GroundTruth(
            fires=fires,
            survivor=s.survivor,
            decoy=s.decoy,
            debris=list(self._debris.values()),
            compromised=list(s.compromised),
            dead_drones=[k for k, d in self._drones.items() if not d["alive"]],
        )

    def get_state(self) -> bytes:
        return pickle.dumps(
            {
                "t": self._t,
                "next_id": self._next_id,
                "drones": self._drones,
                "ff": self._ff,
                "debris": [h.model_dump() for h in self._debris.values()],
                "fires": [h.model_dump() for h in self._fires.values()],
            }
        )

    def set_state(self, b: bytes) -> None:
        st = pickle.loads(b)
        self._t, self._next_id = st["t"], st["next_id"]
        self._drones, self._ff = st["drones"], st["ff"]
        self._debris = {d["id"]: Handle(**d) for d in st["debris"]}
        self._fires = {d["id"]: Handle(**d) for d in st["fires"]}
        self._rebuild_blocked()

    # ================================================================ internals
    def _new_id(self, kind: str) -> str:
        self._next_id += 1
        return f"{kind}-{self._next_id}"

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor(x / self.res)), int(math.floor(y / self.res))

    def _is_blocked(self, x: float, y: float) -> bool:
        ix, iy = self._cell(x, y)
        if ix < 0 or iy < 0 or ix >= self.w or iy >= self.h:
            return True
        return bool(self._blocked[iy, ix])

    def _fits(self, x: float, y: float, r: float) -> bool:
        return not any(
            self._is_blocked(x + dx, y + dy) for dx, dy in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r))
        )

    def _slide(self, p: list[float], dx: float, dy: float, r: float) -> None:
        """Move p by (dx, dy) in small substeps, sliding along walls axis by axis."""
        k = max(1, math.ceil(math.hypot(dx, dy) / 0.05))
        sx, sy = dx / k, dy / k
        for _ in range(k):
            if self._fits(p[0] + sx, p[1], r):
                p[0] += sx
            if self._fits(p[0], p[1] + sy, r):
                p[1] += sy

    def _raycast(self, x: float, y: float, angles: np.ndarray) -> np.ndarray:
        r = np.arange(_RAY_STEP, config.LIDAR_MAX + 1e-9, _RAY_STEP)
        xs = x + np.cos(angles)[:, None] * r
        ys = y + np.sin(angles)[:, None] * r
        ix = np.floor(xs / self.res).astype(int)
        iy = np.floor(ys / self.res).astype(int)
        oob = (ix < 0) | (iy < 0) | (ix >= self.w) | (iy >= self.h)
        hit = oob.copy()
        ok = ~oob
        hit[ok] = self._blocked[iy[ok], ix[ok]]
        first = hit.argmax(axis=1)
        out = np.where(hit.any(axis=1), r[first], config.LIDAR_MAX)
        return out.astype(np.float32)

    def _heat_at(self, x: float, y: float) -> float:
        t = self.scenario.heat.ambient_c
        pts = [(f.x, f.y) for f in self.scenario.fires] + [(h.x, h.y) for h in self._fires.values()]
        for fx, fy in pts:
            d2 = (x - fx) ** 2 + (y - fy) ** 2
            t += 380.0 * math.exp(-d2 / (2 * 1.5**2))
        return t

    def _cam_info(self, drone: str) -> CamInfo:
        f = (config.CAM_H / 2) / math.tan(math.radians(config.CAM_FOVY_DEG) / 2)
        return CamInfo(
            width=config.CAM_W,
            height=config.CAM_H,
            fovy_deg=config.CAM_FOVY_DEG,
            fx=f,
            fy=f,
            cx=config.CAM_W / 2,
            cy=config.CAM_H / 2,
            pose=self.drone_pose(drone),
            pitch_deg=config.CAM_PITCH_DEG,
        )

    def _placeable(self, x: float, y: float) -> bool:
        if self._is_blocked(x, y):
            return False
        others = [d["pose"][:2] for d in self._drones.values() if d["alive"]]
        others += [self._ff["pose"][:2], (self.scenario.survivor.x, self.scenario.survivor.y)]
        return all(math.hypot(x - ox, y - oy) >= _CLICK_CLEARANCE for ox, oy in others)

    def _cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        xs = (np.arange(self.w) + 0.5) * self.res
        ys = (np.arange(self.h) + 0.5) * self.res
        return np.meshgrid(xs, ys)  # [iy, ix]

    def _rasterize_compromised(self) -> np.ndarray:
        """Each compromised edge gets a debris band 1.0 m deep across the whole corridor."""
        mask = np.zeros_like(self._walls)
        nodes = {n.id: n for n in self.scenario.graph.nodes}
        edges = {e.id: e for e in self.scenario.graph.edges}
        X, Y = self._cell_centres()
        for c in self.scenario.compromised:
            e = edges[c.edge]
            a, b = nodes[e.a], nodes[e.b]
            dx, dy = b.x - a.x, b.y - a.y
            length = math.hypot(dx, dy) or 1.0
            ux, uy = dx / length, dy / length
            px, py = a.x + dx * c.at, a.y + dy * c.at
            along = (X - px) * ux + (Y - py) * uy
            across = -(X - px) * uy + (Y - py) * ux
            mask |= (np.abs(along) <= 0.5) & (np.abs(across) <= e.width / 2 + 0.5)
        return mask

    def _rebuild_blocked(self) -> None:
        blocked = self._walls | self._static_debris
        if self._debris:
            X, Y = self._cell_centres()
            for h in self._debris.values():  # 1.0 x 1.0 m box
                blocked |= (np.abs(X - h.x) <= 0.5) & (np.abs(Y - h.y) <= 0.5)
        self._blocked = blocked


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
