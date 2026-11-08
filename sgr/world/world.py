"""World facade (spec §4): MuJoCo physics, LiDAR, heat and drone cameras.

Ground truth lives here. sgr/swarm and sgr/guidance must never call
ground_truth() (tests/test_honesty.py).

Threading: create and call everything on the main thread. The one
mujoco.Renderer is created lazily on the first render_drone() call (it must be
on the main thread), so headless runs never open a GL context.
"""

from __future__ import annotations

import math
import os
import pickle
import zlib
import threading
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from . import fog, mjcf, shapes
from sgr.contracts import XY, CamInfo, Decoy, Fire, GroundTruth, Handle, Pose, Scenario

from .heat import temperature_at
from .walls import world_to_cell

LIDAR_N = 36
LIDAR_RANGE = 6.0
LIDAR_GROUPS = np.array([1, 0, 0, 0, 1, 0], np.uint8)  # walls + debris only (walls move to group 4 with the scan)
PIXEL_RAY_GROUPS = np.array([1, 1, 0, 0, 0, 0], np.uint8)  # what the camera draws: debris, visuals, scan replica
WALL_GROUP_WITH_SCAN = 4  # box walls + plain floor with the scan replica on: physics/LiDAR/LOS only, never drawn
PIXEL_RAY_MAX = 30.0
RENDER_GROUPS = (1, 1, 1, 0, 0, 0)  # group 3 = hidden, 4 = walls/collision (never drawn)
SAFE_DIST = 1.0  # click safety (§10.2)
FF_WALK = 1.4  # m/s (§9)
FF_YAW_GAIN = 4.0
FF_PUSH = 1.0
DEAD_FALL_VZ = -2.0
DEAD_RGBA = (0.08, 0.08, 0.08, 1.0)
DEBRIS_TILT_DEG = 10.0  # §7.1 "slightly rotated"
CAM_ZNEAR_M = 0.02  # absolute camera clip planes (metres)
CAM_ZFAR_M = 120.0
FLICKER = 0.25  # +-fraction of fire colour per tick


@dataclass
class _Placed:
    handle: Handle
    slot: int  # mocap pool index
    scenario: bool  # True = from scenario.json (fires / compromised edges), False = click or flare-up
    size: float = 1.5  # fire size (ground truth only)


def _smooth(e0: float, e1: float, x):
    """GLSL smoothstep."""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class World:
    def __init__(self, map_dir: Path, scenario: Scenario, seed: int = 0, scan_vis: bool | None = None):
        """scan_vis: show W's textured scan replica (maps/<name>/scan_vis.obj + .png) in the drone cameras.
        None = on when the map has the files, unless SGR_SCAN_VIS=0. Physics, LiDAR and line of sight always
        use the box walls; the replica is visual only (it doesn't change the physics state, so snapshots
        made with and without it are interchangeable)."""
        self.map_dir = Path(map_dir)
        self.scenario = scenario
        self.res = float(scenario.grid.resolution)
        self.origin = tuple(float(v) for v in scenario.grid.origin)
        img = np.asarray(Image.open(self.map_dir / scenario.grid.file).convert("L"))
        self.free = img >= 128
        self.grid_h, self.grid_w = self.free.shape
        self.ambient = float(scenario.heat.ambient_c)
        self.drones: list[str] = list(scenario.drones.ids)

        if scan_vis is None:
            scan_vis = os.environ.get("SGR_SCAN_VIS", "1").lower() not in ("0", "false", "off", "no")
        obj, png = self.map_dir / "scan_vis.obj", self.map_dir / "scan_vis.png"
        vis = (obj.resolve().as_posix(), png.resolve().as_posix()) if scan_vis and obj.is_file() and png.is_file() else None
        xml, self.build_info = mjcf.build_mjcf(self.free, self.res, self.origin, scenario, self.drones, scan_vis=vis)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        # Camera clip planes are relative to the model extent, which the z=-50 mocap pool and the floor
        # inflate to ~70 m: the default near plane was 0.7 m, so drone cameras saw THROUGH nearby walls.
        m.vis.map.znear = CAM_ZNEAR_M / m.stat.extent
        m.vis.map.zfar = CAM_ZFAR_M / m.stat.extent

        def jnt(name):
            j = m.joint(name)
            return int(m.jnt_qposadr[j.id]), int(m.jnt_dofadr[j.id])

        self._d = {}
        for d in self.drones:
            bid = m.body(d).id
            self._d[d] = {
                "body": bid,
                "q": [jnt(f"{d}_{ax}")[0] for ax in ("x", "y", "z", "yaw")],
                "v": [jnt(f"{d}_{ax}")[1] for ax in ("x", "y", "z", "yaw")],
                "act": [m.actuator(f"{d}_{a}").id for a in ("vx", "vy", "vz", "wz")],
                "cam": m.camera(f"{d}_cam").id,
                "geoms": np.flatnonzero(m.geom_bodyid == bid),
            }
        self._ff = {
            "body": m.body("FF").id,
            "q": [jnt(f"FF_{ax}")[0] for ax in ("x", "y", "yaw")],
            "v": [jnt(f"FF_{ax}")[1] for ax in ("x", "y", "yaw")],
            "act": [m.actuator(f"FF_{a}").id for a in ("vx", "vy", "wz")],
        }
        self._orig_gravcomp = m.body_gravcomp.copy()
        if self.build_info.scan_vis:  # the replica replaces the plain walls + floor in the cameras only
            for g in range(m.ngeom):
                name = m.geom(g).name
                if name.startswith("wall") or name == "floor":
                    m.geom_rgba[g, 3] = 0.0  # not drawn
                    # Own group: LiDAR / LOS still hit them, camera pixel rays don't. Where W closed a doorway in
                    # grid.png the scan still shows it open: the camera sees a fire through it, and a pixel ray
                    # that stopped at the invisible box wall put the fire in the doorway (5 m short).
                    m.geom_group[g] = WALL_GROUP_WITH_SCAN
        self._orig_rgba = m.geom_rgba.copy()

        # mocap pools
        self._pool_mocap = {
            "debris": [int(m.body_mocapid[m.body(f"debris{i:02d}").id]) for i in range(mjcf.N_DEBRIS)],
            "fire": [int(m.body_mocapid[m.body(f"fire{i:02d}").id]) for i in range(mjcf.N_FIRE)],
        }
        self._fire_geoms = [np.flatnonzero(m.geom_bodyid == m.body(f"fire{i:02d}").id) for i in range(mjcf.N_FIRE)]
        # animated fire parts (W's FireFX look), per pool slot, in the order of shapes.fire_particle_params()
        gid = lambda nm: m.geom(nm).id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm) >= 0 else -1  # noqa: E731
        self._fire_fx = []
        for i in range(mjcf.N_FIRE):
            pre = f"fire{i:02d}"
            fx = {k: np.array([gid(f"{pre}_{k}{j}") for j in range(n)], int)
                  for k, n in (("p", shapes.FIRE_N_PARTICLES), ("e", shapes.FIRE_N_EMBERS))}
            fx["core"] = np.array([g for g in self._fire_geoms[i] if m.geom(g).name.split("_")[-1].startswith(("c", "embers"))], int)
            self._fire_fx.append(fx)
        self._fire_params = shapes.fire_particle_params()
        self._survivor_xy = (scenario.survivor.x, scenario.survivor.y)

        # runtime state (pickled with snapshots)
        self._rng = np.random.default_rng(seed)
        self._placed: dict[str, _Placed] = {}  # by handle id
        self._next_id = 0
        self._cmd = {d: np.zeros(4) for d in self.drones}
        self._ff_cmd = (0.0, 0.0, None)
        self._dead: set[str] = set()

        self._renderer: mujoco.Renderer | None = None
        self._scene_opt: mujoco.MjvOption | None = None
        self._ray_gid = np.zeros(1, np.int32)
        self._lidar_gid = np.zeros(LIDAR_N, np.int32)
        self._lidar_dist = np.zeros(LIDAR_N)
        k = np.arange(LIDAR_N) * (2 * math.pi / LIDAR_N)
        self._lidar_angles = k

        self._reset_bodies()
        self._place_scenario_objects()
        mujoco.mj_forward(m, self.data)

    # ------------------------------------------------------------ setup
    def _reset_bodies(self) -> None:
        sc, q = self.scenario, self.data.qpos
        alt = float(sc.drones.altitude)
        for d, start in zip(self.drones, sc.drones.start):
            qa = self._d[d]["q"]  # drones start facing the entrance heading, like the firefighter
            q[qa[0]], q[qa[1]], q[qa[2]], q[qa[3]] = float(start[0]), float(start[1]), alt, float(sc.entrance.yaw)
        e = sc.entrance
        fq = self._ff["q"]
        q[fq[0]], q[fq[1]], q[fq[2]] = e.x, e.y, e.yaw
        for mid in self._pool_mocap["debris"] + self._pool_mocap["fire"]:
            self.data.mocap_pos[mid] = (0.0, 0.0, mjcf.PARK_Z)

    def _place_scenario_objects(self) -> None:
        sc = self.scenario
        for f in sc.fires:
            self._place("fire", f.x, f.y, True, None, scenario=True, size=f.size)
        nodes = {n.id: n for n in sc.graph.nodes}
        edges = {e.id: e for e in sc.graph.edges}
        for c in sc.compromised:
            e = edges.get(c.edge)
            if e is None or e.a not in nodes or e.b not in nodes:
                continue
            a, b = nodes[e.a], nodes[e.b]
            ax, ay, bx, by = a.x, a.y, b.x, b.y
            px, py = ax + c.at * (bx - ax), ay + c.at * (by - ay)
            ang = math.atan2(by - ay, bx - ax)
            nx, ny = -math.sin(ang), math.cos(ang)
            size = shapes.DEBRIS_SIZE[0]
            count = max(1, math.ceil(e.width / size))
            for k in range(count):
                off = (k - (count - 1) / 2) * size
                # tilt from the edge id + block index, so editing one edge never moves another's blocks
                tilt = (zlib.crc32(f"{c.edge}:{k}".encode()) % 2001 - 1000) / 1000 * math.radians(DEBRIS_TILT_DEG)
                self._place("debris", px + off * nx, py + off * ny, True, ang + tilt, scenario=True, jitter=False)

    # ------------------------------------------------------------ sim
    def step(self, n: int) -> None:
        m, d = self.model, self.data
        for name in self.drones:
            dd = self._d[name]
            cmd = self._cmd[name] if name not in self._dead else (0.0, 0.0, DEAD_FALL_VZ, 0.0)
            for a, v in zip(dd["act"], cmd):
                d.ctrl[a] = v
        fwd, strafe, yaw_t = self._ff_cmd
        fq = self._ff["q"]
        yaw = d.qpos[fq[2]]
        c, s = math.cos(yaw), math.sin(yaw)
        d.ctrl[self._ff["act"][0]] = FF_WALK * (fwd * c + strafe * s)
        d.ctrl[self._ff["act"][1]] = FF_WALK * (fwd * s - strafe * c)
        d.ctrl[self._ff["act"][2]] = 0.0 if yaw_t is None else FF_YAW_GAIN * _wrap(yaw_t - yaw)
        self._animate_fires()
        mujoco.mj_step(m, d, nstep=int(n))

    def _animate_fires(self) -> None:
        """Flicker the flame core; move W-style flame particles and embers (fire.tsx shader maths)."""
        m, t = self.model, float(self.data.time)
        flames, embers = self._fire_params
        for p in self._placed.values():
            if p.handle.kind != "fire":
                continue
            fx = self._fire_fx[p.slot]
            core = fx["core"]
            if len(core):
                k = 1.0 + FLICKER * (self._rng.random(len(core)) * 2 - 1)
                m.geom_rgba[core, :3] = np.clip(self._orig_rgba[core, :3] * k[:, None], 0, 1)
            if (fx["p"] >= 0).all():
                ph, an, rf, sp = flames.T
                life = np.mod(t * sp + ph, 1.0)
                ang = an * 2 * np.pi + t * 1.3
                r = rf * shapes.FLAME_R * (1 - 0.65 * life)
                sway = np.sin(t * 3.1 + ph * 23) * 0.07 * life
                m.geom_pos[fx["p"]] = np.stack([np.cos(ang) * r + sway, np.sin(ang) * r, 0.15 + life * shapes.FLAME_H], 1)
                # bright throughout: MuJoCo has no additive blending, so dim red particles would read as soot
                c0, c1, c2 = np.array([1.0, 0.9, 0.5]), np.array([1.0, 0.58, 0.1]), np.array([1.0, 0.32, 0.05])
                a = _smooth(0, 0.35, life)[:, None]
                col = c0 * (1 - a) + c1 * a
                b = _smooth(0.45, 0.9, life)[:, None]
                col = col * (1 - b) + c2 * b
                alpha = _smooth(0, 0.06, life) * (1 - _smooth(0.35, 0.75, life)) * 0.8
                m.geom_rgba[fx["p"]] = np.column_stack([col, alpha])
                sz = 0.11 * (1 - 0.5 * life)
                m.geom_size[fx["p"]] = np.stack([sz, sz, sz * 1.4], 1)
            if (fx["e"] >= 0).all():
                ph, an, rf, sp = embers.T
                life = np.mod(t * sp + ph, 1.0)
                ang = an * 2 * np.pi + t * 0.7
                r = rf * 0.6 + 0.15 * np.sin(t * 2.0 + ph * 10)
                m.geom_pos[fx["e"]] = np.stack([np.cos(ang) * r, np.sin(ang) * r, 0.3 + life * 2.2], 1)
                m.geom_rgba[fx["e"], 3] = 0.9 * (1 - _smooth(0.6, 1.0, life))

    def set_drone_cmd(self, drone: str, vx: float, vy: float, vz: float, wz: float) -> None:
        self._cmd[drone] = np.array([vx, vy, vz, wz], float)

    def drone_pose(self, drone: str) -> Pose:
        q = self.data.qpos
        a = self._d[drone]["q"]
        return Pose(x=float(q[a[0]]), y=float(q[a[1]]), z=float(q[a[2]]), yaw=_wrap(float(q[a[3]])))

    def alive(self, drone: str) -> bool:
        return drone not in self._dead

    def teleport_drone(self, drone: str, x: float, y: float, yaw: float, z: float | None = None) -> None:
        """Debug/test helper (not part of §4): set a drone's pose and zero its velocity."""
        q, v, dd = self.data.qpos, self.data.qvel, self._d[drone]
        q[dd["q"][0]], q[dd["q"][1]], q[dd["q"][3]] = x, y, yaw
        if z is not None:
            q[dd["q"][2]] = z
        v[dd["v"]] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------ sensors
    def lidar(self, drone: str) -> np.ndarray:
        """36 horizontal ranges (m). Ray k points at yaw + k*10 deg (k=0 forward, CCW). 6.0 = no hit."""
        dd = self._d[drone]
        pnt = self.data.xpos[dd["body"]].copy()
        yaw = self.drone_pose(drone).yaw
        a = self._lidar_angles + yaw
        vec = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1).ravel()
        mujoco.mj_multiRay(self.model, self.data, pnt, vec, LIDAR_GROUPS, 1, dd["body"],
                           self._lidar_gid, self._lidar_dist, None, LIDAR_N, LIDAR_RANGE)
        out = self._lidar_dist.copy()
        out[(out < 0) | (out > LIDAR_RANGE)] = LIDAR_RANGE
        return out

    def fires_xy(self) -> list[tuple[float, float]]:
        return [(p.handle.x, p.handle.y) for p in self._placed.values() if p.handle.kind == "fire"]

    def placed_handles(self, kind: str | None = None) -> list[Handle]:
        """Everything placed from the pools, scenario objects included (debug/tests)."""
        return [p.handle for p in self._placed.values() if kind is None or p.handle.kind == kind]

    def temperature(self, drone: str) -> float:
        p = self.drone_pose(drone)
        return temperature_at(p.x, p.y, self.fires_xy(), self.ambient)

    def temperature_at(self, x: float, y: float) -> float:
        return temperature_at(x, y, self.fires_xy(), self.ambient)

    # ------------------------------------------------------------ cameras
    def cam_info(self, drone: str) -> CamInfo:
        """Intrinsics + camera pose (camera centre, drone yaw). pixel_to_world does the exact ray."""
        cid = self._d[drone]["cam"]
        fx, fy, cx, cy = shapes.cam_intrinsics()
        x, y, z = (float(v) for v in self.data.cam_xpos[cid])
        return CamInfo(
            width=shapes.CAM_W, height=shapes.CAM_H, fovy_deg=shapes.CAM_FOVY_DEG,
            fx=fx, fy=fy, cx=cx, cy=cy,
            pose=Pose(x=x, y=y, z=z, yaw=self.drone_pose(drone).yaw),
            pitch_deg=-shapes.CAM_TILT_DEG,
        )

    def render_drone(self, drone: str) -> tuple[np.ndarray, CamInfo]:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("World.render_drone must run on the main thread (GL context)")
        info = self.cam_info(drone)
        if drone in self._dead:  # camera "static"
            return self._rng.integers(0, 256, (shapes.CAM_H, shapes.CAM_W, 3), dtype=np.uint8), info
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, shapes.CAM_H, shapes.CAM_W)
            self._scene_opt = mujoco.MjvOption()
            self._scene_opt.geomgroup[:] = RENDER_GROUPS
        r = self._renderer
        r.update_scene(self.data, camera=self._d[drone]["cam"], scene_option=self._scene_opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        return r.render().copy(), info

    def dashboard_frame(self, drone: str, frame: np.ndarray) -> np.ndarray:
        """The copy of a drone frame shown on the dashboard: smoke fog by depth, like the firefighter view
        (fog.UI_FOG_DENSITY, 0 = off). Call right after render_drone (same pose). Perception always gets
        the clean frame."""
        if fog.UI_FOG_DENSITY <= 0:
            return frame
        depth = self.render_depth(drone)
        return frame if depth is None else fog.apply_fog(frame, depth, fog.UI_FOG_DENSITY)

    def render_depth(self, drone: str) -> np.ndarray | None:
        """Depth in metres for the same view as render_drone (call it right after). For UI effects (the
        dashboard's smoke fog); None for a dead drone (its static is not fogged)."""
        if drone in self._dead:
            return None
        if self._renderer is None:
            self.render_drone(drone)
        r = self._renderer
        r.update_scene(self.data, camera=self._d[drone]["cam"], scene_option=self._scene_opt)
        r.enable_depth_rendering()
        try:
            return r.render().copy()
        finally:
            r.disable_depth_rendering()

    def pixel_to_world(self, drone: str, u: int, v: int) -> XY | None:
        cid = self._d[drone]["cam"]
        fx, fy, cx, cy = shapes.cam_intrinsics()
        d_cam = np.array([(u + 0.5 - cx) / fx, -(v + 0.5 - cy) / fy, -1.0])
        vec = self.data.cam_xmat[cid].reshape(3, 3) @ d_cam
        vec /= np.linalg.norm(vec)
        pnt = self.data.cam_xpos[cid].copy()
        dist = mujoco.mj_ray(self.model, self.data, pnt, vec, PIXEL_RAY_GROUPS, 1,
                             self._d[drone]["body"], self._ray_gid)
        if dist < 0 or dist > PIXEL_RAY_MAX:
            return None
        hit = pnt + dist * vec
        return XY(x=float(hit[0]), y=float(hit[1]))

    def line_of_sight(self, drone: str, x: float, y: float, z: float = 0.0) -> bool:
        """True if no wall or debris blocks the ray from this drone's camera to (x, y, z).

        Hits within the last 0.3 m don't count, so a panel mounted on a wall still counts as visible.
        """
        cid = self._d[drone]["cam"]
        pnt = self.data.cam_xpos[cid].copy()
        vec = np.array([x, y, z]) - pnt
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            return True
        vec /= dist
        hit = mujoco.mj_ray(self.model, self.data, pnt, vec, LIDAR_GROUPS, 1, self._d[drone]["body"], self._ray_gid)
        return hit < 0 or hit >= dist - 0.3

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------ change tool
    def _is_free(self, x: float, y: float) -> bool:
        r, c = world_to_cell(x, y, self.grid_h, self.res, self.origin)
        return 0 <= r < self.grid_h and 0 <= c < self.grid_w and bool(self.free[r, c])

    def _too_close(self, x: float, y: float) -> bool:
        pts = [(p.x, p.y) for d in self.drones if d not in self._dead for p in [self.drone_pose(d)]]
        f = self.ff_pose()
        pts.append((f.x, f.y))
        if self._survivor_xy:
            pts.append(self._survivor_xy)
        return any(math.hypot(x - px, y - py) < SAFE_DIST for px, py in pts)

    def _place(self, kind: str, x: float, y: float, force: bool, yaw: float | None,
               scenario: bool = False, size: float = 1.5, jitter: bool = True) -> Handle | None:
        if not force and (not self._is_free(x, y) or self._too_close(x, y)):
            return None
        used = {p.slot for p in self._placed.values() if p.handle.kind == kind}
        slot = next((i for i in range(len(self._pool_mocap[kind])) if i not in used), None)
        if slot is None:
            return None
        if yaw is None:
            yaw = self._rng.uniform(-math.pi, math.pi)
        if kind == "debris" and jitter:
            yaw += math.radians(self._rng.uniform(-DEBRIS_TILT_DEG, DEBRIS_TILT_DEG))  # "slightly rotated"
        mid = self._pool_mocap[kind][slot]
        self.data.mocap_pos[mid] = (x, y, 0.0)
        self.data.mocap_quat[mid] = _yaw_quat(yaw)
        self._next_id += 1
        h = Handle(id=f"{kind}-{self._next_id}", kind=kind, x=float(x), y=float(y))
        self._placed[h.id] = _Placed(h, slot, scenario, size)
        mujoco.mj_forward(self.model, self.data)
        return h

    def place_debris(self, x: float, y: float, force: bool = False, yaw: float | None = None) -> Handle | None:
        """None = pool empty, not a free cell, or within 1 m of a drone/firefighter/survivor."""
        return self._place("debris", x, y, force, yaw)

    def place_fire(self, x: float, y: float, force: bool = False) -> Handle | None:
        """force=True skips the safety checks. The flare-up uses it to put fire on its drone (§7.4)."""
        return self._place("fire", x, y, force, None)

    def remove(self, h: Handle) -> None:
        p = self._placed.pop(h.id, None)
        if p is None:
            return
        self.data.mocap_pos[self._pool_mocap[h.kind][p.slot]] = (0.0, 0.0, mjcf.PARK_Z)
        if h.kind == "fire":
            g = self._fire_geoms[p.slot]
            self.model.geom_rgba[g] = self._orig_rgba[g]
        mujoco.mj_forward(self.model, self.data)

    def kill_drone(self, drone: str) -> None:
        """Flare-up victim: drops to the floor, goes dark, camera turns to static."""
        self._dead.add(drone)
        self._apply_dead_visuals()

    def _apply_dead_visuals(self) -> None:
        m = self.model
        for d in self.drones:
            dd = self._d[d]
            dead = d in self._dead
            m.body_gravcomp[dd["body"]] = 0.0 if dead else self._orig_gravcomp[dd["body"]]
            m.geom_rgba[dd["geoms"]] = DEAD_RGBA if dead else self._orig_rgba[dd["geoms"]]

    # ------------------------------------------------------------ firefighter
    def set_ff_input(self, fwd: float, strafe: float, yaw: float) -> None:
        """fwd/strafe in [-1, 1] (x 1.4 m/s, strafe + = right). yaw = absolute target heading (rad)."""
        self._ff_cmd = (float(np.clip(fwd, -1, 1)), float(np.clip(strafe, -1, 1)), float(yaw))

    def ff_pose(self) -> Pose:
        q, a = self.data.qpos, self._ff["q"]
        return Pose(x=float(q[a[0]]), y=float(q[a[1]]), z=0.0, yaw=_wrap(float(q[a[2]])))

    def push_ff_back(self) -> None:
        """Walked into fire: move 1 m back against the direction of travel (or the heading)."""
        q, v = self.data.qpos, self.data.qvel
        fq, fv = self._ff["q"], self._ff["v"]
        vel = np.array([v[fv[0]], v[fv[1]]])
        if np.linalg.norm(vel) > 0.05:
            dirn = vel / np.linalg.norm(vel)
        else:
            dirn = np.array([math.cos(q[fq[2]]), math.sin(q[fq[2]])])
        x0, y0 = q[fq[0]], q[fq[1]]
        for dist in (FF_PUSH, FF_PUSH / 2, 0.0):
            nx, ny = x0 - dirn[0] * dist, y0 - dirn[1] * dist
            if self._is_free(nx, ny):
                break
        q[fq[0]], q[fq[1]] = nx, ny
        v[fv[0]] = v[fv[1]] = v[fv[2]] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------ truth + snapshots
    def ground_truth(self) -> GroundTruth:
        """ONLY for mock perception and the mission summary. Never from sgr/swarm or sgr/guidance."""
        sc = self.scenario
        dp = self.build_info.decoy_pose
        placed = list(self._placed.values())
        return GroundTruth(
            fires=[Fire(x=p.handle.x, y=p.handle.y, size=p.size) for p in placed if p.handle.kind == "fire"],
            survivor=sc.survivor,
            # the panel as actually mounted on the nearest wall face
            decoy=Decoy(x=dp[0], y=dp[1], kind=sc.decoy.kind) if (dp and sc.decoy) else None,
            debris=[p.handle for p in placed if p.handle.kind == "debris" and not p.scenario],  # clicks only
            compromised=list(sc.compromised),
            dead_drones=sorted(self._dead),
        )

    def get_state(self) -> bytes:
        sig = mujoco.mjtState.mjSTATE_INTEGRATION
        st = np.empty(mujoco.mj_stateSize(self.model, sig))
        mujoco.mj_getState(self.model, self.data, st, sig)
        rt = {
            "placed": [(p.handle.model_dump(), p.slot, p.scenario, p.size) for p in self._placed.values()],
            "next_id": self._next_id,
            "cmd": {k: v.copy() for k, v in self._cmd.items()},
            "ff_cmd": self._ff_cmd,
            "dead": set(self._dead),
            "rng": self._rng.bit_generator.state,
        }
        return pickle.dumps({"v": 1, "mj": st, "rt": rt})

    def set_state(self, b: bytes) -> None:
        blob = pickle.loads(b)
        mujoco.mj_setState(self.model, self.data, blob["mj"], mujoco.mjtState.mjSTATE_INTEGRATION)
        rt = blob["rt"]
        self._placed = {h["id"]: _Placed(Handle(**h), slot, scen, size) for h, slot, scen, size in rt["placed"]}
        self._next_id = rt["next_id"]
        self._cmd = {k: np.array(v) for k, v in rt["cmd"].items()}
        self._ff_cmd = tuple(rt["ff_cmd"])
        self._dead = set(rt["dead"])
        self._rng.bit_generator.state = rt["rng"]
        self.model.geom_rgba[:] = self._orig_rgba
        self._apply_dead_visuals()
        mujoco.mj_forward(self.model, self.data)
