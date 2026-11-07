"""Build the MuJoCo model for a map + scenario (spec §7.1).

Collision bits:
- walls, debris: contype=1, conaffinity=1
- drones, firefighter: contype=2, conaffinity=1. They hit walls and debris,
  but not each other.
- floor and visuals (fire, survivor, decoy): contype=0, conaffinity=0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sgr import config

from . import shapes
from .walls import grid_to_boxes, nearest_wall_normal, rect_to_world

N_DEBRIS = 40
N_FIRE = 30
PARK_Z = -50.0

DRONE_RADIUS = config.DRONE_RADIUS
DRONE_MASS = 1.0
DRONE_Z_RANGE = (0.25, 2.5)
DRONE_KV_LIN = 10.0
DRONE_KV_YAW = 0.3
DRONE_VMAX = 2.0
DRONE_WMAX = 3.0
CAM_FORWARD = DRONE_RADIUS + 0.02  # camera sits just in front of the sphere

FF_RADIUS = 0.25
FF_HEIGHT = 1.8
FF_MASS = 80.0
FF_KV_LIN = 800.0
FF_KV_YAW = 25.0

# drone colours avoid orange and yellow-green so drones never trip the fire/person detector
DRONE_RGBAS = ((0.20, 0.45, 1.0, 1), (0.10, 0.80, 0.90, 1), (0.80, 0.25, 0.85, 1), (0.92, 0.92, 0.96, 1))
FF_RGBA = (0.55, 0.12, 0.10, 1)


@dataclass
class BuildInfo:
    n_wall_boxes: int
    decoy_pose: tuple[float, float, float] | None  # x, y, normal_yaw


def _drone(name: str, rgba) -> str:
    c = shapes.rgba(rgba)
    r = DRONE_RADIUS
    rotors = []
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        rotors.append(
            f'<geom name="{name}_rotor{i}" type="cylinder" size="0.10 0.012" '
            f'pos="{sx * 0.21} {sy * 0.21} 0.06" mass="0.01" rgba="0.15 0.15 0.15 1" '
            f'group="2" contype="0" conaffinity="0"/>'
        )
    lo, hi = DRONE_Z_RANGE
    return f"""<body name="{name}" gravcomp="1">
<joint name="{name}_x" type="slide" axis="1 0 0"/>
<joint name="{name}_y" type="slide" axis="0 1 0"/>
<joint name="{name}_z" type="slide" axis="0 0 1" limited="true" range="{lo} {hi}"/>
<joint name="{name}_yaw" type="hinge" axis="0 0 1"/>
<geom name="{name}_body" type="sphere" size="{r}" mass="{DRONE_MASS}" rgba="{c}" group="2" contype="2" conaffinity="1"/>
{chr(10).join(rotors)}
<camera name="{name}_cam" pos="{CAM_FORWARD} 0 0" fovy="{shapes.CAM_FOVY_DEG}" xyaxes="{shapes.camera_xyaxes(0.0)}"/>
</body>"""


def _drone_actuators(name: str) -> str:
    v, w = DRONE_VMAX, DRONE_WMAX
    return "\n".join(
        [f'<velocity name="{name}_v{ax}" joint="{name}_{ax}" kv="{DRONE_KV_LIN}" ctrlrange="{-v} {v}"/>'
         for ax in "xyz"]
        + [f'<velocity name="{name}_wz" joint="{name}_yaw" kv="{DRONE_KV_YAW}" ctrlrange="{-w} {w}"/>']
    )


def _firefighter() -> str:
    r = FF_RADIUS
    return f"""<body name="FF">
<joint name="FF_x" type="slide" axis="1 0 0"/>
<joint name="FF_y" type="slide" axis="0 1 0"/>
<joint name="FF_yaw" type="hinge" axis="0 0 1"/>
<geom name="FF_body" type="capsule" fromto="0 0 {r} 0 0 {FF_HEIGHT - r}" size="{r}" mass="{FF_MASS}"
      rgba="{shapes.rgba(FF_RGBA)}" group="2" contype="2" conaffinity="1"/>
</body>"""


def _ff_actuators() -> str:
    return (
        f'<velocity name="FF_vx" joint="FF_x" kv="{FF_KV_LIN}" ctrlrange="-2 2"/>\n'
        f'<velocity name="FF_vy" joint="FF_y" kv="{FF_KV_LIN}" ctrlrange="-2 2"/>\n'
        f'<velocity name="FF_wz" joint="FF_yaw" kv="{FF_KV_YAW}" ctrlrange="-4 4"/>'
    )


def build_mjcf(free: np.ndarray, res: float, origin, scenario, drone_ids: list[str]) -> tuple[str, BuildInfo]:
    """free: bool grid (True = free), image orientation (row 0 = top)."""
    h, w = free.shape
    wall_rgba = shapes.rgba(shapes.WALL_RGBA)
    wh = shapes.WALL_HEIGHT / 2

    walls = []
    boxes = grid_to_boxes(~free)
    for i, rect in enumerate(boxes):
        cx, cy, hx, hy = rect_to_world(rect, h, res, origin)
        walls.append(
            f'<geom name="wall{i}" type="box" pos="{cx:.3f} {cy:.3f} {wh}" size="{hx:.3f} {hy:.3f} {wh}" '
            f'rgba="{wall_rgba}" material="mat_matte" group="0" contype="1" conaffinity="1"/>'
        )

    pools = []
    for i in range(N_DEBRIS):
        pools.append(
            f'<body name="debris{i:02d}" mocap="true" pos="0 0 {PARK_Z}">'
            f'{shapes.debris_geom(f"debris{i:02d}_g")}</body>'
        )
    for i in range(N_FIRE):
        pools.append(shapes.fire_cluster(f"fire{i:02d}", 0.0, 0.0, PARK_Z, mocap=True))

    statics = []
    sv = scenario.survivor
    statics.append(shapes.survivor("survivor", sv.x, sv.y, sv.yaw))
    decoy_pose = None
    dc = scenario.decoy
    if dc is not None:
        hit = nearest_wall_normal(free, dc.x, dc.y, res, origin)
        if hit is not None:
            _, fx, fy, nyaw = hit
            off = 0.011  # half thickness + epsilon, so the panel sits on the wall face
            px, py = fx + off * np.cos(nyaw), fy + off * np.sin(nyaw)
        else:  # no wall nearby: free-standing, facing -x
            px, py, nyaw = dc.x, dc.y, float(np.pi)
        decoy_pose = (float(px), float(py), float(nyaw))
        statics.append(shapes.decoy_panel("decoy", px, py, nyaw))

    drones = [_drone(d, DRONE_RGBAS[i % len(DRONE_RGBAS)]) for i, d in enumerate(drone_ids)]
    acts = [_drone_actuators(d) for d in drone_ids] + [_ff_actuators()]

    floor = shapes.rgba(shapes.FLOOR_RGBA)
    span = max(w, h) * res
    xml = f"""<mujoco model="sgr_world">
<compiler angle="radian" autolimits="true"/>
<option timestep="0.005" integrator="implicitfast"/>
{shapes.visual_settings()}
<asset>
{shapes.assets()}
</asset>
<worldbody>
<light name="sun" directional="true" pos="0 0 10" dir="0.2 0.1 -1" diffuse="0.55 0.55 0.55" castshadow="false"/>
<geom name="floor" type="plane" pos="{origin[0] + w * res / 2} {origin[1] + h * res / 2} 0" size="{span} {span} 0.1"
      rgba="{floor}" material="mat_matte" group="0" contype="0" conaffinity="0"/>
{chr(10).join(walls)}
{chr(10).join(statics)}
{chr(10).join(pools)}
{chr(10).join(drones)}
{_firefighter()}
</worldbody>
<actuator>
{chr(10).join(acts)}
</actuator>
</mujoco>"""
    return xml, BuildInfo(len(boxes), decoy_pose)
