"""MJCF snippet builders for the simple world look (spec §7.1).

Shared by the World (sgr/world/world.py) and the Gemma bench
(sgr/perception/gemma_bench.py), so Gemma is benchmarked on exactly the
shapes the drone cameras see.

Geom groups (§7.1): 0 = walls/debris (ray-visible), 1 = fire/decoy/survivor
visuals, 2 = drones + firefighter, 3 = hidden.
"""

from __future__ import annotations

import functools
import math
import os
from pathlib import Path

import numpy as np

from sgr import config

# ---------------------------------------------------------------- camera
CAM_W, CAM_H = config.CAM_W, config.CAM_H
CAM_FOVY_DEG = config.CAM_FOVY_DEG
CAM_TILT_DEG = -config.CAM_PITCH_DEG  # degrees below horizontal (config pitch is -15)

# ---------------------------------------------------------------- sizes
WALL_HEIGHT = 2.6
DEBRIS_SIZE = (1.0, 1.0, 2.0)  # spec says 1.2 m tall; raised so the 1.6 m LiDAR plane hits it
DECOY_SIZE = (0.6, 0.9)  # width, height (m)
DECOY_CENTRE_Z = 1.5

# ---------------------------------------------------------------- colours
WALL_RGBA = (0.72, 0.68, 0.60, 1.0)  # grey-beige
FLOOR_RGBA = (0.22, 0.22, 0.23, 1.0)  # dark grey
DEBRIS_RGBA = (0.42, 0.35, 0.28, 1.0)  # grey-brown
DECOY_RGBA = (1.0, 0.45, 0.05, 1.0)  # orange panel
HIVIS_RGBA = (0.75, 1.0, 0.05, 1.0)  # hi-vis yellow-green
SKIN_RGBA = (0.85, 0.64, 0.50, 1.0)
TROUSER_RGBA = (0.15, 0.18, 0.30, 1.0)
# fire cones, yellow -> orange (emissive, so lighting doesn't dim them)
FIRE_RGBAS = (
    (1.0, 0.82, 0.15, 1.0),  # yellow core
    (1.0, 0.45, 0.03, 1.0),  # orange
    (0.95, 0.25, 0.02, 1.0),  # red-orange
    (1.0, 0.60, 0.08, 1.0),  # amber
    (0.95, 0.32, 0.02, 1.0),  # deep orange
)
# No glow on the decoy: tested with gemma4:e4b, a glowing panel reads as "flames" from a grazing
# fly-past (3/3), while the matte panel reads "orange panel, no fire" (10/12 views). The detector's
# low V threshold catches the matte panel at grazing angles instead.
DECOY_EMISSION = 0.0
FIRE_EMISSION = 0.6  # enough to glow; 1.0 + headlight saturates everything to flat yellow

# visual-only geoms: never collide
_VIS = 'contype="0" conaffinity="0"'


def rgba(c) -> str:
    return " ".join(f"{v:g}" for v in c)


def _cone_vertices(radius: float, height: float, n: int = 12) -> str:
    pts = [(0.0, 0.0, height)]
    for k in range(n):
        a = 2 * math.pi * k / n
        pts.append((radius * math.cos(a), radius * math.sin(a), 0.0))
    return " ".join(f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in pts)


# Fire cluster scale. 1.35 keeps a fire >= 0.5 % of the frame (detector threshold) out to ~5 m,
# the coverage range of §8.2. Spec §7.3 allows bigger shapes.
FIRE_SCALE = 1.35

# (name, radius, height). MuJoCo has no cone primitive, so these are convex meshes.
_CONES = tuple((n, r * FIRE_SCALE, h * FIRE_SCALE)
               for n, r, h in (("cone_tall", 0.26, 1.15), ("cone_mid", 0.22, 0.85), ("cone_short", 0.18, 0.55)))

# cone layout of one fire cluster, in the cluster body frame: (mesh, x, y, colour index)
_FIRE_LAYOUT = (
    ("cone_tall", 0.00, 0.00, 0),
    ("cone_mid", 0.30, 0.14, 1),
    ("cone_short", -0.26, 0.20, 2),
    ("cone_mid", 0.06, -0.30, 3),
    ("cone_short", -0.20, -0.18, 4),
)


# W's quadcopter model, shared with the dashboard's 3D view (which draws it at DRONE_SCALE = 1.25).
DRONE_MODEL = Path(__file__).resolve().parents[2] / "web" / "src" / "scene3d" / "dressing" / "drone.glb"
DRONE_MODEL_SCALE = 1.25
_DRONE_EMISSION_MATS = {0.6: "mat_dm_e60", 1.0: "mat_dm_e100"}


@functools.lru_cache(maxsize=1)
def drone_model_parts() -> tuple:
    """((part, vertices z-up m, faces, rgba, emission, material_name), ...) from drone.glb; () if unavailable.

    glTF is y-up with +x forward; the sim is z-up: (x, y, z) -> (x, -z, y), the same +90 deg X turn the
    3D view applies.
    """
    if not DRONE_MODEL.is_file():
        return ()
    try:
        import trimesh

        scene = trimesh.load(DRONE_MODEL)
        parts = []
        for node in sorted(scene.graph.nodes_geometry):
            T, gname = scene.graph[node]
            g = scene.geometry[gname]
            v = trimesh.transformations.transform_points(np.asarray(g.vertices, float), T)
            v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1) * DRONE_MODEL_SCALE
            mat = g.visual.material
            col = np.asarray(getattr(mat, "baseColorFactor", (128, 128, 128, 255)), float)
            col = col / 255.0 if col.max() > 1.0 else col
            em = getattr(mat, "emissiveFactor", None)
            emission = 0.0 if em is None else (1.0 if max(em) > 0.8 else 0.6 if max(em) > 0.1 else 0.0)
            parts.append((node, v, np.asarray(g.faces, int), tuple(float(c) for c in col), emission,
                          str(getattr(mat, "name", ""))))
        return tuple(parts)
    except Exception:
        return ()


def drone_model_front() -> float:
    """Forward extent of the model (m): the drone camera sits just ahead of it."""
    parts = drone_model_parts()
    return max(float(p[1][:, 0].max()) for p in parts) if parts else 0.25


def _drone_mesh_assets() -> list[str]:
    out = [f'<material name="{m}" emission="{e}" specular="0" shininess="0"/>' for e, m in _DRONE_EMISSION_MATS.items()]
    for node, v, f, *_ in drone_model_parts():
        vs = " ".join(f"{x:.4f}" for x in v.ravel())
        fs = " ".join(str(i) for i in f.ravel())
        out.append(f'<mesh name="dm_{node}" vertex="{vs}" face="{fs}" inertia="shell"/>')
    return out


def drone_model_geoms(name: str, colour) -> str:
    """Geoms of one drone from the shared model (visual only, group 2). The accent part gets `colour`."""
    out = []
    for node, _v, _f, col, emission, mat in drone_model_parts():
        c = colour if mat == "accent" else col
        material = _DRONE_EMISSION_MATS.get(emission, "mat_matte")
        out.append(f'<geom name="{name}_m_{node}" type="mesh" mesh="dm_{node}" rgba="{rgba(c)}" material="{material}" '
                   f'group="2" contype="0" conaffinity="0" mass="0"/>')
    return chr(10).join(out)


def assets() -> str:
    """<asset> children needed by the snippets below."""
    out = [f'<mesh name="{n}" vertex="{_cone_vertices(r, h)}"/>' for n, r, h in _CONES]
    out.append(f'<material name="mat_fire" emission="{FIRE_EMISSION}" specular="0" shininess="0"/>')
    out.append('<material name="mat_fire_hot" emission="1" specular="0" shininess="0"/>')  # flame particles
    out.append('<material name="mat_matte" specular="0" shininess="0" reflectance="0"/>')
    out.extend(_drone_mesh_assets())
    out.extend(_rubble_mesh_assets())
    out.append(f'<material name="mat_decoy" emission="{DECOY_EMISSION}" specular="0" shininess="0" reflectance="0"/>')
    return "\n".join(out)


def visual_settings() -> str:
    """<visual> block: no shadows, decent light, offscreen buffer big enough for 320x240."""
    return (
        "<visual>"
        '<quality shadowsize="0"/>'
        '<headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0 0 0"/>'
        '<global offwidth="640" offheight="480"/>'
        "</visual>"
    )


# Flame look: tapered flame tongues (white-yellow core -> orange -> red tips) on a glowing ember bed.
# Against the photoreal scan replica gemma4:e4b read the cartoon cones as "safety
# cones" / "yellow markers" (6/24 views called fire), so the fire has to look like fire.
# (x, y, z_centre, rx, ry, rz, colour) per tongue, in the cluster frame (metres)
_FLAME_TONGUES = (
    (0.00, 0.00, 0.55, 0.16, 0.16, 0.55, (1.0, 0.92, 0.55, 1.0)),   # white-yellow core
    (0.18, 0.08, 0.42, 0.13, 0.13, 0.42, (1.0, 0.72, 0.15, 1.0)),
    (-0.15, 0.12, 0.38, 0.12, 0.12, 0.38, (1.0, 0.55, 0.05, 1.0)),
    (0.05, -0.18, 0.45, 0.13, 0.12, 0.45, (1.0, 0.62, 0.08, 1.0)),
    (-0.12, -0.12, 0.30, 0.11, 0.11, 0.30, (0.98, 0.38, 0.03, 1.0)),
    (0.25, -0.10, 0.28, 0.10, 0.10, 0.28, (0.95, 0.30, 0.02, 1.0)),
    (-0.25, 0.00, 0.25, 0.10, 0.10, 0.25, (0.92, 0.25, 0.02, 1.0)),
    (0.02, 0.02, 1.10, 0.09, 0.09, 0.30, (1.0, 0.50, 0.04, 1.0)),    # licking tip
    (0.12, -0.04, 0.95, 0.07, 0.07, 0.22, (0.95, 0.28, 0.02, 1.0)),
    (-0.10, 0.06, 0.88, 0.07, 0.07, 0.20, (0.95, 0.30, 0.02, 1.0)),
)
# W's FireFX look around the proven flame core (web/src/scene3d/dressing/fire.tsx): animated flame
# particles and embers, re-positioned every tick by World. No smoke: MuJoCo draws it as hard-edged grey
# spheres, and with them gemma4:e4b called fire in 21/24 views ("figures in yellow"); without them 24/24.
FIRE_N_PARTICLES = 24
FIRE_N_EMBERS = 6
FLAME_H = 0.75 * 1.5 + 0.4  # W's uHeight at size 1.5
FLAME_R = 0.32 * 1.5  # W's uRadius


@functools.lru_cache(maxsize=1)
def fire_particle_params():
    """Per-particle (phase, angle, radius fraction, speed) like W's seeds(); deterministic."""
    from .rubble import rng

    r = rng(7331)
    flames = np.array([[r(), r(), r(), 0.9 + r() * 0.7] for _ in range(FIRE_N_PARTICLES)])
    for _ in range(8 * 4):  # W's 8 smoke seeds: keep drawing them so the embers stay as Gemma-tested
        r()
    embers = np.array([[r(), r(), r(), 0.35 + r() * 0.3] for _ in range(FIRE_N_EMBERS)])
    return flames, embers


FIRE_LOOK =os.environ.get("SGR_FIRE_LOOK", "flames").lower()  # "flames" | "cones" (the old look)


def _flame_geoms(prefix: str) -> str:
    s = FIRE_SCALE * 0.9
    out = [
        f'<geom name="{prefix}_embers" type="cylinder" size="{0.55 * s:.3f} 0.01" pos="0 0 0.01" '
        f'material="mat_fire" rgba="0.85 0.22 0.02 1" group="1" {_VIS}/>'
    ]
    for i, (x, y, z, rx, ry, rz, c) in enumerate(_FLAME_TONGUES):
        out.append(
            f'<geom name="{prefix}_c{i}" type="ellipsoid" size="{rx * s:.3f} {ry * s:.3f} {rz * s:.3f}" '
            f'pos="{x * s:.3f} {y * s:.3f} {z * s:.3f}" material="mat_fire" rgba="{rgba(c)}" group="1" {_VIS}/>'
        )
    for i in range(FIRE_N_PARTICLES):  # positions / colours / sizes set by World every tick
        out.append(f'<geom name="{prefix}_p{i}" type="ellipsoid" size="0.11 0.11 0.15" pos="0 0 0.3" '
                   f'material="mat_fire_hot" rgba="1 0.6 0.1 0.8" group="1" {_VIS}/>')
    for i in range(FIRE_N_EMBERS):
        out.append(f'<geom name="{prefix}_e{i}" type="sphere" size="0.018" pos="0 0 1.0" '
                   f'material="mat_fire" rgba="1 0.6 0.24 0.9" group="1" {_VIS}/>')
    return "\n".join(out)


def fire_cluster_geoms(prefix: str) -> str:
    """Geoms of one fire cluster. Put them inside a body at floor level."""
    if FIRE_LOOK != "cones":
        return _flame_geoms(prefix)
    out = []
    for i, (mesh, x, y, ci) in enumerate(_FIRE_LAYOUT):
        out.append(
            f'<geom name="{prefix}_c{i}" type="mesh" mesh="{mesh}" pos="{x * FIRE_SCALE:.3f} {y * FIRE_SCALE:.3f} 0" '
            f'material="mat_fire" rgba="{rgba(FIRE_RGBAS[ci])}" group="1" {_VIS}/>'
        )
    return "\n".join(out)


def fire_cluster(name: str, x: float, y: float, z: float = 0.0, mocap: bool = False) -> str:
    m = ' mocap="true"' if mocap else ""
    return f'<body name="{name}" pos="{x} {y} {z}"{m}>\n{fire_cluster_geoms(name)}\n</body>'


def yaw_quat(yaw: float) -> str:
    """quat for a rotation about z. Used instead of euler, whose unit depends on <compiler angle>."""
    return f"{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}"


def survivor(name: str, x: float, y: float, yaw: float) -> str:
    """Capsule humanoid lying on its back; body frame +x = head direction."""
    hv, sk, tr = rgba(HIVIS_RGBA), rgba(SKIN_RGBA), rgba(TROUSER_RGBA)
    g = f'group="1" material="mat_matte" {_VIS}'
    return f"""<body name="{name}" pos="{x} {y} 0" quat="{yaw_quat(yaw)}">
<geom name="{name}_torso" type="capsule" fromto="-0.25 0 0.16 0.30 0 0.16" size="0.17" rgba="{hv}" {g}/>
<geom name="{name}_head" type="sphere" pos="0.62 0 0.13" size="0.13" rgba="{sk}" {g}/>
<geom name="{name}_neck" type="capsule" fromto="0.42 0 0.13 0.50 0 0.13" size="0.06" rgba="{sk}" {g}/>
<geom name="{name}_arm_l" type="capsule" fromto="0.36 0.24 0.12 -0.10 0.40 0.08" size="0.065" rgba="{hv}" {g}/>
<geom name="{name}_arm_r" type="capsule" fromto="0.36 -0.24 0.12 -0.05 -0.42 0.08" size="0.065" rgba="{hv}" {g}/>
<geom name="{name}_hand_l" type="sphere" pos="-0.16 0.42 0.07" size="0.06" rgba="{sk}" {g}/>
<geom name="{name}_hand_r" type="sphere" pos="-0.11 -0.45 0.07" size="0.06" rgba="{sk}" {g}/>
<geom name="{name}_leg_l" type="capsule" fromto="-0.35 0.10 0.10 -1.05 0.16 0.08" size="0.085" rgba="{tr}" {g}/>
<geom name="{name}_leg_r" type="capsule" fromto="-0.35 -0.10 0.10 -1.05 -0.20 0.08" size="0.085" rgba="{tr}" {g}/>
<geom name="{name}_hivis_l" type="capsule" fromto="-0.55 0.12 0.10 -0.70 0.13 0.10" size="0.09" rgba="{hv}" {g}/>
<geom name="{name}_hivis_r" type="capsule" fromto="-0.55 -0.13 0.10 -0.70 -0.15 0.10" size="0.09" rgba="{hv}" {g}/>
</body>"""


def decoy_panel(name: str, x: float, y: float, normal_yaw: float) -> str:
    """Flat orange panel, centre at (x, y, 1.5). normal_yaw = direction the face points (rad)."""
    w, h = DECOY_SIZE
    # box local x = normal (thin), local y = width, local z = height
    return (
        f'<body name="{name}" pos="{x} {y} {DECOY_CENTRE_Z}" quat="{yaw_quat(normal_yaw)}">'
        f'<geom name="{name}_g" type="box" size="0.01 {w / 2} {h / 2}" rgba="{rgba(DECOY_RGBA)}" '
        f'material="mat_decoy" group="1" {_VIS}/></body>'
    )


DEBRIS_COLLISION_GROUP = 4  # the 2 m block: physics + LiDAR + line of sight only, never drawn


def _rubble_mesh_assets() -> list[str]:
    from . import rubble

    out = []
    for seed in range(1, rubble.N_VARIANTS + 1):
        for col, (v, f) in rubble.pile(seed).items():
            vs = " ".join(f"{x:.4f}" for x in v.ravel())
            fs = " ".join(str(i) for i in f.ravel())
            out.append(f'<mesh name="rb{seed}_{col}" vertex="{vs}" face="{fs}" inertia="shell"/>')
    return out


def debris_geom(name: str, contype: int = 1, conaffinity: int = 1, variant: int = 1) -> str:
    """Debris: an invisible 2 m collision block (what LiDAR and the drones hit), dressed as W's rubble pile."""
    from . import rubble

    sx, sy, sz = DEBRIS_SIZE
    out = [f'<geom name="{name}" type="box" size="{sx / 2} {sy / 2} {sz / 2}" pos="0 0 {sz / 2}" '
           f'rgba="{rgba(DEBRIS_RGBA)}" material="mat_matte" group="{DEBRIS_COLLISION_GROUP}" '
           f'contype="{contype}" conaffinity="{conaffinity}"/>']
    seed = (variant - 1) % rubble.N_VARIANTS + 1
    for col in rubble.pile(seed):
        out.append(f'<geom name="{name}_rb_{col}" type="mesh" mesh="rb{seed}_{col}" '
                   f'rgba="{rgba((*rubble.COLOURS[col], 1.0))}" material="mat_matte" group="1" {_VIS}/>')
    return "\n".join(out)


def camera_xyaxes(yaw: float = 0.0, tilt_deg: float = CAM_TILT_DEG) -> str:
    """xyaxes for a camera looking along heading `yaw`, tilted `tilt_deg` down.

    MuJoCo cameras look along their local -z, x = right, y = up.
    """
    t = math.radians(tilt_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    right = (s, -c, 0.0)
    up = (math.sin(t) * c, math.sin(t) * s, math.cos(t))
    return " ".join(f"{v:.6f}" for v in (*right, *up))


def cam_intrinsics(w: int = CAM_W, h: int = CAM_H, fovy_deg: float = CAM_FOVY_DEG):
    """(fx, fy, cx, cy) for a MuJoCo camera with square pixels."""
    f = (h / 2) / math.tan(math.radians(fovy_deg) / 2)
    return f, f, w / 2, h / 2
