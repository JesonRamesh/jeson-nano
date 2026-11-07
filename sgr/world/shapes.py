"""MJCF snippet builders for the simple world look (spec §7.1).

Shared by the World (sgr/world/world.py) and the Gemma bench
(sgr/perception/gemma_bench.py), so Gemma is benchmarked on exactly the
shapes the drone cameras see.

Geom groups (§7.1): 0 = walls/debris (ray-visible), 1 = fire/decoy/survivor
visuals, 2 = drones + firefighter, 3 = hidden.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------- camera
CAM_W, CAM_H = 320, 240
CAM_FOVY_DEG = 90.0
CAM_TILT_DEG = 15.0  # drone camera looks this far below horizontal

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


def assets() -> str:
    """<asset> children needed by the snippets below."""
    out = [f'<mesh name="{n}" vertex="{_cone_vertices(r, h)}"/>' for n, r, h in _CONES]
    out.append(f'<material name="mat_fire" emission="{FIRE_EMISSION}" specular="0" shininess="0"/>')
    out.append('<material name="mat_matte" specular="0" shininess="0" reflectance="0"/>')
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


def fire_cluster_geoms(prefix: str) -> str:
    """Geoms of one fire cluster (3-5 emissive cones). Put them inside a body at floor level."""
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
        f'material="mat_matte" group="1" {_VIS}/></body>'
    )


def debris_geom(name: str, contype: int = 1, conaffinity: int = 1) -> str:
    sx, sy, sz = DEBRIS_SIZE
    return (
        f'<geom name="{name}" type="box" size="{sx / 2} {sy / 2} {sz / 2}" pos="0 0 {sz / 2}" '
        f'rgba="{rgba(DEBRIS_RGBA)}" material="mat_matte" group="0" '
        f'contype="{contype}" conaffinity="{conaffinity}"/>'
    )


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
