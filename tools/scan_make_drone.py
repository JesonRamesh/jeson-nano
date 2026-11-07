#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "trimesh"]
# ///
"""Generate the quadcopter model for the 3D view -> web/src/scene3d/dressing/drone.glb

    python tools/scan_make_drone.py

Procedural, so it's our own work: released as CC0 (see dressing/README.md).
Conventions (glTF standard):
  * +Y up, **+X forward** (camera + white nav lights at the front, red at the back,
    accent arrow on top pointing forward), origin at the body centre;
  * span 0.50 m prop tip to prop tip (sim DRONE_RADIUS = 0.25 m), ~0.13 m tall;
  * nodes prop_FL / prop_FR / prop_RL / prop_RR: propellers, each centred on its own
    origin, spin them about their local +Y;
  * material "accent": LED ring + top arrow, white base colour -> tint per drone;
  * materials "nav_front" / "nav_rear": emissive lights.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

T = trimesh.transformations
# design frame: x forward, y left, z up. Converted to glTF y-up at the end.
TO_GLTF = T.rotation_matrix(-math.pi / 2, [1, 0, 0])      # (x, y, z) -> (x, z, -y)

ARM = 0.165          # motor centre distance from the body centre along x and y
PROP_R = 0.085       # propeller radius  -> span = 2 * (ARM + PROP_R) = 0.50 m


def mat(name, rgb, metal=0.0, rough=0.6, emissive=None, alpha=1.0):
    kw = {}
    if emissive is not None:
        kw["emissiveFactor"] = emissive
    if alpha < 1:
        kw["alphaMode"] = "BLEND"
    return PBRMaterial(name=name, baseColorFactor=[*rgb, alpha], metallicFactor=metal, roughnessFactor=rough,
                       doubleSided=alpha < 1, **kw)


def paint(mesh, m):
    mesh.visual = trimesh.visual.TextureVisuals(material=m)
    return mesh


def at(mesh, xyz=(0, 0, 0), rot=None):
    M = T.translation_matrix(xyz)
    if rot is not None:
        M = M @ rot
    mesh.apply_transform(M)
    return mesh


def build() -> trimesh.Scene:
    body_m = mat("body", (0.16, 0.17, 0.21), metal=0.2, rough=0.45)
    arm_m = mat("arm", (0.10, 0.10, 0.12), rough=0.5)
    motor_m = mat("motor", (0.55, 0.56, 0.60), metal=0.9, rough=0.3)
    prop_m = mat("prop", (0.85, 0.87, 0.92), rough=0.4, alpha=0.45)
    accent_m = mat("accent", (1.0, 1.0, 1.0), rough=0.3, emissive=(0.6, 0.6, 0.6))
    lens_m = mat("lens", (0.02, 0.02, 0.03), metal=0.5, rough=0.1)
    navf_m = mat("nav_front", (1.0, 1.0, 1.0), emissive=(1.0, 1.0, 1.0))
    navr_m = mat("nav_rear", (1.0, 0.1, 0.1), emissive=(1.0, 0.05, 0.05))

    parts = {}
    # body: low box + rounded top shell
    parts["body"] = paint(trimesh.util.concatenate([
        at(trimesh.creation.box(extents=(0.17, 0.11, 0.045))),
        at(trimesh.creation.icosphere(subdivisions=3, radius=1.0).apply_scale((0.085, 0.055, 0.035)), (0, 0, 0.02)),
        at(trimesh.creation.box(extents=(0.07, 0.07, 0.03)), (-0.02, 0, -0.035)),            # battery
    ]), body_m)
    # arms (X layout) + motors
    arms, motors, guards = [], [], []
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        L = math.hypot(ARM, ARM)
        ang = math.atan2(sy * ARM, sx * ARM)
        arms.append(at(trimesh.creation.box(extents=(L, 0.022, 0.016)), (sx * ARM / 2, sy * ARM / 2, 0.0),
                       T.rotation_matrix(ang, [0, 0, 1])))
        motors.append(at(trimesh.creation.cylinder(radius=0.022, height=0.032, sections=24), (sx * ARM, sy * ARM, 0.012)))
    parts["arms"] = paint(trimesh.util.concatenate(arms), arm_m)
    parts["motors"] = paint(trimesh.util.concatenate(motors), motor_m)
    # landing skids
    skids = []
    for sy in (1, -1):
        skids.append(at(trimesh.creation.cylinder(radius=0.006, height=0.18, sections=12), (0, sy * 0.05, -0.065),
                        T.rotation_matrix(math.pi / 2, [0, 1, 0])))
        for sx in (1, -1):
            skids.append(at(trimesh.creation.cylinder(radius=0.005, height=0.045, sections=10),
                            (sx * 0.05, sy * 0.045, -0.045), T.rotation_matrix(sy * 0.25, [1, 0, 0])))
    parts["skids"] = paint(trimesh.util.concatenate(skids), arm_m)
    # front camera gimbal (forward = +x)
    parts["camera"] = paint(trimesh.util.concatenate([
        at(trimesh.creation.icosphere(subdivisions=2, radius=0.028), (0.095, 0, -0.025)),
    ]), body_m)
    parts["lens"] = paint(at(trimesh.creation.cylinder(radius=0.013, height=0.012, sections=24), (0.122, 0, -0.025),
                             T.rotation_matrix(math.pi / 2, [0, 1, 0])), lens_m)
    # accent: LED ring around the body + arrow on top pointing forward
    ring = trimesh.creation.annulus(r_min=0.072, r_max=0.08, height=0.006, sections=48)
    ring.apply_scale((1.0, 0.72, 1.0))
    arrow = trimesh.creation.extrude_triangulation(np.array([[0.06, 0], [-0.01, 0.028], [-0.01, -0.028]]),
                                                   np.array([[0, 1, 2]]), height=0.004)
    parts["accent"] = paint(trimesh.util.concatenate([at(ring, (0, 0, 0.0)), at(arrow, (0.0, 0, 0.054))]), accent_m)
    # nav lights: white front pair, red rear
    parts["nav_front"] = paint(trimesh.util.concatenate(
        [at(trimesh.creation.icosphere(subdivisions=1, radius=0.009), (0.083, s * 0.035, 0.005)) for s in (1, -1)]),
        navf_m)
    parts["nav_rear"] = paint(at(trimesh.creation.icosphere(subdivisions=1, radius=0.01), (-0.087, 0, 0.005)), navr_m)

    scene = trimesh.Scene()
    for name, m in parts.items():
        m.apply_transform(TO_GLTF)
        scene.add_geometry(m, geom_name=name, node_name=name)
    # props: own nodes so they can spin about their local +Y (glTF up)
    for tag, sx, sy in (("FL", 1, 1), ("FR", 1, -1), ("RL", -1, 1), ("RR", -1, -1)):
        blade = trimesh.util.concatenate([
            trimesh.creation.cylinder(radius=PROP_R, height=0.003, sections=40),           # motion disc
            at(trimesh.creation.box(extents=(2 * PROP_R * 0.95, 0.014, 0.004)), (0, 0, 0.002)),  # blade
            at(trimesh.creation.cylinder(radius=0.008, height=0.01, sections=12), (0, 0, 0.004)),
        ])
        blade.apply_transform(TO_GLTF)
        paint(blade, prop_m)
        pos = (TO_GLTF @ np.array([sx * ARM, sy * ARM, 0.032, 1.0]))[:3]
        scene.add_geometry(blade, geom_name=f"prop_{tag}", node_name=f"prop_{tag}",
                           transform=T.translation_matrix(pos))
    return scene


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("web/src/scene3d/dressing/drone.glb"))
    a = ap.parse_args()
    scene = build()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(scene.export(file_type="glb"))
    lo, hi = scene.bounds
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e3:.0f} kB, {sum(len(g.faces) for g in scene.geometry.values())} "
          f"triangles); bounds (glTF, +Y up, +X forward) x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}] "
          f"z[{lo[2]:.3f},{hi[2]:.3f}]")


if __name__ == "__main__":
    main()
