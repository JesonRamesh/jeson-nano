#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "trimesh"]
# ///
"""Generate low-poly person figures for the 3D view -> web/src/scene3d/dressing/{firefighter,survivor}.glb

    python tools/scan_make_people.py

Procedural, so it's our own work: CC0 (see dressing/README.md). Cheap on purpose: smooth
capsules/ellipsoids at adult proportions (1.78 m), no textures, no skeleton; parts sharing a
colour are merged, so each figure is a handful of draw calls and a few thousand triangles.

Conventions (glTF standard, same as drone.glb): +Y up, +X forward (the way the person faces),
origin at the floor.
  * firefighter.glb: standing, origin between the feet. Navy turnout gear with hi-vis
    reflective bands, yellow helmet, dark face mask, breathing-air cylinder on the back.
  * survivor.glb: lying on the back along X (head toward +X), origin under the body centre.
    Hi-vis yellow-green jacket (matches the drone cameras' survivor colour), jeans.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

T = trimesh.transformations
TO_GLTF = T.rotation_matrix(-math.pi / 2, [1, 0, 0])  # design (x fwd, y left, z up) -> glTF y-up


def ellipsoid(c, r, sub=2):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=1.0)
    m.apply_scale(r)
    m.apply_translation(c)
    return m


def capsule(a, b, r, sections=14):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    m = trimesh.creation.capsule(height=max(L, 1e-3), radius=r, count=[sections, sections // 2 + 2])
    # trimesh capsule runs along z, centred: rotate z onto (b - a), move to the midpoint
    z = np.array([0, 0, 1.0])
    d = (b - a) / max(L, 1e-9)
    axis = np.cross(z, d)
    if np.linalg.norm(axis) > 1e-9:
        m.apply_transform(T.rotation_matrix(math.acos(np.clip(np.dot(z, d), -1, 1)), axis))
    elif d[2] < 0:
        m.apply_transform(T.rotation_matrix(math.pi, [1, 0, 0]))
    m.apply_translation((a + b) / 2)
    return m


def band(center_a, center_b, r, w=0.035, sections=20):
    """A reflective band: a short cylinder around a limb at its point a->b, 8 mm proud of the
    surface (same radius would z-fight)."""
    a, b = np.asarray(center_a, float), np.asarray(center_b, float)
    d = (b - a) / np.linalg.norm(b - a)
    m = trimesh.creation.cylinder(radius=r + 0.008, height=w, sections=sections)
    z = np.array([0, 0, 1.0])
    axis = np.cross(z, d)
    if np.linalg.norm(axis) > 1e-9:
        m.apply_transform(T.rotation_matrix(math.acos(np.clip(np.dot(z, d), -1, 1)), axis))
    m.apply_translation(a)
    return m


def boot(x0, x1, y, h, wdt):
    b = trimesh.creation.box(extents=(x1 - x0, wdt, h))
    b.apply_translation(((x0 + x1) / 2, y, h / 2))
    toe = ellipsoid((x1 - 0.02, y, h * 0.45), (0.05, wdt / 2, h * 0.45), sub=1)
    return [b, toe]


def body(parts, pose="stand"):
    """Shared adult body (1.78 m), parts dict: material -> list of meshes. Returns joint dict."""
    J = {}
    for s in (1, -1):
        hip, knee, ankle = (0, s * 0.1, 0.92), (0.02, s * 0.1, 0.5), (0, s * 0.1, 0.09)
        sh = (0, s * 0.215, 1.43)
        if pose == "stand":
            el, wr = (0.01, s * 0.255, 1.15), (0.1, s * 0.24, 0.93)
        else:  # lying: arms a little out from the body
            el, wr = (0.0, s * 0.33, 1.17), (0.03, s * 0.42, 0.93)
        J[s] = dict(hip=hip, knee=knee, ankle=ankle, sh=sh, el=el, wr=wr)
        parts["legs"] += [capsule(hip, knee, 0.085), capsule(knee, ankle, 0.066)]
        parts["arms"] += [capsule(sh, el, 0.056), capsule(el, wr, 0.046)]
        parts["hands"].append(ellipsoid((wr[0] + 0.02, wr[1], wr[2] - 0.06), (0.045, 0.03, 0.07), sub=1))
        parts["feet"] += boot(-0.07, 0.19, s * 0.1, 0.11, 0.1)
    parts["torso"] += [ellipsoid((0, 0, 1.25), (0.135, 0.205, 0.27)),          # chest
                       ellipsoid((0, 0, 1.0), (0.125, 0.18, 0.12))]             # pelvis/waist
    parts["legs"].append(ellipsoid((0, 0, 0.93), (0.12, 0.17, 0.09), sub=1))    # crotch
    parts["skin"] += [capsule((0, 0, 1.47), (0, 0, 1.56), 0.05),                  # neck
                      ellipsoid((0.005, 0, 1.665), (0.1, 0.085, 0.118)),           # head
                      ellipsoid((0.1, 0, 1.655), (0.02, 0.015, 0.025), sub=1)]     # nose
    return J


def scene_from(parts, mats, M=np.eye(4)) -> trimesh.Scene:
    """One mesh per MATERIAL (parts sharing a colour are merged: fewer draw calls)."""
    by_mat: dict[str, list] = defaultdict(list)
    for name, meshes in parts.items():
        by_mat[mats[name].name] += meshes
    first = {m.name: m for m in mats.values()}
    sc = trimesh.Scene()
    for mname, meshes in by_mat.items():
        if not meshes:
            continue
        m = trimesh.util.concatenate(meshes)
        m.apply_transform(TO_GLTF @ M)
        m.visual = trimesh.visual.TextureVisuals(material=first[mname])
        sc.add_geometry(m, geom_name=mname, node_name=mname)
    return sc


def mat(name, rgb, rough=0.8, metal=0.0, emissive=None):
    kw = {"emissiveFactor": emissive} if emissive else {}
    return PBRMaterial(name=name, baseColorFactor=[*rgb, 1.0], roughnessFactor=rough, metallicFactor=metal, **kw)


def firefighter() -> trimesh.Scene:
    parts = defaultdict(list)
    J = body(parts, "stand")
    # turnout coat: bulkier torso + collar, coat hem over the hips
    parts["torso"] += [ellipsoid((0, 0, 1.12), (0.15, 0.22, 0.2)), capsule((0, -0.1, 1.5), (0, 0.1, 1.5), 0.06)]
    # helmet: dome + brim (wider at the back) + visor/mask over the face
    parts["helmet"] += [ellipsoid((0, 0, 1.71), (0.135, 0.125, 0.105)),
                        ellipsoid((-0.01, 0, 1.66), (0.165, 0.15, 0.018), sub=2)]
    parts["mask"] += [ellipsoid((0.075, 0, 1.635), (0.045, 0.08, 0.075), sub=2),
                      capsule((0.1, 0, 1.58), (0.13, 0, 1.55), 0.025)]  # regulator
    # breathing-air cylinder + backplate
    parts["tank"] += [capsule((-0.22, 0, 1.0), (-0.22, 0, 1.45), 0.075)]
    parts["gear_dark"] += [trimesh.creation.box(extents=(0.03, 0.28, 0.5)).apply_translation((-0.15, 0, 1.22)),
                           capsule((-0.12, 0.13, 1.42), (0.1, 0.13, 1.1), 0.018),     # shoulder straps
                           capsule((-0.12, -0.13, 1.42), (0.1, -0.13, 1.1), 0.018)]
    # reflective bands: arms, wrists, legs, chest, hem
    for s in (1, -1):
        j = J[s]
        parts["reflective"] += [band(np.add(j["sh"], np.subtract(j["el"], j["sh"]) * 0.45), j["el"], 0.062),
                                band(np.add(j["el"], np.subtract(j["wr"], j["el"]) * 0.75), j["wr"], 0.052),
                                band(np.add(j["knee"], np.subtract(j["ankle"], j["knee"]) * 0.55), j["ankle"], 0.072),
                                band(np.add(j["hip"], np.subtract(j["knee"], j["hip"]) * 0.75), j["knee"], 0.088)]
    parts["reflective"] += [band((0, 0, 1.3), (0, 0, 1.6), 0.208, w=0.045, sections=28),
                            band((0, 0, 0.98), (0, 0, 1.3), 0.19, w=0.045, sections=28)]
    parts["reflective"] = [m.apply_scale((0.7, 1, 1)) if m.bounds[1][1] > 0.15 else m for m in parts["reflective"]]
    mats = {
        "legs": mat("turnout", (0.11, 0.13, 0.18)), "arms": mat("turnout", (0.11, 0.13, 0.18)),
        "torso": mat("turnout", (0.11, 0.13, 0.18)), "gear_dark": mat("gear", (0.05, 0.05, 0.06)),
        "hands": mat("gloves", (0.06, 0.06, 0.06)), "feet": mat("boots", (0.04, 0.04, 0.04), rough=0.5),
        "skin": mat("skin", (0.78, 0.58, 0.46)), "helmet": mat("helmet", (0.95, 0.76, 0.1), rough=0.35),
        "mask": mat("mask", (0.03, 0.03, 0.04), rough=0.2), "tank": mat("tank", (0.78, 0.8, 0.83), rough=0.45),
        "reflective": mat("reflective", (0.82, 0.95, 0.25), rough=0.3, emissive=(0.25, 0.3, 0.06)),
    }
    return scene_from(parts, mats)


def survivor() -> trimesh.Scene:
    parts = defaultdict(list)
    body(parts, "lie")
    # jacket: a little bulkier over the chest; hair cap
    parts["torso"].append(ellipsoid((0.0, 0, 1.18), (0.14, 0.215, 0.2)))
    parts["hair"].append(ellipsoid((-0.015, 0, 1.69), (0.098, 0.088, 0.1)))
    mats = {
        "torso": mat("jacket", (0.75, 1.0, 0.05), rough=0.6), "arms": mat("jacket", (0.75, 1.0, 0.05), rough=0.6),
        "legs": mat("jeans", (0.17, 0.24, 0.38)), "feet": mat("shoes", (0.08, 0.07, 0.07)),
        "hands": mat("skin", (0.78, 0.58, 0.46)), "skin": mat("skin", (0.78, 0.58, 0.46)),
        "hair": mat("hair", (0.12, 0.08, 0.06)),
    }
    # lie on the back: rotate so +z (head) -> +x and +x (front) -> +z, centre, rest on the floor
    R = T.rotation_matrix(math.pi / 2, [0, 1, 0])  # x' = z, z' = -x  (front would face down)
    R = T.rotation_matrix(math.pi, [1, 0, 0]) @ R  # flip so the front faces up, head stays +x
    M = T.translation_matrix([-0.88, 0, 0.14]) @ R
    return scene_from(parts, mats, M)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("web/src/scene3d/dressing"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in (("firefighter", firefighter), ("survivor", survivor)):
        sc = fn()
        p = a.out_dir / f"{name}.glb"
        p.write_bytes(sc.export(file_type="glb"))
        lo, hi = sc.bounds
        print(f"wrote {p} ({p.stat().st_size / 1e3:.0f} kB, {sum(len(g.faces) for g in sc.geometry.values())} "
              f"triangles, {len(sc.geometry)} meshes); glTF bounds x[{lo[0]:.2f},{hi[0]:.2f}] "
              f"y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}]")


if __name__ == "__main__":
    main()
