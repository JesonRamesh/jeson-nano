#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "trimesh", "pillow", "opencv-python-headless"]
# ///
"""Generate a synthetic, textured, Y-up GLB that looks like a phone LiDAR scan.

Used to test tools/scan_to_grid.py and tools/propose_graph.py before a real
Scaniverse scan exists. It mimics the things a real scan does:

* glTF Y-up, origin = where the phone started (at the entrance, ~1.4 m above
  the floor), arbitrary heading (yaw) and small vertex noise;
* several meshes/materials with embedded textures;
* a floor that stops at the open entrance door (outside is not scanned);
* partial ceiling, a table (upward faces at 0.75 m) and a tall shelf.

Layout (design frame, metres, z up, before yaw/offset):
    corridor 24 x 1.8 m, entrance at its west end; rooms A and B north of it
    (B has two doors -> loop, A-B internal door -> second loop); T-branch south
    to a dead end with side room C; big room D at the east end.

Usage:
    python tools/scan_make_synthetic.py maps/synthetic/raw/scan_raw.glb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

RES = 0.05          # design raster resolution (m)
WALL_T = 0.15       # wall thickness (m)
WALL_H = 2.6        # wall height (m)

# Free-space rectangles (x0, y0, x1, y1) in the design frame.
ROOMS = [
    (0.0, 0.0, 24.0, 1.8),      # main corridor (entrance at x=0)
    (2.0, 1.95, 7.0, 6.8),      # room A
    (7.15, 1.95, 13.0, 6.8),    # room B
    (15.0, -8.0, 16.8, -0.15),  # south branch corridor (dead end at y=-8)
    (16.95, -6.0, 21.0, -2.0),  # side room C
    (24.0, -2.0, 30.0, 4.0),    # room D (open end of corridor)
]
DOORS = [  # openings cut through walls
    (3.5, 1.7, 4.4, 2.05),      # corridor -> A
    (8.0, 1.7, 8.9, 2.05),      # corridor -> B (west)
    (11.5, 1.7, 12.4, 2.05),    # corridor -> B (east)  => loop
    (6.9, 5.0, 7.25, 5.9),      # A -> B                => second loop
    (15.0, -0.25, 16.8, 0.1),   # corridor -> south branch (full width)
    (16.7, -4.5, 17.05, -3.6),  # south branch -> C
    (-0.4, 0.0, 0.1, 1.8),      # entrance (open to the unscanned outside)
]
TABLE = (9.0, 4.0, 10.2, 4.8, 0.75)   # x0, y0, x1, y1, height  (room B)
SHELF = (27.0, 3.3, 29.0, 3.8, 1.9)   # tall shelf in room D -> obstacle at 1.2 m


def _texture(kind: str, size: int = 256) -> Image.Image:
    rng = np.random.default_rng(1 if kind == "wall" else 2)
    img = np.zeros((size, size, 3), np.uint8)
    if kind == "wall":   # pale plaster with faint brick lines
        img[:] = (205, 200, 190)
        img = (img + rng.integers(-12, 12, img.shape)).clip(0, 255).astype(np.uint8)
        for y in range(0, size, 32):
            img[y:y + 2] = (170, 160, 150)
    else:                # grey floor tiles
        img[:] = (120, 125, 130)
        img = (img + rng.integers(-15, 15, img.shape)).clip(0, 255).astype(np.uint8)
        img[::64] = img[:, ::64] = (80, 80, 85)
    return Image.fromarray(img)


def _quad(p0, p1, p2, p3, normal) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([p0, p1, p2, p3], float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    if np.dot(np.cross(v[1] - v[0], v[2] - v[0]), normal) < 0:
        f = f[:, ::-1]
    return v, f


def _merge(quads) -> trimesh.Trimesh:
    vs, fs, off = [], [], 0
    for v, f in quads:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return trimesh.Trimesh(np.vstack(vs), np.vstack(fs), process=False)


def _runs(row: np.ndarray):
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))


def _wall_surface(mask: np.ndarray, x0: float, y0: float, z0: float, z1: float) -> trimesh.Trimesh:
    """Vertical quads on the boundary of a wall raster (a surface mesh, like a scan)."""
    m = np.pad(mask.astype(np.int8), 1)
    quads = []
    for r in range(1, m.shape[0]):            # y-facing faces at y = y0 + (r-1)*RES
        d = m[r, 1:-1] - m[r - 1, 1:-1]
        y = y0 + (r - 1) * RES
        for s in (1, -1):
            for a, b in _runs(d == s):
                xa, xb = x0 + a * RES, x0 + b * RES
                quads.append(_quad((xa, y, z0), (xb, y, z0), (xb, y, z1), (xa, y, z1), (0, -s, 0)))
    for c in range(1, m.shape[1]):            # x-facing faces at x = x0 + (c-1)*RES
        d = m[1:-1, c] - m[1:-1, c - 1]
        x = x0 + (c - 1) * RES
        for s in (1, -1):
            for a, b in _runs(d == s):
                ya, yb = y0 + a * RES, y0 + b * RES
                quads.append(_quad((x, ya, z0), (x, yb, z0), (x, yb, z1), (x, ya, z1), (-s, 0, 0)))
    return _merge(quads)


def _flat_surface(mask: np.ndarray, x0: float, y0: float, z: float, up: bool) -> trimesh.Trimesh:
    """Horizontal quads (row runs) facing up (floor) or down (ceiling)."""
    quads = []
    for r in range(mask.shape[0]):
        ya, yb = y0 + r * RES, y0 + (r + 1) * RES
        for a, b in _runs(mask[r]):
            xa, xb = x0 + a * RES, x0 + b * RES
            quads.append(_quad((xa, ya, z), (xb, ya, z), (xb, yb, z), (xa, yb, z), (0, 0, 1 if up else -1)))
    return _merge(quads)


def _box(x0, y0, x1, y1, z0, z1) -> trimesh.Trimesh:
    return trimesh.creation.box(
        extents=[x1 - x0, y1 - y0, z1 - z0],
        transform=trimesh.transformations.translation_matrix([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]))


def _textured(mesh: trimesh.Trimesh, tex: Image.Image, name: str) -> trimesh.Trimesh:
    # Split vertices per face so UVs can be planar per face orientation.
    mesh = mesh.copy()
    mesh.unmerge_vertices()
    v = mesh.vertices
    n = np.zeros_like(v)
    n[mesh.faces.ravel()] = np.repeat(mesh.face_normals, 3, axis=0)
    horizontal = np.abs(n[:, 2]) > 0.5
    uv = np.where(horizontal[:, None], v[:, :2], np.c_[v[:, 0] + v[:, 1], v[:, 2]]) / 2.0
    mat = trimesh.visual.material.PBRMaterial(name=name, baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=0.9)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    return mesh


def build(yaw_deg: float, offset_xy: tuple[float, float], phone_h: float, noise: float, seed: int) -> trimesh.Scene:
    # ---- design-frame occupancy raster -------------------------------------
    gx0, gy0, gx1, gy1 = -1.0, -9.0, 31.0, 8.0
    w, h = int((gx1 - gx0) / RES), int((gy1 - gy0) / RES)
    free = np.zeros((h, w), np.uint8)

    def fill(m, rect):
        x0, y0, x1, y1 = rect
        c0, c1 = int(round((x0 - gx0) / RES)), int(round((x1 - gx0) / RES))
        r0, r1 = int(round((y0 - gy0) / RES)), int(round((y1 - gy0) / RES))
        m[r0:r1, c0:c1] = 1

    for r in ROOMS:
        fill(free, r)
    t = int(round(WALL_T / RES))
    walls = cv2.dilate(free, np.ones((2 * t + 1, 2 * t + 1), np.uint8)) & (1 - free)
    doors = np.zeros_like(free)
    for d in DOORS:
        fill(doors, d)
    walls &= 1 - doors
    floor = (free | doors) & 1
    floor[:, : int(round((0.0 - gx0) / RES))] = 0  # nothing scanned outside the entrance

    wall_mesh = _wall_surface(walls.astype(bool), gx0, gy0, 0.0, WALL_H)
    floor_mesh = _flat_surface(floor.astype(bool), gx0, gy0, 0.0, up=True)
    ceil = (free & 1).astype(bool)
    ceil[:, int(round((14.0 - gx0) / RES)):] = False  # ceiling only partly scanned
    ceil_mesh = _flat_surface(ceil, gx0, gy0, WALL_H, up=False)
    furn = trimesh.util.concatenate([
        _box(*TABLE[:4], TABLE[4] - 0.04, TABLE[4]),
        *[_box(x, y, x + 0.05, y + 0.05, 0, TABLE[4] - 0.04) for x in (TABLE[0], TABLE[2] - 0.05) for y in (TABLE[1], TABLE[3] - 0.05)],
        _box(*SHELF[:4], 0.0, SHELF[4]),
    ])

    wall_tex, floor_tex = _texture("wall"), _texture("floor")
    parts = {
        "walls": _textured(wall_mesh, wall_tex, "wall"),
        "ceiling": _textured(ceil_mesh, wall_tex, "ceiling"),
        "floor": _textured(floor_mesh, floor_tex, "floor"),
        "furniture": _textured(furn, floor_tex, "furniture"),
    }

    # ---- design frame (z up) -> "scan" raw frame (glTF Y-up) ---------------
    # origin at the entrance (phone start: x=0.9, y=0.9, z=phone_h), heading = yaw
    start = np.array([0.9, 0.9, phone_h])
    T = trimesh.transformations
    design_to_zup = (T.translation_matrix([*offset_xy, 0.0]) @ T.rotation_matrix(np.radians(yaw_deg), [0, 0, 1])
                     @ T.translation_matrix(-start))
    zup_to_yup = T.rotation_matrix(np.radians(-90), [1, 0, 0])   # (x,y,z) -> (x, z, -y)
    rng = np.random.default_rng(seed)
    scene = trimesh.Scene()
    for name, m in parts.items():
        m.apply_transform(zup_to_yup @ design_to_zup)
        m.vertices = m.vertices + rng.normal(0, noise, m.vertices.shape)
        scene.add_geometry(m, geom_name=name, node_name=name)
    return scene


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--yaw", type=float, default=17.0, help="heading of the building in the scan frame (deg)")
    ap.add_argument("--offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"))
    ap.add_argument("--phone-height", type=float, default=1.4, help="scan origin height above the floor (m)")
    ap.add_argument("--noise", type=float, default=0.01, help="vertex noise sigma (m)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    scene = build(a.yaw, tuple(a.offset), a.phone_height, a.noise, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(scene.export(file_type="glb"))
    tris = sum(len(g.faces) for g in scene.geometry.values())
    print(f"wrote {a.out}  ({a.out.stat().st_size / 1e6:.2f} MB, {tris} triangles, {len(scene.geometry)} meshes)")
    print("design-frame entrance = scan origin (0, 0); floor is at raw Y = -phone_height")


if __name__ == "__main__":
    main()
