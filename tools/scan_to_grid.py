#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "trimesh", "opencv-python-headless", "pillow"]
# ///
"""sim.glb (phone LiDAR scan) -> grid.png + transform.json + preview.png   (spec §6.2 step 5)

    python tools/scan_to_grid.py maps/<name>/sim.glb --entrance 0 0

Pipeline:
  1. Load every mesh in the GLB (node transforms applied), rotate +90 deg about X
     (glTF is Y-up; trimesh does not convert it) so the frame is Z-up.
  2. Floor z = area-weighted peak height of upward-facing faces; shift so floor = 0.
  3. Slice at z = 1.2 m; draw the section segments in world XY at 0.1 m/px.
  4. MORPH_CLOSE the walls, then flood-fill free space from the entrance seed
     (4-connected, so 1-px diagonal walls do not leak). Unreached = wall.
     By default the fill is also limited to cells that have scanned floor under
     them (holes in the floor such as furniture footprints are filled first);
     this stops leaks out of open doors into unscanned space. --no-floor-mask to disable.
  5. Write grid.png (0 = wall, 255 = free, bottom-left pixel = world (0,0)),
     transform.json (§6.3) and preview.png, and create/update the grid/scan/entrance
     fields of scenario.json (other fields are kept).

Frames
  raw    : the GLB as exported (Y-up, origin = where the scan started).
  scan   : raw rotated to Z-up, floor at z = 0, same XY origin (x = raw X, y = -raw Z).
           The scan start point is (0, 0) here, so if you start scanning at the
           entrance the default seed --entrance 0 0 is right.
  world  : the sim frame (§3): scan frame [optionally yawed by --align-walls],
           shifted so the grid's bottom-left corner is (0, 0).
  --entrance is given in the scan frame by default; use --seed-frame world to
  pass world coordinates read off a previous preview.png instead.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh

# Preview colours (BGR) -- Tenyks palette
C_WALL = (0x1E, 0x10, 0x0D)        # #0D101E void
C_FREE = (0xF5, 0xF0, 0xF1)        # #F1F0F5 pearl
C_FLOOR_UNREACHED = (0x9B, 0x9B, 0x9B)  # #9B9B9B muted gray
C_SECTION = (0xE3, 0x3B, 0x40)     # #403BE3 violet
C_SEED = (0xFB, 0xFF, 0x78)        # #78FFFB cyan
C_TEXT = (0xFF, 0xFF, 0xFF)

AXES = "XYZ"


# --------------------------------------------------------------------------- loading
def load_mesh(path: Path) -> trimesh.Trimesh:
    """All triangle meshes of the file, node transforms applied, geometry only."""
    scene = trimesh.load(path, force="scene", process=False)
    verts, faces, off = [], [], 0
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        verts.append(trimesh.transform_points(g.vertices, T))
        faces.append(g.faces + off)
        off += len(g.vertices)
    if not verts:
        raise SystemExit(f"no triangle meshes in {path}")
    return trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)


def rot_to_z(axis: int, sign: int) -> np.ndarray:
    """4x4 rotation taking the unit vector sign*e_axis to +Z."""
    T = trimesh.transformations
    if axis == 2:
        return np.eye(4) if sign > 0 else T.rotation_matrix(math.pi, [1, 0, 0])
    if axis == 1:  # +Y -> +Z is +90 deg about X (spec §6.2.5.1)
        return T.rotation_matrix(sign * math.pi / 2, [1, 0, 0])
    return T.rotation_matrix(-sign * math.pi / 2, [0, 1, 0])  # +X -> +Z is -90 deg about Y


def detect_up(mesh: trimesh.Trimesh, forced: str) -> tuple[int, int, dict]:
    """Up = the axis whose facing triangles pile up at ONE height (the floor).

    Walls face many directions at many positions; the floor is a single large
    flat sheet. Score each axis by the largest 5 cm slab of area facing along it.
    """
    n, a, c = mesh.face_normals, mesh.area_faces, mesh.triangles_center
    stats = {}
    for i, ax in enumerate(AXES):
        best = []
        for s in (1, -1):
            sel = s * n[:, i] > 0.9
            if sel.sum() == 0:
                best.append(0.0)
                continue
            v = c[sel, i]
            nb = max(1, int(np.ceil((v.max() - v.min()) / 0.05)))
            best.append(float(np.histogram(v, bins=nb, weights=a[sel])[0].max()))
        stats[ax] = tuple(best)
    if forced != "auto":
        sign = -1 if forced.startswith("-") else 1
        return AXES.index(forced[-1].upper()), sign, stats
    axis = max(range(3), key=lambda i: max(stats[AXES[i]]))
    if axis == 1:
        sign = 1  # glTF mandates +Y up; don't let a ceiling-heavy scan flip it
    else:
        pos, neg = stats[AXES[axis]]
        sign = 1 if pos >= neg else -1  # the floor slab is usually bigger than the ceiling slab
    return axis, sign, stats


def weighted_peak(values: np.ndarray, weights: np.ndarray, bin_m: float = 0.02) -> float:
    lo, hi = values.min(), values.max()
    if hi - lo < bin_m:
        return float(np.average(values, weights=weights))
    hist, edges = np.histogram(values, bins=int(np.ceil((hi - lo) / bin_m)), range=(lo, hi), weights=weights)
    hist = np.convolve(hist, [0.25, 0.5, 0.25], mode="same")
    i = int(hist.argmax())
    sel = (values >= edges[max(i - 1, 0)]) & (values <= edges[min(i + 2, len(edges) - 1)])
    return float(np.average(values[sel], weights=weights[sel]))


def dominant_wall_yaw(mesh: trimesh.Trimesh) -> float:
    """Dominant wall direction modulo 90 deg, in degrees in [-45, 45)."""
    n, a = mesh.face_normals, mesh.area_faces
    wall = np.abs(n[:, 2]) < 0.2
    th = np.arctan2(n[wall, 1], n[wall, 0])
    z = np.sum(a[wall] * np.exp(4j * th))
    return float(np.degrees(np.angle(z) / 4))


# --------------------------------------------------------------------------- raster helpers
def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill regions of 0 not connected to the image border."""
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = mask
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    return (pad | (1 - ff))[1:-1, 1:-1]


class Raster:
    def __init__(self, origin: tuple[float, float], size: tuple[int, int], pitch: float):
        self.ox, self.oy = origin
        self.w, self.h = size
        self.p = pitch

    def px(self, xy: np.ndarray) -> np.ndarray:
        """world XY (m, relative to the grid origin = 0,0) -> integer (col, row)."""
        col = np.floor(xy[..., 0] / self.p)
        row = self.h - 1 - np.floor(xy[..., 1] / self.p)
        return np.stack([col, row], -1).astype(np.int32)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb", type=Path, help="uncompressed sim.glb (trimesh cannot read meshopt)")
    ap.add_argument("--entrance", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"),
                    help="flood-fill seed in metres (default: 0 0 = scan start point)")
    ap.add_argument("--seed-frame", choices=["scan", "world"], default="scan")
    ap.add_argument("--out", type=Path, help="output map dir (default: the GLB's directory)")
    ap.add_argument("--pitch", type=float, default=0.1, help="m per pixel")
    ap.add_argument("--slice", type=float, nargs="+", default=[1.2], metavar="Z",
                    help="slice height(s) above the floor (m); several are unioned")
    ap.add_argument("--up", default="auto", choices=["auto", "y", "z", "x", "-y", "-z", "-x"])
    ap.add_argument("--align-walls", action="store_true",
                    help="yaw the world frame so the dominant wall direction is axis-aligned")
    ap.add_argument("--close", type=int, default=3, help="MORPH_CLOSE kernel on walls (px, odd)")
    ap.add_argument("--wall-dilate", type=int, default=0, help="extra wall dilation (px)")
    ap.add_argument("--margin", type=float, default=1.0, help="border around the scan bbox (m)")
    ap.add_argument("--no-floor-mask", action="store_true", help="don't restrict free space to scanned floor")
    ap.add_argument("--floor-band", type=float, default=0.15, help="|z| tolerance for floor faces (m)")
    ap.add_argument("--min-obstacle", type=float, default=0.04,
                    help="wall specks fully inside free space smaller than this (m^2) become free")
    ap.add_argument("--name", help="scenario name (default: map dir name)")
    a = ap.parse_args()

    out = a.out or a.glb.parent
    out.mkdir(parents=True, exist_ok=True)
    mesh = load_mesh(a.glb)
    print(f"loaded {a.glb}: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles")
    print(f"raw bbox      min {np.round(mesh.bounds[0], 2)}  max {np.round(mesh.bounds[1], 2)}  (raw GLB frame)")

    # 1. up axis
    axis, sign, stats = detect_up(mesh, a.up)
    desc = ", ".join(f"{k} +{p:.0f}/-{q:.0f}" for k, (p, q) in stats.items())
    print(f"up-axis       {'+' if sign > 0 else '-'}{AXES[axis]}  ({'forced' if a.up != 'auto' else 'detected'}; "
          f"largest flat 5 cm slab facing each axis, m2: {desc})")
    if axis != 1 or sign != 1:
        print("  WARNING: glTF should be +Y up. Check the preview; override with --up y if this is wrong.")
    R_up = rot_to_z(axis, sign)
    m = mesh.copy()
    m.apply_transform(R_up)

    # 2. floor
    n, ar = m.face_normals, m.area_faces
    up = n[:, 2] > 0.9
    if not up.any():
        raise SystemExit("no upward-facing faces: wrong up axis? try --up")
    zc = m.triangles_center[:, 2]
    floor_z = weighted_peak(zc[up], ar[up])
    down = (n[:, 2] < -0.9) & (zc > floor_z + 1.5)
    ceil_txt = f"{weighted_peak(zc[down], ar[down]) - floor_z:.2f} m above floor" if down.any() else "not found"
    print(f"floor height  {floor_z:+.3f} m in the up-rotated raw frame "
          f"(raw {AXES[axis]} = {sign * floor_z:+.3f});  ceiling {ceil_txt}")
    if not (-2.2 < floor_z < -0.3):
        print("  note: Scaniverse origin is the phone, so the floor is usually 0.8-1.8 m below it.")

    # yaw
    yaw = dominant_wall_yaw(m)
    print(f"wall yaw      dominant wall direction {yaw:+.1f} deg (mod 90) in the scan frame"
          + ("  -> aligning" if a.align_walls else "  (use --align-walls to square the grid)"))
    T = trimesh.transformations
    R_yaw = T.rotation_matrix(-math.radians(yaw), [0, 0, 1]) if a.align_walls else np.eye(4)
    S = R_yaw @ T.translation_matrix([0, 0, -floor_z]) @ R_up   # raw -> scan(+yaw), floor at 0
    m = mesh.copy()
    m.apply_transform(S)

    # bounding box (only geometry between floor and slice-ish height matters, but keep all)
    lo, hi = m.bounds
    print(f"scan bbox     x [{lo[0]:.2f}, {hi[0]:.2f}]  y [{lo[1]:.2f}, {hi[1]:.2f}]  z [{lo[2]:.2f}, {hi[2]:.2f}]  "
          f"-> {hi[0] - lo[0]:.1f} x {hi[1] - lo[1]:.1f} m footprint, {hi[2] - lo[2]:.1f} m tall")
    ox, oy = lo[0] - a.margin, lo[1] - a.margin
    w = int(math.ceil((hi[0] - lo[0] + 2 * a.margin) / a.pitch))
    h = int(math.ceil((hi[1] - lo[1] + 2 * a.margin) / a.pitch))
    M = T.translation_matrix([-ox, -oy, 0]) @ S                  # raw -> world
    m.apply_transform(T.translation_matrix([-ox, -oy, 0]))
    ras = Raster((0.0, 0.0), (w, h), a.pitch)
    print(f"grid          {w} x {h} px = {w * a.pitch:.1f} x {h * a.pitch:.1f} m at {a.pitch} m/px")

    # 3. slice -> walls
    walls = np.zeros((h, w), np.uint8)
    nseg = 0
    for z in a.slice:
        sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            print(f"  WARNING: slice at {z} m hit nothing")
            continue
        # Section vertices are already in world coordinates (don't use to_2D, it changes the frame).
        # NB: sec.discrete only returns CLOSED loops and silently drops open fragments (most walls
        # of a holey scan), so rasterise every entity instead.
        polys = [ras.px(e.discrete(sec.vertices)[:, :2]) for e in sec.entities]
        cv2.polylines(walls, polys, isClosed=False, color=1, thickness=1, lineType=cv2.LINE_8)
        nseg += sum(len(p) - 1 for p in polys)
    print(f"section       {nseg} segments at z = {a.slice} m")
    section_px = walls.copy()
    if a.close > 1:
        walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, np.ones((a.close, a.close), np.uint8))
    if a.wall_dilate > 0:
        walls = cv2.dilate(walls, np.ones((2 * a.wall_dilate + 1,) * 2, np.uint8))

    # floor mask
    floor = np.ones((h, w), np.uint8)
    if not a.no_floor_mask:
        n = m.face_normals
        fz = m.triangles_center[:, 2]
        sel = (n[:, 2] > 0.7) & (np.abs(fz) < a.floor_band)
        tris = ras.px(m.triangles[sel][:, :, :2])
        floor = np.zeros((h, w), np.uint8)
        cv2.fillPoly(floor, list(tris), 1)
        floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        floor = fill_holes(floor)            # furniture footprints, scan holes
        print(f"floor mask    {floor.sum() * a.pitch ** 2:.1f} m2 of scanned floor ({sel.sum()} faces)")

    # 4. flood fill from the seed (4-connected)
    cand = ((1 - walls) & floor).astype(np.uint8)
    seed = np.array(a.entrance, float)
    if a.seed_frame == "scan":
        seed_w = (R_yaw @ np.r_[seed, 0, 1])[:2] - [ox, oy]
    else:
        seed_w = seed
    c, r = ras.px(seed_w)
    if not (0 <= c < w and 0 <= r < h):
        raise SystemExit(f"entrance {tuple(seed)} ({a.seed_frame}) -> world {np.round(seed_w, 2)} is outside the grid")
    if not cand[r, c]:
        ys, xs = np.nonzero(cand)
        if len(xs) == 0:
            raise SystemExit("no free space at all")
        k = np.argmin((xs - c) ** 2 + (ys - r) ** 2)
        d = math.hypot(xs[k] - c, ys[k] - r) * a.pitch
        if d > 1.0:
            raise SystemExit(f"entrance seed is in a wall and the nearest free cell is {d:.2f} m away; check --entrance")
        print(f"  seed was on a wall/unscanned cell; moved {d:.2f} m to the nearest free cell")
        c, r = int(xs[k]), int(ys[k])
        seed_w = np.array([(c + 0.5) * a.pitch, (h - 1 - r + 0.5) * a.pitch])
    _, lab = cv2.connectedComponents(cand, connectivity=4)
    free = (lab == lab[r, c]).astype(np.uint8)

    # remove tiny enclosed wall specks
    if a.min_obstacle > 0:
        nlab, wl, st, _ = cv2.connectedComponentsWithStats(1 - free, connectivity=8)
        border = set(np.unique(np.r_[wl[0], wl[-1], wl[:, 0], wl[:, -1]]))
        small = [i for i in range(1, nlab) if i not in border and st[i, cv2.CC_STAT_AREA] * a.pitch ** 2 < a.min_obstacle]
        free[np.isin(wl, small)] = 1
        if small:
            print(f"cleanup       removed {len(small)} wall specks < {a.min_obstacle} m2")

    grid = (free * 255).astype(np.uint8)
    area = free.sum() * a.pitch ** 2
    print(f"free space    {area:.1f} m2 reachable from the entrance at world ({seed_w[0]:.2f}, {seed_w[1]:.2f})")
    if free[0].any() or free[-1].any() or free[:, 0].any() or free[:, -1].any():
        print("  WARNING: free space touches the grid border -> the fill leaked out of the building "
              "(glass / open door / scan hole). Paint the gap shut in grid.png or adjust --close.")
    unreached_floor = floor.astype(bool) & ~free.astype(bool) & ~walls.astype(bool)
    if not a.no_floor_mask and unreached_floor.sum() * a.pitch ** 2 > 2.0:
        print(f"  note: {unreached_floor.sum() * a.pitch ** 2:.1f} m2 of scanned floor is not reachable "
              "(grey in preview.png) -- closed doors or furniture blocking a doorway?")

    # 5. outputs
    cv2.imwrite(str(out / "grid.png"), grid)
    transform = {
        "M": [[round(float(v), 9) for v in row] for row in M],
        "pitch": a.pitch,
        "grid_origin": [0, 0],
        "grid_size": [w, h],
        "meta": {
            "source": a.glb.name,
            "up_axis": f"{'+' if sign > 0 else '-'}{AXES[axis]}",
            "floor_raw": round(sign * floor_z, 4),
            "wall_yaw_deg": round(yaw, 2),
            "aligned": bool(a.align_walls),
            "slice_z": a.slice,
            "entrance_world": [round(float(v), 3) for v in seed_w],
        },
    }
    txt = json.dumps(transform, indent=2)
    for row in transform["M"]:  # one matrix row per line, easier to eyeball
        txt = txt.replace(json.dumps(row, indent=2).replace("\n", "\n    "), json.dumps(row), 1)
    (out / "transform.json").write_text(txt + "\n")
    write_preview(out / "preview.png", grid, section_px, unreached_floor, seed_w, a.pitch)
    update_scenario(out / "scenario.json", a.name or out.name, seed_w, a.pitch)
    print(f"wrote         {out / 'grid.png'}, {out / 'transform.json'}, {out / 'preview.png'}, {out / 'scenario.json'}")


def write_preview(path: Path, grid, section_px, unreached, seed_w, pitch: float, scale: int | None = None) -> None:
    h, w = grid.shape
    scale = scale or max(2, min(8, 1600 // max(w, h)))
    img = np.empty((h, w, 3), np.uint8)
    img[:] = C_WALL
    img[unreached] = C_FLOOR_UNREACHED
    img[grid > 0] = C_FREE
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    sec = cv2.resize(section_px, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST).astype(bool)
    img[sec] = (0.5 * img[sec] + 0.5 * np.array(C_SECTION)).astype(np.uint8)
    draw_frame(img, pitch, scale)
    cx, cy = int(seed_w[0] / pitch * scale), int((h - seed_w[1] / pitch) * scale)
    cv2.circle(img, (cx, cy), 2 * scale, C_SEED, -1, cv2.LINE_AA)
    cv2.putText(img, "entrance", (cx + 3 * scale, cy - 2 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_SEED, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def draw_frame(img: np.ndarray, pitch: float, scale: int) -> None:
    """1 m tick marks along the bottom/left edges, 5 m labels, and a 1 m scale bar."""
    H, W = img.shape[:2]
    px_per_m = scale / pitch
    for i in range(int(W / px_per_m) + 1):
        x = int(i * px_per_m)
        L = 10 if i % 5 == 0 else 5
        cv2.line(img, (x, H - 1), (x, H - 1 - L), C_TEXT, 1)
        if i % 5 == 0 and i:
            cv2.putText(img, f"{i}", (x + 2, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT, 1, cv2.LINE_AA)
    for j in range(int(H / px_per_m) + 1):
        y = int(H - 1 - j * px_per_m)
        L = 10 if j % 5 == 0 else 5
        cv2.line(img, (0, y), (L, y), C_TEXT, 1)
        if j % 5 == 0 and j:
            cv2.putText(img, f"{j}", (12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, "world m (x east, y north; origin bottom-left)", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT, 1, cv2.LINE_AA)


def update_scenario(path: Path, name: str, entrance, pitch: float) -> None:
    sc = json.loads(path.read_text()) if path.exists() else {"version": 1, "name": name}
    sc["grid"] = {"file": "grid.png", "resolution": pitch, "origin": [0, 0]}
    sc["scan"] = {"file": "scan.glb", "transform": "transform.json"}
    old = sc.get("entrance", {})
    sc["entrance"] = {"x": round(float(entrance[0]), 2), "y": round(float(entrance[1]), 2), "yaw": old.get("yaw", 0.0)}
    path.write_text(json.dumps(sc, indent=2) + "\n")


if __name__ == "__main__":
    main()
