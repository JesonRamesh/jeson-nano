#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "trimesh", "opencv-python-headless", "pillow"]
# ///
"""Textured top-down (plan) view of the scan, in the same world frame as grid.png.

    python tools/scan_topdown.py maps/<name>            # -> maps/<name>/topdown.jpg + topdown.json

Uses transform.json (run scan_to_grid.py first), so pixel (0, H-1) is world (0, 0)
exactly like grid.png, just at a finer resolution (--res, default 0.02 m/px = 5x the grid).
Everything above --cut (default 1.6 m) is dropped so the ceiling doesn't hide the floor;
wall faces are drawn darker so the walls read as lines. Source: raw/scan_raw.glb if it
exists (full-res textures), else sim.glb.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh


def load_colored_triangles(path: Path, M: np.ndarray, max_edge: float = 0.06):
    """World-frame triangles (n,3,3) and per-triangle RGB colours (n,3) uint8."""
    scene = trimesh.load(path, force="scene", process=False)
    tris, cols = [], []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        v = trimesh.transform_points(g.vertices, M @ T)
        faces = g.faces
        vis = g.visual
        vc = None
        try:
            if isinstance(vis, trimesh.visual.TextureVisuals) and vis.uv is not None:
                # subdivide big triangles (xyz + uv together) so one colour per triangle
                # still shows the texture; uv rides along at 1e-6 scale so it doesn't
                # affect edge lengths
                uv = np.asarray(vis.uv, float)
                v5, faces = trimesh.remesh.subdivide_to_size(np.c_[v, uv * 1e-6], faces, max_edge=max_edge,
                                                              max_iter=12)
                v, uv = v5[:, :3], v5[:, 3:] * 1e6
                vc = trimesh.visual.color.uv_to_color(uv, vis.material.baseColorTexture
                                                      if hasattr(vis.material, "baseColorTexture")
                                                      else vis.material.image)[:, :3]
            elif hasattr(vis, "vertex_colors"):
                vc = np.asarray(vis.vertex_colors)[:, :3]
        except Exception as e:  # texture missing/unreadable -> grey
            print(f"  {gname}: no usable texture ({e}); grey")
        if vc is None or len(vc) != len(v):
            vc = np.full((len(v), 3), 160, np.uint8)
        tris.append(v[faces])
        cols.append(vc[faces].mean(axis=1).astype(np.uint8))
    return np.concatenate(tris), np.concatenate(cols)


def render(tris, cols, size_wh, res, cut, wall_shade=0.35):
    W, H = size_wh
    z = tris[:, :, 2]
    keep = z.max(axis=1) <= cut
    tris, cols = tris[keep], cols[keep]
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    nz = np.abs(n[:, 2]) / (np.linalg.norm(n, axis=1) + 1e-12)
    cols = cols.astype(np.float32)
    vertical = nz < 0.3
    cols[vertical] *= wall_shade                       # walls read as dark lines
    order = np.argsort(tris[:, :, 2].mean(axis=1))     # painter's algorithm: low first
    tris, cols, vertical = tris[order], cols[order], vertical[order]
    px = np.empty(tris.shape[:2] + (2,), np.int32)
    px[..., 0] = np.floor(tris[..., 0] / res)
    px[..., 1] = H - 1 - np.floor(tris[..., 1] / res)
    img = np.zeros((H, W, 3), np.uint8)
    cover = np.zeros((H, W), np.uint8)
    bgr = cols[:, ::-1].clip(0, 255).astype(np.uint8)
    for p, c, vert in zip(px, bgr, vertical):
        col = tuple(int(x) for x in c)
        if vert:
            cv2.polylines(img, [p], True, col, 1)
            cv2.polylines(cover, [p], True, 1, 1)
        else:
            cv2.fillConvexPoly(img, p, col)
            cv2.fillConvexPoly(cover, p, 1)
    # fill the pinholes between tiny triangles
    holes = cv2.morphologyEx(cover, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) & (1 - cover)
    if holes.any():
        img = cv2.inpaint(img, holes, 2, cv2.INPAINT_TELEA)
    return img, cover | holes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir", type=Path)
    ap.add_argument("--glb", type=Path, help="override the source GLB")
    ap.add_argument("--res", type=float, default=0.02, help="m per pixel (default 0.02)")
    ap.add_argument("--cut", type=float, default=1.6, help="drop geometry above this height (m)")
    ap.add_argument("--max-edge", type=float, default=0.06, help="subdivide triangles to this size before colouring (m)")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality")
    a = ap.parse_args()

    tf = json.loads((a.map_dir / "transform.json").read_text())
    M = np.array(tf["M"], float)
    gw, gh = tf["grid_size"]
    pitch = tf["pitch"]
    W, H = int(round(gw * pitch / a.res)), int(round(gh * pitch / a.res))
    src = a.glb or next((p for p in (a.map_dir / "raw" / "scan_raw.glb", a.map_dir / "sim.glb") if p.exists()), None)
    if src is None:
        raise SystemExit("no raw/scan_raw.glb or sim.glb in the map dir")
    t0 = time.time()
    tris, cols = load_colored_triangles(src, M, a.max_edge)
    print(f"loaded {src} ({len(tris)} triangles) in {time.time() - t0:.1f}s")
    t0 = time.time()
    img, cover = render(tris, cols, (W, H), a.res, a.cut)
    print(f"rendered {W}x{H} px at {a.res} m/px (cut at {a.cut} m) in {time.time() - t0:.1f}s")
    out = a.map_dir / "topdown.jpg"
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
    meta = {"image": "topdown.jpg", "res": a.res, "origin": [0, 0], "size": [W, H], "cut": a.cut,
            "source": src.name, "note": "bottom-left pixel = world (0,0), same frame as grid.png"}
    (a.map_dir / "topdown.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), {a.map_dir / 'topdown.json'}")


if __name__ == "__main__":
    main()
