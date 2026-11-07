#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "trimesh", "pillow"]
# ///
"""Textured scan for the MuJoCo drone cameras (spec §6.2 step 9, strong tier).

    python tools/scan_to_mujoco.py maps/<name>          # -> scan_vis.obj + scan_vis.png

Input : the ORIGINAL scan (maps/<name>/raw/scan_raw.glb, else scan.glb if it is the
        uncompressed original, else sim.glb) + transform.json. Default = exact replica:
        every triangle and the full-resolution texture.
Output: one OBJ (v, vt, f a/a b/b c/c) already in the sim frame (z up, floor z = 0, grid
        origin = (0, 0)) and its base-colour texture as PNG. If the GLB has several
        textured meshes, they're written as scan_vis_0.obj/.png, scan_vis_1... instead.
UVs are written as trimesh loads them (v already flipped for OBJ/OpenGL), so the OBJ and
PNG pair renders the same as the GLB. P loads them into MuJoCo (visual-only geom,
contype=0 conaffinity=0, group 1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def write_obj(path: Path, v: np.ndarray, uv: np.ndarray, f: np.ndarray, mtl_png: str | None = None) -> None:
    with open(path, "w") as fh:
        fh.write(f"# {path.name}: sim frame (z up, floor z=0, grid origin), {len(f)} faces\n")
        if mtl_png:
            fh.write(f"# texture: {mtl_png}\n")
        np.savetxt(fh, v, fmt="v %.4f %.4f %.4f")
        np.savetxt(fh, uv, fmt="vt %.6f %.6f")
        fi = f + 1
        np.savetxt(fh, np.repeat(fi, 2, axis=1), fmt="f %d/%d %d/%d %d/%d")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir", type=Path)
    ap.add_argument("--glb", type=Path, help="default: raw/scan_raw.glb, then scan.glb, then sim.glb")
    ap.add_argument("--max-z", type=float, default=None,
                    help="optionally drop faces entirely above this height (m), e.g. to remove the ceiling")
    a = ap.parse_args()
    md = a.map_dir
    glb = a.glb
    if glb is None:
        for cand in (md / "raw" / "scan_raw.glb", md / "scan.glb", md / "sim.glb"):
            if cand.exists() and b"EXT_meshopt_compression" not in cand.read_bytes()[:200000]:
                glb = cand
                break
    if glb is None:
        raise SystemExit("no uncompressed GLB found (trimesh can't read meshopt); pass --glb")
    print(f"source: {glb}")
    M = np.array(json.loads((md / "transform.json").read_text())["M"], float)
    scene = trimesh.load(glb, force="scene", process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        vis = g.visual
        tex = getattr(getattr(vis, "material", None), "baseColorTexture", None) or \
            getattr(getattr(vis, "material", None), "image", None)
        if not isinstance(vis, trimesh.visual.TextureVisuals) or vis.uv is None or tex is None:
            print(f"  skip {gname}: no UV texture")
            continue
        m = g.copy()
        m.apply_transform(M @ T)                    # raw GLB -> sim frame
        f = m.faces
        if a.max_z is not None:
            f = f[m.vertices[f][:, :, 2].min(axis=1) <= a.max_z]
        parts.append((gname, m.vertices, np.asarray(vis.uv, float), f, tex))
    if not parts:
        raise SystemExit(f"no textured meshes in {glb}")
    single = len(parts) == 1
    for k, (gname, v, uv, f, tex) in enumerate(parts):
        stem = "scan_vis" if single else f"scan_vis_{k}"
        png = md / f"{stem}.png"
        tex.convert("RGB").save(png)
        write_obj(md / f"{stem}.obj", v, uv, f, png.name)
        lo, hi = v[np.unique(f)].min(0), v[np.unique(f)].max(0)
        print(f"wrote {stem}.obj ({len(f)} faces, {(md / f'{stem}.obj').stat().st_size / 1e6:.1f} MB) + {png.name} "
              f"{tex.size[0]}x{tex.size[1]}; bounds x[{lo[0]:.1f},{hi[0]:.1f}] y[{lo[1]:.1f},{hi[1]:.1f}] "
              f"z[{lo[2]:.2f},{hi[2]:.2f}]")


if __name__ == "__main__":
    main()
