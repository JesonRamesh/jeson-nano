#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python-headless", "scikit-image"]
# ///
"""Close doors: paint doorways shut in grid.png and drop those branches from the graph.

    python tools/scan_close_doors.py maps/<name> n3 n5 n10 [--dry-run]

Each argument is the node BEHIND the door (the room / alcove node at the end of a branch).
For each one:
  * the doorway = the narrowest wall-to-wall cross-section along that node's edge
    (from its junction to the node, ignoring the first/last 5%);
  * a wall line is painted across it in grid.png (--thickness px), extended a little past
    the jambs so no diagonal gap is left;
  * free space no longer reachable from the entrance becomes wall (same rule as scan_to_grid);
  * the branch (the node, and anything only reachable through it) is removed from the graph,
    together with its compromised entries and checkpoints. Other ids are kept unchanged.
Refuses a door wider than --max-door (default 3.5 m): that's an open area, not a doorway.
Writes grid.png, scenario.json, preview.png and route_preview.jpg (if topdown.jpg exists),
and keeps a backup grid.before_doors.png the first time.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir", type=Path)
    ap.add_argument("nodes", nargs="+", help="node ids behind the doors")
    ap.add_argument("--thickness", type=int, default=2, help="wall thickness (px of grid.png)")
    ap.add_argument("--max-door", type=float, default=3.5, help="refuse doorways wider than this (m)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    md = a.map_dir
    sc = json.loads((md / "scenario.json").read_text())
    grid = cv2.imread(str(md / "grid.png"), cv2.IMREAD_GRAYSCALE)
    pitch = float(sc.get("grid", {}).get("resolution", 0.1))
    h, w = grid.shape
    nodes = {n["id"]: n for n in sc["graph"]["nodes"]}
    edges = {e["id"]: e for e in sc["graph"]["edges"]}
    ent = next(n["id"] for n in sc["graph"]["nodes"] if n["kind"] == "entrance")
    pos = {i: np.array((n["x"], n["y"]), float) for i, n in nodes.items()}

    def rc(p):
        return h - 1 - int(p[1] / pitch), int(p[0] / pitch)

    def free(p, g):
        r, c = rc(p)
        return 0 <= r < h and 0 <= c < w and g[r, c] > 127

    def ray(p, d, g):
        t = 0.0
        while free(p + d * t, g) and t < 20:
            t += pitch / 4
        return t

    def reach_nodes(edge_set, start):
        adj = {i: [] for i in nodes}
        for e in edge_set.values():
            adj[e["a"]].append(e["b"]); adj[e["b"]].append(e["a"])
        seen, st = set(), [start]
        while st:
            n = st.pop()
            if n not in seen:
                seen.add(n); st += adj[n]
        return seen

    g = grid.copy()
    drop_nodes, drop_edges = set(), set()
    for leaf in a.nodes:
        if leaf not in nodes:
            raise SystemExit(f"unknown node {leaf}")
        if leaf == ent:
            raise SystemExit(f"{leaf} is the entrance")
        inc = [e for e in edges.values() if leaf in (e["a"], e["b"])]
        # the door edge = the incident edge on the way to the entrance
        others = {k: v for k, v in edges.items()}
        door = None
        for e in inc:
            rest = {k: v for k, v in others.items() if k != e["id"]}
            if leaf not in reach_nodes(rest, ent):      # removing e cuts the leaf side off
                door = e
                break
        if door is None:
            raise SystemExit(f"{leaf} is on a loop: there is no single door edge to close")
        root = door["a"] if door["b"] == leaf else door["b"]
        pa, pb = pos[root], pos[leaf]
        u = (pb - pa) / np.linalg.norm(pb - pa)
        nrm = np.array([-u[1], u[0]])
        best = min((ray(pa + (pb - pa) * t, nrm, grid) + ray(pa + (pb - pa) * t, -nrm, grid), t)
                   for t in np.linspace(0.05, 0.95, 37))
        width, t = best
        if width > a.max_door:
            print(f"  SKIP {leaf}: narrowest opening on {door['id']} is {width:.1f} m wide, that's an open area, "
                  f"not a door (use --max-door to force)")
            continue
        m = pa + (pb - pa) * t
        q1 = m + nrm * (ray(m, nrm, grid) + 2 * pitch)
        q2 = m - nrm * (ray(m, -nrm, grid) + 2 * pitch)
        (r1, c1), (r2, c2) = rc(q1), rc(q2)
        cv2.line(g, (c1, r1), (c2, r2), 0, a.thickness)
        cut = reach_nodes({k: v for k, v in edges.items() if k != door["id"]}, ent)
        gone = set(nodes) - cut
        drop_nodes |= gone
        drop_edges |= {k for k, e in edges.items() if e["a"] in gone or e["b"] in gone}
        print(f"  door {leaf}: {width:.1f} m wide doorway on {door['id']}, {t * np.linalg.norm(pb - pa):.1f} m from "
              f"{root} at ({m[0]:.2f}, {m[1]:.2f}); removes nodes {sorted(gone)}")

    # seal: free cells no longer reachable from the entrance -> wall
    free_m = (g > 127).astype(np.uint8)
    _, lab = cv2.connectedComponents(free_m, connectivity=4)
    er, ec = rc(np.array((sc["entrance"]["x"], sc["entrance"]["y"])))
    sealed = (free_m == 1) & (lab != lab[er, ec])
    g[sealed] = 0
    print(f"  sealed {sealed.sum() * pitch * pitch:.1f} m2 of free space behind the doors")

    keep_nodes = [n for n in sc["graph"]["nodes"] if n["id"] not in drop_nodes]
    keep_edges = [e for e in sc["graph"]["edges"] if e["id"] not in drop_edges]
    # check: every remaining node still on free space and reachable
    bad = [n["id"] for n in keep_nodes if not free(pos[n["id"]], g)]
    if bad:
        raise SystemExit(f"refusing: nodes {bad} would end up on wall cells")
    sc["graph"] = {"nodes": keep_nodes, "edges": keep_edges}
    sc["compromised"] = [c for c in sc.get("compromised", []) if c["edge"] not in drop_edges]
    sc["checkpoints"] = [c for c in sc.get("checkpoints", []) if c.get("node") not in drop_nodes]
    if a.dry_run:
        print("dry run: nothing written")
        return
    backup = md / "grid.before_doors.png"
    if not backup.exists():
        shutil.copy(md / "grid.png", backup)
    cv2.imwrite(str(md / "grid.png"), g)
    (md / "scenario.json").write_text(json.dumps(sc, indent=2) + "\n")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from propose_graph import draw_preview
    draw_preview(md / "preview.png", g, sc["graph"], pitch)
    if (md / "topdown.json").exists():
        try:
            import scan_route_draw as E
            app = E.App(md, fresh=False, size=(800, 600))
            cv2.imwrite(str(md / "route_preview.jpg"), app.render_full(), [cv2.IMWRITE_JPEG_QUALITY, 88])
        except Exception as e:  # GUI-less is fine; render_full doesn't open a window
            print(f"  (route_preview.jpg not refreshed: {e})")
    print(f"wrote {md / 'grid.png'} (backup {backup.name}), scenario.json, preview.png, route_preview.jpg: "
          f"{len(keep_nodes)} nodes, {len(keep_edges)} edges, {len(sc['compromised'])} debris")


if __name__ == "__main__":
    main()
