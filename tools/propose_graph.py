#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python-headless", "scikit-image"]
# ///
"""grid.png -> route graph proposal in scenario.json   (spec §6.2 step 7)

    python tools/propose_graph.py maps/<name>            # writes scenario.json "graph" + preview.png
    python tools/propose_graph.py maps/<name> --stdout   # print {"nodes", "edges"} JSON only

Skeletonise the free space, turn junctions/endpoints into nodes and skeleton
branches into edges, then clean it up so a human can finish it in /editor:

  * spurs (short dead-end twigs, e.g. into room corners) are pruned when shorter
    than max(--min-spur, --spur-k x the corridor half-width at their junction);
  * junctions closer than --merge are merged;
  * curved branches are split into straight segments (Douglas-Peucker, then a
    line-of-sight check on the free grid) -> intermediate "waypoint" nodes, so
    every edge is a straight line that stays in free space;
  * the entrance (scenario.json "entrance", or --entrance) becomes node n1.

Node kinds (sgr/contracts.py NodeKind): entrance; junction (>= 3 edges, or a bend);
room (dead end, settled into the open middle of the room); door (bend/pass-through
narrower than --door-width). "stairs" is never proposed: set it by hand in /editor.
If scenario.json has no survivor / drones yet, placeholders are added so the map
loads (survivor at the dead end farthest from the entrance, drones around the
entrance). Move them in /editor.
Edge "width" = 2 x the 10th-percentile clearance along the edge (m), i.e. its
narrowest typical point (doors). Ids are in breadth-first order from the entrance.
Edge ids are regenerated, so re-check "compromised" / "beats" afterwards
(a warning is printed if they point at ids that no longer exist).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

C_WALL = (0x1E, 0x10, 0x0D)        # #0D101E
C_FREE = (0xF5, 0xF0, 0xF1)        # #F1F0F5
C_EDGE = (0xE3, 0x3B, 0x40)        # #403BE3 Tenyks violet
C_NODE = (0xFF, 0x92, 0xD2)        # #D292FF peach
C_WAYPOINT = (0xF3, 0xCB, 0xCB)    # #CBCBF3 light violet
C_ENTRANCE = (0xFB, 0xFF, 0x78)    # #78FFFB cyan
C_TEXT = (0x1E, 0x10, 0x0D)

NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# --------------------------------------------------------------------------- pixel graph
class G:
    """Undirected multigraph; nodes are (row, col) pixels, edges carry pixel paths."""

    def __init__(self):
        self.nodes: dict[int, tuple[int, int]] = {}
        self.edges: dict[int, dict] = {}   # id -> {"a", "b", "path": [(r, c), ...] from a to b}
        self.protected: set[int] = set()
        self._n = self._e = 0

    def add_node(self, rc) -> int:
        self._n += 1
        self.nodes[self._n] = (int(rc[0]), int(rc[1]))
        return self._n

    def add_edge(self, a, b, path) -> int:
        self._e += 1
        self.edges[self._e] = {"a": a, "b": b, "path": [tuple(map(int, p)) for p in path]}
        return self._e

    def inc(self, n) -> list[int]:
        return [e for e, d in self.edges.items() if d["a"] == n or d["b"] == n]

    def deg(self, n) -> int:
        return sum((d["a"] == n) + (d["b"] == n) for d in self.edges.values())

    def path_from(self, e, n):
        d = self.edges[e]
        return d["path"] if d["a"] == n else d["path"][::-1]

    def other(self, e, n):
        d = self.edges[e]
        return d["b"] if d["a"] == n else d["a"]


def plen(path) -> float:
    p = np.asarray(path, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def build_pixel_graph(skel: np.ndarray) -> G:
    h, w = skel.shape
    sk = skel.astype(np.uint8)
    # Crossing number (0->1 transitions around the 8-ring): 1 = endpoint, 2 = line, >=3 = junction.
    # Plain neighbour counts mislabel staircase pixels of diagonal lines as junctions.
    pad = np.pad(sk, 1)
    ring = [pad[1 + dr: 1 + dr + h, 1 + dc: 1 + dc + w] for dr, dc in
            ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))]
    cn = sum(((ring[i] == 0) & (ring[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))
    node_px = (sk == 1) & (cn != 2)
    ncl, lab = cv2.connectedComponents(node_px.astype(np.uint8), connectivity=8)
    g = G()
    cl2node = {}
    for i in range(1, ncl):
        rr, cc = np.nonzero(lab == i)
        k = np.argmin((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2)   # a real pixel near the centroid
        cl2node[i] = g.add_node((rr[k], cc[k]))

    def nbrs(r, c):
        for dr, dc in NB8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and sk[rr, cc]:
                yield rr, cc

    visited = np.zeros_like(sk, bool)
    seen_direct = set()

    def walk(start_lab, p, q):
        path = [p, q]
        visited[q] = True
        prev, cur = p, q
        while True:
            cand = [n for n in nbrs(*cur) if n != prev and n not in path[-3:]]
            # other pixels of the start cluster touch the first steps: not a loop back
            hit = [n for n in cand if lab[n] and not (lab[n] == start_lab and len(path) < 4)]
            if hit:
                end = hit[0]
                path.append(end)
                return cl2node[lab[end]], path
            nxt = [n for n in cand if not visited[n]]
            if not nxt:          # isolated stub without endpoint (shouldn't happen) -> make a node
                return g.add_node(cur), path
            nxt.sort(key=lambda n: abs(n[0] - cur[0]) + abs(n[1] - cur[1]))   # prefer 4-neighbours
            prev, cur = cur, nxt[0]
            visited[cur] = True
            path.append(cur)

    rr, cc = np.nonzero(node_px)
    for p in zip(rr, cc):
        a = cl2node[lab[p]]
        for q in nbrs(*p):
            if lab[q]:
                b = cl2node[lab[q]]
                if b != a and (min(a, b), max(a, b)) not in seen_direct:
                    seen_direct.add((min(a, b), max(a, b)))
                    g.add_edge(a, b, [p, q])
            elif not visited[q]:
                b, path = walk(lab[p], p, q)
                g.add_edge(a, b, path)
    # pure loops with no junction/endpoint (e.g. around a pillar): one node per such component
    ncomp, comp = cv2.connectedComponents(sk, connectivity=8)
    has_node = set(np.unique(comp[node_px]))
    for k in range(1, ncomp):
        if k in has_node:
            continue
        p = tuple(map(int, np.argwhere(comp == k)[0]))
        a = g.add_node(p)
        visited[p] = True
        q = next((n for n in nbrs(*p) if not visited[n]), None)
        if q is None:
            continue
        lab[p] = -1
        cl2node[-1] = a
        _, path = walk(-1, p, q)
        lab[p] = 0
        g.add_edge(a, a, path)
    # replace node-cluster pixels by the node's representative pixel at both path ends
    for d in g.edges.values():
        d["path"][0], d["path"][-1] = g.nodes[d["a"]], g.nodes[d["b"]]
    return g


# --------------------------------------------------------------------------- simplification
def dissolve_deg2(g: G) -> None:
    changed = True
    while changed:
        changed = False
        for n in list(g.nodes):
            if n in g.protected:
                continue
            inc = g.inc(n)
            if len(inc) == 2 and inc[0] != inc[1]:
                e1, e2 = inc
                a, b = g.other(e1, n), g.other(e2, n)
                path = g.path_from(e1, a) + g.path_from(e2, n)[1:]
                del g.edges[e1], g.edges[e2], g.nodes[n]
                g.add_edge(a, b, path)
                changed = True


def drop_isolated(g: G) -> None:
    for n in [n for n in g.nodes if g.deg(n) == 0 and n not in g.protected]:
        del g.nodes[n]


def prune_spurs(g: G, dist: np.ndarray, min_spur_px: float, k: float) -> int:
    removed = 0
    changed = True
    while changed:
        changed = False
        for e, d in list(g.edges.items()):
            a, b = d["a"], d["b"]
            if a == b:                                   # tiny self-loop (blob around a speck)
                if plen(d["path"]) < 4 * min_spur_px:
                    del g.edges[e]; removed += 1; changed = True
                continue
            da, db = g.deg(a), g.deg(b)
            for leaf, root, dl, dr in ((a, b, da, db), (b, a, db, da)):
                if dl == 1 and (dr >= 3 or root in g.protected) and leaf not in g.protected:
                    r_clear = dist[g.nodes[root]]
                    if plen(d["path"]) < max(min_spur_px, k * r_clear):
                        del g.edges[e], g.nodes[leaf]
                        removed += 1
                        changed = True
                        break
            if changed:
                break
        if changed:
            dissolve_deg2(g)
    return removed


def merge_close(g: G, merge_px: float, dist: np.ndarray) -> int:
    merged = 0
    changed = True
    while changed:
        changed = False
        for e, d in list(g.edges.items()):
            a, b = d["a"], d["b"]
            if a == b or plen(d["path"]) >= merge_px:
                continue
            if g.deg(a) < 3 or g.deg(b) < 3:
                continue
            keep, gone = (a, b) if (a in g.protected or dist[g.nodes[a]] >= dist[g.nodes[b]]) else (b, a)
            if gone in g.protected:
                continue
            del g.edges[e]
            for e2 in g.inc(gone):
                d2 = g.edges[e2]
                if d2["a"] == gone:
                    d2["a"] = keep; d2["path"][0] = g.nodes[keep]
                if d2["b"] == gone:
                    d2["b"] = keep; d2["path"][-1] = g.nodes[keep]
            del g.nodes[gone]
            merged += 1
            changed = True
            break
    return merged


def settle_leaves(g: G, dist: np.ndarray) -> None:
    """Slide each dead-end node back along its branch while clearance keeps growing, so
    room dead ends sit in the open middle of the room instead of in a corner."""
    for n in list(g.nodes):
        if n in g.protected or g.deg(n) != 1:
            continue
        e = g.inc(n)[0]
        path = g.path_from(e, n)                     # leaf -> root
        clear = np.array([dist[p] for p in path])
        # walk back while clearance stays within 2 px of the best seen so far (tolerates small
        # dips such as an alcove mouth), and stop at the most open point of that stretch
        # ...but never further than 1.5 x the best half-width (corridors that slowly widen)
        steps = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.asarray(path, float), axis=0), axis=1))]
        i, best = 0, 0
        while (i + 1 < len(path) - 1 and clear[i + 1] >= clear[best] - 2.0
               and steps[i + 1] <= 1.5 * max(clear[best], clear[i + 1])):
            i += 1
            if clear[i] > clear[best]:
                best = i
        i = best
        if 0 < i < len(path) - 2:
            g.nodes[n] = path[i]
            new = path[i:]
            d = g.edges[e]
            d["path"] = new if d["a"] == n else new[::-1]


def line_free(free: np.ndarray, p, q) -> bool:
    n = int(max(abs(p[0] - q[0]), abs(p[1] - q[1]))) + 1
    rr = np.rint(np.linspace(p[0], q[0], n)).astype(int)
    cc = np.rint(np.linspace(p[1], q[1], n)).astype(int)
    return bool(free[rr, cc].all())


def straighten(path, free: np.ndarray, tol_px: float) -> list[int]:
    """Indices into path of the vertices of a straight-segment approximation that stays free."""
    pts = np.asarray(path, np.int32)[:, ::-1].reshape(-1, 1, 2)   # (x=col, y=row)
    approx = cv2.approxPolyDP(pts, tol_px, False).reshape(-1, 2)[:, ::-1]
    idx = []
    j = 0
    for v in approx:                                   # map approx vertices back to path indices
        while j < len(path) and tuple(path[j]) != tuple(v):
            j += 1
        idx.append(min(j, len(path) - 1))
    idx = sorted(set([0, *idx, len(path) - 1]))
    out = [idx[0]]
    for i0, i1 in zip(idx, idx[1:]):
        stack = [(i0, i1)]
        segs = []
        while stack:                                   # recursive split until line-of-sight holds
            s, t = stack.pop()
            if t - s > 1 and not line_free(free, path[s], path[t]):
                seg = np.asarray(path[s:t + 1], float)
                u, r = seg[-1] - seg[0], seg - seg[0]
                d = np.abs(u[0] * r[:, 1] - u[1] * r[:, 0]) / (np.linalg.norm(u) + 1e-9)
                m = s + int(d[1:-1].argmax()) + 1
                stack += [(m, t), (s, m)]
            else:
                segs.append(t)
        out += segs
    return out


# --------------------------------------------------------------------------- main pipeline
def propose(grid: np.ndarray, pitch: float, entrance_xy, a) -> tuple[dict, dict]:
    h, w = grid.shape
    free = (grid > 127).astype(np.uint8)
    smooth = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dist = cv2.distanceTransform(smooth, cv2.DIST_L2, 5)               # clearance in px
    skel = skeletonize(smooth.astype(bool))
    skel &= dist * pitch * 2 >= a.min_width                           # too narrow to fly through
    g = build_pixel_graph(skel)
    stats = {"raw_nodes": len(g.nodes), "raw_edges": len(g.edges)}

    dissolve_deg2(g)
    stats["pruned"] = prune_spurs(g, dist, a.min_spur / pitch, a.spur_k)
    stats["merged"] = merge_close(g, a.merge / pitch, dist)
    dissolve_deg2(g)
    stats["pruned"] += prune_spurs(g, dist, a.min_spur / pitch, a.spur_k)
    drop_isolated(g)

    # keep only the component that will contain the entrance
    er, ec = h - 1 - int(entrance_xy[1] / pitch), int(entrance_xy[0] / pitch)
    best = None
    for e, d in g.edges.items():
        p = np.asarray(d["path"])
        i = int(((p - [er, ec]) ** 2).sum(1).argmin())
        dd = float(np.hypot(*(p[i] - [er, ec])))
        if best is None or dd < best[0]:
            best = (dd, e, i)
    if best is None:
        raise SystemExit("skeleton is empty: is grid.png free (255) where it should be?")
    _, e, i = best
    d = g.edges[e]
    path = d["path"]
    if 0 < i < len(path) - 1:                                          # split edge at the nearest point
        s = g.add_node(path[i])
        del g.edges[e]
        g.add_edge(d["a"], s, path[: i + 1])
        g.add_edge(s, d["b"], path[i:])
    else:
        s = d["a"] if i == 0 else d["b"]
    if np.hypot(er - g.nodes[s][0], ec - g.nodes[s][1]) * pitch > 0.3 and free[er, ec] and line_free(free, (er, ec), g.nodes[s]):
        ent = g.add_node((er, ec))
        n = int(max(abs(er - g.nodes[s][0]), abs(ec - g.nodes[s][1]))) + 1
        g.add_edge(ent, s, list(zip(np.rint(np.linspace(er, g.nodes[s][0], n)).astype(int),
                                    np.rint(np.linspace(ec, g.nodes[s][1], n)).astype(int))))
    else:
        ent = s
    g.protected.add(ent)
    dissolve_deg2(g)
    for _ in range(4):   # stub next to the entrance; leaves that slid back onto their junction
        n_before = len(g.nodes)
        prune_spurs(g, dist, a.min_spur / pitch, a.spur_k)
        drop_isolated(g)
        settle_leaves(g, dist)
        if len(g.nodes) == n_before:
            break
    reach, q = {ent}, deque([ent])
    while q:
        n = q.popleft()
        for e in g.inc(n):
            o = g.other(e, n)
            if o not in reach:
                reach.add(o); q.append(o)
    for e in [e for e, d in g.edges.items() if d["a"] not in reach]:
        del g.edges[e]
    for n in [n for n in g.nodes if n not in reach]:
        del g.nodes[n]

    # straight segments + waypoints
    final_nodes: dict[int, tuple[int, int]] = {}
    final_edges: list[tuple[int, int, float]] = []
    nid = {}

    def fnode(rc):
        key = tuple(map(int, rc))
        if key not in nid:
            nid[key] = len(nid) + 1
            final_nodes[nid[key]] = key
        return nid[key]

    for n, rc in g.nodes.items():
        fnode(rc)
    for d in g.edges.values():
        path = d["path"]
        idx = straighten(path, free.astype(bool), a.dp_tol / pitch)
        for i0, i1 in zip(idx, idx[1:]):
            seg = np.asarray(path[i0:i1 + 1])
            clear = dist[seg[:, 0], seg[:, 1]]
            width = 2 * float(np.percentile(clear, 10)) * pitch
            u, v = fnode(path[i0]), fnode(path[i1])
            if u != v:
                final_edges.append((u, v, round(max(width, a.min_width), 1)))
    ent_f = nid[g.nodes[ent]]

    # BFS order from the entrance for human-friendly ids
    adj = {n: [] for n in final_nodes}
    for u, v, _ in final_edges:
        adj[u].append(v); adj[v].append(u)
    order, q, seen = [], deque([ent_f]), {ent_f}
    while q:
        n = q.popleft(); order.append(n)
        for o in sorted(adj[n], key=lambda o: (final_nodes[o][1], -final_nodes[o][0])):
            if o not in seen:
                seen.add(o); q.append(o)
    rank = {n: i + 1 for i, n in enumerate(order)}

    def xy(rc):
        return round((rc[1] + 0.5) * pitch, 2), round((h - 1 - rc[0] + 0.5) * pitch, 2)

    nodes = []
    for n in order:
        k = len(adj[n])
        narrow = 2 * dist[final_nodes[n]] * pitch < a.door_width
        kind = ("entrance" if n == ent_f else "junction" if k >= 3 else "room" if k == 1
                else "door" if narrow else "junction")
        x, y = (round(float(entrance_xy[0]), 2), round(float(entrance_xy[1]), 2)) if n == ent_f and free[er, ec] else xy(final_nodes[n])
        nodes.append({"id": f"n{rank[n]}", "x": x, "y": y, "kind": kind})
    edges = []
    for u, v, wd in sorted(final_edges, key=lambda t: (min(rank[t[0]], rank[t[1]]), max(rank[t[0]], rank[t[1]]))):
        a_, b_ = sorted((rank[u], rank[v]))
        edges.append({"id": f"e{len(edges) + 1}", "a": f"n{a_}", "b": f"n{b_}", "width": wd})
    stats["nodes"], stats["edges"] = len(nodes), len(edges)
    stats["loops"] = len(edges) - len(nodes) + 1
    stats["kinds"] = {k: sum(n["kind"] == k for n in nodes) for k in ("entrance", "junction", "room", "door")}
    stats["deg"] = {f"n{rank[n]}": len(adj[n]) for n in order}
    return {"nodes": nodes, "edges": edges}, stats


def draw_preview(path: Path, grid: np.ndarray, graph: dict, pitch: float) -> None:
    h, w = grid.shape
    scale = max(2, min(8, 1600 // max(w, h)))
    img = np.empty((h, w, 3), np.uint8)
    img[:] = C_WALL
    img[grid > 127] = C_FREE
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from scan_to_grid import draw_frame
        draw_frame(img, pitch, scale)
    except ImportError:
        pass
    P = lambda n: (int(n["x"] / pitch * scale), int((h - n["y"] / pitch) * scale))
    byid = {n["id"]: n for n in graph["nodes"]}
    deg_of = {}
    for e in graph["edges"]:
        deg_of[e["a"]] = deg_of.get(e["a"], 0) + 1
        deg_of[e["b"]] = deg_of.get(e["b"], 0) + 1
    for e in graph["edges"]:
        a, b = P(byid[e["a"]]), P(byid[e["b"]])
        cv2.line(img, a, b, C_EDGE, max(2, scale // 2), cv2.LINE_AA)
    for n in graph["nodes"]:
        c = P(n)
        if deg_of.get(n["id"], 0) == 2:
            cv2.circle(img, c, scale, C_WAYPOINT, -1, cv2.LINE_AA)
            cv2.circle(img, c, scale, C_EDGE, 1, cv2.LINE_AA)
            continue
        col = C_ENTRANCE if n["kind"] == "entrance" else C_NODE
        cv2.circle(img, c, 2 * scale, col, -1, cv2.LINE_AA)
        cv2.circle(img, c, 2 * scale, C_EDGE, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(n["id"], cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx = c[0] + 2 * scale + 2 if c[0] + 2 * scale + 2 + tw < img.shape[1] else c[0] - 2 * scale - 2 - tw
        ty = c[1] - 2 * scale if c[1] - 2 * scale - th > 0 else c[1] + 2 * scale + th
        cv2.rectangle(img, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 3), C_FREE, -1)
        cv2.putText(img, n["id"], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def add_placeholders(sc: dict, graph: dict, grid: np.ndarray, pitch: float, deg: dict) -> None:
    """Fill the fields Scenario requires (survivor, drones) if missing, so the map loads."""
    h = grid.shape[0]
    is_free = lambda x, y: 0 <= int(y / pitch) < h and 0 <= int(x / pitch) < grid.shape[1] and \
        grid[h - 1 - int(y / pitch), int(x / pitch)] > 127
    ex, ey = sc["entrance"]["x"], sc["entrance"]["y"]
    # keep only placeholder names whose field is still there (and still unedited)
    sc["_placeholders"] = sorted({k for k in sc.get("_placeholders", []) if k in sc})
    if "survivor" not in sc:
        adj = {n["id"]: [] for n in graph["nodes"]}
        pos = {n["id"]: (n["x"], n["y"]) for n in graph["nodes"]}
        for e in graph["edges"]:
            L = float(np.hypot(*np.subtract(pos[e["a"]], pos[e["b"]])))
            adj[e["a"]].append((e["b"], L)); adj[e["b"]].append((e["a"], L))
        import heapq
        dist, pq = {"n1": 0.0}, [(0.0, "n1")]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, 1e18):
                continue
            for v, L in adj[u]:
                if d + L < dist.get(v, 1e18):
                    dist[v] = d + L; heapq.heappush(pq, (d + L, v))
        ends = [i for i in dist if deg.get(i) == 1] or list(dist)
        far = max(ends, key=dist.get)
        sc["survivor"] = {"x": pos[far][0], "y": pos[far][1], "yaw": 0.0}
        sc["_placeholders"] = sorted({*sc["_placeholders"], "survivor"})
        print(f"  placeholder: survivor at {far} {pos[far]} ({dist[far]:.1f} m by graph) -- move it in /editor",
              file=sys.stderr)
    if "drones" not in sc:
        starts = []
        for r in np.arange(0.4, 3.0, 0.2):
            for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                x, y = ex + r * np.cos(ang), ey + r * np.sin(ang)
                if is_free(x, y) and all(np.hypot(x - sx, y - sy) >= 0.5 for sx, sy in starts):
                    starts.append((round(float(x), 2), round(float(y), 2)))
                if len(starts) == 4:
                    break
            if len(starts) == 4:
                break
        sc["drones"] = {"ids": ["D1", "D2", "D3", "D4"], "start": [list(p) for p in starts], "altitude": 1.6}
        sc["_placeholders"] = sorted({*sc["_placeholders"], "drones"})
        print(f"  placeholder: drone starts {starts} -- move them in /editor", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir", type=Path)
    ap.add_argument("--entrance", type=float, nargs=2, metavar=("X", "Y"), help="world m (default: scenario.json)")
    ap.add_argument("--min-spur", type=float, default=1.0, help="always prune dead-end twigs shorter than this (m)")
    ap.add_argument("--spur-k", type=float, default=1.6, help="prune twigs shorter than k x junction half-width")
    ap.add_argument("--merge", type=float, default=0.8, help="merge junctions closer than this (m)")
    ap.add_argument("--dp-tol", type=float, default=0.3, help="max deviation of straight edges from the skeleton (m)")
    ap.add_argument("--min-width", type=float, default=0.6, help="ignore passages narrower than this (m)")
    ap.add_argument("--door-width", type=float, default=1.2, help="pass-throughs narrower than this are 'door' (m)")
    ap.add_argument("--stdout", action="store_true", help="print the graph JSON, don't write any file")
    a = ap.parse_args()

    grid = cv2.imread(str(a.map_dir / "grid.png"), cv2.IMREAD_GRAYSCALE)
    if grid is None:
        raise SystemExit(f"cannot read {a.map_dir / 'grid.png'}")
    sc_path = a.map_dir / "scenario.json"
    sc = json.loads(sc_path.read_text()) if sc_path.exists() else {"version": 1, "name": a.map_dir.name}
    pitch = float(sc.get("grid", {}).get("resolution", 0.1))
    if a.entrance:
        ent = a.entrance
    elif "entrance" in sc:
        ent = (sc["entrance"]["x"], sc["entrance"]["y"])
    else:
        raise SystemExit("no entrance: pass --entrance X Y (world m) or run scan_to_grid.py first")

    graph, st = propose(grid, pitch, ent, a)
    if a.stdout:
        print(json.dumps(graph))
        return
    print(f"skeleton      {st['raw_nodes']} raw nodes / {st['raw_edges']} raw edges; "
          f"pruned {st['pruned']} spurs, merged {st['merged']} junctions", file=sys.stderr)
    print(f"graph         {st['nodes']} nodes ({', '.join(f'{v} {k}' for k, v in st['kinds'].items())}), "
          f"{st['edges']} edges, {st['loops']} independent loop(s)", file=sys.stderr)
    deg = {n["id"]: 0 for n in graph["nodes"]}
    for e in graph["edges"]:
        deg[e["a"]] += 1; deg[e["b"]] += 1
    bad = [n["id"] for n in graph["nodes"] if grid[grid.shape[0] - 1 - int(n["y"] / pitch), int(n["x"] / pitch)] <= 127]
    if bad:
        print(f"  WARNING: nodes on wall cells: {bad}", file=sys.stderr)

    sc["graph"] = graph
    add_placeholders(sc, graph, grid, pitch, st["deg"])
    eids = {e["id"] for e in graph["edges"]}
    nids = {n["id"] for n in graph["nodes"]}
    stale = [c["edge"] for c in sc.get("compromised", []) if c.get("edge") not in eids]
    stale += [v for k, v in sc.get("beats", {}).items() if k.endswith("node") and v not in nids]
    if stale:
        print(f"  WARNING: compromised/beats refer to ids that no longer exist: {stale} -- fix them in /editor",
              file=sys.stderr)
    sc_path.write_text(json.dumps(sc, indent=2) + "\n")
    draw_preview(a.map_dir / "preview.png", grid, graph, pitch)
    print(f"wrote         {sc_path} (graph), {a.map_dir / 'preview.png'}", file=sys.stderr)


if __name__ == "__main__":
    main()
