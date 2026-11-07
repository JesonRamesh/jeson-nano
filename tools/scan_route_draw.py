#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "opencv-python", "scikit-image"]
# ///
"""Draw the route graph by hand on the textured top-down view of the scan (OpenCV window).

    uv run tools/scan_route_draw.py maps/<name> [--fresh]

Needs topdown.jpg (tools/scan_topdown.py) and grid.png/scenario.json (tools/scan_to_grid.py).
--fresh starts from an empty graph (otherwise the current scenario.json graph is loaded).
Needs the GUI build of OpenCV (`opencv-python`, not -headless): `uv run` on this file gets it.

MODES (press the key)                    MOUSE
  r  route      click = add node + edge   click / drag         act (drag a node to move it)
                from the last node; click  e / Esc / right-click end the current chain
                a node to connect/continue   (Mac: two-finger click or ctrl+click)
                from it; click an edge to  two-finger scroll    pan  (arrows / right-drag too)
                branch off it              + - / ctrl+scroll    zoom
  c  checkpoint click a node to toggle it  KEYS
                as a drone checkpoint;      1..5  set kind of node under cursor:
                d over one = cycle drone          1 entrance 2 junction 3 door 4 room 5 stairs
  b  blocked    click an edge = blocked     u     undo
                (compromised) at that spot; a     toggle grid.png wall overlay (alignment check)
                click again to clear        f     fit view       h  help on/off
  x  delete     click node/edge to delete   s     save           q / Esc  quit

SAVES (s) into scenario.json: graph.nodes/edges (edge width = measured from grid.png
clearance, min 0.6 m), entrance (= the entrance node), compromised (blocked edges, kind
"debris"), checkpoints [{x, y, drone, node}]. Also writes route_preview.jpg (top-down +
route) and preview.png (grid + route). Warnings are printed for nodes on walls / edges
crossing walls in grid.png (fix the grid or the route, your call).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

KINDS = ["entrance", "junction", "door", "room", "stairs"]
# BGR, Tenyks palette
VIOLET = (0xE3, 0x3B, 0x40)
PEACH = (0xFF, 0x92, 0xD2)
CYAN = (0xFB, 0xFF, 0x78)
WHITE = (255, 255, 255)
DARK = (0x1E, 0x10, 0x0D)
RED = (0x40, 0x40, 0xF0)
AMBER = (0x30, 0xC0, 0xFF)
KIND_COL = {"entrance": CYAN, "junction": PEACH, "door": (0xF3, 0xCB, 0xCB), "room": (0xA0, 0xE0, 0x90),
            "stairs": AMBER}
DRONES = [None, "D1", "D2", "D3", "D4"]


class Route:
    """The editable data, in world metres. Pure logic, no GUI (testable)."""

    def __init__(self, grid: np.ndarray, pitch: float):
        self.grid, self.pitch = grid, pitch
        self.free = (grid > 127).astype(np.uint8)
        self.clear = cv2.distanceTransform(self.free, cv2.DIST_L2, 5) * pitch   # m to nearest wall
        self.nodes: dict[str, dict] = {}
        self.edges: dict[str, dict] = {}
        self.blocked: dict[str, float] = {}       # edge id -> at
        self.checkpoints: list[dict] = []          # {"node", "drone"}
        self.history: list = []
        self.dirty = False
        self.keep_ids = True        # --renumber turns this off

    # ---- persistence -------------------------------------------------------
    def load(self, sc: dict) -> None:
        g = sc.get("graph", {})
        self.nodes = {n["id"]: {"x": n["x"], "y": n["y"], "kind": n.get("kind", "junction")} for n in g.get("nodes", [])}
        self.edges = {e["id"]: {"a": e["a"], "b": e["b"], "width": e.get("width", 1.8)} for e in g.get("edges", [])}
        self.blocked = {c["edge"]: c.get("at", 0.5) for c in sc.get("compromised", []) if c.get("edge") in self.edges}
        self.checkpoints = [{"node": c["node"], "drone": c.get("drone")} for c in sc.get("checkpoints", [])
                            if c.get("node") in self.nodes]

    def snapshot(self) -> None:
        self.history.append(copy.deepcopy((self.nodes, self.edges, self.blocked, self.checkpoints)))
        self.history = self.history[-200:]
        self.dirty = True

    def undo(self) -> bool:
        if not self.history:
            return False
        self.nodes, self.edges, self.blocked, self.checkpoints = self.history.pop()
        self.dirty = True
        return True

    # ---- helpers -----------------------------------------------------------
    def _new_id(self, prefix: str, used) -> str:
        k = 1 + max([int(i[1:]) for i in used if i[1:].isdigit()] or [0])
        return f"{prefix}{k}"

    def cell(self, x, y):
        h, w = self.grid.shape
        c, r = int(x / self.pitch), h - 1 - int(y / self.pitch)
        return (r, c) if 0 <= r < h and 0 <= c < w else None

    def is_free(self, x, y) -> bool:
        rc = self.cell(x, y)
        return rc is not None and bool(self.free[rc])

    def samples(self, a, b, step=0.05):
        (x0, y0), (x1, y1) = a, b
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step) + 1)
        return zip(np.linspace(x0, x1, n), np.linspace(y0, y1, n))

    def edge_width(self, a: str, b: str) -> float:
        pa, pb = self.xy(a), self.xy(b)
        cl = [self.clear[rc] for rc in (self.cell(x, y) for x, y in self.samples(pa, pb)) if rc is not None]
        cl = [c for c in cl if c > 0]
        if not cl:
            return 1.8
        return round(max(0.6, 2 * float(np.percentile(cl, 10))), 1)

    def wall_fraction(self, a: str, b: str) -> float:
        s = [self.is_free(x, y) for x, y in self.samples(self.xy(a), self.xy(b))]
        return 1 - sum(s) / len(s)

    def xy(self, n: str):
        return self.nodes[n]["x"], self.nodes[n]["y"]

    def entrance(self) -> str | None:
        return next((i for i, n in self.nodes.items() if n["kind"] == "entrance"), None)

    # ---- edits -------------------------------------------------------------
    def add_node(self, x, y, kind=None) -> str:
        i = self._new_id("n", self.nodes)
        if kind is None:
            kind = "entrance" if self.entrance() is None else "junction"
        self.nodes[i] = {"x": round(float(x), 2), "y": round(float(y), 2), "kind": kind}
        return i

    def has_edge(self, a, b) -> bool:
        return any({e["a"], e["b"]} == {a, b} for e in self.edges.values())

    def add_edge(self, a: str, b: str) -> str | None:
        if a == b or self.has_edge(a, b):
            return None
        i = self._new_id("e", self.edges)
        self.edges[i] = {"a": a, "b": b, "width": self.edge_width(a, b)}
        return i

    def move_node(self, n, x, y) -> None:
        self.nodes[n]["x"], self.nodes[n]["y"] = round(float(x), 2), round(float(y), 2)
        for e in self.edges.values():
            if n in (e["a"], e["b"]):
                e["width"] = self.edge_width(e["a"], e["b"])

    def set_kind(self, n, kind) -> None:
        if kind == "entrance":
            for m in self.nodes.values():
                if m["kind"] == "entrance":
                    m["kind"] = "junction"
        self.nodes[n]["kind"] = kind

    def delete_node(self, n) -> None:
        for e in [e for e, d in self.edges.items() if n in (d["a"], d["b"])]:
            self.delete_edge(e)
        self.checkpoints = [c for c in self.checkpoints if c["node"] != n]
        del self.nodes[n]

    def delete_edge(self, e) -> None:
        self.blocked.pop(e, None)
        del self.edges[e]

    def split_edge(self, e, t: float) -> str:
        d = self.edges[e]
        (x0, y0), (x1, y1) = self.xy(d["a"]), self.xy(d["b"])
        n = self.add_node(x0 + t * (x1 - x0), y0 + t * (y1 - y0), "junction")
        at = self.blocked.pop(e, None)
        a, b = d["a"], d["b"]
        del self.edges[e]
        e1, e2 = self.add_edge(a, n), self.add_edge(n, b)
        if at is not None:   # keep the blockage on the half that contains it
            if at < t:
                self.blocked[e1] = round(at / t, 2)
            else:
                self.blocked[e2] = round((at - t) / (1 - t), 2)
        return n

    def toggle_blocked(self, e, at) -> None:
        if e in self.blocked:
            del self.blocked[e]
        else:
            self.blocked[e] = round(float(np.clip(at, 0.05, 0.95)), 2)

    def toggle_checkpoint(self, n) -> None:
        if any(c["node"] == n for c in self.checkpoints):
            self.checkpoints = [c for c in self.checkpoints if c["node"] != n]
        else:
            self.checkpoints.append({"node": n, "drone": None})

    def cycle_drone(self, n) -> None:
        for c in self.checkpoints:
            if c["node"] == n:
                c["drone"] = DRONES[(DRONES.index(c["drone"]) + 1) % len(DRONES)]

    # ---- hit tests (world metres) -----------------------------------------
    def node_at(self, x, y, r) -> str | None:
        best = min(self.nodes, key=lambda i: math.hypot(self.nodes[i]["x"] - x, self.nodes[i]["y"] - y), default=None)
        if best and math.hypot(self.nodes[best]["x"] - x, self.nodes[best]["y"] - y) <= r:
            return best
        return None

    def edge_at(self, x, y, r):
        best = None
        for i, d in self.edges.items():
            (x0, y0), (x1, y1) = self.xy(d["a"]), self.xy(d["b"])
            vx, vy = x1 - x0, y1 - y0
            L2 = vx * vx + vy * vy or 1e-9
            t = max(0.0, min(1.0, ((x - x0) * vx + (y - y0) * vy) / L2))
            dd = math.hypot(x0 + t * vx - x, y0 + t * vy - y)
            if dd <= r and (best is None or dd < best[1]):
                best = (i, dd, t)
        return (best[0], best[2]) if best else (None, None)

    # ---- output ------------------------------------------------------------
    def renumbered(self, keep: bool = False):
        """Ids n1.. in BFS order from the entrance (n1 = entrance), e1.. likewise.
        keep=True: keep the existing ids (other people refer to them), order by number."""
        if keep:
            num = lambda i: int(i[1:]) if i[1:].isdigit() else 10 ** 9
            order = sorted(self.nodes, key=num)
            eorder = sorted(self.edges, key=num)
            return {n: n for n in order}, {e: e for e in eorder}, order, eorder
        ent = self.entrance() or next(iter(self.nodes), None)
        adj = {n: [] for n in self.nodes}
        for d in self.edges.values():
            adj[d["a"]].append(d["b"]); adj[d["b"]].append(d["a"])
        order, seen = [], set()
        for root in [ent] + list(self.nodes):
            if root is None or root in seen:
                continue
            q = [root]; seen.add(root)
            while q:
                n = q.pop(0); order.append(n)
                for o in sorted(adj[n], key=lambda o: (self.nodes[o]["x"], self.nodes[o]["y"])):
                    if o not in seen:
                        seen.add(o); q.append(o)
        nmap = {old: f"n{i + 1}" for i, old in enumerate(order)}
        rank = {old: i for i, old in enumerate(order)}
        eorder = sorted(self.edges, key=lambda e: tuple(sorted((rank[self.edges[e]["a"]], rank[self.edges[e]["b"]]))))
        emap = {old: f"e{i + 1}" for i, old in enumerate(eorder)}
        return nmap, emap, order, eorder

    def to_scenario(self, sc: dict) -> tuple[dict, list[str]]:
        warn = []
        if not self.nodes:
            raise ValueError("nothing to save: the graph is empty")
        if self.entrance() is None:
            warn.append("no entrance node: making the first node the entrance")
            self.nodes[next(iter(self.nodes))]["kind"] = "entrance"
        nmap, emap, order, eorder = self.renumbered(keep=self.keep_ids)
        ent0 = self.entrance()
        if self.keep_ids and ent0 and order[0] != ent0:
            order.remove(ent0); order.insert(0, ent0)
        nodes = [{"id": nmap[n], "x": self.nodes[n]["x"], "y": self.nodes[n]["y"], "kind": self.nodes[n]["kind"]}
                 for n in order]
        edges = []
        for e in eorder:
            d = self.edges[e]
            a, b = sorted((nmap[d["a"]], nmap[d["b"]]), key=lambda s: int(s[1:]))
            flip = a != nmap[d["a"]]
            edges.append({"id": emap[e], "a": a, "b": b, "width": d["width"]})
            if e in self.blocked and flip:
                self.blocked[e] = round(1 - self.blocked[e], 2)
        for n in order:
            if not self.is_free(*self.xy(n)):
                warn.append(f"{nmap[n]} is on a wall cell of grid.png")
        for e in eorder:
            f = self.wall_fraction(self.edges[e]["a"], self.edges[e]["b"])
            if f > 0:
                warn.append(f"{emap[e]} crosses wall cells of grid.png ({f:.0%} of its length)")
        deg = {n: 0 for n in self.nodes}
        for d in self.edges.values():
            deg[d["a"]] += 1; deg[d["b"]] += 1
        iso = [nmap[n] for n, k in deg.items() if k == 0 and len(self.nodes) > 1]
        if iso:
            warn.append(f"unconnected nodes: {iso}")
        # connectivity from the entrance
        reach = set(); stack = [order[0]]
        while stack:
            n = stack.pop()
            if n in reach:
                continue
            reach.add(n)
            stack += [d["b"] if d["a"] == n else d["a"] for d in self.edges.values() if n in (d["a"], d["b"])]
        if len(reach) < len(self.nodes):
            warn.append(f"{len(self.nodes) - len(reach)} node(s) not reachable from the entrance")

        sc = dict(sc)
        sc["graph"] = {"nodes": nodes, "edges": edges}
        sc["graph_source"] = "manual"
        ent = self.entrance()
        sc["entrance"] = {"x": self.nodes[ent]["x"], "y": self.nodes[ent]["y"],
                          "yaw": sc.get("entrance", {}).get("yaw", 0.0)}
        # blocked edges: keep other compromised kinds that point at edges we don't know (none expected)
        sc["compromised"] = [{"edge": emap[e], "at": at, "kind": "debris"} for e, at in self.blocked.items()]
        sc["checkpoints"] = [{"x": self.nodes[c["node"]]["x"], "y": self.nodes[c["node"]]["y"],
                              "drone": c["drone"], "node": nmap[c["node"]]} for c in self.checkpoints]
        return sc, warn


# --------------------------------------------------------------------------- GUI
class View:
    def __init__(self, bg: np.ndarray, res: float, size=(1400, 900)):
        self.bg, self.res = bg, res
        self.W, self.H = size
        self.fit()

    def fit(self):
        h, w = self.bg.shape[:2]
        self.zoom = min(self.W / w, self.H / h)
        self.ox = (w - self.W / self.zoom) / 2      # image px at the screen's left edge
        self.oy = (h - self.H / self.zoom) / 2

    def w2s(self, x, y):
        u, v = x / self.res, self.bg.shape[0] - y / self.res
        return int(round((u - self.ox) * self.zoom)), int(round((v - self.oy) * self.zoom))

    def s2w(self, sx, sy):
        u, v = sx / self.zoom + self.ox, sy / self.zoom + self.oy
        return u * self.res, (self.bg.shape[0] - v) * self.res

    def zoom_at(self, sx, sy, f):
        u, v = sx / self.zoom + self.ox, sy / self.zoom + self.oy
        self.zoom = float(np.clip(self.zoom * f, 0.05, 8))
        self.ox, self.oy = u - sx / self.zoom, v - sy / self.zoom

    def pan(self, dsx, dsy):
        self.ox -= dsx / self.zoom
        self.oy -= dsy / self.zoom

    def base(self, overlay=None) -> np.ndarray:
        A = np.array([[self.zoom, 0, -self.ox * self.zoom], [0, self.zoom, -self.oy * self.zoom]], np.float32)
        interp = cv2.INTER_NEAREST if self.zoom > 2 else cv2.INTER_AREA if self.zoom < 1 else cv2.INTER_LINEAR
        src = self.bg if overlay is None else overlay
        return cv2.warpAffine(src, A, (self.W, self.H), flags=interp, borderValue=(20, 16, 13))


class App:
    PICK_PX = 12

    def __init__(self, map_dir: Path, fresh: bool, size, renumber: bool = False):
        self.map_dir = map_dir
        meta = json.loads((map_dir / "topdown.json").read_text())
        bg = cv2.imread(str(map_dir / meta["image"]))
        if bg is None:
            raise SystemExit(f"cannot read {map_dir / meta['image']}: run tools/scan_topdown.py first")
        self.sc = json.loads((map_dir / "scenario.json").read_text())
        pitch = float(self.sc.get("grid", {}).get("resolution", 0.1))
        grid = cv2.imread(str(map_dir / "grid.png"), cv2.IMREAD_GRAYSCALE)
        self.route = Route(grid, pitch)
        self.route.keep_ids = not renumber
        if not fresh:
            self.route.load(self.sc)
        self.view = View(bg, meta["res"], size)
        walls = cv2.resize((grid <= 127).astype(np.uint8), (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.bg_overlay = bg.copy()
        self.bg_overlay[walls > 0] = (0.5 * self.bg_overlay[walls > 0] + 0.5 * np.array(RED)).astype(np.uint8)
        self.mode, self.head = "route", None
        self.show_grid, self.show_help = False, True
        self.mouse = (0, 0)
        self.press = None          # (button, sx, sy, node, dragging)
        self.msg, self.quit_armed = "", False

    # ---- events ------------------------------------------------------------
    def on_mouse(self, ev, sx, sy, flags, _):
        R, V = self.route, self.view
        self.mouse = (sx, sy)
        wx, wy = V.s2w(sx, sy)
        rad = self.PICK_PX / V.zoom * V.res
        if ev in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
            d = flags >> 16          # == cv2.getMouseWheelDelta (missing from some Python builds)
            if flags & cv2.EVENT_FLAG_CTRLKEY:          # ctrl + scroll = zoom
                V.zoom_at(sx, sy, 1.1 if d > 0 else 1 / 1.1)
            elif ev == cv2.EVENT_MOUSEWHEEL:              # two-finger scroll = pan
                V.pan(0, 25 * np.sign(d))
            else:
                V.pan(-25 * np.sign(d), 0)
        elif ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            btn = "L" if ev == cv2.EVENT_LBUTTONDOWN else "R"
            if btn == "L" and flags & cv2.EVENT_FLAG_CTRLKEY:   # macOS ctrl+click = right click
                btn = "R"
            self.press = [btn, sx, sy, R.node_at(wx, wy, rad) if btn == "L" else None, False]
        elif ev == cv2.EVENT_MOUSEMOVE and self.press:
            btn, px, py, node, dragging = self.press
            if not dragging and math.hypot(sx - px, sy - py) > 4:
                self.press[4] = dragging = True
                if btn == "L" and node and self.mode in ("route", "checkpoint"):
                    R.snapshot()
            if dragging:
                if btn == "R":
                    V.pan(sx - px, sy - py)
                    self.press[1], self.press[2] = sx, sy
                elif node and self.mode in ("route", "checkpoint"):
                    R.move_node(node, wx, wy)
        elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP) and self.press:
            btn, _, _, _, dragging = self.press
            self.press = None
            if dragging:
                return
            if btn == "R":
                self.head = None
                self.msg = "chain ended"
            else:
                self.click(wx, wy, rad)

    def click(self, wx, wy, rad):
        R = self.route
        n = R.node_at(wx, wy, rad)
        e, t = (None, None) if n else R.edge_at(wx, wy, rad)
        if self.mode == "route":
            R.snapshot()
            if n is None and e is not None:
                n = R.split_edge(e, t)
                self.msg = f"branched off {e}"
            if n is None:
                n = R.add_node(wx, wy)
                if not R.is_free(wx, wy):
                    self.msg = "note: node is on a wall cell of grid.png"
            if self.head and self.head in R.nodes and self.head != n:
                R.add_edge(self.head, n)
            self.head = n
        elif self.mode == "checkpoint":
            if n is None and e is not None:
                R.snapshot(); n = R.split_edge(e, t)
            if n:
                R.snapshot(); R.toggle_checkpoint(n)
            else:
                self.msg = "checkpoints go on nodes or edges"
        elif self.mode == "blocked":
            e2, t2 = R.edge_at(wx, wy, rad)
            if e2:
                R.snapshot(); R.toggle_blocked(e2, t2)
            else:
                self.msg = "click on an edge to block it"
        elif self.mode == "delete":
            if n:
                R.snapshot(); R.delete_node(n)
                if self.head == n:
                    self.head = None
            elif e:
                R.snapshot(); R.delete_edge(e)

    def on_key(self, k) -> bool:
        """Return False to quit."""
        R, V = self.route, self.view
        if k < 0:
            return True
        ch = chr(k & 0xFF).lower() if (k & 0xFF) < 128 else ""
        sx, sy = self.mouse
        wx, wy = V.s2w(sx, sy)
        under = R.node_at(wx, wy, self.PICK_PX / V.zoom * V.res)
        arrows = {63232: (0, 60), 2490368: (0, 60), 65362: (0, 60), 63233: (0, -60), 2621440: (0, -60),
                  65364: (0, -60), 63234: (60, 0), 2424832: (60, 0), 65361: (60, 0), 63235: (-60, 0),
                  2555904: (-60, 0), 65363: (-60, 0)}
        if k in arrows:
            V.pan(*arrows[k]); return True
        if (k == 27 or ch == "e") and self.head:
            self.head = None; self.msg = "chain ended"; return True
        if ch in ("q",) or k == 27:
            if R.dirty and not self.quit_armed:
                self.quit_armed = True
                self.msg = "unsaved changes: press s to save, or q again to quit without saving"
                return True
            return False
        self.quit_armed = False
        if ch in "rcbx" and ch:
            self.mode = {"r": "route", "c": "checkpoint", "b": "blocked", "x": "delete"}[ch]
            self.head = None if ch != "r" else self.head
        elif ch in "12345" and ch:
            if under:
                R.snapshot(); R.set_kind(under, KINDS[int(ch) - 1])
            else:
                self.msg = "hover over a node, then press 1-5"
        elif ch == "d" and under:
            R.snapshot(); R.cycle_drone(under)
        elif ch in ("u", "z"):
            self.msg = "undone" if R.undo() else "nothing to undo"
            if self.head not in R.nodes:
                self.head = None
        elif ch == "a":
            self.show_grid = not self.show_grid
        elif ch == "f":
            V.fit()
        elif ch == "h" or ch == "?":
            self.show_help = not self.show_help
        elif ch in ("+", "="):
            V.zoom_at(V.W // 2, V.H // 2, 1.25)
        elif ch in ("-", "_"):
            V.zoom_at(V.W // 2, V.H // 2, 0.8)
        elif ch == "s":
            self.save()
        return True

    # ---- drawing -----------------------------------------------------------
    def frame(self) -> np.ndarray:
        R, V = self.route, self.view
        img = V.base(self.bg_overlay if self.show_grid else None)
        th = max(2, int(3 * min(V.zoom, 2)))
        cps = {c["node"]: c for c in R.checkpoints}
        for i, d in R.edges.items():
            a, b = V.w2s(*R.xy(d["a"])), V.w2s(*R.xy(d["b"]))
            cv2.line(img, a, b, DARK, th + 3, cv2.LINE_AA)
            cv2.line(img, a, b, VIOLET, th, cv2.LINE_AA)
            if i in R.blocked:
                t = R.blocked[i]
                p = (int(a[0] + t * (b[0] - a[0])), int(a[1] + t * (b[1] - a[1])))
                s = 9
                cv2.line(img, (p[0] - s, p[1] - s), (p[0] + s, p[1] + s), RED, 4, cv2.LINE_AA)
                cv2.line(img, (p[0] - s, p[1] + s), (p[0] + s, p[1] - s), RED, 4, cv2.LINE_AA)
        if self.mode == "route" and self.head in R.nodes:
            cv2.line(img, V.w2s(*R.xy(self.head)), self.mouse, VIOLET, 1, cv2.LINE_AA)
        nmap = R.renumbered(keep=R.keep_ids)[0] if R.nodes else {}
        for i, n in R.nodes.items():
            p = V.w2s(n["x"], n["y"])
            r = 7 if n["kind"] != "entrance" else 10
            if i in cps:
                cv2.circle(img, p, r + 6, AMBER, 3, cv2.LINE_AA)
            cv2.circle(img, p, r, KIND_COL[n["kind"]], -1, cv2.LINE_AA)
            cv2.circle(img, p, r, DARK, 2, cv2.LINE_AA)
            if i == self.head:
                cv2.circle(img, p, r + 3, WHITE, 1, cv2.LINE_AA)
            label = nmap.get(i, i)
            if i in cps:
                label += " CP" + str(list(cps).index(i) + 1) + (f" {cps[i]['drone']}" if cps[i]["drone"] else "")
            self.text(img, label, (p[0] + r + 4, p[1] - r), 0.45)
        # HUD
        nblk = len(R.blocked)
        bar = (f"MODE {self.mode.upper()}  |  nodes {len(R.nodes)}  edges {len(R.edges)}  blocked {nblk}  "
               f"checkpoints {len(R.checkpoints)}  |  {'UNSAVED' if R.dirty else 'saved'}  |  h = help")
        cv2.rectangle(img, (0, 0), (V.W, 28), DARK, -1)
        cv2.putText(img, bar, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        wx, wy = V.s2w(*self.mouse)
        cv2.putText(img, f"x {wx:6.2f}  y {wy:6.2f} m", (V.W - 190, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1,
                    cv2.LINE_AA)
        if self.msg:
            self.text(img, self.msg, (10, V.H - 12), 0.55)
        if self.show_help:
            lines = ["r route: click = node+edge from last, click node = connect, click edge = branch",
                     "   end chain: e / Esc / right-click (two-finger or ctrl+click); drag node = move",
                     "c checkpoint: click node/edge = toggle checkpoint, d over it = cycle drone",
                     "b blocked: click edge = mark blocked door/debris (again = clear)",
                     "x delete node/edge     1-5 over node: entrance/junction/door/room/stairs",
                     "u undo  a grid overlay  f fit  +/- or ctrl+scroll zoom  two-finger scroll/arrows pan",
                     "s save   q quit"]
            y0 = 40
            cv2.rectangle(img, (6, y0 - 4), (620, y0 + 20 * len(lines)), DARK, -1)
            for j, l in enumerate(lines):
                cv2.putText(img, l, (12, y0 + 12 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.47, WHITE, 1, cv2.LINE_AA)
        return img

    @staticmethod
    def text(img, s, org, scale):
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(img, (org[0] - 2, org[1] - th - 3), (org[0] + tw + 2, org[1] + 4), DARK, -1)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1, cv2.LINE_AA)

    # ---- save --------------------------------------------------------------
    def save(self) -> None:
        R = self.route
        try:
            sc, warn = R.to_scenario(self.sc)
        except ValueError as e:
            self.msg = str(e); return
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:   # refresh placeholder survivor / drone starts around the (new) entrance
            from propose_graph import add_placeholders, draw_preview
            for k in sc.get("_placeholders", []):
                sc.pop(k, None)          # placeholders follow the new entrance / graph
            sc["_placeholders"] = []
            deg = {n["id"]: 0 for n in sc["graph"]["nodes"]}
            for e in sc["graph"]["edges"]:
                deg[e["a"]] += 1; deg[e["b"]] += 1
            add_placeholders(sc, sc["graph"], R.grid, R.pitch, deg)
            draw_preview(self.map_dir / "preview.png", R.grid, sc["graph"], R.pitch)
        except ImportError:
            pass
        (self.map_dir / "scenario.json").write_text(json.dumps(sc, indent=2) + "\n")
        self.sc = sc
        R.load(sc)                      # adopt the renumbered ids
        self.head = None
        R.history.clear()
        R.dirty = False
        full = self.render_full()
        cv2.imwrite(str(self.map_dir / "route_preview.jpg"), full, [cv2.IMWRITE_JPEG_QUALITY, 88])
        for w in warn:
            print("  WARNING:", w)
        print(f"saved {self.map_dir / 'scenario.json'}: {len(sc['graph']['nodes'])} nodes, "
              f"{len(sc['graph']['edges'])} edges, {len(sc['compromised'])} blocked, "
              f"{len(sc['checkpoints'])} checkpoints; route_preview.jpg, preview.png")
        self.msg = f"saved ({len(warn)} warning(s) in the terminal)" if warn else "saved"

    def render_full(self) -> np.ndarray:
        """Route over the full-resolution top-down image."""
        V = self.view
        saved = (V.W, V.H, V.zoom, V.ox, V.oy, self.show_help, self.msg, self.head, self.show_grid)
        self.show_grid = False
        V.W, V.H = V.bg.shape[1], V.bg.shape[0]
        V.zoom, V.ox, V.oy = 1.0, 0.0, 0.0
        self.show_help, self.msg, self.head = False, "", None
        img = self.frame()
        V.W, V.H, V.zoom, V.ox, V.oy, self.show_help, self.msg, self.head, self.show_grid = saved
        return img

    def run(self) -> None:
        win = f"route editor - {self.map_dir.name}"
        try:
            cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        except cv2.error:
            raise SystemExit("This OpenCV build has no GUI (opencv-python-headless). Run with\n"
                             "  uv run tools/scan_route_draw.py ...   (installs opencv-python for this script)\n"
                             "or pip install opencv-python in a separate venv.")
        cv2.setMouseCallback(win, self.on_mouse)
        while True:
            cv2.imshow(win, self.frame())
            k = cv2.waitKeyEx(20)
            if not self.on_key(k):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                if self.route.dirty:
                    print("window closed with unsaved changes (not saved)")
                break
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir", type=Path)
    ap.add_argument("--fresh", action="store_true", help="start from an empty graph")
    ap.add_argument("--renumber", action="store_true",
                    help="renumber ids n1.. in BFS order from the entrance on save (default: keep ids)")
    ap.add_argument("--size", type=int, nargs=2, default=(1400, 900), metavar=("W", "H"), help="window size")
    a = ap.parse_args()
    App(a.map_dir, a.fresh, tuple(a.size), a.renumber).run()


if __name__ == "__main__":
    main()
