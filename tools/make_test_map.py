"""Draw the synthetic test map: `uv run python tools/make_test_map.py`.

Writes maps/test/grid.png (30 x 20 m at 0.1 m/px, 0 = wall, 255 = free) and
maps/test/scenario.json, then checks the §5 scenario requirements and fails loudly
if any is broken. Owned by S (integration fixture).

Layout (metres, x east, y north). Corridors are 1.8 m wide.

    y=17   n6 ==================== n7
            ||                      ||
    y=10   n4 ==== room n5         n8 ==== survivor room n9
     (left dead-end room)           ||
            ||                      ||
    y=5.6 n11 (closet)
          ||
    y=3  n1 == n2 ====[e2 debris]== n3 ======= n10 (fire spur)
       entrance

- Loop: n2-n4-n6-n7-n8-n3-n2.
- Naive shortest route n1-n2-n3-n8-n9 uses e2, which is compromised (LiDAR re-plan).
- After e2 is blocked the survivor is still reachable via n4-n6-n7-n8.
- Wrong-way branch next to the final route: e4 (n4 -> left dead-end room).
- Branches from the entrance region: e11 (closet), e2 (east), e3 (north).
"""

from __future__ import annotations

import heapq
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sgr.contracts import (  # noqa: E402
    Beats,
    Compromise,
    Decoy,
    DroneSetup,
    Entrance,
    Fire,
    Flare,
    Graph,
    GraphEdge,
    GraphNode,
    GridRef,
    Heat,
    Scenario,
    Survivor,
)

W_M, H_M, RES = 30.0, 20.0, 0.1
CORRIDOR = 1.8
OUT = ROOT / "maps" / "test"

NODES = [
    GraphNode(id="n1", x=2.0, y=3.0, kind="entrance"),
    GraphNode(id="n2", x=8.0, y=3.0, kind="junction"),
    GraphNode(id="n3", x=20.0, y=3.0, kind="junction"),
    GraphNode(id="n4", x=8.0, y=10.0, kind="junction"),
    GraphNode(id="n5", x=2.75, y=10.0, kind="room"),
    GraphNode(id="n6", x=8.0, y=17.0, kind="junction"),
    GraphNode(id="n7", x=20.0, y=17.0, kind="junction"),
    GraphNode(id="n8", x=20.0, y=10.0, kind="junction"),
    GraphNode(id="n9", x=26.0, y=10.0, kind="room"),
    GraphNode(id="n10", x=27.0, y=3.0, kind="door"),
    GraphNode(id="n11", x=2.0, y=5.6, kind="room"),
]
EDGES = [
    GraphEdge(id="e1", a="n1", b="n2", width=CORRIDOR),
    GraphEdge(id="e2", a="n2", b="n3", width=CORRIDOR),
    GraphEdge(id="e3", a="n2", b="n4", width=CORRIDOR),
    GraphEdge(id="e4", a="n4", b="n5", width=CORRIDOR),
    GraphEdge(id="e5", a="n4", b="n6", width=CORRIDOR),
    GraphEdge(id="e6", a="n6", b="n7", width=CORRIDOR),
    GraphEdge(id="e7", a="n7", b="n8", width=CORRIDOR),
    GraphEdge(id="e8", a="n8", b="n3", width=CORRIDOR),
    GraphEdge(id="e9", a="n8", b="n9", width=CORRIDOR),
    GraphEdge(id="e10", a="n3", b="n10", width=CORRIDOR),
    GraphEdge(id="e11", a="n1", b="n11", width=CORRIDOR),
]
ROOMS = [  # (x0, y0, x1, y1) metres
    (0.5, 7.0, 5.0, 13.0),  # left dead-end room (n5)
    (23.0, 6.0, 29.0, 14.0),  # survivor room (n9)
    (0.8, 4.6, 3.8, 6.4),  # side closet next to the entrance (n11)
]


def build_scenario() -> Scenario:
    return Scenario(
        version=1,
        name="test",
        grid=GridRef(file="grid.png", resolution=RES, origin=(0.0, 0.0)),
        scan=None,
        entrance=Entrance(x=2.0, y=3.0, yaw=0.0),
        graph=Graph(nodes=NODES, edges=EDGES),
        compromised=[Compromise(edge="e2", at=0.5, kind="debris")],
        fires=[Fire(x=25.5, y=3.0, size=1.5)],
        survivor=Survivor(x=26.5, y=11.0, yaw=1.2),
        decoy=Decoy(x=8.7, y=6.5, kind="orange_panel"),
        drones=DroneSetup(
            ids=["D1", "D2", "D3", "D4"],
            start=[(2.0, 2.5), (2.0, 3.5), (2.5, 2.5), (2.5, 3.5)],
            altitude=1.6,
        ),
        flare=Flare(t=65.0, target="D3"),
        heat=Heat(limit_c=70.0, ambient_c=20.0),
        beats=Beats(decoy_drone="D2", d3_area_node="n10"),
    )


# --------------------------------------------------------------------- drawing
def draw_grid(sc: Scenario) -> np.ndarray:
    """Return free mask [iy, ix] (row 0 = bottom)."""
    w, h = round(W_M / RES), round(H_M / RES)
    free = np.zeros((h, w), dtype=bool)
    nodes = {n.id: n for n in sc.graph.nodes}

    def fill_rect(x0: float, y0: float, x1: float, y1: float) -> None:
        free[round(y0 / RES) : round(y1 / RES), round(x0 / RES) : round(x1 / RES)] = True

    half = CORRIDOR / 2
    for e in sc.graph.edges:  # all edges are axis-aligned here
        a, b = nodes[e.a], nodes[e.b]
        fill_rect(min(a.x, b.x) - half, min(a.y, b.y) - half, max(a.x, b.x) + half, max(a.y, b.y) + half)
    for r in ROOMS:
        fill_rect(*r)
    return free


def save_png(free: np.ndarray, path: Path) -> None:
    img = np.where(free[::-1], 255, 0).astype(np.uint8)  # flip: image row 0 = top
    Image.fromarray(img, mode="L").save(path)


# --------------------------------------------------------------------- §5 checks
def _is_free(free: np.ndarray, x: float, y: float) -> bool:
    ix, iy = int(x / RES), int(y / RES)
    return 0 <= iy < free.shape[0] and 0 <= ix < free.shape[1] and bool(free[iy, ix])


def _shortest(sc: Scenario, src: str, dst: str, banned: set[str]) -> list[str] | None:
    """Dijkstra on the route graph; returns the list of edge ids, or None."""
    nodes = {n.id: n for n in sc.graph.nodes}
    adj: dict[str, list[tuple[str, str, float]]] = {n: [] for n in nodes}
    for e in sc.graph.edges:
        if e.id in banned:
            continue
        L = math.dist((nodes[e.a].x, nodes[e.a].y), (nodes[e.b].x, nodes[e.b].y))
        adj[e.a].append((e.b, e.id, L))
        adj[e.b].append((e.a, e.id, L))
    best = {src: (0.0, [])}
    pq: list[tuple[float, str]] = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            return best[u][1]
        if d > best[u][0]:
            continue
        for v, eid, L in adj[u]:
            nd = d + L
            if v not in best or nd < best[v][0]:
                best[v] = (nd, best[u][1] + [eid])
                heapq.heappush(pq, (nd, v))
    return None


def _flood_reaches(free: np.ndarray, start: tuple[float, float], goal: tuple[float, float]) -> bool:
    h, w = free.shape
    sx, sy = int(start[0] / RES), int(start[1] / RES)
    gx, gy = int(goal[0] / RES), int(goal[1] / RES)
    seen = np.zeros_like(free)
    q = deque([(sx, sy)])
    seen[sy, sx] = True
    while q:
        x, y = q.popleft()
        if (x, y) == (gx, gy):
            return True
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and free[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((nx, ny))
    return False


def _debris_mask(sc: Scenario, shape: tuple[int, int]) -> np.ndarray:
    nodes = {n.id: n for n in sc.graph.nodes}
    edges = {e.id: e for e in sc.graph.edges}
    ys, xs = (np.arange(shape[0]) + 0.5) * RES, (np.arange(shape[1]) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    mask = np.zeros(shape, dtype=bool)
    for c in sc.compromised:
        e = edges[c.edge]
        a, b = nodes[e.a], nodes[e.b]
        L = math.dist((a.x, a.y), (b.x, b.y))
        ux, uy = (b.x - a.x) / L, (b.y - a.y) / L
        px, py = a.x + (b.x - a.x) * c.at, a.y + (b.y - a.y) * c.at
        along = (X - px) * ux + (Y - py) * uy
        across = -(X - px) * uy + (Y - py) * ux
        mask |= (np.abs(along) <= 0.5) & (np.abs(across) <= e.width / 2 + 0.5)
    return mask


def validate(sc: Scenario, free: np.ndarray) -> list[str]:
    errors: list[str] = []
    nodes = {n.id: n for n in sc.graph.nodes}
    entrance = next(n for n in sc.graph.nodes if n.kind == "entrance")
    survivor_node = min(sc.graph.nodes, key=lambda n: math.dist((n.x, n.y), (sc.survivor.x, sc.survivor.y)))

    # All nodes and placements on free cells; edges run through free space.
    pts = [(f"node {n.id}", n.x, n.y) for n in sc.graph.nodes]
    pts += [("survivor", sc.survivor.x, sc.survivor.y), ("entrance", sc.entrance.x, sc.entrance.y)]
    pts += [(f"fire {i}", f.x, f.y) for i, f in enumerate(sc.fires)]
    pts += [(f"drone start {i}", x, y) for i, (x, y) in enumerate(sc.drones.start)]
    if sc.decoy:
        pts.append(("decoy", sc.decoy.x, sc.decoy.y))
    errors += [f"{name} at ({x}, {y}) is not on a free cell" for name, x, y in pts if not _is_free(free, x, y)]
    for e in sc.graph.edges:
        a, b = nodes[e.a], nodes[e.b]
        for k in range(51):
            t = k / 50
            if not _is_free(free, a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t):
                errors.append(f"edge {e.id} leaves free space")
                break

    # >= 3 branches from the entrance region: the choices a drone has at the entrance and at
    # the junction(s) next to it (the corridor joining entrance -> junction is not a choice).
    near = {e.b if e.a == entrance.id else e.a for e in sc.graph.edges if entrance.id in (e.a, e.b)}
    near_junctions = {n for n in near if nodes[n].kind == "junction"}
    region = {entrance.id} | near_junctions
    approach = {e.id for e in sc.graph.edges if {e.a, e.b} <= region}
    branches = {e.id for e in sc.graph.edges if (e.a in region or e.b in region) and e.id not in approach}
    if len(branches) < 3:
        errors.append(f"only {len(branches)} branches from the entrance region {sorted(branches)} (need >= 3)")

    # Naive shortest route uses a compromised edge; survivor reachable once they're applied.
    compromised = {c.edge for c in sc.compromised}
    naive = _shortest(sc, entrance.id, survivor_node.id, banned=set())
    if naive is None:
        errors.append("survivor not reachable on the open graph")
    elif not compromised & set(naive):
        errors.append(f"naive shortest route {naive} uses no compromised edge")
    final = _shortest(sc, entrance.id, survivor_node.id, banned=compromised)
    if final is None:
        errors.append("survivor unreachable on the graph after compromised edges")
    if not _flood_reaches(free & ~_debris_mask(sc, free.shape), (sc.entrance.x, sc.entrance.y), (sc.survivor.x, sc.survivor.y)):
        errors.append("survivor unreachable in the grid after debris is placed")

    # At least one wrong-way branch next to the final route.
    if final:
        route_edges = set(final)
        route_nodes = {nid for e in sc.graph.edges if e.id in route_edges for nid in (e.a, e.b)}
        side = [e.id for e in sc.graph.edges if e.id not in route_edges and (e.a in route_nodes or e.b in route_nodes)]
        if not side:
            errors.append("no wrong-way branch next to the final route")
    return errors


def main() -> int:
    sc = build_scenario()
    free = draw_grid(sc)
    errors = validate(sc, free)
    if errors:
        print("make_test_map: scenario requirements FAILED:")
        for e in errors:
            print("  -", e)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    save_png(free, OUT / "grid.png")
    (OUT / "scenario.json").write_text(sc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"make_test_map: OK -> {OUT / 'grid.png'} ({free.shape[1]}x{free.shape[0]} px), {OUT / 'scenario.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
