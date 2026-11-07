"""Search areas (spec §8.2): a Voronoi split of free cells by route-graph nodes, and greedy assignment.

The split is geodesic (multi-source BFS over free cells), so an area never leaks through a wall
into the room next door. The graph is public structure; which edges are compromised is not.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sgr.contracts import Graph

from .belief import BeliefGrid

CLAIM_MIN_UNSEARCHED = 0.20  # §8.2: claim only areas with > 20 % unsearched
MOPUP_MIN_CELLS = 6  # fallback: any area with a few unsearched cells left (fewer = not worth a trip)


@dataclass
class Area:
    id: str  # = graph node id
    mask: np.ndarray  # bool [j, i]
    centre: tuple[float, float]  # node position
    n_cells: int

    def unsearched(self, belief: BeliefGrid) -> np.ndarray:
        return self.mask & ~belief.searched

    def unsearched_fraction(self, belief: BeliefGrid) -> float:
        return float(self.unsearched(belief).sum()) / max(1, self.n_cells)


def split_areas(belief: BeliefGrid, graph: Graph) -> dict[str, Area]:
    free = belief.reachable
    label = np.full(free.shape, -1, np.int32)
    q: deque[tuple[int, int]] = deque()
    nodes = list(graph.nodes)
    for k, n in enumerate(nodes):
        i, j = belief.world_to_cell(n.x, n.y)
        if belief.inside(i, j) and free[j, i] and label[j, i] < 0:
            label[j, i] = k
            q.append((i, j))
    while q:
        i, j = q.popleft()
        lab = label[j, i]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < belief.w and 0 <= b < belief.h and free[b, a] and label[b, a] < 0:
                label[b, a] = lab
                q.append((a, b))
    areas = {}
    for k, n in enumerate(nodes):
        m = label == k
        if m.any():
            areas[n.id] = Area(id=n.id, mask=m, centre=(n.x, n.y), n_cells=int(m.sum()))
    return areas


def assign(
    idle: list[str],
    areas: dict[str, Area],
    claims: dict[str, str],
    belief: BeliefGrid,
    dist: Callable[[str, Area], float],
    excluded: dict[str, set[str]] | None = None,
    helping: set[str] | None = None,
) -> dict[str, tuple[str, float, float]]:
    """Greedy assignment. `claims` maps area id -> drone (updated in place).

    Returns {drone: (area id, distance, unsearched fraction)} for the new claims.
    `excluded[drone]` lists areas that drone gave up on (no reachable unsearched cell).
    `helping` lists areas that already have a mop-up helper.
    """
    excluded = excluded or {}
    out: dict[str, tuple[str, float, float]] = {}
    pending = list(idle)
    # global greedy: repeatedly take the closest (drone, area) pair
    while pending:
        best = None
        for d in pending:
            for a in areas.values():
                if a.id in claims or a.id in excluded.get(d, set()):
                    continue
                frac = a.unsearched_fraction(belief)
                if frac <= CLAIM_MIN_UNSEARCHED:
                    continue
                dd = dist(d, a)
                if np.isfinite(dd) and (best is None or dd < best[0]):
                    best = (dd, d, a.id, frac)
        if best is None:
            break
        dd, d, aid, frac = best
        claims[aid] = d
        out[d] = (aid, dd, frac)
        pending.remove(d)
    # mop-up: leftover drones help in the nearest area that still has unsearched cells
    # (may be claimed by someone else, but at most one helper per area per round)
    helped: set[str] = set(helping or ())
    for d in pending:
        best = None
        for a in areas.values():
            if a.id in excluded.get(d, set()) or a.id in helped:
                continue
            left = int(a.unsearched(belief).sum())
            if left < MOPUP_MIN_CELLS:
                continue
            dd = dist(d, a)
            if np.isfinite(dd) and (best is None or dd < best[0]):
                best = (dd, a.id, a.unsearched_fraction(belief))
        if best is not None:
            out[d] = (best[1], best[0], best[2])
            helped.add(best[1])
    return out


def release(claims: dict[str, str], drone: str) -> list[str]:
    """Return a drone's claims to the pool."""
    mine = [a for a, d in claims.items() if d == drone]
    for a in mine:
        del claims[a]
    return mine
