"""Polyline geometry for guidance (pure functions, metres / radians)."""

from __future__ import annotations

import math
from dataclasses import dataclass

Pt = tuple[float, float]


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Projection:
    s: float  # arc length along the polyline to the closest point
    dist: float  # distance from the point to the polyline
    closest: Pt
    seg: int  # index of the segment containing the closest point
    side: float  # > 0: the point is LEFT of the direction of travel, < 0: right


def lengths(poly: list[Pt]) -> list[float]:
    """Cumulative arc length at each vertex."""
    out = [0.0]
    for a, b in zip(poly, poly[1:]):
        out.append(out[-1] + math.dist(a, b))
    return out


def project(poly: list[Pt], p: Pt) -> Projection | None:
    if not poly:
        return None
    if len(poly) == 1:
        return Projection(0.0, math.dist(poly[0], p), poly[0], 0, 0.0)
    cum = lengths(poly)
    best = None
    for i, (a, b) in enumerate(zip(poly, poly[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L2 = dx * dx + dy * dy
        f = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
        c = (a[0] + f * dx, a[1] + f * dy)
        d = math.dist(c, p)
        if best is None or d < best.dist - 1e-9:
            side = dx * (p[1] - a[1]) - dy * (p[0] - a[0])
            best = Projection(cum[i] + f * math.sqrt(L2), d, c, i, side)
    return best


def point_at(poly: list[Pt], s: float) -> Pt:
    """Point at arc length s (clamped to the ends)."""
    if not poly:
        raise ValueError("empty polyline")
    cum = lengths(poly)
    if s <= 0:
        return poly[0]
    for i in range(len(poly) - 1):
        if s <= cum[i + 1]:
            seg = cum[i + 1] - cum[i]
            f = 0.0 if seg == 0 else (s - cum[i]) / seg
            a, b = poly[i], poly[i + 1]
            return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
    return poly[-1]


def seg_dist(p: Pt, a: Pt, b: Pt) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    f = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.dist(p, (a[0] + f * dx, a[1] + f * dy))


def bearing(frm: Pt, to: Pt) -> float:
    return math.atan2(to[1] - frm[1], to[0] - frm[0])
