"""W's procedural rubble pile (web/src/scene3d/dressing/rubble.tsx), ported for the MuJoCo drone cameras.

Same recipe: a faceted dusty mound (~1.2 m across, ~0.75 m high) with broken concrete chunks
and slabs resting on it, a beam leaning onto the top, rebar poking out and gravel spilling
around, in W's brown / brick colours. W bakes per-vertex colours; MuJoCo can't, so each pile
is one mesh per colour group and MuJoCo's lights do the shading. W's mulberry32 rng and noise
are ported exactly, so the same seed gives a similar pile (not vertex-identical: three.js
geometry internals differ).
"""

from __future__ import annotations

import functools
import math

import numpy as np

# W's colours (rubble.tsx)
COLOURS = {
    "concrete": (0x85 / 255, 0x70 / 255, 0x60 / 255),
    "concrete_light": (0x9D / 255, 0x86 / 255, 0x70 / 255),
    "brick": (0x8F / 255, 0x4F / 255, 0x35 / 255),
    "wood": (0x6E / 255, 0x4A / 255, 0x2C / 255),
    "steel": (0x5C / 255, 0x4A / 255, 0x3C / 255),
    "rust": (0x7A / 255, 0x3F / 255, 0x22 / 255),
}
MOUND_R = 0.62
MOUND_H = 0.62
N_VARIANTS = 6  # distinct piles shared by the 40-body debris pool
_M32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return (a * b) & _M32


def rng(seed: int):
    """mulberry32, exactly as hazards.ts rng()."""
    state = [seed & _M32]

    def r() -> float:
        state[0] = (state[0] + 0x6D2B79F5) & _M32
        t = state[0]
        t = _imul(t ^ (t >> 15), t | 1)
        t ^= (t + _imul(t ^ (t >> 7), t | 61)) & _M32
        return ((t ^ (t >> 14)) & _M32) / 4294967296

    return r


def _hash3(x, y, z):
    h = math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453
    return h - math.floor(h)


def _vnoise(x, y, z):
    xi, yi, zi = math.floor(x), math.floor(y), math.floor(z)
    xf, yf, zf = x - xi, y - yi, z - zi
    u, v, w = xf * xf * (3 - 2 * xf), yf * yf * (3 - 2 * yf), zf * zf * (3 - 2 * zf)

    def l(a, b, t):
        return a + (b - a) * t

    def c(dx, dy, dz):
        return _hash3(xi + dx, yi + dy, zi + dz)

    return l(l(l(c(0, 0, 0), c(1, 0, 0), u), l(c(0, 1, 0), c(1, 1, 0), u), v),
             l(l(c(0, 0, 1), c(1, 0, 1), u), l(c(0, 1, 1), c(1, 1, 1), u), v), w)


@functools.lru_cache(maxsize=2)
def _unit_icosphere(detail: int):
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=detail, radius=1.0) if detail else trimesh.creation.icosahedron()
    v = np.asarray(m.vertices, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True), np.asarray(m.faces, int)


def _icosphere(r: float, detail: int):
    v, f = _unit_icosphere(detail)
    return v * r, f


def _rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # three.js Euler 'XYZ'


def _between(p0, p1):
    """Rotation + centre that place geometry built along +x from p0 to p1."""
    d = np.asarray(p1, float) - np.asarray(p0, float)
    L = float(np.linalg.norm(d))
    x = d / L
    a = np.array([0, 0, 1.0]) if abs(x[2]) < 0.9 else np.array([0, 1.0, 0])
    y = np.cross(a, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1), (np.asarray(p0) + np.asarray(p1)) / 2, L


def _box(lx, ly, lz):
    v = np.array([[sx * lx / 2, sy * ly / 2, sz * lz / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    f = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
                  [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
    return v, f


def _cylinder_x(r, length, n=5):
    ang = np.arange(n) * 2 * math.pi / n
    ring = np.stack([np.zeros(n), r * np.cos(ang), r * np.sin(ang)], axis=1)
    v = np.vstack([ring + [-length / 2, 0, 0], ring + [length / 2, 0, 0]])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + i], [j, n + j, n + i]]
    f += [[0, i + 1, i + 2] for i in range(n - 2)] + [[n, n + i + 2, n + i + 1] for i in range(n - 2)]
    return v, np.array(f)


def _mound(rand):
    rings = 5
    seg = 11
    verts = [[0.0, 0.0, MOUND_H]]
    for j in range(1, rings + 1):
        th = j / rings * math.pi / 2
        for i in range(seg):
            ph = i / seg * 2 * math.pi
            verts.append([MOUND_R * math.sin(th) * math.cos(ph), MOUND_R * math.sin(th) * math.sin(ph),
                          MOUND_H * math.cos(th)])
    v = np.array(verts)
    for k in range(len(v)):
        s = 0.85 + rand() * 0.3
        z = v[k, 2]
        v[k] = [v[k, 0] * s, v[k, 1] * s, z * (0.8 + rand() * 0.35) if z > 0.01 else 0.0]
    f = [[0, 1 + i, 1 + (i + 1) % seg] for i in range(seg)]
    for j in range(rings - 1):
        a0, b0 = 1 + j * seg, 1 + (j + 1) * seg
        for i in range(seg):
            i1 = (i + 1) % seg
            f += [[a0 + i, b0 + i, b0 + i1], [a0 + i, b0 + i1, a0 + i1]]
    return v, np.array(f)


def _mound_z(d):
    return MOUND_H * math.sqrt(max(0.0, 1 - (d / MOUND_R) ** 2))


def _chunk(rand, r):
    v, f = _icosphere(r, 1)
    o = rand() * 100
    sx, sy, sz = 0.75 + rand() * 0.55, 0.7 + rand() * 0.5, 0.4 + rand() * 0.35
    out = np.empty_like(v)
    for i, (x, y, z) in enumerate(v):
        d = 0.62 + 0.6 * _vnoise(x / r * 1.4 + o, y / r * 1.4, z / r * 1.4) + 0.25 * (_hash3(x * 97 + o, y * 89, z * 83) - 0.5)
        out[i] = [x * d * sx, y * d * sy, z * d * sz]
    return out, f


def _slab(rand, w, h, t):
    v, f = _box(w, h, t)
    jit = {}
    for i in range(len(v)):
        key = (round(v[i, 0], 4), round(v[i, 1], 4))
        if key not in jit:
            jit[key] = ((rand() - 0.5) * w * 0.35, (rand() - 0.5) * h * 0.35)
        jx, jy = jit[key]
        v[i] = [v[i, 0] + jx, v[i, 1] + jy, v[i, 2] + (rand() - 0.5) * t * 0.4]
    return v, f


CACHE_VERSION = 1


@functools.lru_cache(maxsize=None)
def pile(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{colour group: (vertices (N,3), faces (M,3))}, faceted. Cached on disk (deterministic per seed)."""
    import tempfile
    from pathlib import Path

    cache = Path(tempfile.gettempdir()) / f"sgr_rubble_v{CACHE_VERSION}_s{seed}.npz"
    if cache.is_file():
        try:
            z = np.load(cache)
            return {k[2:]: (z[k], z["f_" + k[2:]]) for k in z.files if k.startswith("v_")}
        except Exception:
            pass
    out = _build_pile(seed)
    try:
        np.savez(cache, **{f"v_{k}": v for k, (v, _f) in out.items()}, **{f"f_{k}": f for k, (_v, f) in out.items()})
    except OSError:
        pass
    return out


def _build_pile(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rand = rng(seed)
    groups: dict[str, list] = {k: [] for k in COLOURS}

    def add(colour, v, f, R=None, tpos=(0.0, 0.0, 0.0)):
        rand(), rand(), rand()  # W's per-part colour jitter (keeps the rng sequence aligned)
        v = np.asarray(v, float)
        if R is not None:
            v = v @ R.T
        groups[colour].append((v + np.asarray(tpos, float), np.asarray(f, int)))

    add("concrete", *_mound(rand))
    for _ in range(26):  # chunks half-buried in the mound surface
        r = 0.06 + rand() * rand() * 0.2
        a = rand() * math.pi * 2
        d = math.sqrt(rand()) * (MOUND_R + 0.15)
        z = _mound_z(d) + r * 0.25
        col = "brick" if rand() < 0.08 else "concrete_light" if rand() < 0.55 else "concrete"
        v, f = _chunk(rand, r)
        add(col, v, f, _rot(rand() * 3, rand() * 3, rand() * 3), (math.cos(a) * d, math.sin(a) * d, z))
    for i in range(4):  # broken slabs on the flank, tilted with the slope
        a = (i / 4) * math.pi * 2 + rand()
        d = 0.3 + rand() * 0.15
        v, f = _slab(rand, 0.42 + rand() * 0.3, 0.3 + rand() * 0.2, 0.06)
        col = "brick" if (i == 0 and rand() < 0.3) else "concrete_light"
        slope = math.atan2(_mound_z(d - 0.05) - _mound_z(d + 0.05), 0.1)
        add(col, v, f, _rot(0, -slope, 0) if False else (_rot(0, 0, a) @ _rot(0, -slope, 0)),
            (math.cos(a) * d, math.sin(a) * d, _mound_z(d) + 0.04))
    a = rand() * math.pi * 2  # a beam from the floor onto the top
    p0 = (math.cos(a) * 0.95, math.sin(a) * 0.95, 0.06)
    p1 = (-math.cos(a) * 0.2, -math.sin(a) * 0.2, MOUND_H + 0.35 + rand() * 0.3)
    R, c, L = _between(p0, p1)
    v, f = _box(L + 0.2, 0.11, 0.11)
    add("wood" if rand() < 0.5 else "steel", v, f, R, c)
    for _ in range(5):  # rebar poking out
        a = rand() * math.pi * 2
        d = rand() * 0.45
        ln = 0.35 + rand() * 0.45
        base = np.array([math.cos(a) * d, math.sin(a) * d, _mound_z(d) - 0.05])
        tip = base + [math.cos(a) * ln * 0.6, math.sin(a) * ln * 0.6, ln * (0.5 + rand() * 0.5)]
        R, c, L = _between(base, tip)
        add("rust", *_cylinder_x(0.011, L), R, c)
    for _ in range(70):  # gravel spilling around, thinning out
        r = 0.018 + rand() * rand() * 0.05
        a = rand() * math.pi * 2
        d = MOUND_R * (0.75 + rand() ** 1.6 * 1.0)
        col = "brick" if rand() < 0.1 else "concrete_light" if rand() < 0.5 else "concrete"
        v, f = _icosphere(r, 0)
        add(col, v, f, _rot(rand() * 3, rand() * 3, rand() * 3), (math.cos(a) * d, math.sin(a) * d, r * 0.35))

    out = {}
    for col, parts in groups.items():
        if not parts:
            continue
        vs, fs = [], []
        n = 0
        for v, f in parts:  # faceted: give every face its own 3 vertices
            tri = v[f].reshape(-1, 3)
            vs.append(tri)
            fs.append(np.arange(len(tri)).reshape(-1, 3) + n)
            n += len(tri)
        out[col] = (np.vstack(vs), np.vstack(fs))
    return out
