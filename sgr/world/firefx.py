"""Realistic fire in the drone cameras (W).

MuJoCo has no additive blending or soft point sprites, so the in-sim fire was an approximation made of
ellipsoids. This draws W's 3D-view fire (web/src/scene3d/dressing/fire.tsx) on top of the MuJoCo frame
instead, with the same maths and the same seeds:
  scorch + orange floor pool (floor pixels, from the depth buffer), smoke puffs (alpha), flame
  particles (additive, softDot sprites), a halo sprite and sparkles.
Occlusion uses MuJoCo's depth buffer per pixel (depthTest on, depthWrite off, like three.js), so a fire
behind a wall stays hidden. Pure numpy; a few ms per frame with a fire in view, ~0 without.

The default (sgr/world/shapes.py FIRE_LOOK; SGR_FIRE_LOOK=flames brings back the old look). The World keeps its own
fire geoms (animated, ray-hit, so fire positions are unchanged) but makes them transparent while a drone
camera renders. Perception gets the composited frame, so the fast detector and Gemma see this fire.
"""

from __future__ import annotations

import functools
import math

import os

import cv2
import numpy as np

from . import shapes
from .rubble import rng

# fire.tsx constants (s = max(0.6, size))
FLAME_N, FLAME_SPEED = 140, (0.9, 1.6)
SMOKE_N, SMOKE_SPEED = 40, (0.06, 0.11)
SMOKE_H, SMOKE_CEIL, SMOKE_OPACITY = 5.5, 2.15, 0.5
SPARK_N, SPARK_SPEED = 24, (0.8, 1.6)
FLAME_CAP_PX = 40.0  # sprite radius cap (a drone right in the fire would otherwise splat huge sprites)
SMOKE_CAP_PX = 40.0  # at half resolution
MAX_RANGE = 18.0  # m: farther fires are a few pixels; skip them
FLOOR_Z = 0.06  # pixels whose world z is below this are floor (scorch, firelight pool)
# Drone-camera tuning (1.0 = exactly fire.tsx). 1.35 keeps the fast detector at the old fire's hit rate (thin flames fall under its 0.5 % area).
SIZE_K = float(os.environ.get("SGR_FIREFX_SIZE", "1.35"))
GAIN = float(os.environ.get("SGR_FIREFX_GAIN", "1.0"))  # flame brightness (additive)


def _js_round(v: float) -> int:
    return math.floor(v + 0.5)


@functools.lru_cache(maxsize=64)
def _seeds(n: int, seed: int, lo: float, hi: float) -> np.ndarray:
    """fire.tsx seeds(): (phase, angle, sqrt(radius), speed) per particle, mulberry32 like the web."""
    r = rng(seed)
    a = np.empty((n, 4))
    for i in range(n):
        a[i, 0] = r()
        a[i, 1] = r()
        a[i, 2] = math.sqrt(r())
        a[i, 3] = lo + r() * (hi - lo)
    return a


def _smooth(e0: float, e1: float, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _ramp(d, stops):
    """Canvas radial gradient alpha at normalised radius d (0 centre, 1 edge)."""
    xs, ys = zip(*stops)
    return np.interp(d, xs, ys, right=0.0)


SOFTDOT = ((0, 1.0), (0.35, 0.6), (1, 0.0))
GLOW = ((0, 1.0), (0.2, 0.6), (0.5, 0.18), (1, 0.0))


class FireCompositor:
    flame_depth: np.ndarray | None = None  # after apply(): nearest flame depth per pixel (inf = no flame)
    flame_cover: np.ndarray | None = None  # after apply(): summed flame sprite weight per pixel (0 = none)

    def __init__(self, w: int = shapes.CAM_W, h: int = shapes.CAM_H) -> None:
        self.w, self.h = w, h
        self.fx, self.fy, self.cx, self.cy = shapes.cam_intrinsics(w, h)
        u = (np.arange(w) + 0.5 - self.cx) / self.fx
        v = -(np.arange(h) + 0.5 - self.cy) / self.fy
        self._ray = np.stack(np.broadcast_arrays(u[None, :], v[:, None], -np.ones((h, w))), -1)  # camera frame

    def warm(self, fires) -> None:
        """Pre-compute the particle seeds of these fires (call at startup, not on the first fire frame)."""
        for x, y, size in fires:
            s, seed = max(0.6, float(size)) * SIZE_K, _js_round(x * 131 + y * 71)
            _seeds(_js_round(FLAME_N * s), seed, *FLAME_SPEED)
            _seeds(_js_round(SMOKE_N * s), seed + 1, *SMOKE_SPEED)
            _seeds(_js_round(SPARK_N * s), seed + 2, *SPARK_SPEED)
        if fires:  # one synthetic pass: first calls into cv2 / numpy happen now, not on the first fire frame
            x, y, size = fires[0]
            cam = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 3 m west of it, facing east
            self.apply(np.zeros((self.h, self.w, 3), np.uint8), np.full((self.h, self.w), 30.0, np.float32),
                       (x - 3.0, y, 1.6), cam, [(x, y, size)], 0.0)

    # ------------------------------------------------------------------ public
    def apply(self, frame: np.ndarray, depth: np.ndarray, cam_pos, cam_mat, fires, t: float) -> np.ndarray:
        """frame: HxWx3 uint8; depth: HxW metres along the view axis; cam_pos (3,), cam_mat (3, 3) MuJoCo
        camera frame (x right, y up, looking down -z); fires: [(x, y, size)]; t: seconds.
        Each layer is drawn for all fires at once (fire.tsx order: floor, smoke, halo, flames + sparks)."""
        pos = np.asarray(cam_pos, float)
        rot = np.asarray(cam_mat, float).reshape(3, 3)
        todo = []
        for x, y, size in fires:
            c = rot.T @ (np.array([x, y, 1.0]) - pos)
            if math.hypot(x - pos[0], y - pos[1]) > MAX_RANGE or -c[2] < -3.0:  # far, or well behind the camera
                continue
            s = max(0.6, float(size)) * SIZE_K
            seed = _js_round(x * 131 + y * 71)
            f = 0.72 + 0.18 * math.sin(t * 11 + seed) * math.sin(t * 6.7) + 0.1 * math.sin(t * 23)
            todo.append((x, y, s, seed, f))
        self.flame_depth = self.flame_cover = None
        if not todo:
            return frame
        self.flame_depth = np.full(depth.shape, np.inf, np.float32)
        self.flame_cover = np.zeros(depth.shape, np.float32)
        img = frame.astype(np.float32) / 255.0
        self._floor(img, depth, pos, rot, todo)
        self._smoke(img, depth, pos, rot, todo, t)
        for x, y, s, _, f in todo:
            self._halo(img, depth, pos, rot, x, y, s, f)
        self._glow(img, depth, pos, rot, todo, t)
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)

    # ------------------------------------------------------------------ pieces
    def _project(self, pts: np.ndarray, pos, rot):
        """World points (N, 3) -> pixel u, v and view depth (positive in front)."""
        c = (pts - pos) @ rot
        z = -c[:, 2]
        zs = np.maximum(z, 1e-3)
        return self.cx + self.fx * c[:, 0] / zs, self.cy - self.fy * c[:, 1] / zs, z

    def _floor(self, img, depth, pos, rot, todo) -> None:
        """Scorch + firelight pool on floor pixels only (world position from the depth buffer)."""
        rx, ry = self._ray[..., 0], self._ray[..., 1]  # camera-frame ray (x, y, -1)
        wz = pos[2] + depth * (rx * rot[2, 0] + ry * rot[2, 1] - rot[2, 2])
        idx = np.flatnonzero(wz < FLOOR_Z)
        if not len(idx):
            return
        dd, ux, uy = depth.reshape(-1)[idx], rx.reshape(-1)[idx], ry.reshape(-1)[idx]
        wx = pos[0] + dd * (ux * rot[0, 0] + uy * rot[0, 1] - rot[0, 2])
        wy = pos[1] + dd * (ux * rot[1, 0] + uy * rot[1, 1] - rot[1, 2])
        flat = img.reshape(-1, 3)
        px = flat[idx]
        for x, y, s, _, f in todo:
            r = np.hypot(wx - x, wy - y)
            near = r < 3.25 * s
            if not near.any():
                continue
            rn = r[near]
            a = 0.3 * _ramp(rn / (1.4 * s), ((0, 1.0), (0.6, 0.4), (1, 0.0)))  # scorch, 2.8 s across
            q = px[near] * (1 - a[:, None]) + np.array([0.03, 0.022, 0.016]) * a[:, None]
            a = 0.55 * f * _ramp(rn / (3.25 * s), GLOW)  # firelight pool: glow(), #ff6410, 0.55 f, 6.5 s across
            px[near] = q * (1 - a[:, None]) + np.array([1.0, 0.392, 0.063]) * a[:, None]
        flat[idx] = px

    def _splat(self, depth, u, v, z, rad, alpha, profile, cap: float):
        """Many round sprites at once, depth-tested. Sprites are bucketed by integer radius so each bucket is
        one numpy pass. Returns (flat pixel index, weight, sprite index) per covered pixel."""
        h, w = depth.shape
        rad = np.minimum(rad, cap)
        keep = ((z > 0.05) & (alpha > 0.004) & (rad >= 0.5) & (u + rad >= 0) & (u - rad < w)
                & (v + rad >= 0) & (v - rad < h))
        out_f, out_a, out_i = [], [], []
        rb = np.ceil(rad).astype(int)
        for r in np.unique(rb[keep]):
            sel = np.flatnonzero(keep & (rb == r))
            o = np.arange(-r, r + 1)
            px = np.floor(u[sel]).astype(int)[:, None, None] + o[None, None, :]
            py = np.floor(v[sel]).astype(int)[:, None, None] + o[None, :, None]
            d = np.hypot(px + 0.5 - u[sel, None, None], py + 0.5 - v[sel, None, None]) / rad[sel, None, None]
            inside = (d < 1) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
            pxc, pyc = np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)
            inside &= depth[pyc, pxc] > z[sel, None, None]
            ii = np.broadcast_to(sel[:, None, None], d.shape)[inside]
            out_f.append((pyc * w + pxc)[inside])
            out_a.append(profile(d[inside]) * alpha[ii])
            out_i.append(ii)
        if not out_f:
            return None
        return np.concatenate(out_f), np.concatenate(out_a), np.concatenate(out_i)

    def _add(self, img, depth, u, v, z, rad, rgb, alpha, profile, cap, zbuf=None, cover=None) -> None:
        """Additive sprites (order-free): sum every sprite's colour into the image. zbuf / cover (flat,
        optional): nearest sprite depth and summed sprite weight per pixel (the dashboard fog uses them)."""
        hit = self._splat(depth, u, v, z, rad, alpha, profile, cap)
        if hit is None:
            return
        f, a, i = hit
        if zbuf is not None:
            vis = a > 0.001
            np.minimum.at(zbuf, f[vis], z[i[vis]].astype(np.float32))
        if cover is not None:
            cover += np.bincount(f, weights=a, minlength=depth.size).astype(np.float32)
        flat, n = img.reshape(-1, 3), depth.size
        for ch in range(3):
            flat[:, ch] += np.bincount(f, weights=rgb[i, ch] * a, minlength=n)

    def _sprites(self, img, depth, u, v, z, size_px, rgb, alpha, additive: bool, profile) -> None:
        """One big billboard at a time (the halo: one per fire), alpha-blended, depth-tested."""
        h, w = depth.shape
        for i in np.argsort(-z):
            if z[i] <= 0.05 or alpha[i] <= 0.004:
                continue
            rad = 0.5 * size_px[i]
            if rad < 0.5:
                continue
            x0, x1 = int(max(0, math.floor(u[i] - rad))), int(min(w, math.ceil(u[i] + rad)))
            y0, y1 = int(max(0, math.floor(v[i] - rad))), int(min(h, math.ceil(v[i] + rad)))
            if x0 >= x1 or y0 >= y1:
                continue
            xs = np.arange(x0, x1) + 0.5 - u[i]
            ys = np.arange(y0, y1) + 0.5 - v[i]
            d = np.hypot(xs[None, :], ys[:, None]) / rad
            a = profile(d) * alpha[i] * (depth[y0:y1, x0:x1] > z[i])
            patch = img[y0:y1, x0:x1]
            patch[:] = patch * (1 - a[..., None]) + rgb[i] * a[..., None]

    def _flame_sprites(self, pos, rot, x, y, s, seed, t):
        sd = _seeds(_js_round(FLAME_N * s), seed, *FLAME_SPEED)
        ph, an, rf, sp = sd.T
        height, radius, part = 0.75 * s + 0.4, 0.32 * s, 0.22 * s
        life = np.mod(t * sp + ph, 1.0)
        ang = an * 2 * np.pi + t * 1.3
        r = rf * radius * (1 + (0.35 - 1) * life)  # mix(1, uSpread, life)
        sway = np.sin(t * 3.1 + ph * 23) * 0.07 * life
        pts = np.stack([x + np.cos(ang) * r + sway, y + np.sin(ang) * r, life * height], 1)
        u, v, z = self._project(pts, pos, rot)
        rad = 0.5 * self.fy * part * (1 + (0.55 - 1) * life) / np.maximum(0.1, z)
        a = _smooth(0, 0.35, life)[:, None]
        c = np.array([1.0, 0.82, 0.45]) * (1 - a) + np.array([1.0, 0.42, 0.06]) * a
        b = _smooth(0.45, 0.9, life)[:, None]
        c = c * (1 - b) + np.array([0.75, 0.12, 0.02]) * b
        alpha = _smooth(0, 0.06, life) * (1 - _smooth(0.55, 1.0, life))
        # three.js AdditiveBlending (SRC_ALPHA, ONE) of gl_FragColor = (c * a * 0.75, a): adds c * 0.75 * a^2,
        # with a = tex * alpha: rgb = c * 0.75 * alpha, sprite alpha = alpha, profile = tex^2.
        return u, v, z, rad, c * 0.75 * GAIN * alpha[:, None], alpha

    def _spark_sprites(self, pos, rot, x, y, s, seed, t):
        sd = _seeds(_js_round(SPARK_N * s), seed + 2, *SPARK_SPEED)
        ph, an, rf, sp = sd.T
        k = np.arange(len(sd))
        jx = np.sin(t * sp * 1.2 + ph * 40) * 0.15
        jy = np.cos(t * sp * 1.1 + an * 40) * 0.15
        jz = np.sin(t * sp * 0.9 + k) * 0.2
        pts = np.stack([x + (ph - 0.5) * 1.2 * s + jx, y + (an - 0.5) * 1.2 * s + jy, 1.3 + (rf - 0.5) * 2.4 + jz], 1)
        u, v, z = self._project(pts, pos, rot)
        rad = 0.5 * np.clip(self.fy * 0.05 / np.maximum(0.1, z), 1.0, 4.0)
        tw = 0.5 + 0.5 * np.sin(t * 6 * sp + ph * 30)
        return u, v, z, rad, np.array([1.0, 0.604, 0.235])[None, :] * (0.9 * tw)[:, None], np.ones(len(sd))

    def _glow(self, img, depth, pos, rot, todo, t) -> None:
        """Flames and sparks of every fire in one additive pass (sparks: softDot, flames: softDot^2)."""
        fl = [self._flame_sprites(pos, rot, x, y, s, seed, t) for x, y, s, seed, _ in todo]
        u, v, z, rad, rgb, al = (np.concatenate(k) for k in zip(*fl))
        self._add(img, depth, u, v, z, rad, rgb, al, lambda d: _ramp(d, SOFTDOT) ** 2, FLAME_CAP_PX,
                  zbuf=self.flame_depth.reshape(-1), cover=self.flame_cover.reshape(-1))
        sk = [self._spark_sprites(pos, rot, x, y, s, seed, t) for x, y, s, seed, _ in todo]
        u, v, z, rad, rgb, al = (np.concatenate(k) for k in zip(*sk))
        self._add(img, depth, u, v, z, rad, rgb, al, lambda d: _ramp(d, SOFTDOT), FLAME_CAP_PX)

    def _smoke(self, img, depth, pos, rot, todo, t) -> None:
        """Smoke puffs of every fire, drawn at half resolution with an order-independent blend
        (transmittance = prod(1 - a), colour = alpha-weighted mean): smoke is soft and low-alpha, so this
        matches three.js's sorted blending closely at a fraction of the cost."""
        us, vs, zs, rads, cs, als = [], [], [], [], [], []
        for x, y, s, seed, _ in todo:
            sd = _seeds(_js_round(SMOKE_N * s), seed + 1, *SMOKE_SPEED)
            ph, an, rf, sp = sd.T
            radius, part = 0.3 * s, 0.9 * s
            life = np.mod(t * sp + ph, 1.0)
            rise = life * SMOKE_H
            over = np.maximum(0.0, rise - SMOKE_CEIL)
            ang = an * 2 * np.pi + t * 0.25
            r = rf * radius * (0.4 + life) + over * (0.8 + rf)
            zw = np.minimum(rise, SMOKE_CEIL) - rf * 0.25 * (over > 0.001)
            pts = np.stack([x + np.cos(ang) * r, y + np.sin(ang) * r, zw], 1)
            u, v, z = self._project(pts, pos, rot)
            k = _smooth(0, 0.4, life)[:, None]
            us.append(u); vs.append(v); zs.append(z)
            rads.append(0.5 * self.fy * part * (0.5 + 1.5 * life) / np.maximum(0.1, z))
            cs.append(np.array([0.36, 0.27, 0.2]) * (1 - k) + np.array([0.17, 0.16, 0.16]) * k)
            als.append(SMOKE_OPACITY * _smooth(0, 0.15, life) * (1 - _smooth(0.6, 1.0, life)))
        u, v, z, rad, c, al = (np.concatenate(k) for k in (us, vs, zs, rads, cs, als))
        half = depth[::2, ::2]
        # smokePuff: soft blobs (peak ~0.35), cut to a circle by smoothstep(0.5, 0.25, |pc - 0.5|)
        hit = self._splat(half, u / 2, v / 2, z, rad / 2, al,
                          lambda d: 0.35 * np.clip(1 - d, 0, 1) * _smooth(1.0, 0.5, d), SMOKE_CAP_PX)
        if hit is None:
            return
        f, a, i = hit
        n = half.size
        a = np.minimum(a, 0.95)
        trans = np.exp(np.bincount(f, weights=np.log1p(-a), minlength=n)).reshape(half.shape)
        wsum = np.bincount(f, weights=a, minlength=n)
        col = np.stack([np.bincount(f, weights=c[i, ch] * a, minlength=n) for ch in range(3)], -1)
        col = (col / np.maximum(wsum, 1e-6)[:, None]).reshape(*half.shape, 3) * (1 - trans)[..., None]
        size = (img.shape[1], img.shape[0])
        trans = cv2.resize(trans.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        col = cv2.resize(col.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        img[:] = img * trans[..., None] + col

    def _halo(self, img, depth, pos, rot, x, y, s, f) -> None:
        u, v, z = self._project(np.array([[x, y, 0.9]]), pos, rot)
        size_px = self.fy * 4.5 * s / np.maximum(0.1, z)
        self._sprites(img, depth, u, v, z, size_px, np.array([[1.0, 0.439, 0.125]]), np.array([0.28 * f]), False,
                      lambda d: _ramp(d, GLOW))
