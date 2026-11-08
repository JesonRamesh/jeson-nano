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

import numpy as np

from . import shapes
from .rubble import rng

# fire.tsx constants (s = max(0.6, size))
FLAME_N, FLAME_SPEED = 140, (0.9, 1.6)
SMOKE_N, SMOKE_SPEED = 40, (0.06, 0.11)
SMOKE_H, SMOKE_CEIL, SMOKE_OPACITY = 5.5, 2.15, 0.5
SPARK_N = 24
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
    def __init__(self, w: int = shapes.CAM_W, h: int = shapes.CAM_H) -> None:
        self.w, self.h = w, h
        self.fx, self.fy, self.cx, self.cy = shapes.cam_intrinsics(w, h)
        u = (np.arange(w) + 0.5 - self.cx) / self.fx
        v = -(np.arange(h) + 0.5 - self.cy) / self.fy
        self._ray = np.stack(np.broadcast_arrays(u[None, :], v[:, None], -np.ones((h, w))), -1)  # camera frame

    # ------------------------------------------------------------------ public
    def apply(self, frame: np.ndarray, depth: np.ndarray, cam_pos, cam_mat, fires, t: float) -> np.ndarray:
        """frame: HxWx3 uint8; depth: HxW metres along the view axis; cam_pos (3,), cam_mat (3, 3) MuJoCo
        camera frame (x right, y up, looking down -z); fires: [(x, y, size)]; t: seconds."""
        pos = np.asarray(cam_pos, float)
        rot = np.asarray(cam_mat, float).reshape(3, 3)
        todo = []
        for x, y, size in fires:
            c = rot.T @ (np.array([x, y, 1.0]) - pos)
            dist = math.hypot(*(np.array([x, y]) - pos[:2]))
            if dist > MAX_RANGE or (-c[2] < -3.0):  # far away, or well behind the camera
                continue
            todo.append((x, y, size))
        if not todo:
            return frame
        img = frame.astype(np.float32) / 255.0
        world = None
        for x, y, size in todo:
            s = max(0.6, float(size)) * SIZE_K
            seed = _js_round(x * 131 + y * 71)
            f = 0.72 + 0.18 * math.sin(t * 11 + seed) * math.sin(t * 6.7) + 0.1 * math.sin(t * 23)
            if world is None:  # back-project every pixel once: floor effects need world positions
                world = pos + (self._ray * depth[..., None]) @ rot.T
            self._floor(img, world, x, y, s, f)
            self._smoke(img, depth, pos, rot, x, y, s, seed, t)
            self._halo(img, depth, pos, rot, x, y, s, f)
            self._flames(img, depth, pos, rot, x, y, s, seed, t)
            self._sparks(img, depth, pos, rot, x, y, s, seed, t)
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)

    # ------------------------------------------------------------------ pieces
    def _project(self, pts: np.ndarray, pos, rot):
        """World points (N, 3) -> pixel u, v and view depth (positive in front)."""
        c = (pts - pos) @ rot
        z = -c[:, 2]
        zs = np.maximum(z, 1e-3)
        return self.cx + self.fx * c[:, 0] / zs, self.cy - self.fy * c[:, 1] / zs, z

    def _floor(self, img, world, x, y, s, f) -> None:
        dx, dy = world[..., 0] - x, world[..., 1] - y
        floor = world[..., 2] < FLOOR_Z
        r = np.hypot(dx, dy)
        # scorch: a dark soot stain, 2.8 s across
        a = np.where(floor, 0.3 * _ramp(r / (1.4 * s), ((0, 1.0), (0.6, 0.4), (1, 0.0))), 0.0)
        img[:] = img * (1 - a[..., None]) + np.array([0.03, 0.022, 0.016]) * a[..., None]
        # firelight pool: glow(), #ff6410, opacity 0.55 f, 6.5 s across
        a = np.where(floor, 0.55 * f * _ramp(r / (3.25 * s), GLOW), 0.0)
        img[:] = img * (1 - a[..., None]) + np.array([1.0, 0.392, 0.063]) * a[..., None]

    def _sprites(self, img, depth, u, v, z, size_px, rgb, alpha, additive: bool, profile) -> None:
        """Round sprites (point sprites / billboards), depth-tested against the scene."""
        h, w = depth.shape
        for i in np.argsort(-z):  # far to near (matters for alpha blending)
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
            if additive:
                patch += rgb[i] * a[..., None]
            else:
                patch[:] = patch * (1 - a[..., None]) + rgb[i] * a[..., None]

    def _flames(self, img, depth, pos, rot, x, y, s, seed, t) -> None:
        sd = _seeds(_js_round(FLAME_N * s), seed, *FLAME_SPEED)
        ph, an, rf, sp = sd.T
        height, radius, part = 0.75 * s + 0.4, 0.32 * s, 0.22 * s
        life = np.mod(t * sp + ph, 1.0)
        ang = an * 2 * np.pi + t * 1.3
        r = rf * radius * (1 + (0.35 - 1) * life)  # mix(1, uSpread, life)
        sway = np.sin(t * 3.1 + ph * 23) * 0.07 * life
        pts = np.stack([x + np.cos(ang) * r + sway, y + np.sin(ang) * r, life * height], 1)
        u, v, z = self._project(pts, pos, rot)
        size_px = self.fy * part * (1 + (0.55 - 1) * life) / np.maximum(0.1, z)
        a = _smooth(0, 0.35, life)[:, None]
        c = np.array([1.0, 0.82, 0.45]) * (1 - a) + np.array([1.0, 0.42, 0.06]) * a
        b = _smooth(0.45, 0.9, life)[:, None]
        c = c * (1 - b) + np.array([0.75, 0.12, 0.02]) * b
        alpha = _smooth(0, 0.06, life) * (1 - _smooth(0.55, 1.0, life))
        # three.js AdditiveBlending (SRC_ALPHA, ONE) of gl_FragColor = (c * a * 0.75, a): adds c * 0.75 * a^2,
        # with a = tex * alpha. Pass rgb = c * 0.75 * alpha and the sprite alpha = alpha (tex applied per pixel).
        self._sprites(img, depth, u, v, z, size_px, c * 0.75 * GAIN * alpha[:, None], alpha, True,
                      lambda d: _ramp(d, SOFTDOT) ** 2)

    def _smoke(self, img, depth, pos, rot, x, y, s, seed, t) -> None:
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
        size_px = self.fy * part * (0.5 + 1.5 * life) / np.maximum(0.1, z)
        k = _smooth(0, 0.4, life)[:, None]
        c = np.array([0.36, 0.27, 0.2]) * (1 - k) + np.array([0.17, 0.16, 0.16]) * k
        alpha = SMOKE_OPACITY * _smooth(0, 0.15, life) * (1 - _smooth(0.6, 1.0, life))
        # smokePuff: soft blobs (peak ~0.35), cut to a circle by smoothstep(0.5, 0.25, |pc - 0.5|)
        self._sprites(img, depth, u, v, z, size_px, c, alpha, False,
                      lambda d: 0.35 * np.clip(1 - d, 0, 1) * _smooth(1.0, 0.5, d))

    def _halo(self, img, depth, pos, rot, x, y, s, f) -> None:
        u, v, z = self._project(np.array([[x, y, 0.9]]), pos, rot)
        size_px = self.fy * 4.5 * s / np.maximum(0.1, z)
        self._sprites(img, depth, u, v, z, size_px, np.array([[1.0, 0.439, 0.125]]), np.array([0.28 * f]), False,
                      lambda d: _ramp(d, GLOW))

    def _sparks(self, img, depth, pos, rot, x, y, s, seed, t) -> None:
        sd = _seeds(_js_round(SPARK_N * s), seed + 2, 0.8, 1.6)
        ph, an, rf, sp = sd.T
        k = np.arange(len(sd))
        jx = np.sin(t * sp * 1.2 + ph * 40) * 0.15
        jy = np.cos(t * sp * 1.1 + an * 40) * 0.15
        jz = np.sin(t * sp * 0.9 + k) * 0.2
        pts = np.stack([x + (ph - 0.5) * 1.2 * s + jx, y + (an - 0.5) * 1.2 * s + jy, 1.3 + (rf - 0.5) * 2.4 + jz], 1)
        u, v, z = self._project(pts, pos, rot)
        size_px = np.clip(self.fy * 0.05 / np.maximum(0.1, z), 1.0, 4.0)
        tw = 0.5 + 0.5 * np.sin(t * 6 * sp + ph * 30)
        rgb = np.array([1.0, 0.604, 0.235])[None, :] * (0.9 * tw)[:, None]
        self._sprites(img, depth, u, v, z, size_px, rgb, np.ones(len(sd)), True, lambda d: _ramp(d, SOFTDOT))
