"""Fast colour detector (spec §7.2).

HSV thresholds, a few ms per 320x240 frame:
- fire: bright yellow-orange, high V, area >= 0.5 %
- person: hi-vis yellow-green, area >= 0.15 %

The orange decoy panel deliberately passes the fire threshold. It scores
lower than real fire because it is one flat hue, while fire spans
red -> orange -> yellow. Gemma has the final say.

FastDetector adds the stateful parts: world position via pixel_to_world,
1.0 m de-duplication per label, and a 60 s blacklist for rejected spots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

# OpenCV HSV: H in 0..179, S and V in 0..255
FIRE_H = (0, 33)
FIRE_S_MIN = 150
FIRE_V_MIN = 100  # the matte decoy panel is V 100-190 at grazing angles; S > 150 still rejects walls/floor/debris/skin
PERSON_H = (34, 50)
PERSON_S_MIN = 150
PERSON_V_MIN = 90
MIN_AREA_FRAC = 0.005  # fire (§7.2)
PERSON_MIN_AREA_FRAC = 0.0015  # hi-vis is distinctive; survivor detectable to ~5 m

# score_fast = base + w_area * min(1, area / area_full) + w_cue * cue
FIRE_SCORE = dict(base=0.40, w_area=0.20, area_full=0.02, w_cue=0.39, cue_full=6.0)  # cue = hue std
PERSON_SCORE = dict(base=0.55, w_area=0.40, area_full=0.01)

MERGE_DIST = 1.0
BLACKLIST_S = 60.0

_OPEN = np.ones((3, 3), np.uint8)  # remove speckle
_CLOSE = np.ones((9, 9), np.uint8)  # join parts of one object (separate cones, hi-vis bands)


@dataclass
class FastBox:
    label: str  # "fire" | "person"
    bbox: tuple[int, int, int, int]  # x, y, w, h (pixels)
    area_frac: float  # matching-colour pixels / frame pixels
    score_fast: float
    anchor: tuple[int, int]  # (u, v) interior pixel nearest the blob centroid
    samples: tuple[tuple[int, int], ...] = ()  # interior pixels for the world rays (median of their hits)


def _blobs(mask: np.ndarray, min_frac: float):
    """Yield (pixels, bbox, area_frac) per object. Area counts real colour pixels, not the closing."""
    h, w = mask.shape
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN)
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    for i in range(1, n):
        px = (lab == i) & (mask > 0)
        area = float(px.sum()) / (h * w)
        if area >= min_frac:
            yield px, tuple(int(v) for v in stats[i, :4]), area


N_RAY_SAMPLES = 9
_ERODE = np.ones((3, 3), np.uint8)


def _ray_pixels(px: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    """(anchor, samples): interior object pixels to cast world rays through.

    Edge pixels are dropped (eroded). Where the object peeks around a wall corner, the anti-aliased
    edge pixels are object-coloured but their ray clips the corner, which put detections metres
    short of the target. If erosion leaves nothing (a tiny blob), the raw pixels are used.
    """
    core = cv2.erode(px.astype(np.uint8), _ERODE) > 0
    ys, xs = np.nonzero(core if core.any() else px)
    k = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
    anchor = (int(xs[k]), int(ys[k]))
    idx = np.linspace(0, len(xs) - 1, min(N_RAY_SAMPLES, len(xs))).astype(int)
    order = np.lexsort((xs, ys))  # spread the samples over the object, top to bottom
    samples = tuple((int(xs[order[i]]), int(ys[order[i]])) for i in idx)
    return anchor, samples


def detect(frame: np.ndarray) -> list[FastBox]:
    """frame: (H, W, 3) uint8 RGB. Returns boxes, biggest first."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    out: list[FastBox] = []

    fire = cv2.inRange(hsv, (FIRE_H[0], FIRE_S_MIN, FIRE_V_MIN), (FIRE_H[1], 255, 255))
    for px, bbox, area in _blobs(fire, MIN_AREA_FRAC):
        hue_std = float(hsv[..., 0][px].std())
        p = FIRE_SCORE
        score = p["base"] + p["w_area"] * min(1.0, area / p["area_full"]) + p["w_cue"] * min(1.0, hue_std / p["cue_full"])
        out.append(FastBox("fire", bbox, area, round(min(score, 0.99), 3), *_ray_pixels(px)))

    person = cv2.inRange(hsv, (PERSON_H[0], PERSON_S_MIN, PERSON_V_MIN), (PERSON_H[1], 255, 255))
    for px, bbox, area in _blobs(person, PERSON_MIN_AREA_FRAC):
        p = PERSON_SCORE
        score = p["base"] + p["w_area"] * min(1.0, area / p["area_full"])
        out.append(FastBox("person", bbox, area, round(min(score, 0.99), 3), *_ray_pixels(px)))

    out.sort(key=lambda b: -b.area_frac)
    return out


@dataclass
class Candidate:
    det_id: str
    label: str
    drone: str
    t: float
    frame_id: str
    box: FastBox
    world_xy: tuple[float, float]
    is_new: bool  # False = merged into an existing detection (no new Gemma job)


@dataclass
class _Track:
    det_id: str
    label: str
    xy: tuple[float, float]
    last_t: float
    n: int = 1


@dataclass
class FastDetector:
    merge_dist: float = MERGE_DIST
    blacklist_s: float = BLACKLIST_S
    _tracks: list[_Track] = field(default_factory=list)
    _blacklist: list[tuple[str, tuple[float, float], float]] = field(default_factory=list)  # label, xy, until
    _next: int = 1

    def process(self, drone: str, frame: np.ndarray, t: float, frame_id: str,
                pixel_to_world: Callable[[str, int, int], object]) -> list[Candidate]:
        """pixel_to_world(drone, u, v) -> object with .x/.y, or None (e.g. World.pixel_to_world)."""
        self._blacklist = [b for b in self._blacklist if b[2] > t]
        out = []
        for box in detect(frame):
            xy = self._locate(drone, box, pixel_to_world)
            if xy is None:
                continue
            if self.is_blacklisted(box.label, xy, t):
                continue
            tr = self._nearest(box.label, xy)
            if tr is not None:
                tr.xy = ((tr.xy[0] * tr.n + xy[0]) / (tr.n + 1), (tr.xy[1] * tr.n + xy[1]) / (tr.n + 1))
                tr.n += 1
                tr.last_t = t
                out.append(Candidate(tr.det_id, box.label, drone, t, frame_id, box, tr.xy, False))
            else:
                tr = _Track(f"det{self._next:04d}", box.label, xy, t)
                self._next += 1
                self._tracks.append(tr)
                out.append(Candidate(tr.det_id, box.label, drone, t, frame_id, box, xy, True))
        return out

    @staticmethod
    def _locate(drone: str, box: FastBox, pixel_to_world) -> tuple[float, float] | None:
        """Median of the world hits of the interior sample pixels (robust to a stray ray)."""
        hits = [h for h in (pixel_to_world(drone, u, v) for u, v in (box.samples or (box.anchor,))) if h is not None]
        if not hits:
            return None
        return float(np.median([h.x for h in hits])), float(np.median([h.y for h in hits]))

    def _nearest(self, label: str, xy) -> _Track | None:
        best, bd = None, self.merge_dist
        for tr in self._tracks:
            if tr.label != label:
                continue
            d = math.dist(tr.xy, xy)
            if d <= bd:
                best, bd = tr, d
        return best

    def blacklist(self, label: str, xy, t: float) -> None:
        """Rejected location: ignore same-label hits within merge_dist for 60 s and drop its track."""
        self._blacklist.append((label, (float(xy[0]), float(xy[1])), t + self.blacklist_s))
        self._tracks = [tr for tr in self._tracks if not (tr.label == label and math.dist(tr.xy, xy) <= self.merge_dist)]

    def is_blacklisted(self, label: str, xy, t: float) -> bool:
        return any(lb == label and until > t and math.dist(bxy, xy) <= self.merge_dist
                   for lb, bxy, until in self._blacklist)

    def get_state(self) -> dict:
        return {"tracks": [vars(t).copy() for t in self._tracks], "blacklist": list(self._blacklist), "next": self._next}

    def set_state(self, s: dict) -> None:
        self._tracks = [_Track(**d) for d in s["tracks"]]
        self._blacklist = list(s["blacklist"])
        self._next = s["next"]
