"""Mock perception (spec §7.3): geometric, scripted, no frames needed.

A ground-truth target inside a drone camera's frustum, within range and with
line of sight, produces a detection. Gemma answers after a simulated 1.0 s of
sim time. The scripted answers go through the same fusion as the live mode:

- fire:     fast 0.80, Gemma agrees 0.88 -> combined 0.85 -> confirmed
- survivor: fast 0.85, Gemma agrees 0.93 -> combined 0.90 -> confirmed
- decoy:    reported as "fire" (it passes the colour threshold).
  First look: Gemma weakly agrees 0.35 -> combined 0.45 -> low_conf.
  Re-look: Gemma disagrees -> combined 0.18 -> rejected (blacklisted 60 s).

Ground truth is used here and only here (honesty rule).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from sgr import config
from sgr.contracts import XY, CamInfo, Detection, GemmaResult, GroundTruth

from .fusion import fuse

LATENCY_S = 1.0  # default simulated Gemma latency (sim time). Live gemma4:e4b is ~5-8 s: use latency_s=7 to rehearse it
RANGE_M = 5.5  # about the real detector's range at 320x240 (fire ~6 m, person ~5 m)
MERGE_M = 1.0
BLACKLIST_S = 60.0
DEFAULT_CAM_Z = 1.6
_HALF_VFOV = math.radians(config.CAM_FOVY_DEG) / 2
_HALF_HFOV = math.atan(math.tan(_HALF_VFOV) * config.CAM_W / config.CAM_H)

# kind -> (label, height of the target point, score_fast)
_KINDS = {"fire": ("fire", 0.6, 0.80), "survivor": ("person", 0.15, 0.85), "decoy": ("fire", 1.5, 0.60)}
_MODEL = "mock"


def _gemma(kind: str, relooks: int, latency: float = LATENCY_S) -> GemmaResult:
    if kind == "fire":
        return GemmaResult(fire=True, person=False, confidence=0.88, caption="Flames on the floor ahead.",
                           latency_s=latency, model_tag=_MODEL)
    if kind == "survivor":
        return GemmaResult(fire=False, person=True, confidence=0.93, caption="A person lying on the floor.",
                           latency_s=latency, model_tag=_MODEL)
    if relooks == 0:  # decoy, first look from afar
        return GemmaResult(fire=True, person=False, confidence=0.35, caption="Orange glow, unclear.",
                           latency_s=latency, model_tag=_MODEL)
    return GemmaResult(fire=False, person=False, confidence=0.8, caption="An orange panel on a wall, no flames.",
                       latency_s=latency, model_tag=_MODEL)


@dataclass
class _Track:
    det: Detection
    kind: str
    due: float | None = None  # sim time the pending Gemma answer arrives


@dataclass
class MockPerception:
    ground_truth: Callable[[], GroundTruth]
    line_of_sight: Callable[[str, float, float, float], bool] | None = None
    latency_s: float = LATENCY_S
    now: float = 0.0
    _tracks: dict[str, _Track] = field(default_factory=dict)
    _blacklist: list[tuple[str, float, float, float]] = field(default_factory=list)  # kind, x, y, until
    _out: list[Detection] = field(default_factory=list)
    _next: int = 1
    _latencies: list[float] = field(default_factory=list)

    # ---------------------------------------------------------------- facade
    def submit_frame(self, drone: str, cam: CamInfo, t: float, frame_id: str) -> None:
        self.now = max(self.now, t)
        self._blacklist = [b for b in self._blacklist if b[3] > t]
        for kind, x, y in self._targets():
            label, z, score = _KINDS[kind]
            if not self._visible(drone, cam, x, y, z):
                continue
            if any(k == kind and math.hypot(x - bx, y - by) <= MERGE_M for k, bx, by, _ in self._blacklist):
                continue
            if any(tr.kind == kind and math.hypot(x - tr.det.world_xy.x, y - tr.det.world_xy.y) <= MERGE_M
                   for tr in self._tracks.values()):
                continue  # merged into an existing detection: no new job
            det = Detection(id=f"det{self._next:04d}", drone=drone, t=t, label=label, world_xy=XY(x=x, y=y),
                            score_fast=score, status="pending", frame_id=frame_id)
            self._next += 1
            self._tracks[det.id] = _Track(det, kind, due=t + self.latency_s)
            self._out.append(det.model_copy())

    def poll(self) -> list[Detection]:
        for tr in self._tracks.values():
            if tr.due is not None and self.now >= tr.due:
                tr.due = None
                g = _gemma(tr.kind, tr.det.relooks, self.latency_s)
                combined, status = fuse(tr.det.label, tr.det.score_fast, g, tr.det.relooks)
                tr.det = tr.det.model_copy(update={"gemma": g, "combined": combined, "status": status})
                self._latencies.append(self.latency_s)
                if status == "rejected":
                    self._blacklist.append((tr.kind, tr.det.world_xy.x, tr.det.world_xy.y, self.now + BLACKLIST_S))
                self._out.append(tr.det.model_copy())
        self._tracks = {k: v for k, v in self._tracks.items() if v.det.status != "rejected"}
        out, self._out = self._out, []
        return out

    def request_relook(self, detection_id: str) -> None:
        tr = self._tracks.get(detection_id)
        if tr is None or tr.due is not None or tr.det.status not in ("low_conf", "unconfirmed"):
            return
        tr.det = tr.det.model_copy(update={"relooks": tr.det.relooks + 1, "status": "pending"})
        tr.due = self.now + self.latency_s
        self._out.append(tr.det.model_copy())

    def queue_len(self) -> int:
        return sum(tr.due is not None for tr in self._tracks.values())

    # ---------------------------------------------------------------- geometry
    def _targets(self):
        gt = self.ground_truth()
        for f in gt.fires:
            yield "fire", f.x, f.y
        yield "survivor", gt.survivor.x, gt.survivor.y
        if gt.decoy is not None:
            yield "decoy", gt.decoy.x, gt.decoy.y

    def _visible(self, drone: str, cam: CamInfo, x: float, y: float, z: float) -> bool:
        p = cam.pose
        cz = p.z if p.z > 0.05 else DEFAULT_CAM_Z  # a stub CamInfo may carry z = 0
        dx, dy, dz = x - p.x, y - p.y, z - cz
        d = math.hypot(dx, dy)
        if d > RANGE_M or d < 0.3:
            return False
        bearing = math.atan2(dy, dx) - p.yaw
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        if abs(bearing) > _HALF_HFOV:
            return False
        elev = math.atan2(dz, d) - math.radians(cam.pitch_deg)
        if abs(elev) > _HALF_VFOV:
            return False
        return self.line_of_sight is None or self.line_of_sight(drone, x, y, z)

    # ---------------------------------------------------------------- snapshots
    def get_state(self) -> dict:
        return {
            "now": self.now, "next": self._next, "blacklist": list(self._blacklist),
            "tracks": [(tr.det.model_dump(), tr.kind, tr.due) for tr in self._tracks.values()],
        }

    def set_state(self, s: dict) -> None:
        self.now, self._next, self._blacklist = s["now"], s["next"], list(s["blacklist"])
        self._tracks = {d["id"]: _Track(Detection(**d), kind, due) for d, kind, due in s["tracks"]}
        self._out = []
