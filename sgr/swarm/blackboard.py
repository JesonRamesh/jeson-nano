"""Blackboard (spec §8.1): the in-process shared state of the swarm.

Each drone publishes its own entries (heartbeat, pose, claims, hazards it saw); the belief grid is
the merged map layer. A lost drone simply stops publishing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sgr import config
from sgr.contracts import (
    Checkpoint,
    Detection,
    EventMsg,
    OldRoute,
    Pose,
    Route,
    SurvivorState,
)


@dataclass
class Hazard:
    id: str
    kind: str  # blocked | fire | danger | unsafe
    xy: tuple[float, float]
    source_sensor: str
    t: float
    confidence: float = 1.0


@dataclass
class Blackboard:
    heartbeats: dict[str, float] = field(default_factory=dict)  # drone -> last sim time it published
    poses: dict[str, Pose] = field(default_factory=dict)
    lost: set[str] = field(default_factory=set)
    detections: dict[str, Detection] = field(default_factory=dict)  # upsert by id
    hazards: dict[str, Hazard] = field(default_factory=dict)
    claims: dict[str, str] = field(default_factory=dict)  # area id -> drone
    route: Route = field(default_factory=Route)
    old_routes: list[OldRoute] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    survivor: SurvivorState = field(default_factory=lambda: SurvivorState(status="unknown"))
    events: list[EventMsg] = field(default_factory=list)
    _hz_next: int = 1

    # ---------------------------------------------------------------- publishing
    def heartbeat(self, drone: str, t: float, pose: Pose) -> None:
        self.heartbeats[drone] = t
        self.poses[drone] = pose

    def upsert_detection(self, det: Detection) -> tuple[Detection | None, Detection]:
        """Insert or replace by id. Returns (previous version or None, current)."""
        prev = self.detections.get(det.id)
        self.detections[det.id] = det
        return prev, det

    def add_hazard(self, kind: str, xy: tuple[float, float], sensor: str, t: float, conf: float = 1.0) -> Hazard:
        h = Hazard(id=f"hz{self._hz_next:04d}", kind=kind, xy=xy, source_sensor=sensor, t=t, confidence=conf)
        self._hz_next += 1
        self.hazards[h.id] = h
        return h

    def event(self, kind: str, drone: str | None = None, xy: tuple[float, float] | None = None) -> None:
        self.events.append(EventMsg(kind=kind, drone=drone, xy=xy))

    def pop_events(self) -> list[EventMsg]:
        out, self.events = self.events, []
        return out

    # ---------------------------------------------------------------- queries
    def newly_lost(self, t: float, timeout: float = config.HEARTBEAT_TIMEOUT) -> list[str]:
        """Drones whose heartbeat is older than `timeout` and not yet declared lost."""
        return [d for d, hb in self.heartbeats.items() if d not in self.lost and t - hb > timeout]

    def alive_poses(self, exclude: str | None = None) -> list[Pose]:
        return [p for d, p in self.poses.items() if d not in self.lost and d != exclude]

    def claims_of(self, drone: str) -> list[str]:
        return [a for a, d in self.claims.items() if d == drone]
