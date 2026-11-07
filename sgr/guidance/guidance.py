"""Guidance facade (owned by P, Day 2). §4 signatures, v1.

STUB: no guidance yet. Rules are in §9. Never read hidden world truth here (honesty rule, CLAUDE.md).
"""

from __future__ import annotations

from sgr.contracts import GuidanceMsg, Pose, SwarmState, VelCmd


class Guidance:
    def __init__(self) -> None:
        pass

    def tick(self, t: float, ff: Pose, swarm: SwarmState) -> tuple[GuidanceMsg, dict[str, VelCmd]]:
        return GuidanceMsg(), {}
