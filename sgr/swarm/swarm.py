"""Swarm facade (owned by S). §4 signatures, v1.

STUB: every drone hovers. Real logic arrives in block H1-H4 (branch s/swarm-core).
The swarm learns ONLY through `readings` and `detections` (honesty rule, CLAUDE.md).
"""

from __future__ import annotations

from sgr.contracts import Detection, PriorMap, Readings, SwarmDrone, SwarmState, TraceRow, VelCmd


class Swarm:
    def __init__(self, scenario_prior: PriorMap, drones: list[str]) -> None:
        self.prior = scenario_prior  # pre-incident plan ONLY
        self.drones = list(drones)

    def tick(
        self, t: float, readings: dict[str, Readings], detections: list[Detection]
    ) -> tuple[dict[str, VelCmd], list[TraceRow], SwarmState]:
        cmds = {d: VelCmd() for d in readings}
        state = SwarmState(
            t=t,
            phase="stub",
            drones=[
                SwarmDrone(id=d, status="IDLE" if d in readings else "LOST", alive=d in readings)
                for d in self.drones
            ],
        )
        return cmds, [], state
