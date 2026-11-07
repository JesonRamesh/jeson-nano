"""Swarm facade (owned by S). §4 signatures, v1.

The swarm learns ONLY through `readings` and `detections` (honesty rule, CLAUDE.md). The prior map is
the pre-incident plan: walls + the route graph with every edge open.

Block H1-H4: belief grid, camera coverage, area split + greedy assignment, exploration, reflexes,
heartbeat loss -> reassignment, LiDAR change detection -> blocked cells.
H4-H7 (s/inspect): inspect / re-look from 2 m, fire + survivor from detections, heat danger + RETREAT.
Next: survivor route + REPLAN + chain.
"""

from __future__ import annotations

import math

import numpy as np

from sgr.contracts import (
    Detection,
    Evidence,
    PriorMap,
    Readings,
    SwarmDrone,
    SwarmState,
    TraceRow,
    VelCmd,
)

from .agent import Agent
from .areas import Area, assign, release, split_areas
from .belief import BeliefGrid
from .blackboard import Blackboard
from .coverage import camera_footprint
from .hazards import Hazards
from .planner import DistanceField, path_hits, snap, cost_map
from .trace import Trace

SAME_BLOCKAGE_M = 2.5  # new blocked cells this close to a reported blockage extend it (one trace row each)


class Swarm:
    def __init__(self, scenario_prior: PriorMap, drones: list[str]) -> None:
        self.prior = scenario_prior  # pre-incident plan ONLY
        self.drones = list(drones)
        self.belief = BeliefGrid.from_prior(scenario_prior)
        self.areas: dict[str, Area] = split_areas(self.belief, scenario_prior.graph)
        self.bb = Blackboard()
        self.trace = Trace()
        self.hazards = Hazards(self.belief, self.bb, self.trace)
        self.agents = {d: Agent(d, scenario_prior.altitude) for d in self.drones}
        self._excluded: dict[str, set[str]] = {d: set() for d in self.drones}
        self._excluded_version = self.belief.version
        self.t = 0.0

    # ------------------------------------------------------------------ facade
    def tick(
        self, t: float, readings: dict[str, Readings], detections: list[Detection]
    ) -> tuple[dict[str, VelCmd], list[TraceRow], SwarmState]:
        self.t = t
        for d, r in readings.items():
            self.bb.heartbeat(d, t, r.pose)
        self._check_lost(t)
        alive = {d: r for d, r in readings.items() if d not in self.bb.lost}

        self._sense(t, alive)
        poses = {d: r.pose for d, r in alive.items()}
        for d, r in alive.items():
            self.hazards.heat(t, d, r.pose, r.temperature, self.agents[d])
        for det in detections:
            self.bb.upsert_detection(det)
            self.hazards.on_detection(t, det, self.agents, poses)
        relooks = self.hazards.tick(t, self.agents, poses)
        self._assign(t, alive)

        cmds: dict[str, VelCmd] = {}
        for d, r in alive.items():
            ag = self.agents[d]
            area = self.areas.get(ag.area) if ag.area else None
            cmd, done = ag.step(t, r, self.belief, area, self.bb.alive_poses(exclude=d))
            if done and area is not None:
                self._excluded[d].add(area.id)
                release(self.bb.claims, d)
                ag.set_area(None)
            cmds[d] = cmd

        state = SwarmState(
            t=t,
            phase="explore",
            drones=[
                SwarmDrone(id=d, status=self.agents[d].mode, task=self.agents[d].task, alive=d not in self.bb.lost)
                for d in self.drones
            ],
            route=self.bb.route,
            old_routes=[o for o in self.bb.old_routes if o.until > t],
            checkpoints=self.bb.checkpoints,
            survivor=self.bb.survivor,
            map_delta=self.belief.pop_map_delta(),
            searched_delta=self.belief.pop_searched_delta(),
            events=self.bb.pop_events(),
            relook_requests=relooks,
        )
        return cmds, self.trace.drain(), state

    def coverage(self) -> float:
        return self.belief.coverage()

    # ------------------------------------------------------------------ steps
    def _check_lost(self, t: float) -> None:
        for d in self.bb.newly_lost(t):
            age = t - self.bb.heartbeats[d]
            self.bb.lost.add(d)
            ag = self.agents[d]
            ag.mode = "LOST"
            last = self.bb.poses.get(d)
            xy = (last.x, last.y) if last else None
            self.trace.log(t, d, "DRONE_LOST", f"{d} silent for {age:.1f} s: declared lost",
                           Evidence(sensor="heartbeat", value=round(age, 2)))
            self.bb.event("drone_lost", drone=d, xy=xy)
            freed = release(self.bb.claims, d)
            if ag.area and ag.area not in freed:
                freed.append(ag.area)
            ag.set_area(None)
            if freed:
                self.trace.log(t, "swarm", "REASSIGN", f"{d} lost: area {', '.join(freed)} back to the pool",
                               Evidence(sensor="heartbeat", value={"drone": d, "areas": freed}))

    def _sense(self, t: float, alive: dict[str, Readings]) -> None:
        free = self.belief.free_mask()
        for d, r in alive.items():
            blocked, cleared = self.belief.update_lidar(d, r.pose, r.lidar)
            if blocked:
                xs = [self.belief.cell_center(i, j) for i, j in blocked]
                cx, cy = float(np.mean([p[0] for p in xs])), float(np.mean([p[1] for p in xs]))
                known = [h for h in self.bb.hazards.values() if h.kind == "blocked"
                         and min(math.hypot(h.xy[0] - x, h.xy[1] - y) for x, y in xs) < SAME_BLOCKAGE_M]
                if known:  # more cells of a blockage already reported: the map grows, the trace doesn't
                    for h in known:
                        h.t = t
                else:
                    self.trace.log(t, d, "MARK_BLOCKED",
                                   f"{d} LiDAR: path blocked near ({cx:.1f}, {cy:.1f}), not on the plan",
                                   Evidence(sensor="lidar", value={"cells": [list(c) for c in blocked]},
                                            scan=[round(x, 2) for x in r.lidar]))
                    self.bb.add_hazard("blocked", (cx, cy), "lidar", t)
                    self.bb.event("blocked", drone=d, xy=(cx, cy))
            if blocked or cleared:
                for ag in self.agents.values():
                    if ag.path and path_hits(ag.path, blocked + cleared, self.belief):
                        ag.invalidate()
            self.belief.mark_searched(camera_footprint(self.belief, r.pose) & free)
        if self.belief.version != self._excluded_version:  # the map changed: give up-areas another chance
            self._excluded_version = self.belief.version
            for s in self._excluded.values():
                s.clear()
            for ag in self.agents.values():
                ag.forget_given_up()

    def _assign(self, t: float, alive: dict[str, Readings]) -> None:
        idle = [d for d in alive if self.agents[d].area is None]
        if not idle:
            return
        cm = cost_map(self.belief)
        fields = {d: DistanceField(self.belief, (alive[d].pose.x, alive[d].pose.y)) for d in idle}
        centre_cell: dict[str, tuple[int, int] | None] = {
            a.id: snap(self.belief, cm, *a.centre) for a in self.areas.values()
        }

        def dist(d: str, a: Area) -> float:
            c = centre_cell[a.id]
            if c is not None and np.isfinite(fields[d].cost[c[1], c[0]]):
                return float(fields[d].cost[c[1], c[0]]) * self.belief.res
            m = np.where(a.mask, fields[d].cost, np.inf)  # node itself not reachable: nearest cell of the area
            return float(m.min()) * self.belief.res

        helping = {ag.area for d, ag in self.agents.items()
                   if ag.area and d not in self.bb.lost and self.bb.claims.get(ag.area) != d}
        got = assign(idle, self.areas, self.bb.claims, self.belief, dist, self._excluded, helping)
        for d, (aid, dd, frac) in got.items():
            self.agents[d].set_area(aid)
            shared = self.bb.claims.get(aid) != d
            self.trace.log_change(
                "area", aid, t, d, "ASSIGN_AREA",
                f"{d} {'helping in' if shared else 'takes'} area {aid}: {frac:.0%} unsearched, {dd:.1f} m away",
                Evidence(sensor="planner", value={"area": aid, "unsearched": round(frac, 3), "dist_m": round(dd, 1)}),
            )
