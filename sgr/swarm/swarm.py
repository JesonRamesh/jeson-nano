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
import pickle

import numpy as np

from sgr.contracts import (
    Detection,
    Evidence,
    OldRoute,
    Pose,
    PriorMap,
    Readings,
    SwarmDrone,
    SwarmState,
    TraceRow,
    VelCmd,
)

from .agent import Agent, goal_mask
from .areas import Area, assign, release, split_areas
from .belief import FREE, UNSAFE, BeliefGrid
from .blackboard import Blackboard
from .coverage import camera_footprint
from .hazards import Hazards
from .rescue import Rescue
from .planner import DistanceField, cost_map, path_hits, path_length, snap
from .trace import Trace

EXCLUDE_RETRY_S = 30.0  # a drone retries areas it gave up on after this
ASSIGN_RETRY_S = 1.0  # idle drones with nothing they can reach: look for work again this often
REVISIT_LOOK_S = 8.0  # a revisiting drone watches the spot this long (a new sighting starts a normal inspection)
REVISIT_GIVE_UP_S = 45.0  # couldn't get to a viewpoint in this long: try again later
DEAD_END_TTL_S = 45.0  # a "couldn't see it" dead-end mark is retried after this
SAME_BLOCKAGE_S = 10.0  # ... and seen within this long (otherwise it is a new blockage, e.g. a judge's click)
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
        self.rescue = Rescue(self.belief, self.bb, self.trace, scenario_prior.graph,
                             (scenario_prior.entrance.x, scenario_prior.entrance.y))
        self.agents = {d: Agent(d, scenario_prior.altitude) for d in self.drones}
        self._excluded: dict[str, set[str]] = {d: set() for d in self.drones}
        self._excluded_version = self.belief.open_version
        self._excluded_t = 0.0
        self._path_replans: set[str] = set()  # blockages already reported as a REPLAN
        self._idle_key: tuple[str, ...] | None = None  # idle drones the last fruitless assignment saw
        self._assign_retry_t = 0.0
        self._revisit_drone: str | None = None  # flying back to look at a ruled-out "person" spot
        self._revisit_xy: tuple[float, float] = (0.0, 0.0)
        self._revisit_t0 = 0.0
        self._revisit_seen_t: float | None = None
        self.t = 0.0

    # ------------------------------------------------------------------ facade
    def tick(
        self, t: float, readings: dict[str, Readings], detections: list[Detection], ff: Pose | None = None
    ) -> tuple[dict[str, VelCmd], list[TraceRow], SwarmState]:
        """`ff` = the firefighter's tracked pose in Act 2 (None in Act 1): re-routes start from it."""
        self.t = t
        self.belief.now = t
        self.rescue.ff = (ff.x, ff.y) if ff is not None else None
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
        for sensor, d, cells in self.hazards.fresh_fire:
            self.rescue.on_new_cells(sensor, d, cells)
        self.hazards.fresh_fire.clear()
        phase = self.rescue.tick(t, self.agents, poses)
        self._assign(t, alive)
        self._revisit(t, alive)

        cmds: dict[str, VelCmd] = {}
        for d, r in alive.items():
            ag = self.agents[d]
            area = self.areas.get(ag.area) if ag.area else None
            cmd, done = ag.step(t, r, self.belief, area, self.bb.alive_poses(exclude=d))
            if done and area is not None:
                self._excluded[d].add(area.id)
                release(self.bb.claims, d)
                ag.set_area(None)
                self._area_done(t, d, area)
            cmds[d] = cmd

        state = SwarmState(
            t=t,
            phase=phase,
            drones=[
                SwarmDrone(id=d, status=self.agents[d].mode, task=self.agents[d].task, alive=d not in self.bb.lost)
                for d in self.drones
            ],
            route=self.bb.route,
            old_routes=[o for o in self.bb.old_routes if o.until > t],
            checkpoints=self.bb.checkpoints,
            survivor=self.bb.survivor,
            chain_formed=self.bb.chain_formed,
            lead=self.bb.lead,
            map_delta=self.belief.pop_map_delta(),
            searched_delta=self.belief.pop_searched_delta(),
            events=self.bb.pop_events(),
            relook_requests=relooks,
        )
        return cmds, self.trace.drain(), state

    def coverage(self) -> float:
        return self.belief.coverage()

    # ------------------------------------------------------------------ snapshots (§8.6)
    def get_state(self) -> bytes:
        """Blackboard, belief, agents, trace, hazards, rescue: everything but the prior map."""
        return pickle.dumps({k: v for k, v in self.__dict__.items() if k != "prior"},
                            protocol=pickle.HIGHEST_PROTOCOL)

    def set_state(self, b: bytes) -> None:
        self.__dict__.update(pickle.loads(b))

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
            ag.unstation()
            unsafe = self._mark_unsafe(last)  # §8.4: its last area is unsafe until re-observed
            if unsafe and unsafe not in freed:
                freed.append(unsafe)
            if freed:
                note = f"; area {unsafe} unsafe until re-observed" if unsafe else ""
                self.trace.log(t, "swarm", "REASSIGN",
                               f"{d} lost: area {', '.join(freed)} back to the pool{note}",
                               Evidence(sensor="heartbeat", value={"drone": d, "areas": freed, "unsafe": unsafe}))

    def _mark_unsafe(self, pose: Pose | None) -> str | None:
        """The area a lost drone was in becomes `unsafe` and unsearched, so another drone re-checks it."""
        if pose is None:
            return None
        i, j = self.belief.world_to_cell(pose.x, pose.y)
        area = next((a for a in self.areas.values() if self.belief.inside(i, j) and a.mask[j, i]), None)
        if area is None:
            return None
        cells = area.mask & (self.belief.state == FREE)
        jj, ii = np.nonzero(cells)
        for a, b in zip(ii.tolist(), jj.tolist()):
            self.belief.set_state(a, b, "unsafe")
        self.belief.searched[cells] = False
        return area.id

    def _sense(self, t: float, alive: dict[str, Readings]) -> None:
        free = self.belief.free_mask()
        for d, r in alive.items():
            blocked, cleared = self.belief.update_lidar(d, r.pose, r.lidar)
            if blocked:
                xs = [self.belief.cell_center(i, j) for i, j in blocked]
                cx, cy = float(np.mean([p[0] for p in xs])), float(np.mean([p[1] for p in xs]))
                known = [h for h in self.bb.hazards.values() if h.kind == "blocked" and t - h.t < SAME_BLOCKAGE_S
                         and min(math.hypot(h.xy[0] - x, h.xy[1] - y) for x, y in xs) < SAME_BLOCKAGE_M]
                # (only a blockage still being mapped: debris that appears later next to old rubble is new)
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
            if blocked:
                self.rescue.on_new_cells("lidar", d, blocked)
            if blocked or cleared:
                for ag in self.agents.values():
                    if ag.path and path_hits(ag.path, blocked + cleared, self.belief):
                        if blocked and ag.id not in self.bb.lost:
                            self._path_replan(t, d, ag, blocked)
                        ag.invalidate()
            seen = camera_footprint(self.belief, r.pose) & free
            self.belief.mark_searched(seen)
            jj, ii = np.nonzero(seen & (self.belief.state == UNSAFE))  # re-observed: safe again
            for a, b in zip(ii.tolist(), jj.tolist()):
                self.belief.set_state(a, b, "free")
        # "couldn't see it from here" marks expire (often just heat or a drone in the way at the time);
        # rooms cut off by debris stay out for good through belief.reachable_now()
        expired = self.belief.dead & (self.belief.dead_t < t - DEAD_END_TTL_S)
        if t - self._excluded_t > EXCLUDE_RETRY_S:  # areas a drone gave up on get another look now and then
            self._excluded_t = t
            for s in self._excluded.values():
                s.clear()
        if expired.any():
            self.belief.dead &= ~expired
            for s in self._excluded.values():
                s.clear()
        if self.belief.open_version != self._excluded_version:  # the map OPENED up (debris gone): re-try dead ends
            self._excluded_version = self.belief.open_version
            for s in self._excluded.values():
                s.clear()
            self.belief.dead[:] = False

    def _area_done(self, t: float, d: str, area: Area) -> None:
        """Nothing left this drone can search in the area: say whether it was a dead end, then move on."""
        cut = area.cut_off(self.belief)
        dead = int((area.mask & self.belief.dead & ~self.belief.searched).sum())
        if cut >= 5:
            why = f"area {area.id} is a dead end: blocked by debris ({cut * self.belief.res ** 2:.0f} m² cut off)"
        elif dead >= 5:
            why = f"area {area.id} is a dead end: the rest can't be seen from here"
        else:
            why = f"area {area.id} searched"
        self.trace.log_change("area_done", area.id, t, d, "EXPLORE", f"{d}: {why}, turning back to explore elsewhere",
                              Evidence(sensor="planner", value={"area": area.id, "cut_off_cells": cut,
                                                                "dead_end_cells": dead,
                                                                "unsearched": round(area.unsearched_fraction(self.belief), 3)}))

    def _path_replan(self, t: float, seen_by: str, ag: Agent, blocked: list[tuple[int, int]]) -> None:
        """A drone's planned path just got blocked by debris: REPLAN (once per blockage)."""
        xs = [self.belief.cell_center(i, j) for i, j in blocked]
        cx, cy = float(np.mean([p[0] for p in xs])), float(np.mean([p[1] for p in xs]))
        hz = min((h for h in self.bb.hazards.values() if h.kind == "blocked"),
                 key=lambda h: math.hypot(h.xy[0] - cx, h.xy[1] - cy), default=None)
        key = hz.id if hz else f"{cx:.0f},{cy:.0f}"  # one row per blockage; other drones re-route quietly
        if key in self._path_replans:
            return
        self._path_replans.add(key)
        where = f"to area {ag.area}" if ag.area else ""
        row = self.trace.log(
            t, ag.id, "REPLAN",
            f"{ag.id} path {where} blocked by debris near ({cx:.1f}, {cy:.1f}) ({seen_by} LiDAR): re-routing",
            Evidence(sensor="lidar", value={"seen_by": seen_by, "cells": [list(c) for c in blocked[:20]],
                                            "old_path_m": round(path_length(ag.path), 1)}),
        )
        if row is not None:
            self.bb.old_routes.append(OldRoute(points=[(round(x, 2), round(y, 2)) for x, y in ag.path],
                                               until=t + 6.0))
            self.bb.event("replan", drone=ag.id, xy=(cx, cy))

    def _revisit(self, t: float, alive: dict[str, Readings]) -> None:
        """Nothing reachable left to search and still no survivor: fly an idle drone back to look at a spot
        where a "person" was ruled out (Gemma can be wrong from a bad angle). The perception layer reports it
        again if the camera still sees it, and the normal inspection takes over from there."""
        d = self._revisit_drone
        if d is not None:
            ag = self.agents[d]
            handed_over = any(ins.drone == d for ins in self.hazards.active.values())
            if d not in alive or handed_over or self.bb.survivor.status == "confirmed" or ag.stationed:
                if d in alive and not handed_over and not ag.stationed:
                    ag.release()
                self._revisit_drone = None
                return
            if ag.in_position and self._revisit_seen_t is None:
                self._revisit_seen_t = t
            looked = self._revisit_seen_t is not None and t - self._revisit_seen_t > REVISIT_LOOK_S
            if looked or t - self._revisit_t0 > REVISIT_GIVE_UP_S:
                ag.release()
                self._revisit_drone = None
            return
        if self.bb.survivor.status == "confirmed" or self.hazards.active:
            return
        idle = [x for x in alive if self.agents[x].area is None and not self.agents[x].stationed
                and self.agents[x].mode == "IDLE"]
        if not idle:
            return
        spot = self.hazards.revisit_due(t)
        if spot is None:
            return
        x, y, since = spot
        d = min(idle, key=lambda q: math.hypot(alive[q].pose.x - x, alive[q].pose.y - y))
        view = self.hazards.viewpoint((x, y), alive[d].pose)
        if view is None:
            return
        self.agents[d].approach(view, (x, y))
        self.hazards.mark_revisited((x, y), t)
        self._revisit_drone, self._revisit_xy, self._revisit_t0, self._revisit_seen_t = d, (x, y), t, None
        self.trace.log(t, d, "INSPECT",
                       f"{d}: nothing reachable left to search and no survivor yet: going back to look at "
                       f"({x:.1f}, {y:.1f}), where a person was ruled out at {since:.0f} s",
                       Evidence(sensor="planner", value={"spot": [round(x, 1), round(y, 1)],
                                                         "ruled_out_t": round(since, 1),
                                                         "viewpoint": [round(view[0], 1), round(view[1], 1)]}))

    def _assign(self, t: float, alive: dict[str, Readings]) -> None:
        idle = [d for d in alive if self.agents[d].area is None and not self.agents[d].stationed
                and d != self._revisit_drone]
        if not idle:
            return
        # nothing could be handed out last time: try again when someone new is idle, else once a second
        key = tuple(sorted(idle))
        if key == self._idle_key and t < self._assign_retry_t:
            return
        self._idle_key, self._assign_retry_t = key, t + ASSIGN_RETRY_S
        cm = cost_map(self.belief)
        fields = {d: DistanceField(self.belief, (alive[d].pose.x, alive[d].pose.y)) for d in idle}
        centre_cell: dict[str, tuple[int, int] | None] = {
            a.id: snap(self.belief, cm, *a.centre) for a in self.areas.values()
        }

        goals: dict[str, np.ndarray] = {}

        def dist(d: str, a: Area) -> float:
            # only areas this drone can actually see into: a drone-sized way to a viewpoint (else it is offered,
            # finds nothing to fly to, gives it up and gets it offered again: a livelock)
            if a.id not in goals:
                goals[a.id] = goal_mask(self.belief, a)
            if not np.isfinite(np.where(goals[a.id], fields[d].cost, np.inf)).any():
                return math.inf
            c = centre_cell[a.id]
            if c is not None and np.isfinite(fields[d].cost[c[1], c[0]]):
                return float(fields[d].cost[c[1], c[0]]) * self.belief.res
            m = np.where(a.mask, fields[d].cost, np.inf)  # node itself not reachable: nearest cell of the area
            return float(m.min()) * self.belief.res

        helping = {ag.area for d, ag in self.agents.items()
                   if ag.area and d not in self.bb.lost and self.bb.claims.get(ag.area) != d}
        got = assign(idle, self.areas, self.bb.claims, self.belief, dist, self._excluded, helping)
        if got:
            self._idle_key = None  # the idle set changes now: the next idle drone is served at once
        for d, (aid, dd, frac) in got.items():
            self.agents[d].set_area(aid)
            shared = self.bb.claims.get(aid) != d
            self.trace.log_change(
                "area", aid, t, d, "ASSIGN_AREA",
                f"{d} {'helping in' if shared else 'takes'} area {aid}: {frac:.0%} unsearched, {dd:.1f} m away",
                Evidence(sensor="planner", value={"area": aid, "unsearched": round(frac, 3), "dist_m": round(dd, 1)}),
            )
