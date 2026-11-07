"""Rescue: fire watch, the entrance -> survivor route, re-planning, and the checkpoint chain (spec §8.3).

- Fire watch: when the first fire is confirmed, one drone keeps its camera on it from ~3.3 m (warm,
  below the 70 °C retreat limit), at a spot away from the route-graph corridors so it never blocks them.
- Route: A* entrance -> survivor once the survivor is confirmed. Re-planned (REPLAN) when a new
  blocked/fire cell lands within 0.5 m of it; the old route stays in old_routes for 6 s.
  No route -> status `none`, NO_ROUTE, the chain holds and the route is re-checked every 2 s.
- Chain: route -> RDP -> turn points (> 35°). D_survivor hovers over the survivor, D_lead goes to the
  entrance to meet the firefighter, any other drone takes the largest turns as a sentinel (yaw sweep).
  Rebuilt when the route changes or a drone is lost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sgr.contracts import Checkpoint, Evidence, Graph, OldRoute, Pose, Route

from .agent import Agent
from .belief import BeliefGrid
from .blackboard import Blackboard
from .planner import DistanceField, astar, cost_map, path_hits, path_length, rdp, turn_points
from .trace import Trace

ROUTE_MARGIN_M = 0.5  # new blocked/fire cell this close to the route -> re-plan
OLD_ROUTE_S = 6.0
NO_ROUTE_RECHECK_S = 2.0
WATCH_M = 3.3  # fire-watch distance: ~55 °C on the heat model, under the 70 °C retreat limit
TURN_DEG = 35.0
CHAIN_MAX = 3  # survivor + lead + one sentinel (the three checkpoints); extra drones keep exploring
_RING = np.radians(np.arange(0, 360, 10))

Pt = tuple[float, float]


@dataclass
class Rescue:
    belief: BeliefGrid
    bb: Blackboard
    trace: Trace
    graph: Graph
    entrance: Pt
    path: list[Pt] | None = None  # full-resolution route (cell centres)
    roles: dict[str, str] = field(default_factory=dict)  # drone -> survivor | lead | sentinel
    watcher: str | None = None
    watch_fire: Pt | None = None
    _watch_done: bool = False  # one watch per run: the flare-up victim is not replaced
    _pending: tuple[str, str, list[tuple[int, int]], Pt] | None = None  # (sensor, drone, cells, xy)
    _next_check: float = 0.0
    _chain_key: tuple | None = None

    # ------------------------------------------------------------------ inputs
    def on_new_cells(self, sensor: str, drone: str, cells: list[tuple[int, int]]) -> None:
        """New blocked_new / fire cells this tick: re-plan if they touch the route."""
        if self.path and self.bb.route.status == "ok" and path_hits(self.path, cells, self.belief, ROUTE_MARGIN_M):
            xs = [self.belief.cell_center(i, j) for i, j in cells]
            xy = (float(np.mean([p[0] for p in xs])), float(np.mean([p[1] for p in xs])))
            self._pending = (sensor, drone, cells, xy)

    # ------------------------------------------------------------------ tick
    def tick(self, t: float, agents: dict[str, Agent], alive: dict[str, Pose]) -> str:
        """Returns the swarm phase: explore | chain | no_route."""
        self._fire_watch(t, agents, alive)
        sv = self.bb.survivor
        if sv.status == "confirmed" and sv.x is not None:
            target = (sv.x, sv.y)
            if self.bb.route.status == "idle":
                self._plan(t, target, first=True)
            elif self._pending is not None:
                self._plan(t, target)
            elif self.bb.route.status == "none" and t >= self._next_check:
                self._plan(t, target)
            if self.bb.route.status == "ok":
                self._chain(t, agents, alive, target)
        self._pending = None
        if self.bb.route.status == "none":
            return "no_route"
        return "chain" if self.roles else "explore"

    # ------------------------------------------------------------------ route
    def _plan(self, t: float, target: Pt, first: bool = False) -> None:
        old = self.bb.route
        new = astar(self.belief, self.entrance, target)
        if new is None:
            self._next_check = t + NO_ROUTE_RECHECK_S
            if old.status != "none":
                if self.path:
                    self.bb.old_routes.append(OldRoute(points=old.points, until=t + OLD_ROUTE_S))
                self.bb.route = Route(points=[], status="none", version=old.version + 1)
                self.path = None
                why = self._why()
                self.trace.log(t, "swarm", "NO_ROUTE", f"No safe route to the survivor{why}. Holding.",
                               self._evidence(fallback={"from": list(self.entrance), "to": list(target)}))
                self.bb.event("no_route", xy=target)
            return
        length = path_length(new)
        pts = [(round(x, 2), round(y, 2)) for x, y in rdp(new, 0.1)]
        if not first:  # the first plan is not a re-plan (the chain rows follow it)
            if old.points:
                self.bb.old_routes.append(OldRoute(points=old.points, until=t + OLD_ROUTE_S))
            before = path_length(self.path) if self.path else None
            delta = f" ({length - before:+.1f} m)" if before else ""
            reason = (f"Route blocked{self._why()}: new route {length:.1f} m{delta}" if old.status == "ok"
                      else f"Safe route found again: {length:.1f} m")
            self.trace.log(t, "swarm", "REPLAN", reason,
                           self._evidence(fallback={"length_m": round(length, 1)}))
            self.bb.event("replan", xy=self._pending[3] if self._pending else None)
        self.path = new
        self.bb.route = Route(points=pts, status="ok", version=old.version + 1)

    def _why(self) -> str:
        if self._pending is None:
            return ""
        sensor, drone, _, (x, y) = self._pending
        what = "fire" if sensor in ("gemma", "camera") else "debris"
        return f" by {what} near ({x:.1f}, {y:.1f}) ({drone})"

    def _evidence(self, fallback: dict) -> Evidence:
        if self._pending is None:
            return Evidence(sensor="planner", value=fallback)
        sensor, drone, cells, _ = self._pending
        return Evidence(sensor=sensor, value={"drone": drone, "cells": [list(c) for c in cells[:20]], **fallback})

    # ------------------------------------------------------------------ chain
    def _chain(self, t: float, agents: dict[str, Agent], alive: dict[str, Pose], target: Pt) -> None:
        cands = sorted(d for d in alive if d != self.watcher)
        key = (self.bb.route.version, tuple(cands))
        if key == self._chain_key or not cands or not self.path:
            return
        self._chain_key = key
        simp = rdp(self.path, 0.3)
        turns = sorted(turn_points(simp, TURN_DEG), key=lambda tp: -tp[1])

        def near(p: Pt):
            return lambda d: math.hypot(alive[d].x - p[0], alive[d].y - p[1])

        free = list(cands)
        keep = [d for d in free if d in self.roles]  # prefer drones already in the chain (less shuffling)
        surv = min(keep or free, key=near(target)) if keep and self.roles.get(min(keep, key=near(target))) == "survivor"             else min(free, key=near(target))
        free.remove(surv)
        plan: list[tuple[str, str, Pt, dict]] = [(surv, "survivor", target, {"task": "over the survivor"})]
        if free:
            lead = min(free, key=near(self.entrance))
            free.remove(lead)
            face = self.path[min(5, len(self.path) - 1)]
            plan.append((lead, "lead", self.entrance, {"face": face, "task": "lead: meet the firefighter"}))
        # sentinels: largest turns first (nearest free drone each); spare drones go to the route's midpoint
        for k in range(max(0, CHAIN_MAX - len(plan))):
            if not free:
                break
            if k < len(turns):
                (px, py), ang = turns[k]
                xy, task = (px, py), f"sentinel: {ang:.0f}° turn"
            else:
                xy, task = self.path[len(self.path) // 2], "sentinel: mid-route"
            d = min(free, key=near(xy))
            free.remove(d)
            plan.append((d, "sentinel", xy, {"sweep": True, "task": task}))
        for d in free:  # not needed in the chain: back to exploring
            if d in self.roles:
                agents[d].unstation()

        first = not self.roles
        self.roles = {}
        for d, role, xy, kw in plan:
            self.roles[d] = role
            agents[d].release()
            agents[d].set_area(None)
            agents[d].station(xy, face=kw.get("face"), sweep=kw.get("sweep", False), mode="CHAIN", task=kw["task"])
            self.trace.log_change("chain", (role, round(xy[0], 1), round(xy[1], 1)), t, d, "CHAIN_POSITION",
                                  f"{d} -> {kw['task']} at ({xy[0]:.1f}, {xy[1]:.1f})",
                                  Evidence(sensor="planner", value={"role": role, "xy": [round(xy[0], 2),
                                                                                         round(xy[1], 2)]}))
        self.bb.claims.clear()  # exploring is over for the chain drones
        order = {"lead": 0, "sentinel": 1, "survivor": 2}
        along = {xy: self._along(xy) for _, _, xy, _ in plan}
        self.bb.checkpoints = [Checkpoint(x=xy[0], y=xy[1], drone=d)
                               for d, role, xy, _ in sorted(plan, key=lambda p: (order[p[1]], along[p[2]]))]
        lead = next((d for d, r in self.roles.items() if r == "lead"), None)
        self.bb.lead = lead
        if first:
            self.bb.chain_formed = True
            self.bb.event("chain_formed", drone=lead, xy=target)

    def _along(self, xy: Pt) -> float:
        pts = np.asarray(self.path)
        return float(np.argmin(np.hypot(pts[:, 0] - xy[0], pts[:, 1] - xy[1])))

    # ------------------------------------------------------------------ fire watch
    def _fire_watch(self, t: float, agents: dict[str, Agent], alive: dict[str, Pose]) -> None:
        if self.watcher is not None and self.watcher not in alive:
            self.watcher, self._watch_done = None, True  # the watcher was lost (flare-up): don't replace it
            return
        if self.watcher is not None or self._watch_done:
            return
        fires = [h for h in self.bb.hazards.values() if h.kind == "fire"]
        if not fires:
            return
        fire = fires[0].xy
        cands = [d for d in alive if d not in self.roles and not agents[d].inspecting and agents[d].mode != "RETREAT"]
        if not cands:
            return
        d = min(cands, key=lambda d: math.hypot(alive[d].x - fire[0], alive[d].y - fire[1]))
        post = self._watch_post(fire, alive[d])
        if post is None:
            self._watch_done = True
            return
        self.watcher, self.watch_fire = d, fire
        agents[d].release()
        agents[d].set_area(None)
        self.bb.claims = {a: o for a, o in self.bb.claims.items() if o != d}
        agents[d].station(post, face=fire, mode="WATCHING", task="watching the fire")
        self.trace.log(t, d, "INSPECT", f"{d} keeps its camera on the fire from {WATCH_M:.1f} m "
                                        f"at ({post[0]:.1f}, {post[1]:.1f})",
                       Evidence(sensor="planner", value={"fire": [round(fire[0], 2), round(fire[1], 2)],
                                                         "post": [round(post[0], 2), round(post[1], 2)]}))

    def _watch_post(self, fire: Pt, pose: Pose) -> Pt | None:
        """Reachable spot ~WATCH_M from the fire with line of sight, as far as possible from the corridors."""
        cm = cost_map(self.belief)
        df = DistanceField(self.belief, (pose.x, pose.y))
        best, score = None, -math.inf
        for r in (WATCH_M, WATCH_M + 0.3, WATCH_M - 0.2):
            for a in _RING:
                px, py = fire[0] + r * math.cos(a), fire[1] + r * math.sin(a)
                i, j = self.belief.world_to_cell(px, py)
                if not (self.belief.inside(i, j) and cm.traversable[j, i] and np.isfinite(df.cost[j, i])):
                    continue
                if not self._los((px, py), fire):
                    continue
                s = self._corridor_dist((px, py)) - 0.01 * df.cost[j, i] * self.belief.res
                if s > score:
                    best, score = (px, py), s
            if best is not None:
                return best
        return None

    def _corridor_dist(self, p: Pt) -> float:
        nodes = {n.id: (n.x, n.y) for n in self.graph.nodes}
        best = math.inf
        for e in self.graph.edges:
            if e.a in nodes and e.b in nodes:
                best = min(best, _seg_dist(p, nodes[e.a], nodes[e.b]))
        return best

    def _los(self, a: Pt, b: Pt) -> bool:
        d = math.dist(a, b)
        rng = self.belief.march(a[0], a[1], np.array([math.atan2(b[1] - a[1], b[0] - a[0])]), d)[3][0]
        return rng >= d - 0.3


def _seg_dist(p: Pt, a: Pt, b: Pt) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L = dx * dx + dy * dy
    u = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L))
    return math.hypot(p[0] - (ax + u * dx), p[1] - (ay + u * dy))
