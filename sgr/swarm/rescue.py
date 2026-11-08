"""Rescue: fire watch, the entrance -> survivor route, re-planning, and the checkpoint chain (spec §8.3).

- Fire watch: when the first fire is confirmed, one drone keeps its camera on it from ~3.3 m (warm,
  below the 70 °C retreat limit), at a spot away from the route-graph corridors so it never blocks them.
- Route: A* entrance -> survivor once the survivor is confirmed. Re-planned (REPLAN) when a new
  blocked/fire cell lands within 0.5 m of it; the old route stays in old_routes for 6 s.
  No route -> status `none`, NO_ROUTE, the chain holds and the route is re-checked every 2 s; one drone
  (a sentinel, never the lead or the survivor drone) flies to ~2.5 m from the blockage so its LiDAR can
  see whether it is still there (after a judge's Undo, nothing else would ever clear it).
- Chain: route -> RDP -> turn points (> 35°). D_survivor hovers over the survivor, D_lead goes to the
  entrance to meet the firefighter, every other drone becomes a checkpoint (sentinel): sharpest turns first,
  then the middle of the longest stretch between checkpoints, so the drones mark out the path.
  Rebuilt when the route changes or a drone is lost.
- Act 1, once the chain is ready (act2_start already saved with everyone at their posts): the lead and one
  sentinel split the route and patrol it, LiDAR sweeping as they go, so a judge's debris click anywhere on
  the route gets noticed; the survivor drone and any extra checkpoint drones hold their posts. In Act 2 they hold their posts again (P's
  handover logic expects fixed checkpoints, and the lead meets the firefighter at the entrance).
- Act 2 (the firefighter's tracked position `ff` is known): re-plans start at the firefighter, only
  blockages AHEAD of them on the route count, the lead is the drone nearest the firefighter and
  sentinels take the turns still ahead (§9 "Route becomes unsafe").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sgr.contracts import Checkpoint, Evidence, Graph, OldRoute, Pose, Route

from .agent import Agent
from .belief import BeliefGrid
from .blackboard import Blackboard
from .planner import NO_STAY, DistanceField, astar, cost_map, path_hits, path_length, rdp, turn_points
from .trace import Trace

ROUTE_MARGIN_M = 0.5  # new blocked/fire cell this close to the route -> re-plan
SAME_ROUTE_LEN_M = 0.5  # a re-plan this close in length ...
SAME_ROUTE_SKIP_M = 1.5  # ignore the start of a re-plan when comparing (it starts at the firefighter)
SAME_ROUTE_DEV_M = 1.5  # ... and never further than this from the old line is the same route (equal-cost lanes in open halls are ~1 m apart)
OLD_ROUTE_S = 6.0
REPLAN_SETTLE_S = 1.0  # after a re-plan, more of the same blockage is folded into one follow-up (first one: instant)
NO_ROUTE_RECHECK_S = 2.0
WATCH_GIVEUP_S = 40.0  # keep re-trying a safe watch post this long (more fires get confirmed meanwhile)
WATCH_REACH_S = 15.0  # a watcher that hasn't reached its post by then gets another one
REPOST_S = 5.0  # a watcher pushed off its post (too hot) gets a new one after this
WATCH_MIN_CORRIDOR_M = 2.5  # a watch post is at least this far from any through-corridor: a flare-up fire there
#                             leaves ~115 °C on the corridor centreline, under the 150 °C drones may cross (the loop
#                             re-checks the real summed heat before the flare)
WATCH_MIN_M, WATCH_MAX_M = 3.0, 5.5  # watch-post distance from the fire (camera range ~5.5 m)
WATCH_MAX_C = 66.0  # summed modelled heat of all confirmed fires at a watch post (retreat limit is 70 °C)
WATCH_MAX_HEAT = 0.92  # belief heat (measured danger) at a watch post  # heat_norm limit at a watch post (~60 °C: under the 70 °C retreat limit)
WATCH_M = 3.3  # fire-watch distance: ~55 °C on the heat model, under the 70 °C retreat limit
TURN_DEG = 35.0
CHAIN_READY_M = 0.8  # a chain drone this close to its post is in position
CHAIN_READY_MAX_S = 40.0  # ... or call the chain ready this long after it formed anyway
PATROL_END_M = 4.0  # patrol stops this far from the lead / survivor drone (they watch their own ends)
PATROL_STEP_M = 1.0  # patrol waypoint spacing along the route
PATROL_OVERLAP_M = 4.0  # neighbouring patrol stretches overlap this much (no blind corner at the boundary)
PATROL_MAX_HEAT = 0.3  # skip waypoints where the belief already holds this much heat (retreat ping-pong)
# every drone not watching a fire joins the chain: survivor + lead + checkpoints along the route
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
    _watch_post_xy: Pt | None = None
    _watch_t: float = 0.0
    _watch_start: float = 0.0
    _pending: tuple[str, str, list[tuple[int, int]], Pt] | None = None  # (sensor, drone, cells, xy)
    _next_check: float = 0.0
    _chain_key: tuple | None = None
    _chain_t: float | None = None  # when the chain formed
    _last_replan: float = -1e9  # sim time of the last REPLAN / NO_ROUTE
    _ready: bool = False  # chain_ready reached (latched: the patrol takes the sentinel off its post)
    _blockage: Pt | None = None  # where the last NO_ROUTE came from (re-check target)
    _checker: str | None = None  # the drone sent to re-check it
    _patrol_key: tuple | None = None  # patrol stretches are recomputed only when route / map / drones change
    ff: Pt | None = None  # Act 2: the firefighter's tracked position (None in Act 1)

    # ------------------------------------------------------------------ inputs
    def on_new_cells(self, sensor: str, drone: str, cells: list[tuple[int, int]]) -> None:
        """New blocked_new / fire cells this tick: re-plan if they touch the route (Act 2: the route ahead)."""
        if self.path and self.bb.route.status == "ok" and path_hits(self._ahead(self.path), cells, self.belief,
                                                                    ROUTE_MARGIN_M):
            if self._pending is not None:  # more of the same blockage before we re-planned: merge it
                sensor, drone, cells = self._pending[0], self._pending[1], self._pending[2] + cells
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
                if t - self._last_replan < REPLAN_SETTLE_S:
                    return self._phase(t, agents, alive)  # just re-planned: let the rest of it come in first
                self._plan(t, target)
            elif self.bb.route.status == "none" and t >= self._next_check:
                self._plan(t, target)
            if self.bb.route.status == "ok":
                self._chain(t, agents, alive, target)
        self._pending = None
        return self._phase(t, agents, alive)

    def _phase(self, t: float, agents: dict[str, Agent], alive: dict[str, Pose]) -> str:
        if self.bb.route.status == "none":
            self._recheck(t, agents, alive)
            return "no_route"
        self._checker = None
        if not self.roles:
            return "explore"
        if not self._ready and self._chain_ready(t, agents, alive):
            self._ready = True
        self._patrols(agents, alive)
        return "chain_ready" if self._ready else "chain"

    def _recheck(self, t: float, agents: dict[str, Agent], alive: dict[str, Pose]) -> None:
        """NO_ROUTE: send one drone to look at the blockage again, so a cleared one gets noticed."""
        if self._blockage is None or (self._checker is not None and self._checker in alive):
            return
        lead = self.bb.lead
        cands = [d for d, r in self.roles.items() if r == "sentinel" and d in alive]
        cands = cands or [d for d, r in self.roles.items() if r != "survivor" and d != lead and d in alive]
        if not cands:
            return
        bx, by = self._blockage
        d = min(cands, key=lambda d: math.hypot(alive[d].x - bx, alive[d].y - by))
        post = self._view_post(self._blockage, alive[d])
        if post is None:
            return
        self._checker = d
        agents[d].patrol(None)
        agents[d].station(post, face=self._blockage, mode="CHAIN", task="re-checking the blockage")
        self.trace.log(t, d, "INSPECT", f"{d} re-checking the blockage near ({bx:.1f}, {by:.1f}) from "
                                        f"({post[0]:.1f}, {post[1]:.1f})",
                       Evidence(sensor="planner", value={"blockage": [round(bx, 2), round(by, 2)],
                                                         "post": [round(post[0], 2), round(post[1], 2)]}))

    def _view_post(self, target: Pt, pose: Pose) -> Pt | None:
        """Reachable spot ~2.5 m from `target` with line of sight to it, cheapest for the drone."""
        cm = cost_map(self.belief)
        df = DistanceField(self.belief, (pose.x, pose.y))
        best, bc = None, math.inf
        for r in (2.5, 3.0, 2.0, 3.5):
            for a in _RING:
                px, py = target[0] + r * math.cos(a), target[1] + r * math.sin(a)
                i, j = self.belief.world_to_cell(px, py)
                if not (self.belief.inside(i, j) and cm.traversable[j, i] and np.isfinite(df.cost[j, i])):
                    continue
                if self.belief.heat[j, i] < NO_STAY and self._los((px, py), target) and df.cost[j, i] < bc:
                    best, bc = (px, py), df.cost[j, i]
            if best is not None:
                return best
        return None

    def _patrols(self, agents: dict[str, Agent], alive: dict[str, Pose]) -> None:
        """Act 1 + chain ready: the chain drones split the route (lead first, sentinels in the middle,
        the survivor drone the last stretch up to the survivor) and sweep it; Act 2: back to posts."""
        # only the lead and ONE sentinel patrol: the survivor drone stays over the survivor and extra checkpoint
        # drones hold their posts, so the chain still reads as a chain on the projector (W's request)
        lead = [d for d, r in self.roles.items() if r == "lead" and d in alive]
        sentinels = sorted(d for d, r in self.roles.items() if r == "sentinel" and d in alive)
        walkers = lead + sentinels[:1]
        for d in self.roles:
            if d not in walkers:
                agents[d].patrol(None)
        active = self._ready and self.ff is None and self.path and self.bb.route.status == "ok"
        if not active or not walkers:
            for d in self.roles:
                agents[d].patrol(None)
            self._patrol_key = None
            return
        key = (self.bb.route.version, self.belief.version, tuple(walkers))
        if key == self._patrol_key:
            return
        self._patrol_key = key
        pts = rdp(self.path, 0.1)
        total = path_length(pts)
        span = (total - 2 * PATROL_END_M) / len(walkers)  # the survivor drone watches the last metres itself
        for k, d in enumerate(walkers):
            a = PATROL_END_M + k * span
            # overlapping stretches: a corner at the boundary (dropped waypoints next to rubble) is seen by both
            agents[d].patrol(self._patrol_points(pts, a - (PATROL_OVERLAP_M if k else 0.0),
                                                 a + span + (PATROL_OVERLAP_M if k < len(walkers) - 1 else 0.0),
                                                 alive[d]))

    def _patrol_points(self, pts: list[Pt], s0: float, s1: float, pose: Pose) -> list[Pt]:
        """Waypoints every PATROL_STEP_M along the route between arc lengths s0 and s1 that this drone can
        actually reach: not in a wall's inflation or a fire's >70 °C core, not where heat was already
        measured, and reachable from where it is (a fire across a one-way building cuts the route)."""
        cm = cost_map(self.belief)  # drone mode
        df = DistanceField(self.belief, (pose.x, pose.y))
        out, s = [], max(0.0, s0)
        while s <= s1:
            x, y = _point_at(pts, s)
            i, j = self.belief.world_to_cell(x, y)
            if (self.belief.inside(i, j) and cm.traversable[j, i] and self.belief.heat[j, i] < PATROL_MAX_HEAT
                    and np.isfinite(df.cost[j, i])):
                out.append((round(x, 2), round(y, 2)))
            s += PATROL_STEP_M
        return out

    def _chain_ready(self, t: float, agents: dict[str, Agent], alive: dict[str, Pose]) -> bool:
        """Every chain drone at its post (Act 2 starts here: the lead is at the entrance)."""
        if self._chain_t is not None and t - self._chain_t >= CHAIN_READY_MAX_S:
            return True
        for d in self.roles:
            post, p = agents[d]._station, alive.get(d)
            if post is None or p is None or math.hypot(p.x - post[0], p.y - post[1]) > CHAIN_READY_M:
                return False
        return True

    # ------------------------------------------------------------------ route
    def _plan(self, t: float, target: Pt, first: bool = False) -> None:
        old = self.bb.route
        start = self.ff if self.ff is not None else self.entrance
        new = astar(self.belief, start, target, drone=False)  # firefighter route: §8.3 costs
        if new is None:
            self._next_check = t + NO_ROUTE_RECHECK_S
            if old.status != "none":
                if self.path:
                    self.bb.old_routes.append(OldRoute(points=old.points, until=t + OLD_ROUTE_S))
                self.bb.route = Route(points=[], status="none", version=old.version + 1)
                self.path = None
                why = self._why()
                self._last_replan = t
                self._blockage = self._pending[3] if self._pending else self._blockage
                self.trace.log(t, "swarm", "NO_ROUTE", f"No safe route to the survivor{why}. Holding.",
                               self._evidence(fallback={"from": [round(start[0], 2), round(start[1], 2)],
                                                        "to": list(target)}))
                self.bb.event("no_route", xy=target)
            return
        length = path_length(new)
        pts = [(round(x, 2), round(y, 2)) for x, y in rdp(new, 0.1)]
        if not first and old.status == "ok" and self.path and self._same_route(self._ahead(self.path), new):
            self.path = new  # more cells of an obstacle we already avoid: refresh quietly, keep the version
            self.bb.route = Route(points=pts, status="ok", version=old.version)
            return
        if not first:  # the first plan is not a re-plan (the chain rows follow it)
            if old.points:
                self.bb.old_routes.append(OldRoute(points=old.points, until=t + OLD_ROUTE_S))
            before = path_length(self._ahead(self.path)) if self.path else None
            delta = f" ({length - before:+.1f} m)" if before else ""
            reason = (f"Route blocked{self._why()}: new route {length:.1f} m{delta}" if old.status == "ok"
                      else f"Safe route found again: {length:.1f} m")
            self.trace.log(t, "swarm", "REPLAN", reason,
                           self._evidence(fallback={"length_m": round(length, 1)}))
            self._last_replan = t
            self.bb.event("replan", xy=self._pending[3] if self._pending else None)
        self.path = new
        self.bb.route = Route(points=pts, status="ok", version=old.version + 1)

    def _checkpoint_posts(self, turns, n: int, plan, cm) -> list[tuple[Pt, str]]:
        """n checkpoint positions along the route: turns (sharpest first), then the middle of the longest
        stretch between checkpoints so far, so the drones space themselves out as waypoints."""
        posts: list[tuple[Pt, str]] = []
        for (px, py), ang in turns[:n]:
            posts.append(((px, py), f"checkpoint: {ang:.0f}° turn"))
        taken = sorted({self._along(xy) for _, _, xy, _ in plan} | {self._along(xy) for xy, _ in posts}
                       | {0.0, float(len(self.path) - 1)})
        while len(posts) < n:
            _, a, b = max((b - a, a, b) for a, b in zip(taken, taken[1:]))
            k = int((a + b) / 2)
            cands = sorted(range(len(self.path)), key=lambda q: abs(q - k))  # nearest point a drone can hover at
            q = next((q for q in cands if self._flyable(cm, self.path[q])), k)
            posts.append((self.path[q], "checkpoint: on the route"))
            taken = sorted(taken + [float(q)])
        return posts

    def _flyable(self, cm, xy: Pt) -> bool:
        """A drone can hover here: reachable and under the 70 °C limit (crossing hotter cells is allowed)."""
        i, j = self.belief.world_to_cell(*xy)
        return self.belief.inside(i, j) and bool(cm.traversable[j, i]) and self.belief.heat[j, i] < NO_STAY

    def _ahead(self, path: list[Pt]) -> list[Pt]:
        """The part of `path` still ahead of the firefighter (all of it in Act 1)."""
        if self.ff is None or len(path) < 2:
            return path
        pts = np.asarray(path)
        k = int(np.argmin(np.hypot(pts[:, 0] - self.ff[0], pts[:, 1] - self.ff[1])))
        return path[k:] if len(path) - k >= 2 else path[-2:]

    @staticmethod
    def _same_route(old: list[Pt], new: list[Pt]) -> bool:
        if abs(path_length(new) - path_length(old)) >= SAME_ROUTE_LEN_M:
            return False
        o = np.asarray(old)
        # the first metres always differ in Act 2 (the firefighter is rarely exactly on the line)
        tail = [q for q in new if math.dist(q, new[0]) > SAME_ROUTE_SKIP_M] or new[-1:]
        dev = max(float(np.min(np.hypot(o[:, 0] - x, o[:, 1] - y))) for x, y in tail)
        return dev < SAME_ROUTE_DEV_M

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
        if self.watcher is not None and self.watcher in alive:  # the survivor is found: the rescue path outranks
            agents[self.watcher].unstation()  # watching a fire, so the watcher joins the chain
            self.watcher, self._watch_done = None, True
        cands = sorted(d for d in alive if d != self.watcher)
        key = (self.bb.route.version, tuple(cands))
        if key == self._chain_key or not cands or not self.path:
            return
        self._chain_key = key
        simp = rdp(self._ahead(self.path), 0.3)  # Act 2: only turns still ahead of the firefighter
        cm = cost_map(self.belief)  # drone mode: a sentinel post must be somewhere a drone can hover
        turns = sorted((tp for tp in turn_points(simp, TURN_DEG) if self._flyable(cm, tp[0])), key=lambda tp: -tp[1])

        def near(p: Pt):
            return lambda d: math.hypot(alive[d].x - p[0], alive[d].y - p[1])

        def reach(p: Pt):
            """By flight path (drone rules), not straight line: a drone sealed off by heat can't take this role."""
            df = DistanceField(self.belief, p)

            def cost(d: str) -> float:
                i, j = self.belief.world_to_cell(alive[d].x, alive[d].y)
                c = df.cost[j, i] if self.belief.inside(i, j) else math.inf
                return float(c) if math.isfinite(c) else 1e6 + near(p)(d)  # unreachable: last resort, nearest
            return cost

        free = list(cands)
        # the drone already over the survivor stays there (less shuffling); otherwise the nearest one
        holder = next((d for d, r in self.roles.items() if r == "survivor" and d in free), None)
        surv = holder or min(free, key=reach(target))
        free.remove(surv)
        plan: list[tuple[str, str, Pt, dict]] = [(surv, "survivor", target, {"task": "over the survivor"})]
        if free:
            meet = self.ff if self.ff is not None else self.entrance  # Act 2: lead with the firefighter
            keep = self.bb.lead if self.ff is not None and self.bb.lead in free else None  # Guidance steers it
            lead = keep or min(free, key=reach(meet))
            free.remove(lead)
            face = self.path[min(5, len(self.path) - 1)]
            task = "lead: guiding the firefighter" if self.ff is not None else "lead: meet the firefighter"
            plan.append((lead, "lead", meet, {"face": face, "task": task}))
        # every other drone becomes a checkpoint: the sharpest turns first, then the biggest gaps along the route
        for xy, task in self._checkpoint_posts(turns, len(free), plan, cm):
            d = min(free, key=near(xy))
            free.remove(d)
            plan.append((d, "sentinel", xy, {"sweep": True, "task": task}))

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
            self._chain_t = t
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
        if self.watcher is not None and not self._watch_done:
            ag = agents[self.watcher]
            post_xy = self._watch_post_xy
            stale = post_xy is not None and t - self._watch_t > REPOST_S and (
                self._summed_temp(post_xy) >= WATCH_MAX_C  # a fire confirmed since makes it too hot
                or (t - self._watch_t > WATCH_REACH_S
                    and math.hypot(alive[self.watcher].x - post_xy[0], alive[self.watcher].y - post_xy[1]) > 1.0))
            if stale:
                self._watch_post_xy = None  # re-post below
            if ag._station != self._watch_post_xy and t - self._watch_t > REPOST_S:  # pushed off / post went bad
                post = None
                for h in (h for h in self.bb.hazards.values() if h.kind == "fire"):  # any fire, best post
                    cand = self._watch_post(h.xy, alive[self.watcher])
                    if cand is not None and (post is None or self._corridor_dist(cand) > self._corridor_dist(post)):
                        post, self.watch_fire = cand, h.xy
                if post is not None:
                    self._watch_post_xy, self._watch_t = post, t
                    ag.station(post, face=self.watch_fire, mode="WATCHING", task="watching the fire")
                elif t - self._watch_start > WATCH_GIVEUP_S:  # still nowhere safe: back to exploring (no flare-up)
                    ag.unstation()
                    self.watcher, self._watch_done = None, True
                else:  # hold here (where the retreat ended) and try again in REPOST_S as more fires get confirmed
                    self._watch_post_xy, self._watch_t = None, t
            return
        if self.watcher is not None or self._watch_done:
            return
        fires = [h for h in self.bb.hazards.values() if h.kind == "fire"]
        if not fires:
            return
        cands = [d for d in alive if d not in self.roles and not agents[d].inspecting and agents[d].mode != "RETREAT"]
        if not cands:
            return
        # watch whichever confirmed fire offers the post furthest from the corridors (spread fires: some have none)
        best = None
        for h in fires:
            d = min(cands, key=lambda d: math.hypot(alive[d].x - h.xy[0], alive[d].y - h.xy[1]))
            post = self._watch_post(h.xy, alive[d])
            if post is not None and (best is None or self._corridor_dist(post) > best[0]):
                best = (self._corridor_dist(post), h.xy, d, post)
        if best is None:
            if len(fires) >= 3:  # several fires confirmed and still nowhere safe to watch from: give up the watch
                self._watch_done = True
            return
        _, fire, d, post = best
        self.watcher, self.watch_fire = d, fire
        self._watch_post_xy, self._watch_t, self._watch_start = post, t, t
        agents[d].release()
        agents[d].set_area(None)
        self.bb.claims = {a: o for a, o in self.bb.claims.items() if o != d}
        agents[d].station(post, face=fire, mode="WATCHING", task="watching the fire")
        self.trace.log(t, d, "INSPECT", f"{d} keeps its camera on the fire from {WATCH_M:.1f} m "
                                        f"at ({post[0]:.1f}, {post[1]:.1f})",
                       Evidence(sensor="planner", value={"fire": [round(fire[0], 2), round(fire[1], 2)],
                                                         "post": [round(post[0], 2), round(post[1], 2)]}))

    def _watch_post(self, fire: Pt, pose: Pose) -> Pt | None:
        """Every reachable cell 3-5.5 m from the fire with line of sight to it, cool enough (summed heat of all
        confirmed fires) and clear of the through-corridors: the one furthest from the corridors. A cell-by-cell
        search: with spread fires the allowed region can be just a few cells, which rings of samples miss."""
        cm = cost_map(self.belief)
        df = DistanceField(self.belief, (pose.x, pose.y))
        b = self.belief
        i0, j0 = b.world_to_cell(*fire)
        n = int(math.ceil(WATCH_MAX_M / b.res))
        best, score = None, -math.inf
        for j in range(max(0, j0 - n), min(b.h, j0 + n + 1)):
            for i in range(max(0, i0 - n), min(b.w, i0 + n + 1)):
                if not (cm.traversable[j, i] and np.isfinite(df.cost[j, i])):
                    continue  # (map heat not used here: thermometer estimates double-count fires since confirmed;
                    # the summed heat of confirmed fires below decides, and a too-hot post just makes it re-post)
                px, py = b.cell_center(i, j)
                if not WATCH_MIN_M <= math.hypot(px - fire[0], py - fire[1]) <= WATCH_MAX_M:
                    continue
                cd = self._corridor_dist((px, py))
                if cd < WATCH_MIN_CORRIDOR_M or cd - 0.01 * df.cost[j, i] * b.res <= score:
                    continue  # never in a corridor: the flare-up would put a fire on the way to everything
                if self._summed_temp((px, py)) >= WATCH_MAX_C or not self._los((px, py), fire):
                    continue  # spread fires add up: 3.3 m from this one can be 96 °C next to the others
                best, score = (px, py), cd - 0.01 * df.cost[j, i] * b.res
        return best

    def _summed_temp(self, p: Pt) -> float:
        """Modelled temperature at p from ALL confirmed fires, summed like the physics (not the hottest one)."""
        from .hazards import HEAT_PEAK_C, HEAT_SIGMA_M  # (no import cycle: hazards doesn't import rescue)

        from sgr import config
        t = config.AMBIENT_C
        for h in self.bb.hazards.values():
            if h.kind == "fire":
                d2 = (p[0] - h.xy[0]) ** 2 + (p[1] - h.xy[1]) ** 2
                t += HEAT_PEAK_C * math.exp(-d2 / (2 * HEAT_SIGMA_M ** 2))
        return t

    def _corridor_dist(self, p: Pt) -> float:
        """Distance to the nearest through-corridor (dead-end room edges don't count)."""
        return min((_seg_dist(p, a, b) for a, b in through_edges(self.graph)), default=math.inf)

    def _los(self, a: Pt, b: Pt) -> bool:
        d = math.dist(a, b)
        rng = self.belief.march(a[0], a[1], np.array([math.atan2(b[1] - a[1], b[0] - a[0])]), d)[3][0]
        return rng >= d - 0.3


def _point_at(pts: list[Pt], s: float) -> Pt:
    """Point at arc length s along a polyline."""
    for a, b in zip(pts, pts[1:]):
        d = math.dist(a, b)
        if s <= d and d > 0:
            u = s / d
            return a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])
        s -= d
    return pts[-1]


def _seg_dist(p: Pt, a: Pt, b: Pt) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L = dx * dx + dy * dy
    u = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L))
    return math.hypot(p[0] - (ax + u * dx), p[1] - (ay + u * dy))

def through_edges(graph) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Corridor segments a fire could cut: edges that don't just lead into a dead-end room (a leaf node that
    isn't the entrance). A fire inside a dead-end room can't block the way between other places."""
    nodes = {n.id: n for n in graph.nodes}
    deg: dict[str, int] = {}
    for e in graph.edges:
        deg[e.a] = deg.get(e.a, 0) + 1
        deg[e.b] = deg.get(e.b, 0) + 1
    leaf = {nid for nid, n in nodes.items() if deg.get(nid, 0) <= 1 and n.kind != "entrance"}
    return [((nodes[e.a].x, nodes[e.a].y), (nodes[e.b].x, nodes[e.b].y))
            for e in graph.edges if e.a in nodes and e.b in nodes and e.a not in leaf and e.b not in leaf]
