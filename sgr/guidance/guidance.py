"""Guidance facade (owned by P, Day 2). §4 signature, rules from spec §9.

Inputs are the swarm's beliefs only (SwarmState: route, checkpoints, lead, survivor,
map_delta hazards) plus the firefighter's pose. Never hidden world truth (honesty
rule, CLAUDE.md).

Optional constructor arguments (additive to the §4 stub):
- prior:  PriorMap. Its route graph powers the "nearest edge not on the route" wrong-way rule.
- world:  the World facade, for drone_pose() (lead follow + handover) and push_ff_back()
          (walked into fire). Without it, guidance only gives HUD messages, no drone commands.

Rules (§9), highest priority first:
  complete       within 1.5 m of the confirmed survivor     -> "Survivor reached. Mission complete."
  fire           inside a fire cell                         -> push_ff_back + "Stop. Fire ahead. Turn back."
  danger         within 1.0 m of fire, or in a danger cell  -> alarm, "Stop. Fire ahead. Turn back."
  no route       route.status == "none"                     -> "No safe route. Holding."
  route blocked  route re-planning / replaced under the FF  -> "Stop. Route blocked." then "New route. Follow me."
  wrong way      off-route edge or heading >120 deg off the next checkpoint, for > 2 s
                                                            -> "Wrong way. Turn around."
  drifting       > 2 m off the route for 2 s                -> "Bear left." / "Bear right."
  follow         lead stays 3 m ahead on the route; FF > 6 m behind -> lead stops, "Follow me."
  handover       FF within 2 m of the next chain drone      -> that drone becomes the lead

HUD conventions (for D): arrow_deg = direction to go, relative to the FF heading, degrees,
CCW positive (+90 = turn left). distance = metres left along the route.
`say` is set only when the message changes (speak it once). Subtitles are always set.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sgr import config
from sgr.contracts import Evidence, GuidanceMsg, Pose, PriorMap, SwarmState, TraceRow, VelCmd

from .geometry import Pt, bearing, point_at, project, seg_dist, wrap

LEAD_AHEAD_M = 3.0
FOLLOW_GAP_M = 6.0
HANDOVER_M = 2.0
DRIFT_M = 2.0
DRIFT_S = 2.0
WRONG_S = 2.0
WRONG_HEADING = math.radians(120)
COMPLETE_M = 1.5
FIRE_MARGIN_M = 1.0  # §8.3: fire + 1.0 m margin is impassable
IN_FIRE_M = 0.25
EDGE_ON_ROUTE_M = 1.0
BLOCKED_HOLD_S = 2.0
NEW_ROUTE_HOLD_S = 4.0
LEAD_GAIN = 1.2


class Guidance:
    def __init__(self, prior: PriorMap | None = None, world: Any | None = None) -> None:
        self.prior = prior
        self.world = world
        self.res = config.BELIEF_RES
        self.origin = tuple(prior.origin) if prior is not None else (0.0, 0.0)
        self.altitude = prior.altitude if prior is not None else 1.6
        self._fire: set[tuple[int, int]] = set()
        self._danger: set[tuple[int, int]] = set()
        self._fire_xy = np.zeros((0, 2))
        self.lead: str | None = None
        self.complete = False
        self.corrections = 0
        self._key: str | None = None
        self._drift_since: float | None = None
        self._wrong_since: float | None = None
        self._route_version: int | None = None
        self._blocked = False
        self._blocked_until = -math.inf
        self._new_route_until = -math.inf
        self._prev_status: str | None = None
        self._trace: list[TraceRow] = []
        self._trace_n = 0
        self._walked = 0.0
        self._last_ff: Pt | None = None

    # ---------------------------------------------------------------- belief accumulation
    def observe(self, swarm: SwarmState) -> None:
        """Fold this tick's map_delta into the hazard layers. tick() calls it too.

        The loop should also call it every tick in Act 1, so hazards from before the
        Act 2 snapshot are known (they're also saved in get_state).
        """
        changed = False
        for i, j, state in swarm.map_delta:
            c = (int(i), int(j))
            for layer, name in ((self._fire, "fire"), (self._danger, "danger")):
                if state == name and c not in layer:
                    layer.add(c)
                    changed = True
                elif state != name and c in layer:
                    layer.discard(c)
                    changed = True
        if changed:
            self._fire_xy = np.array([self._cell_xy(c) for c in self._fire]).reshape(-1, 2)

    def _cell_xy(self, c: tuple[int, int]) -> Pt:
        return (self.origin[0] + (c[0] + 0.5) * self.res, self.origin[1] + (c[1] + 0.5) * self.res)

    def _fire_dist(self, p: Pt) -> float:
        if len(self._fire_xy) == 0:
            return math.inf
        return float(np.min(np.hypot(self._fire_xy[:, 0] - p[0], self._fire_xy[:, 1] - p[1])))

    def _in_danger_cell(self, p: Pt) -> bool:
        i = int(math.floor((p[0] - self.origin[0]) / self.res))
        j = int(math.floor((p[1] - self.origin[1]) / self.res))
        return (i, j) in self._danger

    # ---------------------------------------------------------------- main
    def tick(self, t: float, ff: Pose, swarm: SwarmState) -> tuple[GuidanceMsg, dict[str, VelCmd]]:
        self.observe(swarm)
        p: Pt = (ff.x, ff.y)
        if self._last_ff is not None:
            self._walked += math.dist(p, self._last_ff)
        self._last_ff = p

        route = [tuple(q) for q in swarm.route.points]
        status = swarm.route.status
        self._track_route(t, swarm, route)
        if self.lead is None or (swarm.lead and self.lead not in self._chain_drones(swarm)):
            self.lead = swarm.lead or next((c.drone for c in swarm.checkpoints if c.drone), None)

        proj = project(route, p) if len(route) >= 2 else None
        total = proj and sum(math.dist(a, b) for a, b in zip(route, route[1:]))
        remaining = (total - proj.s) if proj else 0.0
        target = self._target(route, proj)
        arrow = math.degrees(wrap(bearing(p, target) - ff.yaw)) if target else 0.0

        cmds: dict[str, VelCmd] = {}
        lead_waits = False
        if proj and not self.complete and self.world is not None:
            self._handover(swarm, route, proj, p)
            lead_waits = self._lead_cmd(route, proj, p, cmds)
            if lead_waits:
                target = self._pose_xy(self.lead) or target
                arrow = math.degrees(wrap(bearing(p, target) - ff.yaw)) if target else arrow

        key, text, sev, alarm = self._rule(t, ff, p, swarm, route, proj, status, lead_waits)
        say = text if key != self._key else None
        if say is not None and key in ("drift", "wrong", "danger", "fire"):
            self.corrections += 1
            self._log(t, "FF_CORRECTION", text, {"rule": key, "ff": [round(ff.x, 2), round(ff.y, 2)],
                                                  "off_route_m": round(proj.dist, 2) if proj else None})
        if say is not None and key == "complete":
            self._log(t, "MISSION_COMPLETE", text, {"ff": [round(ff.x, 2), round(ff.y, 2)],
                                                     "walked_m": round(self._walked, 1)})
        self._key = key
        msg = GuidanceMsg(arrow_deg=round(arrow, 1), distance=round(max(remaining, 0.0), 1), subtitle=text,
                          say=say, alarm=alarm, severity=sev)
        return msg, cmds

    # ---------------------------------------------------------------- rules
    def _rule(self, t, ff: Pose, p: Pt, swarm: SwarmState, route, proj, status, lead_waits):
        sv = swarm.survivor
        if self.complete or (sv and sv.status == "confirmed" and sv.x is not None
                             and math.dist(p, (sv.x, sv.y)) <= COMPLETE_M):
            self.complete = True
            return "complete", "Survivor reached. Mission complete.", "info", False
        fd = self._fire_dist(p)
        if fd <= IN_FIRE_M:
            if self.world is not None and hasattr(self.world, "push_ff_back"):
                self.world.push_ff_back()
            return "fire", "Stop. Fire ahead. Turn back.", "danger", True
        if fd <= FIRE_MARGIN_M or self._in_danger_cell(p):
            return "danger", "Stop. Fire ahead. Turn back.", "danger", True
        if status == "none":
            return "noroute", "No safe route. Holding.", "warn", False
        if self._blocked:
            return "blocked", "Stop. Route blocked.", "warn", True
        if t < self._new_route_until:
            return "newroute", "New route. Follow me.", "info", False
        if not proj:
            return "wait", "Waiting for a safe route.", "info", False

        # wrong way: off-route edge, or facing away from the next checkpoint
        # an off-route edge only counts once the FF has left the route (junctions touch several edges)
        wrong = ((proj.dist > EDGE_ON_ROUTE_M and self._on_wrong_edge(p, route))
                 or self._heading_wrong(ff, swarm, route, proj))
        self._wrong_since = (self._wrong_since if self._wrong_since is not None else t) if wrong else None
        if self._wrong_since is not None and t - self._wrong_since >= WRONG_S:
            return "wrong", "Wrong way. Turn around.", "warn", False

        drifting = proj.dist > DRIFT_M
        self._drift_since = (self._drift_since if self._drift_since is not None else t) if drifting else None
        if self._drift_since is not None and t - self._drift_since >= DRIFT_S:
            # the route is on the opposite side of the FF's offset
            return ("drift", "Bear right." if proj.side > 0 else "Bear left.", "warn", False)

        if lead_waits:
            return "follow", "Follow me.", "info", False
        return "go", f"Follow the drone. {max(0.0, sum(math.dist(a, b) for a, b in zip(route, route[1:])) - proj.s):.0f} metres to go.", "info", False

    def _track_route(self, t: float, swarm: SwarmState, route) -> None:
        """Route replaced (or re-planning) while the FF is being guided: "blocked" for 2 s, then "new route".

        S may re-plan within one tick, so a version change alone triggers it.
        """
        v, status = swarm.route.version, swarm.route.status
        guiding = self._key not in (None, "wait")
        if self._route_version is None:
            self._route_version = v
        elif v != self._route_version:
            self._route_version = v
            if guiding:
                self._blocked_until = t + BLOCKED_HOLD_S
                self._new_route_until = self._blocked_until + NEW_ROUTE_HOLD_S
        if self._prev_status == "replanning" and status == "ok":
            self._new_route_until = max(self._new_route_until, t + NEW_ROUTE_HOLD_S)
        self._prev_status = status
        self._blocked = status == "replanning" or t < self._blocked_until

    def _on_wrong_edge(self, p: Pt, route) -> bool:
        if self.prior is None or not self.prior.graph.edges:
            return False
        nodes = {n.id: (n.x, n.y) for n in self.prior.graph.nodes}
        best, bd = None, math.inf
        for e in self.prior.graph.edges:
            if e.a in nodes and e.b in nodes:
                d = seg_dist(p, nodes[e.a], nodes[e.b])
                if d < bd:
                    best, bd = e, d
        if best is None or bd > best.width:  # not really on any corridor (e.g. in a room)
            return False
        a, b = nodes[best.a], nodes[best.b]
        samples = [(a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])) for f in (0.25, 0.5, 0.75)]
        return not all(project(route, s).dist <= EDGE_ON_ROUTE_M for s in samples)

    def _heading_wrong(self, ff: Pose, swarm: SwarmState, route, proj) -> bool:
        nxt = self._next_checkpoint(swarm, route, proj) or route[-1]
        if math.dist((ff.x, ff.y), nxt) < 1.0:
            return False
        return abs(wrap(bearing((ff.x, ff.y), nxt) - ff.yaw)) > WRONG_HEADING

    # ---------------------------------------------------------------- lead / chain
    def _chain_drones(self, swarm: SwarmState) -> list[str]:
        return [c.drone for c in swarm.checkpoints if c.drone]

    def _next_checkpoint(self, swarm: SwarmState, route, proj) -> Pt | None:
        ahead = [(project(route, (c.x, c.y)).s, (c.x, c.y)) for c in swarm.checkpoints]
        ahead = [a for a in ahead if a[0] > proj.s + 0.5]
        return min(ahead)[1] if ahead else None

    def _target(self, route, proj) -> Pt | None:
        if not proj:
            return None
        return point_at(route, proj.s + LEAD_AHEAD_M)

    def _pose_xy(self, drone: str | None) -> Pt | None:
        if drone is None or self.world is None:
            return None
        q = self.world.drone_pose(drone)
        return (q.x, q.y)

    def _handover(self, swarm: SwarmState, route, proj, p: Pt) -> None:
        """FF within 2 m of the next chain drone -> it becomes the lead."""
        best = None
        for c in swarm.checkpoints:
            if not c.drone or c.drone == self.lead:
                continue
            s = project(route, (c.x, c.y)).s
            if s > proj.s - 1.0 and (best is None or s < best[0]):
                best = (s, c.drone)
        if best is not None:
            q = self._pose_xy(best[1])
            if q is not None and math.dist(p, q) <= HANDOVER_M:
                self.lead = best[1]

    def _lead_cmd(self, route, proj, p: Pt, cmds: dict[str, VelCmd]) -> bool:
        """Drive the lead 3 m ahead on the route. Returns True if it is waiting for the FF."""
        if self.lead is None:
            return False
        q = self.world.drone_pose(self.lead)
        lead_s = project(route, (q.x, q.y)).s
        if lead_s - proj.s > FOLLOW_GAP_M or math.dist(p, (q.x, q.y)) > FOLLOW_GAP_M + 1.0:
            cmds[self.lead] = VelCmd(vz=LEAD_GAIN * (self.altitude - q.z), wz=self._face(q, p))
            return True
        goal = point_at(route, proj.s + LEAD_AHEAD_M)
        vx, vy = LEAD_GAIN * (goal[0] - q.x), LEAD_GAIN * (goal[1] - q.y)
        n = math.hypot(vx, vy)
        if n > config.MAX_SPEED:
            vx, vy = vx * config.MAX_SPEED / n, vy * config.MAX_SPEED / n
        cmds[self.lead] = VelCmd(vx=vx, vy=vy, vz=LEAD_GAIN * (self.altitude - q.z), wz=self._face(q, p))
        return False

    @staticmethod
    def _face(q: Pose, p: Pt) -> float:
        """Yaw rate that turns the lead to face the firefighter (its camera keeps them in view)."""
        return max(-1.5, min(1.5, 2.0 * wrap(bearing((q.x, q.y), p) - q.yaw)))

    # ---------------------------------------------------------------- trace + summary + snapshots
    def _log(self, t: float, action: str, reason: str, value: dict) -> None:
        self._trace_n += 1
        self._trace.append(TraceRow(id=f"g{self._trace_n:04d}", t=t, drone=self.lead or "swarm", action=action,
                                    reason=reason, evidence=Evidence(sensor="planner", value=value)))

    def drain_trace(self) -> list[TraceRow]:
        """FF_CORRECTION / MISSION_COMPLETE rows since the last call (the loop publishes them)."""
        out, self._trace = self._trace, []
        return out

    def stats(self) -> dict:
        """For the §9 summary screen."""
        return {"corrections": self.corrections, "walked_m": round(self._walked, 1), "complete": self.complete}

    def get_state(self) -> dict:
        return {"fire": set(self._fire), "danger": set(self._danger), "lead": self.lead,
                "complete": self.complete, "corrections": self.corrections, "walked": self._walked}

    def set_state(self, s: dict) -> None:
        self._fire, self._danger = set(s["fire"]), set(s["danger"])
        self._fire_xy = np.array([self._cell_xy(c) for c in self._fire]).reshape(-1, 2)
        self.lead, self.complete = s["lead"], s["complete"]
        self.corrections, self._walked = s["corrections"], s["walked"]
        self._key = None
        self._last_ff = None
