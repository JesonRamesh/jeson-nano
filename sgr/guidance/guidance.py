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
  danger         within 1.0 m of fire, or in a danger cell off the route -> alarm, "Stop. Fire ahead. Turn back."
  heat           in a danger cell ON the swarm's route (heat is a route cost, not a wall: spec 8.3)
                                                            -> "Heat on the route. Keep moving."
  no route       route.status == "none"                     -> "No safe route. Holding."
  route blocked  route re-planning / replaced under the FF  -> "Stop. Route blocked." then "New route. Follow me."
  wrong way      off-route edge or heading >120 deg off the next checkpoint, for > 2 s
                                                            -> "Wrong way. Turn around."
  drifting       > 2 m off the route for 2 s                -> "Bear left." / "Bear right."
  follow         lead stays 3 m ahead on the route; FF > 6 m behind -> lead stops, "Follow me."
  returning      lead > 10 m ahead on the route (e.g. the only drone left, still over the survivor)
                 -> it flies back along the route to meet the FF: "Your drone is coming back to you."
  handover       FF within 2 m of the next chain drone      -> that drone becomes the lead

HUD conventions (for D): arrow_deg = direction to go, relative to the FF heading, degrees,
CCW positive (+90 = turn left). distance = metres left along the route.
`say` is set only when the message changes (speak it once). Subtitles are always set.

Drone voices (callouts, voice.py): besides the HUD voice, the drones talk to the firefighter. Short lines, each
said once, at most one every CALLOUT_GAP_S (7 s; a handover may come sooner), never while a danger / correction
message is up, and nothing after mission complete (lines still being voiced are dropped):
  intro          first guided tick                 lead: "D1 here. Follow me to the survivor."
  handover       the lead changes                  new lead: "D2 here, taking over from D1. Follow me."
  fire_side      a believed fire 1.3-6 m away       nearest drone: "Fire on your right, 3 metres."
  survivor_close confirmed survivor 4-8 m away      lead: "Survivor just ahead, 6 metres."
  progress       halfway                           lead: "Halfway there. 30 metres to go."
  lead_wait      the lead has waited 5 s            lead: "Waiting for you. Walk to my light."
  lead_return    the lead starts flying back         lead: "Coming back for you. Follow the arrow."
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sgr import config
from sgr.contracts import Evidence, GuidanceMsg, Pose, PriorMap, SwarmState, TraceRow, VelCmd

from .geometry import Pt, bearing, point_at, project, seg_dist, wrap
from .voice import Callout, template_msg

LEAD_AHEAD_M = 3.0
ARROW_MIN_M = 0.6  # shortest arrow lookahead (tight corners, narrow gaps)
ARROW_CHORD_TOL_M = 0.25  # the straight line to the arrow target may leave the route by at most this
LEAD_CARROT_M = 1.0  # the lead drone steps along the route, never straight through walls
FOLLOW_GAP_M = 6.0
RETURN_GAP_M = 10.0  # a lead this far ahead along the route comes back to the firefighter instead of waiting
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
CALLOUT_GAP_S = 7.0  # at most one drone callout per 7 s (a handover may come sooner): short, never a backlog
FIRE_CALLOUT_M = 6.0
FIRE_CALLOUT_AGAIN_S = 40.0
FIRE_SAME_M = 6.0  # fire cells this close to an announced one are the same fire (a fire spreads over metres)
SURVIVOR_CALLOUT_M = 8.0
SURVIVOR_CALLOUT_MIN_M = 4.0  # closer than this, "Survivor reached" is about to play: say nothing
LEAD_WAIT_CALLOUT_S = 5.0
URGENT_GAP_S = 4.0  # fire / turn warnings may follow another line this soon (others wait CALLOUT_GAP_S)
TURN_CALLOUT_M = (3.0, 8.0)  # announce a corner this far ahead along the route
TURN_MIN_DEG = 50.0
DISTANCE_EVERY_M = 20.0  # "40 metres to go." every 20 m (more than 12 m out)
QUIET_KEYS = {"fire", "danger", "heat", "blocked", "noroute", "wrong", "complete"}  # the HUD voice has the floor
# every fixed HUD line (GuidanceMsg.say); in online mode they are voiced by Gemini, pre-generated at startup
HUD_LINES = ["Survivor reached. Mission complete.", "Stop. Fire ahead. Turn back.", "No safe route. Holding.",
             "Stop. Route blocked.", "New route. Follow me.", "Waiting for a safe route.", "Wrong way. Turn around.",
             "Bear right.", "Bear left.", "Follow me.", "Your drone is coming back to you. Follow the arrow.",
             "Follow the drone.", "Survivor ahead.", "Heat on the route. Keep moving."]


def route_of(swarm: SwarmState) -> list:
    return [tuple(q) for q in swarm.route.points]


class Guidance:
    def __init__(self, prior: PriorMap | None = None, world: Any | None = None, voice: Any | None = None,
                 callouts: bool = True) -> None:
        self.prior = prior
        self.world = world
        self.voice = voice  # VoiceAgent (Gemini) or None: callouts use their fixed lines
        self.voice_online = voice is not None  # the Act 2 switch: online (Gemini voices) / offline (local voices)
        self.callouts_on = callouts
        if voice is not None and hasattr(voice, "prewarm"):
            voice.prewarm(self._fixed_lines())
        self._callouts_out: list = []
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
        self._returning = False
        self._reset_callouts()

    # ---------------------------------------------------------------- voice mode (the Act 2 switch)
    @property
    def voice_mode(self) -> str:
        if self.voice is None:
            return "unavailable"
        return "online" if self.voice_online else "offline"

    def set_voice_mode(self, mode: str) -> None:
        """online: Gemini words + voices for the drones and the HUD; offline: fixed lines, no network (the browser
        shows them as text only, no voice-over)."""
        if self.voice is None:
            return
        online = mode == "online"
        if online != self.voice_online:
            self.voice_online = online
            self.voice.clear()  # nothing voiced in the old mode plays after the switch

    @staticmethod
    def _fixed_lines() -> list[tuple[str, str]]:
        """Lines said word for word (no numbers): voiced once at startup so they play instantly."""
        ids = ["D1", "D2", "D3", "D4"]
        out = [("HUD", t) for t in HUD_LINES]
        out += [(d, f"{d} here, taking over from {o}. Follow me.") for d in ids for o in ids if o != d]
        out += [(d, f"{d} here, I'm with the survivor. Nearly there.") for d in ids]
        out += [(d, f"{d} here, {o} is down. Follow me.") for d in ids for o in ids if o != d]
        out += [(d, "Waiting for you. Walk to my light.") for d in ids]
        out += [(d, "Coming back for you. Follow the arrow.") for d in ids]
        return out

    def _reset_callouts(self) -> None:
        self._cl_last = -math.inf
        self._cl_intro = False
        self._cl_lead: str | None = None
        self._cl_chain: set[str] = set()  # chain drones last tick (one dropping out = failed / lost)
        self._cl_fires: list[tuple[Pt, float]] = []  # (where, when) fires were announced
        self._cl_survivor = False
        self._cl_total: float | None = None
        self._cl_marks: set[str] = set()
        self._cl_wait_since: float | None = None
        self._cl_wait_said = False
        self._cl_return_said = False
        self._cl_done = False
        self._cl_turns: list[Pt] = []
        self._cl_dist_marks: set[int] = set()  # metres-to-go marks already announced
        self._callouts_out = []
        if self.voice is not None:
            self.voice.clear()

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
        if self.lead is None or not self._alive(swarm, self.lead) or (
                swarm.lead and self.lead not in self._chain_drones(swarm)):
            live = swarm.lead if self._alive(swarm, swarm.lead) else None
            self.lead = live or next((c.drone for c in swarm.checkpoints
                                      if c.drone and self._alive(swarm, c.drone)), None)

        proj = project(route, p) if len(route) >= 2 else None
        total = proj and sum(math.dist(a, b) for a, b in zip(route, route[1:]))
        remaining = (total - proj.s) if proj else 0.0
        target = self._target(route, proj, p)
        arrow = math.degrees(wrap(bearing(p, target) - ff.yaw)) if target else 0.0

        cmds: dict[str, VelCmd] = {}
        lead_state = "none"
        if proj and not self.complete and self.world is not None:
            self._handover(swarm, route, proj, p)
            lead_state = self._lead_cmd(route, proj, p, cmds)  # the arrow keeps following the route

        key, text, sev, alarm = self._rule(t, ff, p, swarm, route, proj, status, lead_state)
        say = text if key != self._key else None
        if say is not None and key in ("drift", "wrong", "danger", "fire"):
            self.corrections += 1
            self._log(t, "FF_CORRECTION", text, {"rule": key, "ff": [round(ff.x, 2), round(ff.y, 2)],
                                                  "off_route_m": round(proj.dist, 2) if proj else None})
        if say is not None and key == "complete":
            self._log(t, "MISSION_COMPLETE", text, {"ff": [round(ff.x, 2), round(ff.y, 2)],
                                                     "walked_m": round(self._walked, 1)})
        self._key = key
        if say is not None and self.callouts_on and self.voice is not None and self.voice_online:
            # online: the HUD line is voiced by Gemini too (pre-generated: instant), so all voices match
            spoken = {"go": "Follow the drone.", "near": "Survivor ahead."}.get(key, say)  # distances are on screen
            self.voice.submit(Callout("HUD", "hud", spoken, severity=sev))
        if self.callouts_on:
            self._callouts(t, ff, p, swarm, proj, key, remaining)
        msg = GuidanceMsg(arrow_deg=round(arrow, 1), distance=round(max(remaining, 0.0), 1), subtitle=text,
                          say=say, alarm=alarm, severity=sev)
        return msg, cmds

    # ---------------------------------------------------------------- drone voices (callouts)
    def _say(self, c: Callout, t: float) -> None:
        self._cl_last = t
        if self.voice is not None and self.voice_online:
            self.voice.submit(c)
        else:
            self._callouts_out.append(template_msg(c))

    def drain_callouts(self) -> list:
        """CalloutMsgs ready to publish (the voice agent's finished lines, or fixed lines without one)."""
        out, self._callouts_out = self._callouts_out, []
        if self.voice is not None:
            out += self.voice.poll()
        return out

    def _callouts(self, t: float, ff: Pose, p: Pt, swarm: SwarmState, proj, key: str, remaining: float) -> None:
        lead = self.lead
        if self.complete:  # mission over: the drones go quiet, and lines still being voiced are dropped
            if not self._cl_done:
                self._cl_done = True
                self._callouts_out = []
                if self.voice is not None:
                    self.voice.clear()
            return
        if lead is None or not proj:
            return
        # handover: one short line from the new lead (it breaks the gap: the FF must know who to follow)
        if self._cl_lead is not None and lead != self._cl_lead and key not in ("fire", "danger"):
            if self._at_survivor(swarm, lead):  # the last drone waits with the survivor: nothing left to follow
                self._say(Callout(lead, "handover", f"{lead} here, I'm with the survivor. Nearly there.",
                                  {"from": self._cl_lead, "to": lead}), t)
            elif not self._alive(swarm, self._cl_lead) or (
                    self._cl_lead in self._cl_chain and self._cl_lead not in self._chain_drones(swarm)):
                # the old lead dropped out of the chain (failed / lost): it didn't hand over, it's down
                self._say(Callout(lead, "handover", f"{lead} here, {self._cl_lead} is down. Follow me.",
                                  {"from": self._cl_lead, "to": lead}), t)
            else:
                self._say(Callout(lead, "handover", f"{lead} here, taking over from {self._cl_lead}. Follow me.",
                                  {"from": self._cl_lead, "to": lead}), t)
        self._cl_lead = lead
        self._cl_chain = set(self._chain_drones(swarm))
        if key == "follow":
            self._cl_wait_since = self._cl_wait_since if self._cl_wait_since is not None else t
        else:
            self._cl_wait_since, self._cl_wait_said = None, False
        if key == "returning" and not self._cl_return_said and self._cl_intro:
            self._cl_return_said = True
            self._say(Callout(lead, "lead_return", "Coming back for you. Follow the arrow."), t)
            return
        if key != "returning":
            self._cl_return_said = False
        since = t - self._cl_last
        if key in QUIET_KEYS or since < URGENT_GAP_S:
            return
        if not self._cl_intro:
            self._cl_intro = True
            self._cl_total = remaining
            if key == "returning":  # e.g. the only drone left, still over the survivor
                self._cl_return_said = True
                text = f"{lead} here. Coming back for you, follow the arrow."
            else:
                text = f"{lead} here. Follow me to the survivor."
            self._say(Callout(lead, "intro", text, {"name": lead}), t)
            return
        c = self._fire_callout(t, ff, p, swarm) or self._turn_callout(route_of(swarm), proj)  # urgent: 4 s gap
        if c is None and since >= CALLOUT_GAP_S:
            c = self._progress_callout(remaining)
        waited = self._cl_wait_since is not None and t - self._cl_wait_since >= LEAD_WAIT_CALLOUT_S
        if c is None and waited and not self._cl_wait_said and since >= CALLOUT_GAP_S:
            self._cl_wait_said = True
            c = Callout(lead, "lead_wait", "Waiting for you. Walk to my light.")
        if c is not None:
            self._say(c, t)

    def _fire_callout(self, t: float, ff: Pose, p: Pt, swarm: SwarmState) -> Callout | None:
        if len(self._fire_xy) == 0:
            return None
        d = np.hypot(self._fire_xy[:, 0] - p[0], self._fire_xy[:, 1] - p[1])
        k = int(np.argmin(d))
        dist = float(d[k])
        if not FIRE_MARGIN_M + 0.3 < dist <= FIRE_CALLOUT_M:
            return None
        f = (float(self._fire_xy[k, 0]), float(self._fire_xy[k, 1]))
        if any(math.dist(f, q) < FIRE_SAME_M and t - when < FIRE_CALLOUT_AGAIN_S for q, when in self._cl_fires):
            return None
        rel = math.degrees(wrap(bearing(p, f) - ff.yaw))
        if abs(rel) <= 35:
            side = "ahead"
        elif 35 < rel <= 145:
            side = "left"
        elif -145 <= rel < -35:
            side = "right"
        else:
            return None  # behind: not worth a warning
        self._cl_fires.append((f, t))
        metres = max(1, round(dist))
        text = f"Fire ahead, {metres} metres. Stay with me." if side == "ahead" else f"Fire on your {side}, {metres} metres."
        return Callout(self._nearest_drone(swarm, f) or self.lead, "fire_side", text,
                       {"side": side, "distance_m": metres}, severity="warn")

    @staticmethod
    def _alive(swarm: SwarmState, drone: str | None) -> bool:
        """The swarm's view (it learns a drone is gone from the missing heartbeat)."""
        return drone is not None and all(sd.alive for sd in swarm.drones if sd.id == drone)

    @staticmethod
    def _at_survivor(swarm: SwarmState, drone: str) -> bool:
        """This drone's chain post is at the confirmed survivor (the survivor drone)."""
        sv = swarm.survivor
        if not sv or sv.status != "confirmed" or sv.x is None or sv.y is None:
            return False
        return any(c.drone == drone and math.dist((c.x, c.y), (sv.x, sv.y)) <= 3.0 for c in swarm.checkpoints)

    def _turn_callout(self, route, proj) -> Callout | None:
        """The next corner 3-8 m ahead along the route (heading change > 50 deg over 3 m): "Left turn in 5 metres."
        Each corner once."""
        if len(route) < 2:
            return None
        total = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
        best = None
        s = proj.s + TURN_CALLOUT_M[0]
        while s <= min(proj.s + TURN_CALLOUT_M[1], total - 1.5):
            a, b, c = point_at(route, s - 1.5), point_at(route, s), point_at(route, s + 1.5)
            if math.dist(a, b) > 0.5 and math.dist(b, c) > 0.5:
                turn = math.degrees(wrap(bearing(b, c) - bearing(a, b)))
                if abs(turn) >= TURN_MIN_DEG and (best is None or abs(turn) > abs(best[1])):
                    best = (s, turn, b)
            s += 0.5
        if best is None or any(math.dist(best[2], q) < 4.0 for q in self._cl_turns):
            return None
        self._cl_turns.append(best[2])
        side, metres = ("Left" if best[1] > 0 else "Right"), max(1, round(best[0] - proj.s))
        return Callout(self.lead, "turn", f"{side} turn in {metres} metres.",
                       {"side": side.lower(), "distance_m": metres}, severity="warn")

    def _nearest_drone(self, swarm: SwarmState, xy: Pt) -> str | None:
        """The chain drone (or lead) closest to a fire, if within 8 m: it is the one watching it."""
        best, bd = None, 8.0
        for dr in sorted(d for d in set(self._chain_drones(swarm)) | {self.lead} if self._alive(swarm, d)):
            q = self._pose_xy(dr)
            if q is not None and math.dist(q, xy) < bd:
                best, bd = dr, math.dist(q, xy)
        return best

    def _survivor_callout(self, p: Pt, swarm: SwarmState) -> Callout | None:
        sv = swarm.survivor
        if self._cl_survivor or not sv or sv.status != "confirmed" or sv.x is None or sv.y is None:
            return None
        d = math.dist(p, (sv.x, sv.y))
        if not SURVIVOR_CALLOUT_MIN_M < d <= SURVIVOR_CALLOUT_M:
            return None
        self._cl_survivor = True
        m = max(2, round(d))
        return Callout(self.lead, "survivor_close", f"Survivor just ahead, {m} metres.", {"distance_m": m})

    def _progress_callout(self, remaining: float) -> Callout | None:
        total = self._cl_total
        if not total or total < 20:
            return None
        n = round(remaining)
        if "half" not in self._cl_marks and 15 < remaining <= total / 2:  # the end is the survivor line's job
            self._cl_marks.add("half")
            self._cl_dist_marks.update(k for k in self._dist_marks(total) if remaining <= k)
            return Callout(self.lead, "progress", f"Halfway there. {n} metres to go.", {"metres_left": n})
        # every 20 m mark walked past (40, 20 ... metres to go), while more than 12 m out
        passed = [k for k in self._dist_marks(total) if remaining <= k and k not in self._cl_dist_marks]
        if remaining > 12 and passed:
            self._cl_dist_marks.update(passed)
            return Callout(self.lead, "progress", f"{n} metres to go.", {"metres_left": n})
        return None

    @staticmethod
    def _dist_marks(total: float) -> list[int]:
        """Multiples of 20 m to go, at least 5 m into the walk: 60 m -> [40, 20]."""
        return [int(k * DISTANCE_EVERY_M) for k in range(1, int(total // DISTANCE_EVERY_M) + 1)
                if k * DISTANCE_EVERY_M < total - 5]

    # ---------------------------------------------------------------- rules
    def _rule(self, t, ff: Pose, p: Pt, swarm: SwarmState, route, proj, status, lead_state):
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
        if fd <= FIRE_MARGIN_M:
            return "danger", "Stop. Fire ahead. Turn back.", "danger", True
        if self._in_danger_cell(p):
            # heat the drones felt. The swarm's firefighter route treats heat as a cost, not a wall (spec 8.3),
            # so on its route this is the best way through: keep moving. Off the route: stop.
            if proj is not None and proj.dist <= EDGE_ON_ROUTE_M:
                return "heat", "Heat on the route. Keep moving.", "warn", False
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

        sv = swarm.survivor
        if sv and sv.status == "confirmed" and sv.x is not None and sv.y is not None \
                and math.dist(p, (sv.x, sv.y)) <= SURVIVOR_CALLOUT_M:
            return "near", f"Survivor ahead. {max(2, round(math.dist(p, (sv.x, sv.y))))} metres.", "info", False
        if lead_state == "return":
            return "returning", "Your drone is coming back to you. Follow the arrow.", "info", False
        if lead_state == "wait":
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
        """Facing more than 120 deg away from where the arrow points (the route just ahead).

        Not the straight line to the next checkpoint: with 1-2 drones left the next checkpoint is the survivor,
        and the bearing to it through the walls says nothing about the corridor the firefighter is in."""
        p = (ff.x, ff.y)
        tgt = self._target(route, proj, p)
        if tgt is None or math.dist(p, tgt) < 0.5:
            return False
        return abs(wrap(bearing(p, tgt) - ff.yaw)) > WRONG_HEADING

    # ---------------------------------------------------------------- lead / chain
    def _chain_drones(self, swarm: SwarmState) -> list[str]:
        return [c.drone for c in swarm.checkpoints if c.drone]

    def _next_checkpoint(self, swarm: SwarmState, route, proj) -> Pt | None:
        ahead = [(project(route, (c.x, c.y)).s, (c.x, c.y)) for c in swarm.checkpoints]
        ahead = [a for a in ahead if a[0] > proj.s + 0.5]
        return min(ahead)[1] if ahead else None

    def _target(self, route, proj, p: Pt) -> Pt | None:
        """Arrow target: the farthest route point up to 3 m ahead whose straight line from the firefighter
        stays within 0.25 m of the route. On straights that's 3 m ahead; at a tight corner or a narrow
        gap it shrinks to the corner, so the arrow never points into a wall or debris. A firefighter
        who has drifted off the route gets the short arrow: straight back onto it."""
        if not proj:
            return None
        L = LEAD_AHEAD_M
        while L > ARROW_MIN_M:
            tgt = point_at(route, proj.s + L)
            k = int(L / 0.2)
            if all(seg_dist(point_at(route, proj.s + L * i / k), p, tgt) <= ARROW_CHORD_TOL_M for i in range(1, k)):
                return tgt
            L -= 0.25
        return point_at(route, proj.s + ARROW_MIN_M)

    def _pose_xy(self, drone: str | None) -> Pt | None:
        if drone is None or self.world is None:
            return None
        q = self.world.drone_pose(drone)
        return (q.x, q.y)

    def _handover(self, swarm: SwarmState, route, proj, p: Pt) -> None:
        """FF within 2 m of the next chain drone -> it becomes the lead."""
        best = None
        for c in swarm.checkpoints:
            if not c.drone or c.drone == self.lead or not self._alive(swarm, c.drone):
                continue
            s = project(route, (c.x, c.y)).s
            if s > proj.s - 1.0 and (best is None or s < best[0]):
                best = (s, c.drone)
        if best is not None:
            q = self._pose_xy(best[1])
            if q is not None and math.dist(p, q) <= HANDOVER_M:
                self.lead = best[1]

    def _lead_cmd(self, route, proj, p: Pt, cmds: dict[str, VelCmd]) -> str:
        """Drive the lead 3 m ahead on the route. Returns "lead", "wait" (the FF fell behind: the lead holds)
        or "return" (the lead is far ahead, e.g. the only drone left still over the survivor: it flies back
        along the route until it is in front of the firefighter again)."""
        if self.lead is None:
            return "none"
        q = self.world.drone_pose(self.lead)
        lead_s = project(route, (q.x, q.y)).s
        gap = lead_s - proj.s
        if gap > RETURN_GAP_M:
            self._returning = True
        elif gap <= LEAD_AHEAD_M + 1.0:
            self._returning = False  # back in front of the firefighter: lead as usual
        if not self._returning and (gap > FOLLOW_GAP_M or math.dist(p, (q.x, q.y)) > FOLLOW_GAP_M + 1.0):
            cmds[self.lead] = VelCmd(vz=LEAD_GAIN * (self.altitude - q.z), wz=self._face(q, p))
            return "wait"
        # step along the route towards the goal (3 m ahead of the FF), never straight through walls
        goal_s = proj.s + LEAD_AHEAD_M
        step = max(-LEAD_CARROT_M, min(LEAD_CARROT_M, goal_s - lead_s))
        goal = point_at(route, lead_s + step)
        vx, vy = LEAD_GAIN * (goal[0] - q.x), LEAD_GAIN * (goal[1] - q.y)
        n = math.hypot(vx, vy)
        if n > config.MAX_SPEED:
            vx, vy = vx * config.MAX_SPEED / n, vy * config.MAX_SPEED / n
        cmds[self.lead] = VelCmd(vx=vx, vy=vy, vz=LEAD_GAIN * (self.altitude - q.z), wz=self._face(q, p))
        return "return" if self._returning else "lead"

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
        self._returning = False
        self._reset_callouts()  # a loaded snapshot starts a fresh conversation
