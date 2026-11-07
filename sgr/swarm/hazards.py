"""Acting on detections and heat (spec §8.4, rules when unsure).

Per detection id (mock and live perception re-emit the same id on every status change):

    pending (new)      -> the drone that saw it holds and faces it while Gemma looks   INSPECT
    low_conf           -> fly to ~2 m in front of it, face it, then ask for a re-look   INSPECT
    unconfirmed        -> Gemma down / timed out: same closer second view               UNCONFIRMED
    confirmed fire     -> fire cells on the belief grid (planner keeps 1 m away)        CONFIRM_HAZARD
    confirmed person   -> survivor estimate                                             SURVIVOR_CONFIRMED
    rejected           -> drop it (or still unsure after 2 re-looks)                    REJECT_DETECTION

Heat: > 50 °C marks `danger` cells (planner cost), > 70 °C -> RETREAT 3 m away from the heat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sgr import config
from sgr.contracts import Detection, Evidence, Pose, SurvivorState

from .agent import Agent
from .belief import FIRE, FREE, WALL, BeliefGrid
from .blackboard import Blackboard
from .planner import DistanceField, cost_map, path_hits
from .trace import Trace

FIRE_RADIUS_M = 0.5  # cells this close to a confirmed fire become `fire`
DANGER_RADIUS_M = 0.6
HOLD_TIMEOUT_S = config.GEMMA_TIMEOUT_S + 2.0  # stop holding if Gemma never answers
APPROACH_TIMEOUT_S = 25.0  # can't reach a standoff point: ask from where we are
HOT_M = 3.0  # assume a confirmed fire is too hot (> 70 °C) this close: heat_norm 1 ...
WARM_M = 4.0  # ... fading to 0 here (planner cost; no exploration goals in the hot core)
RETREAT_CLEAR_M = 3.5  # retreat to the nearest reachable spot at least this far from the heat
RETREAT_COOLDOWN_S = 4.0
REJECTED_MERGE_M = 1.0  # a new detection this close to a rejected one (same label) is ignored
_RING = np.radians(np.arange(0, 360, 15))


@dataclass
class Inspection:
    det_id: str
    label: str
    target: tuple[float, float]
    drone: str | None
    phase: str = "hold"  # hold (waiting for Gemma) | approach (moving closer) | wait (re-look pending)
    #                      parked (gave up waiting; the next status update restarts it)
    since: float = 0.0
    standoff: tuple[float, float] | None = None
    asked: int = 0  # re-looks requested by the swarm


@dataclass
class Hazards:
    belief: BeliefGrid
    bb: Blackboard
    trace: Trace
    active: dict[str, Inspection] = field(default_factory=dict)
    done: set[str] = field(default_factory=set)  # detection ids already settled
    _last_retreat: dict[str, float] = field(default_factory=dict)
    rejected: list[tuple[str, float, float]] = field(default_factory=list)  # (label, x, y) ruled out before
    fresh_fire: list[tuple[str, str, list[tuple[int, int]]]] = field(default_factory=list)  # (sensor, drone, cells)

    # ------------------------------------------------------------------ detections
    def on_detection(self, t: float, det: Detection, agents: dict[str, Agent], alive: dict[str, Pose]) -> None:
        if det.id in self.done or det.world_xy is None:
            return
        xy = (det.world_xy.x, det.world_xy.y)
        what = _what(det)
        ins = self.active.get(det.id)
        if det.status == "pending":
            if ins is None and any(lab == det.label and math.hypot(x - xy[0], y - xy[1]) <= REJECTED_MERGE_M
                                   for lab, x, y in self.rejected):
                self.done.add(det.id)  # same spot was already ruled out: don't stop for it again
                return
            if ins is None:
                d = self._pick_drone(det.drone, xy, agents, alive)
                ins = self.active[det.id] = Inspection(det.id, det.label, xy, d, "hold", t)
                if d:
                    agents[d].hold(xy)
                self.trace.log(t, det.drone, "INSPECT",
                               f"{det.drone} sees {what}: fast detector {det.score_fast:.2f}, holding for Gemma",
                               Evidence(sensor="camera", value=round(det.score_fast, 3), frame_id=det.frame_id))
                if det.label == "person" and self.bb.survivor.status == "unknown":
                    self.bb.survivor = SurvivorState(status="suspected", x=xy[0], y=xy[1], conf=det.score_fast)
            else:
                ins.phase, ins.since = "wait", t
        elif det.status in ("low_conf", "unconfirmed"):
            if det.relooks >= config.MAX_RELOOKS:
                self._settle(det, agents)
                self.trace.log(t, ins.drone if ins and ins.drone else det.drone, "REJECT_DETECTION",
                               f"Still unsure about {what} after {det.relooks} closer looks: not acting on it",
                               _gemma_evidence(det))
                self.bb.event("false_alarm", drone=det.drone, xy=xy)
                return
            if ins is None:
                d = self._pick_drone(det.drone, xy, agents, alive)
                ins = self.active[det.id] = Inspection(det.id, det.label, xy, d, "approach", t)
            ins.phase, ins.since = "approach", t
            ins.standoff = self._standoff(xy, alive.get(ins.drone) if ins.drone else None)
            if ins.drone:
                agents[ins.drone].approach(ins.standoff, xy)
            if det.status == "unconfirmed":
                self.trace.log(t, ins.drone or det.drone, "UNCONFIRMED",
                               f"No Gemma answer on {what} (fast {det.score_fast:.2f}): getting a second view",
                               Evidence(sensor="camera", value=round(det.score_fast, 3), frame_id=det.frame_id))
            else:
                g = det.gemma
                self.trace.log(t, ins.drone or det.drone, "INSPECT",
                               f"{ins.drone or det.drone} inspecting {what}: fast {det.score_fast:.2f}, "
                               f"Gemma {g.confidence:.2f} — moving closer" if g else
                               f"{ins.drone or det.drone} inspecting {what}: moving closer",
                               _gemma_evidence(det))
        elif det.status == "confirmed":
            self._settle(det, agents)
            if det.label == "person":
                self.bb.survivor = SurvivorState(status="confirmed", x=xy[0], y=xy[1], conf=det.combined)
                self.trace.log(t, det.drone, "SURVIVOR_CONFIRMED",
                               f"Survivor confirmed at ({xy[0]:.1f}, {xy[1]:.1f}): {_scores(det)}",
                               _gemma_evidence(det))
                self.bb.event("survivor_found", drone=det.drone, xy=xy)
            else:
                new_cells = self.mark_fire(xy, agents)
                cells = len(new_cells)
                self.fresh_fire.append(("gemma" if det.gemma else "camera", det.drone, new_cells))
                self.bb.add_hazard("fire", xy, "gemma" if det.gemma else "camera", t, det.combined or det.score_fast)
                self.trace.log(t, det.drone, "CONFIRM_HAZARD",
                               f"Fire confirmed at ({xy[0]:.1f}, {xy[1]:.1f}): {_scores(det)}; "
                               f"{cells} cells no-go (+1 m margin)",
                               _gemma_evidence(det))
                self.bb.event("fire_confirmed", drone=det.drone, xy=xy)
        elif det.status == "rejected":
            self._settle(det, agents)
            self.trace.log(t, det.drone, "REJECT_DETECTION", f"Not {_noun(det)}: {_scores(det)}",
                           _gemma_evidence(det))
            self.bb.event("false_alarm", drone=det.drone, xy=xy)

    def tick(self, t: float, agents: dict[str, Agent], alive: dict[str, Pose]) -> list[str]:
        """Drive the inspections. Returns the detection ids to re-look at this tick."""
        out: list[str] = []
        for ins in list(self.active.values()):
            if ins.phase == "parked":
                continue
            if ins.drone is not None and ins.drone not in alive:  # inspector lost: hand over
                ins.drone = None
            if ins.drone is None:
                ins.drone = self._pick_drone(None, ins.target, agents, alive)
                if ins.drone is None:
                    continue
                if ins.phase == "approach":
                    ins.standoff = self._standoff(ins.target, alive[ins.drone])
                    agents[ins.drone].approach(ins.standoff, ins.target)
                else:
                    agents[ins.drone].hold(ins.target)
            ag = agents[ins.drone]
            if ins.phase in ("hold", "wait") and t - ins.since > HOLD_TIMEOUT_S:
                ag.release()  # Gemma is slow: carry on exploring; the next status update restarts this
                ins.phase, ins.drone = "parked", None
            elif ins.phase == "approach" and (ag.in_position or t - ins.since > APPROACH_TIMEOUT_S):
                out.append(ins.det_id)
                ins.asked += 1
                ins.phase, ins.since = "wait", t
                ag.hold(ins.target)
        return out

    # ------------------------------------------------------------------ heat
    def heat(self, t: float, drone: str, pose: Pose, temp: float, agent: Agent) -> None:
        if temp > config.HEAT_DANGER_C:
            norm = min(1.0, (temp - config.AMBIENT_C) / (config.HEAT_RETREAT_C - config.AMBIENT_C))
            for i, j in self._disc(pose.x, pose.y, DANGER_RADIUS_M):
                self.belief.heat[j, i] = max(self.belief.heat[j, i], norm)
                if self.belief.state[j, i] == FREE:
                    self.belief.set_state(i, j, "danger")
        if (temp > config.HEAT_RETREAT_C and agent.mode != "RETREAT"
                and t - self._last_retreat.get(drone, -math.inf) > RETREAT_COOLDOWN_S):
            self._last_retreat[drone] = t
            src = self._heat_source(pose)
            agent.retreat(self._retreat_path(pose, src), self._away(pose, src))
            self.trace.log(t, drone, "RETREAT", f"{drone} at {temp:.0f} °C (limit {config.HEAT_RETREAT_C:.0f}): "
                                                "backing off 3 m", Evidence(sensor="temp", value=round(temp, 1)))
            self.bb.add_hazard("danger", (pose.x, pose.y), "temp", t)

    # ------------------------------------------------------------------ helpers
    def mark_fire(self, xy: tuple[float, float], agents: dict[str, Agent]) -> list[tuple[int, int]]:
        for i, j in self._disc(*xy, WARM_M):  # assumed hot zone around the fire
            cx, cy = self.belief.cell_center(i, j)
            norm = min(1.0, max(0.0, (WARM_M - math.hypot(cx - xy[0], cy - xy[1])) / (WARM_M - HOT_M)))
            self.belief.heat[j, i] = max(self.belief.heat[j, i], norm)
        self.belief.version += 1  # heat changed: planner costs must refresh
        new = []
        for i, j in self._disc(*xy, FIRE_RADIUS_M):
            if self.belief.state[j, i] not in (FIRE, WALL) and self.belief.set_state(i, j, "fire"):
                new.append((i, j))
        for ag in agents.values():
            if ag.path and path_hits(ag.path, new, self.belief, margin=1.0):
                ag.invalidate()
        return new

    def _settle(self, det: Detection, agents: dict[str, Agent]) -> None:
        self.done.add(det.id)
        if det.status in ("rejected", "low_conf", "unconfirmed") and det.world_xy is not None:
            self.rejected.append((det.label, det.world_xy.x, det.world_xy.y))
        ins = self.active.pop(det.id, None)
        if ins and ins.drone:
            agents[ins.drone].release()

    def _pick_drone(self, preferred: str | None, xy, agents: dict[str, Agent], alive: dict[str, Pose]) -> str | None:
        busy = {i.drone for i in self.active.values() if i.drone} | {d for d, a in agents.items() if a.stationed}
        if preferred in alive and preferred not in busy and agents[preferred].mode != "RETREAT":
            return preferred
        free = [d for d in alive if d not in busy and agents[d].mode != "RETREAT"]
        if not free:
            return None
        return min(free, key=lambda d: math.hypot(alive[d].x - xy[0], alive[d].y - xy[1]))

    def _standoff(self, target: tuple[float, float], pose: Pose | None) -> tuple[float, float]:
        """A reachable point ~2 m from the target with line of sight to it, cheapest from the drone."""
        cm = cost_map(self.belief)
        r = config.INSPECT_STANDOFF_M
        df = DistanceField(self.belief, (pose.x, pose.y)) if pose else None
        best, bc = None, math.inf
        for a in _RING:
            px, py = target[0] + r * math.cos(a), target[1] + r * math.sin(a)
            i, j = self.belief.world_to_cell(px, py)
            if not (self.belief.inside(i, j) and cm.traversable[j, i]) or not self._los((px, py), target):
                continue
            c = df.cost[j, i] if df else 0.0
            if c < bc:
                best, bc = (px, py), c
        if best is None:  # nothing clean at 2 m: just get as close as the drone can
            return (pose.x, pose.y) if pose else target
        return best

    def _los(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        d = math.dist(a, b)
        ang = np.array([math.atan2(b[1] - a[1], b[0] - a[0])])
        rng = self.belief.march(a[0], a[1], ang, d)[3][0]
        return rng >= d - 0.3  # a target mounted on a wall still counts

    def _disc(self, x: float, y: float, r: float) -> list[tuple[int, int]]:
        b = self.belief
        i0, j0 = b.world_to_cell(x, y)
        n = int(math.ceil(r / b.res))
        out = []
        for j in range(j0 - n, j0 + n + 1):
            for i in range(i0 - n, i0 + n + 1):
                if b.inside(i, j):
                    cx, cy = b.cell_center(i, j)
                    if math.hypot(cx - x, cy - y) <= r:
                        out.append((i, j))
        return out

    def _heat_source(self, pose: Pose) -> tuple[float, float] | None:
        """Where the heat comes from: the nearest confirmed fire, else the hottest danger cell seen."""
        fires = [h.xy for h in self.bb.hazards.values() if h.kind == "fire"]
        if fires:
            return min(fires, key=lambda f: math.hypot(f[0] - pose.x, f[1] - pose.y))
        jj, ii = np.nonzero(self.belief.heat > 0)
        if len(ii) == 0:
            return None
        k = int(np.argmax(self.belief.heat[jj, ii]))
        return self.belief.cell_center(int(ii[k]), int(jj[k]))

    def _away(self, pose: Pose, src: tuple[float, float] | None) -> tuple[float, float]:
        if src is not None:
            dx, dy = pose.x - src[0], pose.y - src[1]
            n = math.hypot(dx, dy)
            if n > 1e-6:
                return dx / n, dy / n
        return -math.cos(pose.yaw), -math.sin(pose.yaw)  # back off the way we were facing

    def _retreat_path(self, pose: Pose, src: tuple[float, float] | None) -> list[tuple[float, float]] | None:
        """Path to the nearest reachable cell >= RETREAT_CLEAR_M from the heat (corridors: not into a wall)."""
        if src is None:
            return None
        b = self.belief
        df = DistanceField(b, (pose.x, pose.y))
        jj, ii = np.mgrid[0 : b.h, 0 : b.w]
        cx = b.origin[0] + (ii + 0.5) * b.res
        cy = b.origin[1] + (jj + 0.5) * b.res
        far = np.hypot(cx - src[0], cy - src[1]) >= RETREAT_CLEAR_M
        goal = df.nearest(far)
        return df.path_to(goal) if goal is not None else None


def _noun(det: Detection) -> str:
    return {"fire": "fire", "person": "a person"}.get(det.label, det.label)


def _what(det: Detection) -> str:
    return {"fire": "possible fire", "person": "a possible person"}.get(det.label, det.label)


def _scores(det: Detection) -> str:
    s = f"fast {det.score_fast:.2f}"
    if det.gemma is not None:
        s += f", Gemma {det.gemma.confidence:.2f} “{det.gemma.caption}”"
    if det.combined is not None:
        s += f" -> {det.combined:.2f}"
    return s


def _gemma_evidence(det: Detection) -> Evidence:
    if det.gemma is not None:
        return Evidence(sensor="gemma", frame_id=det.frame_id,
                        value={"combined": det.combined, "confidence": det.gemma.confidence,
                               "caption": det.gemma.caption, "relooks": det.relooks})
    return Evidence(sensor="camera", value=round(det.score_fast, 3), frame_id=det.frame_id)
