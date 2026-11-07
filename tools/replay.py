"""Replay: run the demo with scripted judge actions, for rehearsing hands-free (spec §12). Owned by S.

    uv run python tools/replay.py                                   # default demo script, headless, fast
    uv run python tools/replay.py tools/replay_demo.json --live     # with the dashboard, real time
    uv run python tools/replay.py my.json --map maps/test --speed 2.5 --live
    uv run python tools/replay.py --mock-latency 7                  # rehearse live-Gemma timing headless

Steps are injected through the same Bus commands the browser sends (click, undo, act2_start, control,
ff_input), so a replay exercises exactly the code a judge would.

Script (JSON):
    {
      "map": "maps/marshgate-route",          (optional; --map overrides)
      "max_t": 300,                           (stop at this sim time)
      "expect": ["SURVIVOR_CONFIRMED", "CHAIN_POSITION", "REPLAN"],   (trace actions / event kinds)
      "steps": [
        {"t": 20, "do": "speed", "value": 2.5},
        {"after": "chain_formed", "wait": 8, "do": "click", "mode": "debris", "on_route": 0.6},
        {"after": "prev", "wait": 10, "do": "undo"},
        {"after": "prev", "wait": 5, "do": "act2"},
        {"after": "prev", "wait": 0, "do": "ff_autopilot"},
        {"after": "MISSION_COMPLETE", "wait": 3, "do": "stop"}
      ]
    }

Triggers: "t" = sim time; "after" = an event kind (chain_formed, replan, drone_lost, ...), a trace action
(SURVIVOR_CONFIRMED, ...), a swarm phase (chain_ready = chain drones at their posts, no_route) or "prev"
(when the previous step ran); "wait" = sim seconds after that.
Actions: click (x/y, on_route = fraction 0..1 along the current route, or ahead_of_ff = metres ahead of
the firefighter along the route), undo, act2, reset, speed,
save (name), ff_input (fwd/strafe/yaw), ff_autopilot (the firefighter follows the guidance arrow), stop.
Every click must get a reaction: the World must accept it (not within 1 m of a drone / the firefighter /
the survivor, on a free cell) and the swarm must notice it (MARK_BLOCKED near it, or a route REPLAN /
NO_ROUTE) within CLICK_REACT_S. A click no drone can see (> 6 m LiDAR range from all of them) is reported,
because live it would be a silent beat.
Exit code 1 if an expected beat never happened, a step never ran, or a click got no reaction.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sgr.bus import Bus  # noqa: E402
from sgr.contracts import (  # noqa: E402
    Act2StartMsg,
    ClickMsg,
    ControlMsg,
    EventMsg,
    FFInputMsg,
    GuidanceMsg,
    PriorMap,
    Scenario,
    SnapshotMsg,
    SummaryMsg,
    TraceMsg,
    UndoMsg,
)
from sgr.guidance import Guidance  # noqa: E402
from sgr.loop import SimLoop  # noqa: E402
from sgr.perception import Perception  # noqa: E402
from sgr.swarm import Swarm  # noqa: E402

DEFAULT_SCRIPT: dict[str, Any] = {
    "map": "maps/marshgate-route",
    "max_t": 320,
    "expect": ["SURVIVOR_CONFIRMED", "CHAIN_POSITION", "chain_formed", "REPLAN", "chain_ready", "MISSION_COMPLETE",
               "summary"],
    "steps": [
        # the judge's change tool: debris on the route once the chain holds, then Undo. Pinned to W's tested
        # spot just south of the n4 junction (wide, D1 in LiDAR range): REPLAN with a ~+1 m detour.
        # The presenter should click the same spot. (on_route 0.55 fell in the n8-n9 corridor out of range.)
        {"after": "chain_formed", "wait": 8, "do": "click", "mode": "debris", "x": 5.8, "y": 24.8},
        {"after": "prev", "wait": 12, "do": "undo"},
        # Act 2 from the snapshot (saved once the chain is at its posts), the firefighter walks the arrow
        {"after": "chain_ready", "wait": 3, "do": "act2"},
        {"after": "prev", "wait": 0.5, "do": "ff_autopilot"},
        # §9 "Route becomes unsafe": debris lands ahead of the walking firefighter, then the judge undoes it
        {"after": "prev", "wait": 15, "do": "click", "mode": "debris", "ahead_of_ff": 5},
        {"after": "prev", "wait": 8, "do": "undo"},
        {"after": "MISSION_COMPLETE", "wait": 3, "do": "stop"},
    ],
}

AUTOPILOT_LOOKAHEAD_M = 1.2
AUTOPILOT_BEHIND_M = 1.0  # stay at least this far behind the lead along the route
CLICK_REACT_S = 30.0  # a judge click must be noticed within this (sim s): the patrol sweeps ~30 m stretches
CLICK_NEAR_M = 2.5  # a MARK_BLOCKED this close to the click counts as noticing it
KEY_ACTIONS = {"CONFIRM_HAZARD", "REJECT_DETECTION", "RETREAT", "DRONE_LOST", "SURVIVOR_CONFIRMED", "REPLAN",
               "NO_ROUTE", "CHAIN_POSITION", "FF_CORRECTION", "MISSION_COMPLETE", "MARK_BLOCKED", "INSPECT"}


class Replay:
    def __init__(self, loop: SimLoop, script: dict[str, Any], verbose: bool = True) -> None:
        self.loop, self.script, self.verbose = loop, script, verbose
        self.steps = [dict(s) for s in script.get("steps", [])]
        self.next = 0
        self.happened: dict[str, float] = {}  # event kind / trace action -> first sim time
        self.prev_t: float | None = 0.0
        self.autopilot = False
        self.stopped = False
        self.guidance: GuidanceMsg | None = None
        self.log: list[str] = []
        self._last_t = loop.t
        self._ff_last: tuple[float, float] | None = None
        self._click: dict[str, Any] | None = None  # the last click, waiting for a reaction
        self.click_failures: list[str] = []
        self._stuck = 0
        self._orig_publish = loop._publish
        loop._publish = self._tap  # see every message the loop publishes, with or without a server

    # ------------------------------------------------------------------ taps
    def _tap(self, msg: Any) -> None:
        t = self.loop.t
        if isinstance(msg, EventMsg):
            self.happened.setdefault(msg.kind, t)
        elif isinstance(msg, TraceMsg):
            self.happened.setdefault(msg.action, t)
            self._check_reaction(msg, t)
            if self.verbose and msg.action in KEY_ACTIONS and msg.t >= t - 1.0:  # not the history re-sent on a load
                self._say(f"{msg.t:7.1f}  {msg.drone:5} {msg.action:18} {msg.reason}")
        elif isinstance(msg, GuidanceMsg):
            if self.verbose and msg.say and (self.guidance is None or msg.say != self.guidance.say):
                self._say(f"{t:7.1f}  VOICE “{msg.say}”")
            self.guidance = msg
        elif isinstance(msg, SummaryMsg):
            self.happened.setdefault("summary", t)
            self._say(f"{t:7.1f}  SUMMARY {msg.time_s:.0f} s, walked {msg.distance_m:.0f} m, "
                      f"{msg.hazards_avoided} hazards avoided, {msg.corrections} corrections, {msg.replans} re-plans")
        self._orig_publish(msg)

    def _say(self, line: str) -> None:
        self.log.append(line)
        print(line, flush=True)

    # ------------------------------------------------------------------ driving
    def before_tick(self) -> None:
        if abs(self.loop.t - self._last_t) > 1.0:  # a snapshot load moved the clock: "prev" counts from now
            self.prev_t = self.loop.t
        self._last_t = self.loop.t + 0.1
        ss = self.loop.swarm_state
        if ss is not None and ss.phase:  # phases are triggers too: explore, chain, chain_ready, no_route
            self.happened.setdefault(ss.phase, self.loop.t)
        self._watch_click()
        while self.next < len(self.steps) and self._due(self.steps[self.next]):
            self._run(self.steps[self.next])
            self.next += 1
        if self.autopilot and self.loop.act == 2:
            self._autopilot()

    def _autopilot(self) -> None:
        """A rehearsal firefighter: walks the swarm's route with a short lookahead (doesn't cut tight
        corners the way a 3 m arrow would), sidesteps if it stops making progress. Falls back to the
        guidance arrow when there is no route."""
        ff = self.loop.world.ff_pose()
        ss = self.loop.swarm_state
        pts = ss.route.points if ss is not None and ss.route.status == "ok" else []
        if len(pts) >= 2:
            dx, dy = _lookahead(pts, (ff.x, ff.y), AUTOPILOT_LOOKAHEAD_M)
            heading = math.atan2(dy, dx)
            done = math.dist((ff.x, ff.y), pts[-1]) < 1.0
        elif self.guidance is not None:
            heading = ff.yaw + math.radians(self.guidance.arrow_deg)
            done = self.guidance.distance <= 0.5
        else:
            return
        err = (heading - ff.yaw + math.pi) % (2 * math.pi) - math.pi
        fwd = 0.0 if done else (1.0 if abs(err) < math.radians(50) else 0.3)
        # a firefighter told "Follow the drone" stays behind it: don't overtake the guidance lead
        lead = getattr(self.loop.guidance, "lead", None)
        if lead and len(pts) >= 2 and lead not in self.loop.dead and not done:
            q = self.loop.world.drone_pose(lead)
            if _arc(pts, (q.x, q.y)) < _arc(pts, (ff.x, ff.y)) + AUTOPILOT_BEHIND_M:
                fwd = min(fwd, 0.15)  # let it get ahead again
        # stuck (snagged on a corner): sidestep for a moment
        moved = math.dist((ff.x, ff.y), self._ff_last) if self._ff_last else 1.0
        self._ff_last = (ff.x, ff.y)
        self._stuck = self._stuck + 1 if (fwd > 0 and moved < 0.01) else 0
        strafe = 0.6 * (1 if (self._stuck // 20) % 2 == 0 else -1) if self._stuck > 10 else 0.0
        self.loop.bus.send_command(FFInputMsg(fwd=fwd, strafe=strafe, yaw=heading))

    def _watch_click(self) -> None:
        c = self._click
        if c is None:
            return
        if not c["checked"] and self.loop.t > c["t"] + 0.05:  # the loop has applied it by now
            c["checked"] = True
            if len(self.loop.click_stack) <= c["n"]:
                self._fail_click(f"click at ({c['x']:.1f}, {c['y']:.1f}) refused by the World "
                                 "(within 1 m of a drone / the firefighter / the survivor, or not a free cell)")
                return
        if self.loop.t - c["t"] > CLICK_REACT_S:
            near = min((math.hypot(p.x - c["x"], p.y - c["y"]) for d in self.loop.scenario.drones.ids
                        if d not in self.loop.dead for p in [self.loop.world.drone_pose(d)]), default=math.inf)
            self._fail_click(f"click at ({c['x']:.1f}, {c['y']:.1f}) not noticed in {CLICK_REACT_S:.0f} s "
                             f"(nearest drone {near:.1f} m away; LiDAR range 6 m): a silent beat live")

    def _fail_click(self, why: str) -> None:
        self._say(f"{self.loop.t:7.1f}  REPLAY WARNING {why}")
        self.click_failures.append(why)
        self._click = None

    def _check_reaction(self, msg: TraceMsg, t: float) -> None:
        c = self._click
        if c is None or t < c["t"]:
            return
        hit = msg.drone == "swarm" and msg.action in ("REPLAN", "NO_ROUTE")
        if msg.action == "MARK_BLOCKED":
            m = re.search(r"near \(([-\d.]+), ([-\d.]+)\)", msg.reason)
            hit = bool(m) and math.hypot(float(m[1]) - c["x"], float(m[2]) - c["y"]) <= CLICK_NEAR_M
        if hit:
            self._say(f"{t:7.1f}  REPLAY click noticed after {t - c['t']:.1f} s ({msg.action} by {msg.drone})")
            self._click = None

    def _due(self, step: dict[str, Any]) -> bool:
        if "t" in step:
            return self.loop.t >= float(step["t"])
        after = step.get("after", "prev")
        base = self.prev_t if after == "prev" else self.happened.get(after)
        return base is not None and self.loop.t >= base + float(step.get("wait", 0.0))

    def _run(self, step: dict[str, Any]) -> None:
        do, bus, t = step["do"], self.loop.bus, self.loop.t
        note = ""
        if do == "click":
            x, y = step.get("x"), step.get("y")
            if "ahead_of_ff" in step:  # Act 2: N metres ahead of the firefighter along the route
                pts = self.loop.swarm_state.route.points if self.loop.swarm_state else []
                if len(pts) < 2:
                    self._say(f"{t:7.1f}  REPLAY click skipped: no route yet")
                    self.prev_t = t
                    return
                ff = self.loop.world.ff_pose()
                dx, dy = _lookahead(pts, (ff.x, ff.y), float(step["ahead_of_ff"]))
                x, y = ff.x + dx, ff.y + dy
            elif "on_route" in step:
                pts = self.loop.swarm_state.route.points if self.loop.swarm_state else []
                if len(pts) < 2:
                    self._say(f"{t:7.1f}  REPLAY click skipped: no route yet")
                    self.prev_t = t
                    return
                x, y = _along(pts, float(step["on_route"]))
            bus.send_command(ClickMsg(mode=step.get("mode", "debris"), x=float(x), y=float(y)))
            note = f"{step.get('mode', 'debris')} at ({x:.1f}, {y:.1f})"
            self._click = {"t": t, "x": float(x), "y": float(y), "n": len(self.loop.click_stack), "checked": False}
        elif do == "undo":
            bus.send_command(UndoMsg())
        elif do == "act2":
            bus.send_command(Act2StartMsg())
        elif do == "reset":
            bus.send_command(ControlMsg(action="reset"))
        elif do == "speed":
            bus.send_command(ControlMsg(action="speed", value=float(step["value"])))
            note = f"x{step['value']}"
        elif do == "save":
            bus.send_command(SnapshotMsg(op="save", name=step["name"]))
            note = step["name"]
        elif do == "ff_input":
            bus.send_command(FFInputMsg(fwd=step.get("fwd", 0.0), strafe=step.get("strafe", 0.0),
                                        yaw=step.get("yaw", self.loop.world.ff_pose().yaw)))
        elif do == "ff_autopilot":
            self.autopilot = bool(step.get("on", True))
        elif do == "stop":
            self.stopped = True
        else:
            raise ValueError(f"unknown replay action: {do}")
        self.prev_t = t
        self._say(f"{t:7.1f}  REPLAY {do} {note}".rstrip())


def _arc(pts: list[tuple[float, float]], p: tuple[float, float]) -> float:
    """Arc length of p's projection onto the polyline."""
    best_s, s_acc, best_d = 0.0, 0.0, math.inf
    for a, b in zip(pts, pts[1:]):
        L = math.dist(a, b)
        u = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / L**2))
        d = math.dist(p, (a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])))
        if d < best_d:
            best_d, best_s = d, s_acc + u * L
        s_acc += L
    return best_s


def _lookahead(pts: list[tuple[float, float]], p: tuple[float, float], ahead: float) -> tuple[float, float]:
    """Vector from p to the route point `ahead` metres past p's projection onto the route."""
    best_s, s_acc, best_d = 0.0, 0.0, math.inf
    for a, b in zip(pts, pts[1:]):
        L = math.dist(a, b)
        u = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / L**2))
        q = (a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]))
        d = math.dist(p, q)
        if d < best_d:
            best_d, best_s = d, s_acc + u * L
        s_acc += L
    gx, gy = _along(pts, min(1.0, (best_s + ahead) / s_acc) if s_acc > 0 else 1.0)
    return gx - p[0], gy - p[1]


def _along(pts: list[tuple[float, float]], frac: float) -> tuple[float, float]:
    """Point at `frac` of the polyline's length."""
    seg = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    goal = max(0.0, min(1.0, frac)) * sum(seg)
    for (a, b), d in zip(zip(pts, pts[1:]), seg):
        if goal <= d and d > 0:
            u = goal / d
            return a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])
        goal -= d
    return pts[-1]


def build(map_dir: Path, args: argparse.Namespace, bus: Bus) -> SimLoop:
    from sgr.world import FakeWorld, World

    sc = Scenario.load(map_dir)
    world = (FakeWorld if args.fake else World)(map_dir, sc)
    prior = PriorMap.from_map(map_dir, sc)
    kw = {"mock_latency_s": args.mock_latency} if args.mock_latency else {}
    perception = Perception(mode=args.perception, ground_truth=world.ground_truth, line_of_sight=world.line_of_sight,
                            pixel_to_world=world.pixel_to_world, **kw)
    return SimLoop(world, perception, Swarm(prior, list(sc.drones.ids)), sc,
                   guidance=Guidance(prior=prior, world=world), bus=bus, speed=args.speed,
                   render=args.live and not args.no_render, realtime=args.live, map_dir=map_dir,
                   snapshot_dir=Path(args.snapshots))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("script", nargs="?", type=Path, help="replay JSON (default: built-in demo script)")
    ap.add_argument("--map", type=Path, default=None)
    ap.add_argument("--live", action="store_true", help="real time + dashboard server (watch it in the browser)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-render", action="store_true", help="with --live: skip camera rendering")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--perception", choices=["mock", "remote", "local"], default="mock")
    ap.add_argument("--mock-latency", type=float, default=None, help="mock Gemma latency in s (live is ~7)")
    ap.add_argument("--fake", action="store_true", help="kinematic FakeWorld instead of MuJoCo (fastest)")
    ap.add_argument("--snapshots", default="snapshots")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    script = json.loads(args.script.read_text(encoding="utf-8")) if args.script else DEFAULT_SCRIPT
    map_dir = args.map or ROOT / script.get("map", "maps/marshgate-route")
    bus = Bus()
    loop = build(Path(map_dir), args, bus)
    if args.live:
        from sgr.server.server import start_server

        start_server(bus, map_dir, loop.scenario, "127.0.0.1", args.port)
        if not loop.perception.wait_ready(timeout=120.0):
            print("replay: WARNING perception not ready, starting anyway")
    rp = Replay(loop, script, verbose=not args.quiet)
    max_t = float(script.get("max_t", 300))
    wall0 = time.perf_counter()
    print(f"replay: {map_dir} {'live' if args.live else 'headless'} speed={args.speed} steps={len(rp.steps)}")
    try:
        while loop.t < max_t and not rp.stopped:
            rp.before_tick()
            if not args.live:
                bus.drain_outbound(100000)  # nobody listens headless: keep the queue empty
            loop.run(max_ticks=loop.ticks + 1) if args.live else loop.tick()
    except KeyboardInterrupt:
        pass

    # ------------------------------------------------------------------ report
    print(f"\nreplay: stopped at t={loop.t:.1f}s ({time.perf_counter() - wall0:.0f} s wall)")
    missing = [e for e in script.get("expect", []) if e not in rp.happened]
    skipped = [s["do"] for s in rp.steps[rp.next:] if s["do"] != "stop"]
    for e in script.get("expect", []):
        print(f"  {'ok  ' if e in rp.happened else 'MISS'} {e:20} {rp.happened.get(e, '')}")
    if skipped:
        print(f"  steps that never ran: {skipped}")
    if rp._click is not None:  # the run ended while still waiting
        rp._fail_click(f"click at ({rp._click['x']:.1f}, {rp._click['y']:.1f}) never noticed before the end")
    for why in rp.click_failures:
        print(f"  CLICK {why}")
    return 1 if missing or skipped or rp.click_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
