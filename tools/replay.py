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
Actions: click (x/y, or on_route = fraction 0..1 along the current route), undo, act2, reset, speed,
save (name), ff_input (fwd/strafe/yaw), ff_autopilot (the firefighter follows the guidance arrow), stop.
Exit code 1 if an expected beat never happened, or a step never ran.
"""

from __future__ import annotations

import argparse
import json
import math
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
    "expect": ["SURVIVOR_CONFIRMED", "CHAIN_POSITION", "chain_formed", "REPLAN", "chain_ready", "MISSION_COMPLETE"],
    "steps": [
        # the judge's change tool: debris on the route once the chain holds, then Undo
        {"after": "chain_formed", "wait": 8, "do": "click", "mode": "debris", "on_route": 0.55},
        {"after": "prev", "wait": 12, "do": "undo"},
        # Act 2 from the snapshot (saved once the chain is at its posts), the firefighter walks the arrow
        {"after": "chain_ready", "wait": 3, "do": "act2"},
        {"after": "prev", "wait": 0.5, "do": "ff_autopilot"},
        {"after": "MISSION_COMPLETE", "wait": 3, "do": "stop"},
    ],
}

AUTOPILOT_LOOKAHEAD_M = 1.2
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
            if self.verbose and msg.action in KEY_ACTIONS and msg.t >= t - 1.0:  # not the history re-sent on a load
                self._say(f"{msg.t:7.1f}  {msg.drone:5} {msg.action:18} {msg.reason}")
        elif isinstance(msg, GuidanceMsg):
            self.guidance = msg
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
        # stuck (snagged on a corner): sidestep for a moment
        moved = math.dist((ff.x, ff.y), self._ff_last) if self._ff_last else 1.0
        self._ff_last = (ff.x, ff.y)
        self._stuck = self._stuck + 1 if (fwd > 0 and moved < 0.01) else 0
        strafe = 0.6 * (1 if (self._stuck // 20) % 2 == 0 else -1) if self._stuck > 10 else 0.0
        self.loop.bus.send_command(FFInputMsg(fwd=fwd, strafe=strafe, yaw=heading))

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
            if "on_route" in step:
                pts = self.loop.swarm_state.route.points if self.loop.swarm_state else []
                if len(pts) < 2:
                    self._say(f"{t:7.1f}  REPLAY click skipped: no route yet")
                    self.prev_t = t
                    return
                x, y = _along(pts, float(step["on_route"]))
            bus.send_command(ClickMsg(mode=step.get("mode", "debris"), x=float(x), y=float(y)))
            note = f"{step.get('mode', 'debris')} at ({x:.1f}, {y:.1f})"
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
    return 1 if missing or skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
