"""Sim loop (owned by S). Runs on the MAIN thread (GL must live here). §3.

One control tick = CONTROL_DT sim seconds:
  1 drain inbound commands   2 timed scenario events (flare-up)
  3 sensors + camera frames  4 perception poll
  5 Swarm.tick               6 Guidance.tick (Act 2)
  7 World.step(SUBSTEPS)     8 publish to the bus

Snapshots (§8.6, when `snapshot_dir` is set): `act1_start` is saved before the first tick and
`act2_start` once the chain drones are at their posts (phase "chain_ready"). Dashboard buttons: Restart Act 1 (control reset) loads
`act1_start`, Jump to Act 2 loads `act2_start` (snapshots/ or the committed maps/<name>/act2_start.pkl).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from sgr import config, snapshot
from sgr.bus import Bus
from sgr.contracts import (
    Act2StartMsg,
    Box,
    CamInfo,
    ClickMsg,
    ControlMsg,
    Detection,
    DetectionMsg,
    DroneState,
    FFInputMsg,
    FFState,
    GemmaView,
    Handle,
    MapDeltaMsg,
    Pose,
    Readings,
    Scenario,
    SnapshotMsg,
    StateMsg,
    SummaryMsg,
    SwarmState,
    TraceMsg,
    TraceRow,
    UndoMsg,
    VelCmd,
)

GUIDANCE_HOT_C = 60.0  # at/above this the swarm (not Guidance) moves the drone: it never enters > 70 °C
GUIDANCE_COOL_C = 45.0  # ... and keeps it until it has cooled below this
RESYNC_S = 5.0  # wall seconds between full belief-map resyncs (late-joining browsers catch up)
FLARE_NEAR_M = 3.7  # flare-up hits the drone nearest a fire if this close: the watcher AT its post (3.3 m)
FLARE_CORRIDOR_M = 2.3  # never flare a drone this close to a through-corridor (posts are >= 2.5 m; a drone holds within 0.4 m) ...
SIGNAL_OFF_CORRIDOR_M = 2.0  # a signal-loss victim this far from the corridors counts as "out of the way"
FLARE_CLEAR_M = 3.0  # no other drone this close to the flare-up victim
FLARE_CORRIDOR_MAX_C = 140.0  # ... nor where the corridor would get this hot (drones may cross up to 150 °C)
FLARE_FALLBACK_M = 6.0  # after the wait: the drone nearest a fire if this close, else no flare-up
FLARE_MAX_WAIT_S = 70.0  # flare.t is a "not before": wait this long for a drone to be near a fire
_WAIT = "__wait__"  # _flare_victim(): nobody near a fire yet, keep the flare pending
_MOCK_FRAME = np.zeros((config.CAM_H, config.CAM_W, 3), dtype=np.uint8)


class SimLoop:
    def __init__(
        self,
        world: Any,
        perception: Any,
        swarm: Any,
        scenario: Scenario,
        guidance: Any | None = None,
        bus: Bus | None = None,
        budget: Any | None = None,
        speed: float = 1.0,
        render: bool = True,
        realtime: bool = True,
        map_dir: Path | None = None,
        snapshot_dir: Path | None = None,
    ) -> None:
        self.world, self.perception, self.swarm, self.guidance = world, perception, swarm, guidance
        self.scenario, self.bus, self.budget = scenario, bus, budget
        self.speed, self.render, self.realtime = speed, render, realtime
        self.t = 0.0
        self.ticks = 0
        self.act = 1
        self.paused = False
        self.dead: set[str] = set()
        self.trace: list[TraceRow] = []
        self.click_stack: list[Handle] = []
        self.swarm_state: SwarmState | None = None
        self._flared = False
        self.flare_fire: Handle | None = None
        self.flare_victim: str | None = None  # the drone that fails in Act 1 (flare-up, or signal loss)
        self.signal_lost: str | None = None  # set when it failed by signal loss instead of the flare-up
        self._person_seen = False  # any detection of the survivor yet (triggers the failure if it's still due)
        self._frame_seq = 0
        self._cam_rr = 0
        self._last_state_pub = -math.inf
        self._last_budget_pub = -math.inf
        self._last_resync = -math.inf
        self.map_dir = Path(map_dir) if map_dir is not None else None
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else None  # None = no snapshots
        self._act2_saved = False
        self.last_snapshot: Path | None = None
        self._reanchor = False  # set when sim time jumps or the speed changes (run() re-paces)
        self._act2_t: float | None = None
        self._hot: set[str] = set()  # drones Guidance may not steer right now (heat)
        self.summary: SummaryMsg | None = None

    # ------------------------------------------------------------------
    def run(self, max_ticks: int | None = None) -> None:
        wall0, sim0 = time.perf_counter(), self.t
        while max_ticks is None or self.ticks < max_ticks:
            if self.paused:
                self._drain_commands()
                time.sleep(0.05)
                wall0, sim0 = time.perf_counter(), self.t
                continue
            self.tick()
            if self._reanchor:  # snapshot load (sim time jumped) or speed change: pace from here
                self._reanchor = False
                wall0, sim0 = time.perf_counter(), self.t
                continue
            if self.realtime and self.speed > 0:
                ahead = (self.t - sim0) / self.speed - (time.perf_counter() - wall0)
                if ahead > 0:
                    time.sleep(ahead)

    def tick(self) -> None:
        w, s = self.world, self.scenario
        if self.ticks == 0 and self.snapshot_dir is not None:
            self._save("act1_start")  # Restart Act 1 comes back here
        # 1 inbound commands
        if self._drain_commands():
            return  # a snapshot was loaded: the next tick starts from the restored state
        # 2 timed events: one drone fails during the exploration (§7.4), always BEFORE the survivor is found:
        # at flare.t, or the moment any drone first glimpses the survivor, whichever comes first. A flare-up if
        # a drone is somewhere safe for one, otherwise that drone loses its signal on the spot (no fire placed).
        if s.flare and not self._flared and (self.t >= s.flare.t or self._person_seen):
            victim = self._flare_victim(waited=FLARE_MAX_WAIT_S)  # decide now: no more waiting
            if victim == _WAIT:
                victim = None
            if victim is not None:
                p = w.drone_pose(victim)
                # force: bypass the 1 m click clearance (the victim drone is AT this spot).
                # Not pushed on click_stack, so Undo can't remove the flare-up fire.
                self.flare_fire = w.place_fire(p.x, p.y, force=True)
                w.kill_drone(victim)
                self.dead.add(victim)
                self.flare_victim = victim
                self._flared = True
            else:  # no safe flare-up: one drone still fails, by losing its signal (no fire, so no route risk)
                victim = self._signal_loss_victim()
                if victim is not None:
                    w.kill_drone(victim)  # radio/power failure: it goes dark and stops publishing
                    self.dead.add(victim)
                    self.flare_victim = victim
                    self.signal_lost = victim
                    print(f"sgr: {victim} lost its signal at t={self.t:.1f}s (no safe spot for a flare-up)")
            self._flared = True
        # 3 sensors (a dead drone publishes nothing = no heartbeat)
        alive = [d for d in s.drones.ids if d not in self.dead]
        readings = {
            d: Readings(
                t=self.t,
                pose=w.drone_pose(d),
                lidar=[float(r) for r in w.lidar(d)],
                temperature=float(w.temperature(d)),
            )
            for d in alive
        }
        if alive:  # one camera per tick, round-robin  TODO(S): wall-clock 10 fps pacing
            d = alive[self._cam_rr % len(alive)]
            self._cam_rr += 1
            self._submit_frame(d, readings[d].pose)
        # 4 perception
        detections = self.perception.poll()
        for det in detections:
            self._publish(_detection_msg(det))
            if det.label == "person":
                self._person_seen = True  # the drone failure must happen before the survivor is confirmed
        # 5 swarm
        ff = w.ff_pose() if self.act == 2 else None  # Act 2: the firefighter's tracker feeds the re-route
        cmds, rows, self.swarm_state = self.swarm.tick(self.t, readings, detections, ff=ff)
        # the swarm asks for a closer look only once a drone is ~2 m in front of the target (§8.4)
        for det_id in self.swarm_state.relook_requests:
            self.perception.request_relook(det_id)
        # 6 guidance: Act 2 steers the firefighter; in Act 1 it only folds in hazards (map_delta)
        if self.guidance is not None:
            if self.act == 2:
                gmsg, gcmds = self.guidance.tick(self.t, w.ff_pose(), self.swarm_state)
                # safety first: Guidance may steer the lead, but never a drone the swarm is pulling out of heat,
                # nor one already at GUIDANCE_HOT_C (it would drive it back in as soon as a retreat ends)
                # (with hysteresis: once vetoed, a drone stays with the swarm until it cooled below GUIDANCE_COOL_C,
                # otherwise the two take turns every tick and the drone jitters at the edge of the heat)
                self._hot |= {sd.id for sd in self.swarm_state.drones if sd.status == "RETREAT"}
                self._hot |= {d for d, r in readings.items() if r.temperature >= GUIDANCE_HOT_C}
                self._hot -= {d for d, r in readings.items() if r.temperature < GUIDANCE_COOL_C}
                retreating = {sd.id for sd in self.swarm_state.drones if sd.status == "RETREAT"}
                for d, c in gcmds.items():
                    if d not in self._hot:
                        cmds[d] = c
                    elif d not in retreating:  # too hot to go on: hold at the edge of the heat (RETREAT wins > 70 °C)
                        cmds[d] = VelCmd(vz=c.vz)
                self._publish(gmsg)
                drain = getattr(self.guidance, "drain_callouts", None)  # drone voices (Gemini-worded when online)
                for cm in drain() if drain else ():
                    self._publish(cm)
            else:
                self.guidance.observe(self.swarm_state)
            rows = rows + self.guidance.drain_trace()
        for d, c in cmds.items():
            if d not in self.dead:
                w.set_drone_cmd(d, c.vx, c.vy, c.vz, c.wz)
        # 7 physics
        w.step(config.SUBSTEPS)
        self.t += config.CONTROL_DT
        self.ticks += 1
        if self.budget is not None:
            self.budget.on_tick(self.t)
        # 8 publish
        for row in rows:
            self.trace.append(row)
            self._publish(TraceMsg(**row.model_dump()))
            if row.action == "MISSION_COMPLETE" and self.act == 2:
                self.summary = self._summary()
                self._publish(self.summary)
        for ev in self.swarm_state.events:
            self._publish(ev)
        ss = self.swarm_state
        if ss.map_delta or ss.searched_delta:
            self._publish(MapDeltaMsg(cells=ss.map_delta, searched=ss.searched_delta))
        now = time.perf_counter()
        if self.bus is not None and now - self._last_resync >= RESYNC_S:
            self._last_resync = now
            self._publish_full_map()
        if ss.phase == "chain_ready" and not self._act2_saved and self.snapshot_dir is not None:
            self._act2_saved = True
            self._save("act2_start")  # Jump to Act 2 starts here: chain drones at their posts, lead at the entrance
        if self.bus is not None and now - self._last_state_pub >= 1.0 / config.STATE_HZ:
            self._last_state_pub = now
            self._publish(self._state_msg(readings))
        if self.budget is not None and self.bus is not None and now - self._last_budget_pub >= 1.0:
            self._last_budget_pub = now
            self._publish(self.budget.message(self.perception.metrics(), self.speed))

    # ------------------------------------------------------------------
    def _flare_victim(self, waited: float = math.inf) -> str | None:
        """The flare-up hits the alive drone nearest a burning fire (the one watching it, §5 beat).

        Simulation side, so reading the World's real fires is fine here (never in sgr/swarm).
        If no drone is near a fire yet (slow Gemma), the flare waits up to FLARE_MAX_WAIT_S, then
        falls back to scenario.flare.target. Returning the sentinel _WAIT keeps it pending.
        """
        alive = [d for d in self.scenario.drones.ids if d not in self.dead]
        if not alive:
            return None
        gt = getattr(self.world, "ground_truth", None)
        fires = gt().fires if gt else []
        if not fires:
            return self.scenario.flare.target if self.scenario.flare.target in alive else None

        def dist(d: str) -> float:
            p = self.world.drone_pose(d)
            return min(math.hypot(p.x - f.x, p.y - f.y) for f in fires)

        # never where the new fire would cut the way to the survivor for good (no Undo for a flare-up): at least
        # FLARE_CORRIDOR_M from every through-corridor, and every corridor still under FLARE_CORRIDOR_MAX_C
        alive = [d for d in alive if self._flare_safe(d, fires)]
        if not alive:
            return _WAIT if waited < FLARE_MAX_WAIT_S else None
        # the beat: the fire flares up and engulfs the drone the swarm has watching it
        ss = self.swarm_state
        watching = [sd.id for sd in (ss.drones if ss else []) if sd.status == "WATCHING" and sd.id in alive]
        if watching:
            return watching[0]
        best = min(alive, key=dist)
        if dist(best) <= FLARE_NEAR_M:
            return best
        if waited < FLARE_MAX_WAIT_S:
            return _WAIT
        if dist(best) <= FLARE_FALLBACK_M:  # nobody at a watch post: the drone nearest the fire, if near enough
            return best
        print(f"sgr: flare-up skipped: no drone within {FLARE_FALLBACK_M:.0f} m of a fire after {waited:.0f} s")
        return None  # a flare-up far from the fires could cut the only route: skip the beat instead

    def _signal_loss_victim(self) -> str | None:
        """Which drone loses its signal when no flare-up is possible: one whose loss shows the swarm adapting
        without wasting a key role -- the fire watcher, else an explorer, else a checkpoint drone; never the
        drone over the survivor or the lead (the swarm would cope, but the demo reads better)."""
        alive = [d for d in self.scenario.drones.ids if d not in self.dead]
        if len(alive) < 2:
            return None  # never the last drone
        info = {sd.id: sd for sd in (self.swarm_state.drones if self.swarm_state else [])}

        def rank(d: str) -> int:
            sd = info.get(d)
            status, task = (sd.status, sd.task) if sd else ("", "")
            if status == "WATCHING":
                return 0
            if status not in ("CHAIN",):
                return 1  # exploring / idle
            if task.startswith("over the survivor") or task.startswith("lead"):
                return 3
            return 2  # checkpoint / sentinel

        gt = getattr(self.world, "ground_truth", None)
        fires = [(f.x, f.y) for f in (gt().fires if gt else [])]

        def near_fire(d: str) -> float:
            p = self.world.drone_pose(d)
            return min((math.hypot(p.x - fx, p.y - fy) for fx, fy in fires), default=0.0)

        # among equals: off the main corridors first ("a little out of the way"), then the one nearest a fire
        return min(alive, key=lambda d: (rank(d), self._corridor_dist(d) < SIGNAL_OFF_CORRIDOR_M, near_fire(d), d))

    def _flare_safe(self, drone: str, fires) -> bool:
        """A flare-up fire at this drone keeps every through-corridor open: far enough from them, and the summed
        heat on their centrelines stays under what a drone may cross (the firefighter's fire + 1 m is less)."""
        if self._corridor_dist(drone) < FLARE_CORRIDOR_M:
            return False
        p0 = self.world.drone_pose(drone)
        for o in self.scenario.drones.ids:  # never right next to another drone (it would be engulfed too)
            if o != drone and o not in self.dead:
                q = self.world.drone_pose(o)
                if math.hypot(q.x - p0.x, q.y - p0.y) < FLARE_CLEAR_M:
                    return False
        from sgr.swarm.rescue import through_edges
        from sgr.world.heat import temperature_at

        p = self.world.drone_pose(drone)
        pts = [(f.x, f.y) for f in fires] + [(p.x, p.y)]
        amb = self.scenario.heat.ambient_c
        for (ax, ay), (bx, by) in through_edges(self.scenario.graph):
            n = max(1, int(math.hypot(bx - ax, by - ay) / 0.25))
            for k in range(n + 1):
                x, y = ax + (bx - ax) * k / n, ay + (by - ay) * k / n
                if math.hypot(x - p.x, y - p.y) < 6.0 and temperature_at(x, y, pts, amb) >= FLARE_CORRIDOR_MAX_C:
                    return False
        return True

    def _corridor_dist(self, drone: str) -> float:
        from sgr.swarm.rescue import _seg_dist, through_edges  # same rule as the swarm's watch posts

        p = self.world.drone_pose(drone)
        return min((_seg_dist((p.x, p.y), a, b) for a, b in through_edges(self.scenario.graph)), default=math.inf)

    def _submit_frame(self, drone: str, pose: Pose) -> None:
        self._frame_seq += 1
        frame_id = f"{drone}-{self._frame_seq}"
        if self.render:
            frame, cam = self.world.render_drone(drone)
            if self.budget is not None:
                self.budget.on_frame()
        else:
            frame, cam = _MOCK_FRAME, _cam_info(pose)
        self.perception.submit_frame(drone, frame, cam, self.t, frame_id)
        if self.render and self.bus is not None:
            import cv2

            # the dashboard's copy gets smoke fog like the firefighter view; perception got the clean frame above
            ui = self.world.dashboard_frame(drone, frame) if hasattr(self.world, "dashboard_frame") else frame
            ok, jpg = cv2.imencode(".jpg", ui[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                channel = self.scenario.drones.ids.index(drone) + 1
                self.bus.publish_frame(drone, channel, self._frame_seq, frame_id, jpg.tobytes())

    def _drain_commands(self) -> bool:
        """Apply browser commands. Returns True if a snapshot was loaded (the tick must restart)."""
        if self.bus is None:
            return False
        for msg in self.bus.drain_inbound():
            if isinstance(msg, Act2StartMsg):
                if self._load("act2_start", act=2):
                    return True
                continue
            if isinstance(msg, SnapshotMsg):
                if msg.op == "save":
                    self._save(msg.name)
                elif self._load(msg.name):
                    return True
                continue
            if isinstance(msg, ControlMsg) and msg.action == "reset":
                if self._load("act1_start", act=1):
                    return True
                continue
            if isinstance(msg, ClickMsg):
                place = self.world.place_debris if msg.mode == "debris" else self.world.place_fire
                h = place(msg.x, msg.y)
                if h is not None:
                    self.click_stack.append(h)
            elif isinstance(msg, UndoMsg) and self.click_stack:
                self.world.remove(self.click_stack.pop())
            elif isinstance(msg, FFInputMsg):
                self.world.set_ff_input(msg.fwd, msg.strafe, msg.yaw)
            elif isinstance(msg, ControlMsg):
                if msg.action == "pause":
                    self.paused = True
                elif msg.action == "start":
                    self.paused = False
                elif msg.action == "speed" and msg.value:
                    self.speed = msg.value
                    self._reanchor = True
        return False

    # ------------------------------------------------------------------ snapshots (§8.6)
    def _save(self, name: str) -> Path | None:
        try:
            directory = self.snapshot_dir or snapshot.SNAPSHOT_DIR
            self.last_snapshot = snapshot.save(name, snapshot.capture(self), directory)
            print(f"sgr: snapshot '{name}' saved at t={self.t:.1f}s -> {self.last_snapshot}")
            return self.last_snapshot
        except Exception as e:  # a failed save must never stop the demo
            print(f"sgr: WARNING snapshot '{name}' not saved: {e}")
            return None

    def _load(self, name: str, act: int | None = None) -> bool:
        path = snapshot.find(name, self.snapshot_dir or snapshot.SNAPSHOT_DIR, self.map_dir)
        if path is None:
            print(f"sgr: WARNING no snapshot '{name}' (snapshots/ or {self.map_dir})")
            return False
        try:
            snapshot.restore(self, snapshot.load_path(path))
        except Exception as e:
            print(f"sgr: WARNING snapshot '{name}' not loaded: {e}")
            return False
        if act is not None:
            self.act = act
        if self.act == 2:
            self._act2_t = self.t  # Act 2 clock for the summary
            self.summary = None
        if name == "act2_start":
            self._act2_saved = True  # don't overwrite it from the replayed chain
        print(f"sgr: snapshot '{name}' loaded -> t={self.t:.1f}s act {self.act}")
        self._reanchor = True
        self._resync_clients()
        return True

    def _resync_clients(self) -> None:
        """After a load: a fresh hello (the dashboard drops its old run), the trace so far, the full map."""
        if self.bus is None:
            return
        if self.map_dir is not None:
            from sgr.server.server import build_hello  # the server's own hello, so the browser resets cleanly

            self._publish(build_hello(self.map_dir, self.scenario))
        for row in self.trace[-200:]:
            self._publish(TraceMsg(**row.model_dump()))
        self._last_resync = -math.inf  # full map on the next tick
        self._last_state_pub = -math.inf

    def _summary(self) -> SummaryMsg:
        """§9 summary screen: Act 2 time, distance walked, hazards avoided, corrections, re-plans."""
        stats = self.guidance.stats() if self.guidance is not None and hasattr(self.guidance, "stats") else {}
        hazards = getattr(getattr(self.swarm, "bb", None), "hazards", {})
        return SummaryMsg(
            time_s=round(self.t - (self._act2_t or 0.0), 1),
            distance_m=float(stats.get("walked_m", 0.0)),
            hazards_avoided=sum(1 for h in hazards.values() if h.kind in ("fire", "blocked")),
            corrections=int(stats.get("corrections", 0)),
            replans=sum(1 for r in self.trace  # route re-plans during the firefighter's walk
                        if r.action == "REPLAN" and r.drone == "swarm" and r.t >= (self._act2_t or 0.0)),
            trace_url=None,  # no trace-file endpoint yet: the full trace is the dashboard's trace list
        )

    def _publish_full_map(self) -> None:
        """Every non-free belief cell + every searched cell (map_delta is additive in the browser)."""
        b = getattr(self.swarm, "belief", None)
        if b is None:
            return
        from sgr.swarm.belief import FREE, STATES, WALL

        jj, ii = np.nonzero((b.state != FREE) & (b.state != WALL))
        cells = [(int(i), int(j), STATES[b.state[j, i]]) for i, j in zip(ii, jj)]
        sj, si = np.nonzero(b.searched)
        self._publish(MapDeltaMsg(cells=cells, searched=list(zip(si.tolist(), sj.tolist()))))

    def _state_msg(self, readings: dict[str, Readings]) -> StateMsg:
        sw = {d.id: d for d in (self.swarm_state.drones if self.swarm_state else [])}
        drones = []
        for d in self.scenario.drones.ids:
            p = self.world.drone_pose(d)
            r = readings.get(d)
            info = sw.get(d)
            drones.append(
                DroneState(
                    id=d,
                    x=p.x,
                    y=p.y,
                    z=p.z,
                    yaw=p.yaw,
                    status=info.status if info else ("LOST" if d in self.dead else "IDLE"),
                    task=info.task if info else "",
                    temp=r.temperature if r else 0.0,
                    lidar=r.lidar if r else [],
                    alive=d not in self.dead,
                )
            )
        ff = None
        if self.act == 2:
            fp = self.world.ff_pose()
            ff = FFState(x=fp.x, y=fp.y, yaw=fp.yaw)
        ss = self.swarm_state
        return StateMsg(
            t=self.t,
            speed=self.speed,
            act=self.act,
            phase=ss.phase if ss else "",
            drones=drones,
            ff=ff,
            route=ss.route if ss else StateMsg.model_fields["route"].default,
            old_routes=ss.old_routes if ss else [],
            checkpoints=ss.checkpoints if ss else [],
            survivor=ss.survivor if ss else None,
            click_handles=list(self.click_stack),
        )

    def _publish(self, msg: Any) -> None:
        if self.bus is not None:
            self.bus.publish(msg)


def _detection_msg(det: Detection) -> DetectionMsg:
    g = det.gemma
    return DetectionMsg(
        drone=det.drone,
        frame_id=det.frame_id,
        boxes=[det.bbox] if isinstance(det.bbox, Box) else [],
        gemma=GemmaView(caption=g.caption if g else None, conf=g.confidence if g else None,
                        pending=det.status == "pending"),
        status=det.status,
    )


def _cam_info(pose: Pose) -> CamInfo:
    f = (config.CAM_H / 2) / math.tan(math.radians(config.CAM_FOVY_DEG) / 2)
    return CamInfo(fx=f, fy=f, cx=config.CAM_W / 2, cy=config.CAM_H / 2, pose=pose)
