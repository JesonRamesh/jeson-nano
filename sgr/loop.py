"""Sim loop (owned by S). Runs on the MAIN thread (GL must live here). §3.

One control tick = CONTROL_DT sim seconds:
  1 drain inbound commands   2 timed scenario events (flare-up)
  3 sensors + camera frames  4 perception poll
  5 Swarm.tick               6 Guidance.tick (Act 2)
  7 World.step(SUBSTEPS)     8 publish to the bus

Snapshots (§8.6, when `snapshot_dir` is set): `act1_start` is saved before the first tick and
`act2_start` when the chain forms. Dashboard buttons: Restart Act 1 (control reset) loads
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
    SwarmState,
    TraceMsg,
    TraceRow,
    UndoMsg,
)

RESYNC_S = 5.0  # wall seconds between full belief-map resyncs (late-joining browsers catch up)
FLARE_NEAR_M = 6.0  # flare-up hits the drone nearest a fire if it is this close, else scenario.flare.target
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
        self.flare_victim: str | None = None
        self._frame_seq = 0
        self._cam_rr = 0
        self._last_state_pub = -math.inf
        self._last_budget_pub = -math.inf
        self._last_resync = -math.inf
        self.map_dir = Path(map_dir) if map_dir is not None else None
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else None  # None = no snapshots
        self._act2_saved = False
        self.last_snapshot: Path | None = None

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
        # 2 timed events: flare-up (§7.4)
        if s.flare and not self._flared and self.t >= s.flare.t:
            victim = self._flare_victim()
            if victim is not None:
                p = w.drone_pose(victim)
                # force: bypass the 1 m click clearance (the victim drone is AT this spot).
                # Not pushed on click_stack, so Undo can't remove the flare-up fire.
                self.flare_fire = w.place_fire(p.x, p.y, force=True)
                w.kill_drone(victim)
                self.dead.add(victim)
                self.flare_victim = victim
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
        # 5 swarm
        cmds, rows, self.swarm_state = self.swarm.tick(self.t, readings, detections)
        # the swarm asks for a closer look only once a drone is ~2 m in front of the target (§8.4)
        for det_id in self.swarm_state.relook_requests:
            self.perception.request_relook(det_id)
        # 6 guidance: Act 2 steers the firefighter; in Act 1 it only folds in hazards (map_delta)
        if self.guidance is not None:
            if self.act == 2:
                gmsg, gcmds = self.guidance.tick(self.t, w.ff_pose(), self.swarm_state)
                cmds = {**cmds, **gcmds}
                self._publish(gmsg)
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
        for ev in self.swarm_state.events:
            self._publish(ev)
        ss = self.swarm_state
        if ss.map_delta or ss.searched_delta:
            self._publish(MapDeltaMsg(cells=ss.map_delta, searched=ss.searched_delta))
        now = time.perf_counter()
        if self.bus is not None and now - self._last_resync >= RESYNC_S:
            self._last_resync = now
            self._publish_full_map()
        if ss.chain_formed and not self._act2_saved and self.snapshot_dir is not None:
            self._act2_saved = True
            self._save("act2_start")  # Jump to Act 2 starts here
        if self.bus is not None and now - self._last_state_pub >= 1.0 / config.STATE_HZ:
            self._last_state_pub = now
            self._publish(self._state_msg(readings))
        if self.budget is not None and self.bus is not None and now - self._last_budget_pub >= 1.0:
            self._last_budget_pub = now
            self._publish(self.budget.message(self.perception.metrics(), self.speed))

    # ------------------------------------------------------------------
    def _flare_victim(self) -> str | None:
        """The flare-up hits the alive drone nearest a burning fire (the one watching it, §5 beat).

        Simulation side, so reading the World's real fires is fine here (never in sgr/swarm).
        Falls back to scenario.flare.target when nothing is burning.
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

        best = min(alive, key=dist)
        if dist(best) <= FLARE_NEAR_M:
            return best
        return self.scenario.flare.target if self.scenario.flare.target in alive else best

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

            ok, jpg = cv2.imencode(".jpg", frame[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                channel = self.scenario.drones.ids.index(drone) + 1
                self.bus.publish_frame(drone, channel, self._frame_seq, frame_id, jpg.tobytes())

    def _drain_commands(self) -> bool:
        """Apply browser commands. Returns True if a snapshot was loaded (the tick must restart)."""
        if self.bus is None:
            return False
        for msg in self.bus.drain_inbound():
            if isinstance(msg, Act2StartMsg):
                if self._load("act2_start"):
                    self.act = 2
                    return True
                continue
            if isinstance(msg, SnapshotMsg):
                if msg.op == "save":
                    self._save(msg.name)
                elif self._load(msg.name):
                    return True
                continue
            if isinstance(msg, ControlMsg) and msg.action == "reset":
                if self._load("act1_start"):
                    self.act = 1
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

    def _load(self, name: str) -> bool:
        path = snapshot.find(name, self.snapshot_dir or snapshot.SNAPSHOT_DIR, self.map_dir)
        if path is None:
            print(f"sgr: WARNING no snapshot '{name}' (snapshots/ or {self.map_dir})")
            return False
        try:
            snapshot.restore(self, snapshot.load_path(path))
        except Exception as e:
            print(f"sgr: WARNING snapshot '{name}' not loaded: {e}")
            return False
        if name == "act2_start":
            self._act2_saved = True  # don't overwrite it from the replayed chain
        print(f"sgr: snapshot '{name}' loaded -> t={self.t:.1f}s act {self.act}")
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
