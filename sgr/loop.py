"""Sim loop (owned by S). Runs on the MAIN thread (GL must live here). §3.

One control tick = CONTROL_DT sim seconds:
  1 drain inbound commands   2 timed scenario events (flare-up)
  3 sensors + camera frames  4 perception poll
  5 Swarm.tick               6 Guidance.tick (Act 2)
  7 World.step(SUBSTEPS)     8 publish to the bus
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from sgr import config
from sgr.bus import Bus
from sgr.contracts import (
    CamInfo,
    ClickMsg,
    ControlMsg,
    DroneState,
    FFInputMsg,
    FFState,
    Handle,
    Pose,
    Readings,
    Scenario,
    StateMsg,
    SwarmState,
    TraceMsg,
    TraceRow,
    UndoMsg,
)

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
        # 1 inbound commands
        self._drain_commands()
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
        now = time.perf_counter()
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

    def _drain_commands(self) -> None:
        if self.bus is None:
            return
        for msg in self.bus.drain_inbound():
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
                # TODO(S): reset -> reload Act 1
            # TODO(S): Act2StartMsg, SnapshotMsg (§8.6)

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


def _cam_info(pose: Pose) -> CamInfo:
    f = (config.CAM_H / 2) / math.tan(math.radians(config.CAM_FOVY_DEG) / 2)
    return CamInfo(fx=f, fy=f, cx=config.CAM_W / 2, cy=config.CAM_H / 2, pose=pose)
