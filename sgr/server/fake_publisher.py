"""Fake sim publisher for dashboard development. Owned by D.

Publishes contract messages through a real ``Bus`` so every dashboard widget can be
exercised before the swarm/perception exist: drones circling the test map, test
JPEGs, belief-map deltas, a scripted decision trace, detections, events, budget.
None of this is used in the demo.
"""

from __future__ import annotations

import io
import math
import threading
import time

from PIL import Image, ImageDraw

from sgr import config
from sgr.bus import Bus
from sgr.contracts import (
    BudgetMsg,
    Box,
    Checkpoint,
    DetectionMsg,
    DroneState,
    EventMsg,
    Evidence,
    GemmaView,
    MapDeltaMsg,
    OldRoute,
    Route,
    Scenario,
    StateMsg,
    SurvivorState,
    TraceMsg,
)

W, H = config.CAM_W, config.CAM_H


def test_jpeg(drone: str, colour: str, t: float, seq: int) -> bytes:
    img = Image.new("RGB", (W, H), (40, 42, 48))
    d = ImageDraw.Draw(img)
    d.rectangle([0, H * 0.55, W, H], fill=(60, 60, 64))
    cx = W / 2 + 110 * math.sin(t * 1.5)
    d.ellipse([cx - 22, 98, cx + 22, 142], fill=colour)
    d.text((8, 6), f"{drone}  TEST FEED", fill=(255, 255, 255))
    d.text((8, H - 18), f"t={t:6.2f}s  seq={seq}", fill=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return buf.getvalue()


_SCRIPT = [  # (t, drone, action, reason, sensor, value)
    (2, "swarm", "ASSIGN_AREA", "Areas assigned to D1–D4 by nearest unclaimed node", "planner", "4 areas"),
    (6, "D2", "INSPECT", "D2 inspecting orange object: fast detector 0.62, Gemma 0.31, moving closer", "camera", 0.62),
    (9, "D2", "REJECT_DETECTION", "D2: Gemma says not fire (0.20) after 2 looks: false alarm", "gemma", 0.2),
    (13, "D3", "CONFIRM_HAZARD", "D3 fire confirmed: fast 0.85, Gemma 0.92, zone marked", "gemma", 0.92),
    (17, "D3", "RETREAT", "D3 temperature 74 °C > 70 °C limit, retreating 3 m", "temp", 74.0),
    (21, "D1", "MARK_BLOCKED", "D1 LiDAR: 1.1 m measured vs 4.8 m expected on 2 scans", "lidar", [1.1, 4.8]),
    (22, "swarm", "REPLAN", "Route v2 avoids blocked corridor (+3.4 m)", "planner", {"version": 2}),
]


def run(bus: Bus, scenario: Scenario, stop: threading.Event, fps: float = 10.0) -> None:
    ids = list(scenario.drones.ids)
    cx = sum(n.x for n in scenario.graph.nodes) / max(1, len(scenario.graph.nodes))
    cy = sum(n.y for n in scenario.graph.nodes) / max(1, len(scenario.graph.nodes))
    t0 = time.monotonic()
    seq = 0
    script_i = 0
    period = 1.0 / fps
    next_budget = 0.0
    route_pts = [(n.x, n.y) for n in scenario.graph.nodes[:5]]
    while not stop.is_set():
        tick_start = time.monotonic()
        t = (tick_start - t0) % 30.0  # loop the script every 30 s
        if t < period:
            script_i = 0
        drones = []
        for i, d in enumerate(ids):
            a = t * 0.3 + i * math.pi / 2
            x, y = cx + 6 * math.cos(a), cy + 4 * math.sin(a)
            alive = not (d == "D3" and t > 19)
            drones.append(
                DroneState(
                    id=d, x=x, y=y, z=1.6, yaw=a + math.pi / 2,
                    status="EXPLORING" if alive else "LOST", task=f"area n{i + 1}",
                    temp=(20 + 3 * t if d == "D3" else 22.0), alive=alive,
                    lidar=[3.0 + 2.5 * math.sin(k * 0.5 + t) for k in range(config.LIDAR_RAYS)],
                )
            )
            if alive:
                bus.publish_frame(d, i + 1, seq, f"{d}-{seq}", test_jpeg(d, config.DRONE_COLOURS.get(d, "#ccc"), t, seq))
            # searched cells under the drone (belief grid, 0.2 m)
            bi, bj = int(x / config.BELIEF_RES), int(y / config.BELIEF_RES)
            bus.publish(MapDeltaMsg(searched=[(bi + di, bj + dj) for di in range(-4, 5) for dj in range(-4, 5)]))
        bus.publish(
            StateMsg(
                t=t, speed=1.0, phase="fake", drones=drones,
                route=Route(points=route_pts, status="ok", version=1 + int(t > 22)),
                old_routes=[OldRoute(points=route_pts[::-1], until=28.0)] if t > 22 else [],
                checkpoints=[Checkpoint(x=p[0], y=p[1], drone=None) for p in route_pts[1:3]],
                survivor=SurvivorState(status="suspected", x=cx + 3, y=cy + 2, conf=0.6) if t > 24 else None,
            )
        )
        while script_i < len(_SCRIPT) and t >= _SCRIPT[script_i][0]:
            st, d, action, reason, sensor, value = _SCRIPT[script_i]
            fid = f"{d}-{seq}" if d != "swarm" else None
            scan = [1.1 if k in (0, 1, 35) else 4.8 for k in range(config.LIDAR_RAYS)] if sensor == "lidar" else None
            bus.publish(TraceMsg(id=f"tr{seq}-{script_i}", t=st, drone=d, action=action, reason=reason,
                                 evidence=Evidence(sensor=sensor, value=value, frame_id=fid, scan=scan)))
            if action == "INSPECT":
                bus.publish(DetectionMsg(drone=d, frame_id=fid, status="low_conf",
                                         boxes=[Box(x=120, y=80, w=60, h=70, label="fire", score=0.62)],
                                         gemma=GemmaView(caption="orange panel on a wall", conf=0.31)))
            if action == "REJECT_DETECTION":
                bus.publish(EventMsg(kind="false_alarm", drone=d))
                bus.publish(DetectionMsg(drone=d, frame_id=fid, status="rejected", boxes=[],
                                         gemma=GemmaView(caption="orange panel, no flames", conf=0.2)))
            if action == "MARK_BLOCKED":
                bj0 = int(cy / config.BELIEF_RES)
                bi0 = int((cx + 4) / config.BELIEF_RES)
                bus.publish(MapDeltaMsg(cells=[(bi0 + k, bj0 + m, "blocked_new") for k in range(5) for m in range(5)]))
                bus.publish(EventMsg(kind="blocked", drone=d))
            if action == "CONFIRM_HAZARD":
                bi0, bj0 = int((cx - 4) / config.BELIEF_RES), int((cy + 2) / config.BELIEF_RES)
                bus.publish(MapDeltaMsg(cells=[(bi0 + k, bj0 + m, "fire") for k in range(6) for m in range(6)]))
                bus.publish(EventMsg(kind="fire_confirmed", drone=d))
            script_i += 1
        if t >= next_budget or t < period:
            next_budget = t + 1.0
            bus.publish(BudgetMsg(network="offline", model_tag="gemma4:e4b (fake)", gemma_p50_s=1.1,
                                  gemma_p95_s=2.3, gemma_queue=1, sim_rtf=1.0, speed=1.0,
                                  control_hz=10.0, camera_fps=10.0))
        seq += 1
        stop.wait(max(0.0, period - (time.monotonic() - tick_start)))
