"""Fake sim publisher for dashboard development (H0–H1 milestone). Owned by D.

Sends a ``hello``, a fake ``state`` at 10 Hz and one test JPEG per drone at
~10 fps, so the web app can be built before World/Swarm exist.
"""

from __future__ import annotations

import io
import math
import queue
import threading
import time

from PIL import Image, ImageDraw

from sgr.server.server import Frame

DRONES = [
    ("D1", "#4fc3f7"),
    ("D2", "#ba68c8"),
    ("D3", "#ffd54f"),
    ("D4", "#81c784"),
]
W, H = 320, 240


def hello_msg() -> dict:
    return {
        "type": "hello",
        "contract_version": 1,
        "map": {
            "name": "fake",
            "grid_url": "/maps/test/grid.png",
            "resolution": 0.1,
            "size": [300, 200],
            "origin": [0, 0],
        },
        "graph": {"nodes": [], "edges": []},
        "entrance": {"x": 2.0, "y": 3.0, "yaw": 0.0},
        "drones": [{"id": d, "color": c} for d, c in DRONES],
        "transform": None,
    }


def state_msg(t: float) -> dict:
    drones = []
    for i, (d, _) in enumerate(DRONES):
        a = t * 0.3 + i * math.pi / 2
        drones.append(
            {
                "id": d,
                "x": 15 + 8 * math.cos(a),
                "y": 10 + 6 * math.sin(a),
                "z": 1.6,
                "yaw": a + math.pi / 2,
                "status": "EXPLORE",
                "task": f"area n{i + 1}",
                "temp": 20 + 5 * (1 + math.sin(t + i)),
                "lidar": [6.0] * 36,
                "alive": True,
            }
        )
    return {
        "type": "state",
        "t": t,
        "speed": 1.0,
        "act": 1,
        "phase": "fake",
        "drones": drones,
        "ff": None,
        "route": None,
        "old_routes": [],
        "checkpoints": [],
        "survivor": None,
        "click_handles": [],
    }


def test_jpeg(drone: str, color: str, t: float, seq: int) -> bytes:
    img = Image.new("RGB", (W, H), (40, 42, 48))
    d = ImageDraw.Draw(img)
    # Floor/wall split + a moving "target" so motion is obvious.
    d.rectangle([0, H * 0.55, W, H], fill=(60, 60, 64))
    cx = W / 2 + 110 * math.sin(t * 1.5)
    d.ellipse([cx - 22, 120 - 22, cx + 22, 120 + 22], fill=color)
    d.text((8, 6), f"{drone}  TEST FEED", fill=(255, 255, 255))
    d.text((8, H - 18), f"t={t:6.2f}s  seq={seq}", fill=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return buf.getvalue()


def run(outbound: queue.Queue, stop: threading.Event, fps: float = 10.0) -> None:
    outbound.put(hello_msg())
    t0 = time.monotonic()
    seq = 0
    period = 1.0 / fps
    while not stop.is_set():
        t = time.monotonic() - t0
        outbound.put(state_msg(t))
        for ch, (d, c) in enumerate(DRONES, start=1):
            outbound.put(Frame(ch, seq, test_jpeg(d, c, t, seq)))
        seq += 1
        stop.wait(max(0.0, period - (time.monotonic() - t0 - t)))
