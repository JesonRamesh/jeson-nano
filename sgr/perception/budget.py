"""Budget metrics (spec §7.5) -> BudgetMsg for the dashboard's budget panel.

- Network probe: TCP connect to 1.1.1.1:443, 1 s timeout, every 5 s -> offline / online.
- VRAM: nvidia-smi every 5 s (None = N/A on this machine).
- Gemma tag, p50/p95, queue length: from Perception.metrics().
- Sim real-time factor, control Hz, camera FPS: from the loop's on_tick() / on_frame() hooks,
  over a sliding wall-clock window.

The probes run on a daemon thread, so a slow probe never stalls the main loop.

Loop wiring (S):
    budget = Budget()                       # once, in app.py
    budget.on_tick(loop.t)                  # every control tick
    budget.on_frame()                       # every rendered camera frame
    bus.publish(budget.message(perception.metrics(), loop.speed))   # at 1 Hz
"""

from __future__ import annotations

import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable

from sgr.contracts import BudgetMsg, PerceptionMetrics

PROBE_INTERVAL_S = 5.0
WINDOW_S = 2.0


def network_online(host: str = "1.1.1.1", port: int = 443, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def vram_mb() -> tuple[float, float] | None:
    """(used, total) MiB from nvidia-smi, or None if there is no NVIDIA GPU / driver."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        ).stdout.strip().splitlines()[0]
        used, total = (float(v) for v in out.split(","))
        return used, total
    except Exception:
        return None


class Budget:
    def __init__(
        self,
        probe_interval_s: float = PROBE_INTERVAL_S,
        net_probe: Callable[[], bool] = network_online,
        vram_probe: Callable[[], tuple[float, float] | None] = vram_mb,
        clock: Callable[[], float] = time.perf_counter,
        start: bool = True,
    ) -> None:
        self._net_probe, self._vram_probe, self._clock = net_probe, vram_probe, clock
        self._interval = probe_interval_s
        self.network: str = "unknown"
        self.vram: tuple[float, float] | None = None
        self._ticks: deque[tuple[float, float]] = deque()  # (wall, sim_t)
        self._frames: deque[float] = deque()  # wall
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if start:
            self._thread = threading.Thread(target=self._run, name="budget-probes", daemon=True)
            self._thread.start()

    # ---------------------------------------------------------------- probes (thread)
    def probe_once(self) -> None:
        online = self._net_probe()
        vram = self._vram_probe()
        with self._lock:
            self.network = "online" if online else "offline"
            self.vram = vram

    def _run(self) -> None:
        while not self._stop.is_set():
            self.probe_once()
            self._stop.wait(self._interval)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    # ---------------------------------------------------------------- loop hooks (main thread)
    def on_tick(self, sim_t: float) -> None:
        now = self._clock()
        self._ticks.append((now, sim_t))
        while self._ticks and now - self._ticks[0][0] > WINDOW_S:
            self._ticks.popleft()

    def on_frame(self) -> None:
        now = self._clock()
        self._frames.append(now)
        while self._frames and now - self._frames[0] > WINDOW_S:
            self._frames.popleft()

    def rates(self) -> tuple[float | None, float | None, float | None]:
        """(control_hz, sim_rtf, camera_fps) over the last WINDOW_S seconds of wall time."""
        hz = rtf = fps = None
        if len(self._ticks) >= 2:
            (w0, s0), (w1, s1) = self._ticks[0], self._ticks[-1]
            if w1 > w0:
                hz = (len(self._ticks) - 1) / (w1 - w0)
                rtf = (s1 - s0) / (w1 - w0)
        if self._frames:
            now = self._clock()
            span = min(WINDOW_S, max(now - self._frames[0], 1e-6))
            fps = len(self._frames) / span if len(self._frames) > 1 else None
        return hz, rtf, fps

    # ---------------------------------------------------------------- message
    def message(self, perception: PerceptionMetrics | None = None, speed: float = 1.0) -> BudgetMsg:
        hz, rtf, fps = self.rates()
        with self._lock:
            network, vram = self.network, self.vram
        r = lambda v, n=2: None if v is None else round(v, n)  # noqa: E731
        return BudgetMsg(
            network=network,
            model_tag=perception.model_tag if perception else None,
            gemma_p50_s=perception.p50_s if perception else None,
            gemma_p95_s=perception.p95_s if perception else None,
            gemma_queue=perception.queue_len if perception else 0,
            vram_used_mb=vram[0] if vram else None,
            vram_total_mb=vram[1] if vram else None,
            sim_rtf=r(rtf),
            speed=speed,
            control_hz=r(hz, 1),
            camera_fps=r(fps, 1),
        )
