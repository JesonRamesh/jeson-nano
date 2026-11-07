"""Budget metrics tests (P), with injected probes and a fake clock (no network, no GPU needed)."""

from __future__ import annotations

from sgr.contracts import BudgetMsg, PerceptionMetrics
from sgr.perception.budget import Budget, vram_mb


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_probes_offline_and_vram():
    b = Budget(net_probe=lambda: False, vram_probe=lambda: (5000.0, 8151.0), start=False)
    assert b.message().network == "unknown"  # before the first probe
    b.probe_once()
    m = b.message()
    assert m.network == "offline" and m.vram_used_mb == 5000.0 and m.vram_total_mb == 8151.0


def test_probe_online_and_no_gpu():
    b = Budget(net_probe=lambda: True, vram_probe=lambda: None, start=False)
    b.probe_once()
    m = b.message()
    assert m.network == "online" and m.vram_used_mb is None


def test_rates_from_hooks():
    clk = Clock()
    b = Budget(net_probe=lambda: False, vram_probe=lambda: None, clock=clk, start=False)
    for k in range(11):  # 10 Hz wall, speed 2.5 -> 0.25 sim s per tick
        b.on_tick(k * 0.25)
        b.on_frame()
        clk.t += 0.1
    m = b.message(speed=2.5)
    assert m.control_hz == 10.0 and m.sim_rtf == 2.5 and m.speed == 2.5
    assert 9.0 <= m.camera_fps <= 11.0


def test_perception_metrics_passthrough():
    b = Budget(net_probe=lambda: False, vram_probe=lambda: None, start=False)
    pm = PerceptionMetrics(mode="local", model_tag="gemma4:e4b", p50_s=4.9, p95_s=5.1, queue_len=2)
    m = b.message(pm)
    assert (m.model_tag, m.gemma_p50_s, m.gemma_p95_s, m.gemma_queue) == ("gemma4:e4b", 4.9, 5.1, 2)
    assert BudgetMsg.model_validate_json(m.model_dump_json()) == m  # WS round-trip


def test_background_thread_probes():
    b = Budget(net_probe=lambda: False, vram_probe=lambda: None, probe_interval_s=0.05)
    import time

    for _ in range(100):
        if b.network != "unknown":
            break
        time.sleep(0.01)
    b.close()
    assert b.network == "offline"


def test_vram_probe_shape():
    v = vram_mb()  # None on machines without nvidia-smi
    assert v is None or (len(v) == 2 and v[0] <= v[1])
