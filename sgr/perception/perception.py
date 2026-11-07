"""Perception facade (owned by P). §4 signatures, v1.

Modes (§7.3):
- mock:   geometric + scripted, no frames needed (mock.py). Uses ground truth; nothing else may.
- local:  fast detector + Gemma on this laptop's Ollama (live.py, gemma.py).
- remote: the same, against the demo laptop's Ollama over the LAN ($SGR_OLLAMA_URL).

poll() returns each Detection whenever it changes (new -> pending -> confirmed /
low_conf / rejected / unconfirmed; request_relook -> pending again with
relooks + 1). The same `id` is kept throughout.

Optional keyword arguments (additive to the S scaffold):
- line_of_sight=world.line_of_sight: mock occlusion. Without it, mock sees through walls.
- pixel_to_world=world.pixel_to_world: required for local/remote (world position of a detection).
- mock_latency_s=7.0: mock Gemma answers after 7 s instead of 1 s, to reproduce live timing headless.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

from sgr.contracts import CamInfo, Detection, GroundTruth, PerceptionMetrics, PerceptionMode

from .gemma import GemmaClient, chat_url
from .live import LivePerception
from .mock import MockPerception


def _pct(xs: list[float], q: float) -> float | None:
    return round(float(np.percentile(xs, q)), 3) if xs else None


class Perception:
    def __init__(
        self,
        mode: PerceptionMode = "mock",
        ground_truth: Callable[[], GroundTruth] | None = None,
        *,
        line_of_sight: Callable[[str, float, float, float], bool] | None = None,
        pixel_to_world: Callable | None = None,
        gemma: GemmaClient | None = None,
        warm: bool = True,
        mock_latency_s: float | None = None,
    ) -> None:
        self.mode = mode
        if mode == "mock":
            if ground_truth is None:
                raise ValueError("mock perception needs ground_truth=world.ground_truth")
            if line_of_sight is None:
                warnings.warn("mock perception without line_of_sight: targets are visible through walls",
                              stacklevel=2)
            self._impl = MockPerception(ground_truth, line_of_sight)
            if mock_latency_s is not None:  # rehearse live-Gemma timing headless (e4b: ~5-8 s)
                self._impl.latency_s = float(mock_latency_s)
        else:
            if pixel_to_world is None:
                raise ValueError(f"{mode} perception needs pixel_to_world=world.pixel_to_world")
            client = gemma or GemmaClient(url=chat_url(remote=mode == "remote"))
            self._impl = LivePerception(pixel_to_world, client, warm=warm)

    def submit_frame(self, drone: str, frame: np.ndarray, cam: CamInfo, t: float, frame_id: str) -> None:
        if self.mode == "mock":
            self._impl.submit_frame(drone, cam, t, frame_id)
        else:
            self._impl.submit_frame(drone, frame, cam, t, frame_id)

    def poll(self) -> list[Detection]:
        return self._impl.poll()

    def request_relook(self, detection_id: str) -> None:
        self._impl.request_relook(detection_id)

    def metrics(self) -> PerceptionMetrics:
        if self.mode == "mock":
            lat = self._impl._latencies
            return PerceptionMetrics(mode="mock", model_tag="mock", p50_s=_pct(lat, 50), p95_s=_pct(lat, 95),
                                     queue_len=self._impl.queue_len())
        live: LivePerception = self._impl
        return PerceptionMetrics(
            mode=self.mode, model_tag=live.client.model_tag, p50_s=_pct(live.latencies, 50),
            p95_s=_pct(live.latencies, 95), queue_len=live.queue_len(), failures=live.failures,
            timeouts=live.timeouts,
        )

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until Gemma is loaded and warmed (~30 s on the demo laptop). Mock is always ready.

        Call before starting the mission clock, so the first checks don't hit the cold-start delay.
        """
        if self.mode == "mock":
            return True
        return self._impl.ready.wait(timeout)

    def get_state(self) -> dict:
        return {"mode": self.mode, "impl": self._impl.get_state()}

    def set_state(self, s: dict) -> None:
        """Restore detections. A snapshot from another mode (e.g. act2_start.pkl recorded with live Gemma
        on the demo laptop, loaded by a teammate running mock) can't be mapped across: start clean."""
        if s.get("mode") != self.mode:
            warnings.warn(f"snapshot perception state is from mode {s.get('mode')!r}, running {self.mode!r}: "
                          "starting with no detections", stacklevel=2)
            return
        self._impl.set_state(s["impl"])

    def close(self) -> None:
        if hasattr(self._impl, "close"):
            self._impl.close()
