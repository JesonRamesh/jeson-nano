"""Perception facade (owned by P). §4 signatures, v1.

STUB: returns no detections. P implements mock / remote / local modes (§7.3).
Mock mode may use the ground_truth callable; nothing else may.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sgr.contracts import CamInfo, Detection, GroundTruth, PerceptionMetrics, PerceptionMode


class Perception:
    def __init__(
        self,
        mode: PerceptionMode = "mock",
        ground_truth: Callable[[], GroundTruth] | None = None,
    ) -> None:
        self.mode = mode
        self._ground_truth = ground_truth  # mock mode only

    def submit_frame(self, drone: str, frame: np.ndarray, cam: CamInfo, t: float, frame_id: str) -> None:
        pass  # TODO(P): fast detector -> Gemma job queue

    def poll(self) -> list[Detection]:
        return []  # TODO(P)

    def request_relook(self, detection_id: str) -> None:
        pass  # TODO(P)

    def metrics(self) -> PerceptionMetrics:
        return PerceptionMetrics(mode=self.mode)
