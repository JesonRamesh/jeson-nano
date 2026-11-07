"""Live perception (spec §7.2 + §7.3): fast detector on real frames, Gemma check on a worker thread.

- submit_frame runs the HSV detector (a few ms) on the main thread. A new
  candidate becomes a `pending` Detection, and its full frame is queued for Gemma.
- ONE worker thread sends Ollama one request at a time.
- poll() collects the finished answers, fuses them (fusion.py) and emits the
  updated Detections.
- Hard wall-clock deadline: if a request runs longer than client.timeout_s,
  poll() marks the check fast-only `unconfirmed` (§7.3) and drops the late
  answer. httpx's own timeout is per read, not total, so it can't do this.
- request_relook(id) waits for the next frame that sees the same object again
  (the drone has moved closer), then queues it. If no such frame arrives within
  RELOOK_WAIT_S of sim time, it re-sends the newest frame it has.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sgr.contracts import XY, Box, CamInfo, Detection, GemmaResult

from .detector import FastDetector
from .fusion import fuse
from .gemma import GemmaClient, GemmaTimeout

RELOOK_WAIT_S = 3.0


@dataclass
class _Track:
    det: Detection
    frame: np.ndarray  # newest frame that saw this object
    frame_id: str
    job: int | None = None  # seq of the Gemma job we are waiting for
    relook_since: float | None = None  # waiting for a fresh frame since this sim time


class LivePerception:
    def __init__(self, pixel_to_world: Callable, client: GemmaClient, warm: bool = True):
        self.client = client
        self.deadline_s = client.timeout_s
        self.fd = FastDetector()
        self.pixel_to_world = pixel_to_world
        self.now = 0.0
        self._tracks: dict[str, _Track] = {}
        self._out: list[Detection] = []
        self._seq = 0
        self._jobs: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._running: tuple[int, float] | None = None  # (seq, wall start) of the request in flight
        self.latencies: list[float] = []  # every successful answer, late ones included
        self.failures = 0
        self.timeouts = 0
        self.ready = threading.Event()  # set once warm-up is done
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, args=(warm,), name="gemma-worker", daemon=True)
        self._worker.start()

    # ---------------------------------------------------------------- main thread
    def submit_frame(self, drone: str, frame: np.ndarray, cam: CamInfo, t: float, frame_id: str) -> None:
        self.now = max(self.now, t)
        for c in self.fd.process(drone, frame, t, frame_id, self.pixel_to_world):
            x, y, w, h = c.box.bbox
            box = Box(x=x, y=y, w=w, h=h, label=c.label, score=c.box.score_fast)
            if c.is_new:
                det = Detection(id=c.det_id, drone=drone, t=t, label=c.label,
                                world_xy=XY(x=c.world_xy[0], y=c.world_xy[1]), score_fast=c.box.score_fast,
                                status="pending", frame_id=frame_id, bbox=box)
                tr = self._tracks[det.id] = _Track(det, frame.copy(), frame_id)
                self._queue(tr)
                self._out.append(det.model_copy())
            elif (tr := self._tracks.get(c.det_id)) is not None:
                tr.frame, tr.frame_id = frame.copy(), frame_id
                tr.det = tr.det.model_copy(update={
                    "world_xy": XY(x=c.world_xy[0], y=c.world_xy[1]),
                    "score_fast": max(tr.det.score_fast, c.box.score_fast), "bbox": box, "drone": drone,
                })
                if tr.relook_since is not None and t > tr.relook_since:
                    self._queue(tr)
        for tr in self._tracks.values():  # re-look fallback: no fresh view came
            if tr.relook_since is not None and self.now - tr.relook_since >= RELOOK_WAIT_S:
                self._queue(tr)

    def _queue(self, tr: _Track) -> None:
        self._seq += 1
        tr.relook_since = None
        tr.job = self._seq
        self._jobs.put((self._seq, tr.det.id, tr.frame, tr.frame_id))

    def _finish(self, tr: _Track, g: GemmaResult | None, frame_id: str) -> None:
        tr.job = None
        combined, status = fuse(tr.det.label, tr.det.score_fast, g, tr.det.relooks)
        tr.det = tr.det.model_copy(update={"gemma": g, "combined": combined, "status": status, "frame_id": frame_id})
        if status == "rejected":
            xy = tr.det.world_xy
            self.fd.blacklist(tr.det.label, (xy.x, xy.y), self.now)
            del self._tracks[tr.det.id]
        self._out.append(tr.det.model_copy())

    def poll(self) -> list[Detection]:
        while True:
            try:
                seq, det_id, frame_id, g, err = self._results.get_nowait()
            except queue.Empty:
                break
            tr = self._tracks.get(det_id)
            if tr is None or tr.job != seq:
                continue  # stale: already timed out, or superseded by a newer job
            if err is not None:
                if err.startswith("timeout"):
                    self.timeouts += 1
                else:
                    self.failures += 1
            self._finish(tr, g, frame_id)
        run = self._running
        if run is not None and time.perf_counter() - run[1] > self.deadline_s:
            for tr in list(self._tracks.values()):
                if tr.job == run[0]:
                    self.timeouts += 1
                    self._finish(tr, None, tr.frame_id)
        out, self._out = self._out, []
        return out

    def request_relook(self, detection_id: str) -> None:
        tr = self._tracks.get(detection_id)
        if tr is None or tr.job is not None or tr.relook_since is not None:
            return
        if tr.det.status not in ("low_conf", "unconfirmed"):
            return
        tr.det = tr.det.model_copy(update={"relooks": tr.det.relooks + 1, "status": "pending"})
        tr.relook_since = self.now
        self._out.append(tr.det.model_copy())

    def queue_len(self) -> int:
        return sum(tr.job is not None for tr in self._tracks.values())  # queued + running

    def close(self) -> None:
        self._stop.set()
        self._jobs.put(None)
        self._worker.join(timeout=2)
        self.client.close()

    # ---------------------------------------------------------------- worker thread
    def _run(self, warm: bool) -> None:
        if warm:
            self.client.warm()
        self.ready.set()
        while not self._stop.is_set():
            job = self._jobs.get()
            if job is None:
                break
            seq, det_id, frame, frame_id = job
            g: GemmaResult | None = None
            err = None
            self._running = (seq, time.perf_counter())
            try:
                g = self.client.ask(frame)
                self.latencies.append(g.latency_s or 0.0)
            except GemmaTimeout as e:
                err = f"timeout: {e}"
            except Exception as e:  # Ollama down, bad JSON...: fast-only, unconfirmed
                err = f"{type(e).__name__}: {e}"
            self._running = None
            self._results.put((seq, det_id, frame_id, g, err))

    # ---------------------------------------------------------------- snapshots
    def get_state(self) -> dict:
        return {"now": self.now, "fd": self.fd.get_state(),
                "tracks": [(tr.det.model_dump(), tr.frame, tr.frame_id) for tr in self._tracks.values()]}

    def set_state(self, s: dict) -> None:
        self.now = s["now"]
        self.fd.set_state(s["fd"])
        self._tracks = {d["id"]: _Track(Detection(**d), f, fid) for d, f, fid in s["tracks"]}
        self._out = []
        for tr in self._tracks.values():  # checks that were in flight at save time
            if tr.det.status == "pending":
                self._queue(tr)
