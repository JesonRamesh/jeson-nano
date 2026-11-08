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

import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sgr.contracts import XY, Box, CamInfo, Detection, GemmaResult

from .detector import FastDetector
from .fusion import MAX_RELOOKS, STRONG_PERSON_FAST, fuse
from .gemma import GemmaClient, GemmaTimeout

RELOOK_WAIT_S = 3.0
REJECT_MAX_VIEW_M = 7.0  # a 'not fire / not a person' verdict only rejects if the view was this close
RECHECK_S = 8.0  # a strong hi-vis person Gemma didn't agree with: ask again at most this often ...
RECHECK_MOVED_M = 1.5  # ... and only from a camera this far from the last view asked about (a new angle)


def _view_m(cam: CamInfo | None, xy) -> float:
    """Camera-to-object distance (m); 0 if the pose is unknown."""
    if cam is None or cam.pose is None:
        return 0.0
    return math.hypot(xy[0] - cam.pose.x, xy[1] - cam.pose.y)


@dataclass
class _Track:
    det: Detection
    frame: np.ndarray  # newest frame that saw this object
    frame_id: str
    job: int | None = None  # seq of the Gemma job we are waiting for
    relook_since: float | None = None  # waiting for a fresh frame since this sim time
    view_m: float = 0.0  # camera-to-object distance of `frame`
    job_can_reject: bool = True  # the job in flight is a fresh, close view (may reject)
    asked_at: float = -math.inf  # sim time of the last Gemma job
    asked_from: tuple[float, float] | None = None  # camera xy of the last Gemma job
    recheck: bool = False  # the job in flight is a quiet re-check of a strong person (see _recheck_due)
    rechecks: int = 0


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
                tr = self._tracks[det.id] = _Track(det, frame.copy(), frame_id, view_m=_view_m(cam, c.world_xy))
                self._queue(tr, fresh=True, cam=cam)
                self._out.append(det.model_copy())
            elif (tr := self._tracks.get(c.det_id)) is not None:
                tr.frame, tr.frame_id, tr.view_m = frame.copy(), frame_id, _view_m(cam, c.world_xy)
                tr.det = tr.det.model_copy(update={
                    "world_xy": XY(x=c.world_xy[0], y=c.world_xy[1]),
                    "score_fast": max(tr.det.score_fast, c.box.score_fast), "bbox": box, "drone": drone,
                })
                if tr.relook_since is not None and t > tr.relook_since:
                    self._queue(tr, fresh=True, cam=cam)
                elif self._recheck_due(tr, cam):
                    tr.recheck = True
                    self._queue(tr, fresh=True, cam=cam)
        for tr in self._tracks.values():  # re-look fallback: no fresh view came, re-send the newest frame
            if tr.relook_since is not None and self.now - tr.relook_since >= RELOOK_WAIT_S:
                self._queue(tr, fresh=False)

    def _recheck_due(self, tr: _Track, cam: CamInfo) -> bool:
        """A strong hi-vis person that Gemma didn't agree with, after the swarm's own closer looks: ask Gemma
        again, quietly, from a new close viewpoint (Gemma misses the real survivor from some angles)."""
        d = tr.det
        if not (d.label == "person" and d.score_fast >= STRONG_PERSON_FAST and d.status in ("low_conf", "unconfirmed")
                and d.relooks >= MAX_RELOOKS and tr.job is None and tr.relook_since is None):
            return False
        if self.now - tr.asked_at < RECHECK_S or tr.view_m > REJECT_MAX_VIEW_M:
            return False
        return tr.asked_from is None or math.dist(tr.asked_from, (cam.pose.x, cam.pose.y)) >= RECHECK_MOVED_M

    def _queue(self, tr: _Track, fresh: bool, cam: CamInfo | None = None) -> None:
        """fresh = this frame is a new sighting. Only a fresh view from within REJECT_MAX_VIEW_M may reject:
        a stale frame or a far-away view can't show that the object isn't there (live run: a real fire
        was rejected from an 18 m / stale view and the swarm never found the survivor)."""
        self._seq += 1
        tr.relook_since = None
        tr.job = self._seq
        tr.job_can_reject = fresh and tr.view_m <= REJECT_MAX_VIEW_M
        tr.asked_at = self.now
        if cam is not None:
            tr.asked_from = (cam.pose.x, cam.pose.y)
        self._jobs.put((self._seq, tr.det.id, tr.frame, tr.frame_id))

    def _finish(self, tr: _Track, g: GemmaResult | None, frame_id: str) -> None:
        tr.job = None
        combined, status = fuse(tr.det.label, tr.det.score_fast, g, tr.det.relooks)
        if tr.recheck:  # quiet re-check: only a confirmation is news
            tr.recheck = False
            tr.det = tr.det.model_copy(update={"gemma": g, "combined": combined, "frame_id": frame_id,
                                               "status": status if status == "confirmed" else tr.det.status})
            if status == "confirmed":
                tr.rechecks += 1
                # a new id: the swarm closed the old one when it stopped looking ("still unsure"), and ignores
                # updates to closed detections; a fresh confirmed detection is taken as is
                self._out.append(tr.det.model_copy(update={"id": f"{tr.det.id}-v{tr.rechecks}"}))
            return
        if status == "rejected" and not tr.job_can_reject:
            status = "unconfirmed"  # couldn't see it properly: needs a second view, not a blacklist
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
                self._queue(tr, fresh=False)
