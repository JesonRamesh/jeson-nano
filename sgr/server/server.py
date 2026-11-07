"""FastAPI server: WS /ws, REST /api/*, static web/dist. Owned by D.

The sim loop (main thread) and this server (background thread) talk only
through two thread-safe queues:

* ``outbound``: sim -> browsers. Items are JSON-able dicts (``{"type": ...}``)
  or :class:`Frame` (a drone camera JPEG).
* ``inbound``: browsers -> sim. Items are the client JSON dicts (``click``,
  ``undo``, ``control``, ...), already parsed.

PROVISIONAL: once S pushes ``sgr/bus.py``, ``start_server`` takes the bus
queues instead of creating its own. The rest stays the same.

Binary frame wire format: ``[u8 channel][u32 frame_seq little-endian][JPEG]``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import struct
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIST = REPO_ROOT / "web" / "dist"
MAPS_DIR = REPO_ROOT / "maps"

FRAME_RING = 300  # frames kept per drone for trace evidence (spec §10.2)
CLIENT_QUEUE = 64  # per-browser backlog before we start dropping


@dataclass(frozen=True)
class Frame:
    channel: int  # 1–4 = D1–D4
    seq: int
    jpeg: bytes

    @property
    def frame_id(self) -> str:
        return f"D{self.channel}-{self.seq}"

    def to_bytes(self) -> bytes:
        return struct.pack("<BI", self.channel, self.seq & 0xFFFFFFFF) + self.jpeg


class Hub:
    """Fans outbound messages out to every connected browser."""

    def __init__(self, outbound: queue.Queue, inbound: queue.Queue) -> None:
        self.outbound = outbound
        self.inbound = inbound
        self.clients: set[asyncio.Queue] = set()
        self.hello: str | None = None  # last hello, replayed on connect
        self.frames: dict[int, deque[Frame]] = {}
        self._frames_lock = threading.Lock()

    # -- frames ring buffer (read by REST from any thread) --
    def store_frame(self, f: Frame) -> None:
        with self._frames_lock:
            self.frames.setdefault(f.channel, deque(maxlen=FRAME_RING)).append(f)

    def find_frame(self, frame_id: str) -> Frame | None:
        try:
            ch_s, seq_s = frame_id.removeprefix("D").split("-", 1)
            ch, seq = int(ch_s), int(seq_s)
        except ValueError:
            return None
        with self._frames_lock:
            for f in reversed(self.frames.get(ch, ())):
                if f.seq == seq:
                    return f
        return None

    # -- broadcast --
    def _broadcast(self, payload: str | bytes) -> None:
        for q in self.clients:
            if q.full():  # slow client: drop the oldest message
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            q.put_nowait(payload)

    async def pump(self) -> None:
        """Move items from the thread queue onto the asyncio side."""
        while True:
            drained = False
            while True:
                try:
                    item: Any = self.outbound.get_nowait()
                except queue.Empty:
                    break
                drained = True
                if isinstance(item, Frame):
                    self.store_frame(item)
                    self._broadcast(item.to_bytes())
                else:
                    text = json.dumps(item, separators=(",", ":"))
                    if item.get("type") == "hello":
                        self.hello = text
                    self._broadcast(text)
            # Always yield so WS senders run even under a steady stream.
            await asyncio.sleep(0 if drained else 0.005)


def create_app(hub: Hub) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(hub.pump())
        yield
        task.cancel()

    app = FastAPI(title="Swarm-Guided Rescue", lifespan=lifespan)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=CLIENT_QUEUE)
        if hub.hello:
            q.put_nowait(hub.hello)
        hub.clients.add(q)

        async def sender() -> None:
            while True:
                payload = await q.get()
                if isinstance(payload, bytes):
                    await ws.send_bytes(payload)
                else:
                    await ws.send_text(payload)

        send_task = asyncio.create_task(sender())
        try:
            while True:
                text = await ws.receive_text()
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict) and "type" in msg:
                    hub.inbound.put(msg)
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(q)
            send_task.cancel()

    @app.get("/api/frames/{frame_id}")
    def get_frame(frame_id: str) -> Response:
        f = hub.find_frame(frame_id)
        if f is None:
            raise HTTPException(404, "frame not in ring buffer")
        return Response(f.jpeg, media_type="image/jpeg")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "clients": len(hub.clients)}

    if MAPS_DIR.is_dir():
        app.mount("/maps", StaticFiles(directory=MAPS_DIR), name="maps")

    if WEB_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        # SPA fallback: "/", "/editor", ... all serve index.html.
        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = WEB_DIST / path
            if path and candidate.is_file() and WEB_DIST in candidate.resolve().parents:
                return FileResponse(candidate)
            return FileResponse(WEB_DIST / "index.html")

    return app


def start_server(
    outbound: queue.Queue,
    inbound: queue.Queue,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> tuple[threading.Thread, Hub]:
    """Run uvicorn on a daemon thread; the sim keeps the main thread (GL)."""
    hub = Hub(outbound, inbound)
    config = uvicorn.Config(create_app(hub), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="sgr-server", daemon=True)
    thread.start()
    return thread, hub
