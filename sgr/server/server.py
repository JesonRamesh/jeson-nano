"""FastAPI server (owned by D): WS /ws, REST /api/*, map files, static web/dist.

Runs uvicorn on a background thread; the sim loop keeps the main thread (GL).
It talks to the sim loop ONLY through ``sgr.bus.Bus``:

* outbound: ``Bus.drain_outbound()`` -> ``JsonOut`` (text) / ``FrameOut`` (binary)
  broadcast to every browser. Frames also go into a ring buffer for trace evidence.
* inbound: browser JSON -> ``parse_client_message`` -> ``Bus.send_command``.

The server builds the ``hello`` message itself from the scenario (structure only,
no hidden fields) and sends it to every browser on connect.

Usage (from sgr/app.py)::

    from sgr.server import start_server
    start_server(bus, args.map, scenario)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from sgr import config
from sgr.bus import Bus, FrameOut, JsonOut
from sgr.contracts import (
    DroneMeta,
    HelloMsg,
    MapMeta,
    PriorMap,
    Scenario,
    Transform,
    parse_client_message,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIST = REPO_ROOT / "web" / "dist"
MAPS_DIR = REPO_ROOT / "maps"
PROPOSE_GRAPH = REPO_ROOT / "tools" / "propose_graph.py"

FRAME_RING = 300  # frames kept per drone for trace evidence (spec §10.2)
CLIENT_QUEUE = 256  # per-browser backlog before we start dropping the oldest
MAP_FILES = {"grid.png", "scan.glb", "transform.json", "preview.png"}


# ---------------------------------------------------------------------------
# hello
# ---------------------------------------------------------------------------


def build_hello(map_dir: Path, scenario: Scenario) -> HelloMsg:
    """Public map info for the browser. Never includes fires/survivor/decoy/compromised."""
    map_dir = Path(map_dir)
    free = PriorMap.load_free_grid(map_dir / scenario.grid.file)
    transform = None
    if scenario.scan is not None:
        tpath = map_dir / scenario.scan.transform
        if tpath.is_file():
            transform = Transform.model_validate_json(tpath.read_text(encoding="utf-8"))
    return HelloMsg(
        map=MapMeta(
            name=map_dir.name,
            grid_url=f"/maps/{map_dir.name}/{scenario.grid.file}",
            resolution=scenario.grid.resolution,
            size=(free.shape[1], free.shape[0]),
            origin=scenario.grid.origin,
            belief_resolution=config.BELIEF_RES,
        ),
        graph=scenario.graph,
        entrance=scenario.entrance,
        drones=[
            DroneMeta(id=d, colour=config.DRONE_COLOURS.get(d, "#cccccc")) for d in scenario.drones.ids
        ],
        transform=transform,
    )


# ---------------------------------------------------------------------------
# Hub: fan-out + frame ring buffer
# ---------------------------------------------------------------------------


class Hub:
    def __init__(self, bus: Bus, hello: HelloMsg | None) -> None:
        self.bus = bus
        self.hello_text = hello.model_dump_json() if hello else None
        self.clients: set[asyncio.Queue] = set()
        self._frames: dict[str, deque[FrameOut]] = {}  # drone -> recent frames
        self._frames_lock = threading.Lock()

    def store_frame(self, f: FrameOut) -> None:
        with self._frames_lock:
            self._frames.setdefault(f.drone, deque(maxlen=FRAME_RING)).append(f)

    def find_frame(self, frame_id: str) -> bytes | None:
        """JPEG bytes for a frame_id still in the ring buffer."""
        with self._frames_lock:
            for ring in self._frames.values():
                for f in reversed(ring):
                    if f.frame_id == frame_id:
                        return f.data[5:]  # strip [u8 channel][u32 seq] header
        return None

    def _broadcast(self, payload: str | bytes) -> None:
        for q in self.clients:
            if q.full():  # slow browser: drop its oldest message
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            q.put_nowait(payload)

    async def pump(self) -> None:
        """Move bus items onto the asyncio side and fan them out."""
        while True:
            items = self.bus.drain_outbound()
            for item in items:
                if isinstance(item, FrameOut):
                    self.store_frame(item)
                    self._broadcast(item.data)
                elif isinstance(item, JsonOut):
                    self._broadcast(item.text)
            # Always yield so WS senders run even under a steady stream.
            await asyncio.sleep(0 if items else 0.005)

    def handle_client_text(self, text: str) -> str | None:
        """Validate a browser message and forward it to the sim. Returns an error or None."""
        try:
            msg = parse_client_message(text)
        except ValidationError as e:
            return f"invalid client message: {e.errors()[0].get('msg', 'bad message')}"
        self.bus.send_command(msg)
        return None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def _map_dir(name: str, active: Path | None) -> Path:
    if active is not None and name == active.name:
        return active
    if not name.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(400, "bad map name")
    d = MAPS_DIR / name
    if not d.is_dir():
        raise HTTPException(404, f"map {name!r} not found")
    return d


def create_app(hub: Hub, map_dir: Path | None = None) -> FastAPI:
    active = Path(map_dir).resolve() if map_dir is not None else None

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(hub.pump())
        yield
        task.cancel()

    app = FastAPI(title="Swarm-Guided Rescue", lifespan=lifespan)

    # ---- WebSocket ----------------------------------------------------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=CLIENT_QUEUE)
        if hub.hello_text:
            q.put_nowait(hub.hello_text)
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
                err = hub.handle_client_text(await ws.receive_text())
                if err:
                    print(f"sgr.server: {err}")
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(q)
            send_task.cancel()

    # ---- REST ---------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "clients": len(hub.clients)}

    @app.get("/api/frames/{frame_id}")
    def get_frame(frame_id: str) -> Response:
        jpeg = hub.find_frame(frame_id)
        if jpeg is None:
            raise HTTPException(404, "frame not in ring buffer")
        return Response(jpeg, media_type="image/jpeg")

    @app.get("/api/maps/{name}/scenario")
    def get_scenario(name: str) -> JSONResponse:
        """Full scenario incl. hidden fields: for the route editor, never the dashboard."""
        path = _map_dir(name, active) / "scenario.json"
        if not path.is_file():
            raise HTTPException(404, "no scenario.json")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.put("/api/maps/{name}/scenario")
    def put_scenario(name: str, body: dict = Body(...)) -> dict:
        try:
            scenario = Scenario.model_validate(body)
        except ValidationError as e:
            raise HTTPException(422, e.errors()) from e
        path = _map_dir(name, active) / "scenario.json"
        path.write_text(scenario.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
        return {"ok": True}

    @app.post("/api/maps/{name}/propose_graph")
    def propose_graph(name: str) -> JSONResponse:
        """Runs W's tools/propose_graph.py on the map, then returns the updated scenario."""
        d = _map_dir(name, active)
        if not PROPOSE_GRAPH.is_file():
            raise HTTPException(501, "tools/propose_graph.py not available yet (W)")
        try:
            r = subprocess.run(
                [sys.executable, str(PROPOSE_GRAPH), str(d)],
                capture_output=True, text=True, timeout=120, cwd=REPO_ROOT,
            )
        except subprocess.TimeoutExpired as e:
            raise HTTPException(504, "propose_graph timed out") from e
        if r.returncode != 0:
            raise HTTPException(500, (r.stderr or r.stdout)[-2000:])
        return JSONResponse(json.loads((d / "scenario.json").read_text(encoding="utf-8")))

    # ---- map files ----------------------------------------------------
    @app.get("/maps/{name}/{file}")
    def map_file(name: str, file: str) -> FileResponse:
        if file not in MAP_FILES:
            raise HTTPException(404)
        path = _map_dir(name, active) / file
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    # ---- built web app (demo) -----------------------------------------
    if WEB_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith(("api/", "maps/", "ws")):
                raise HTTPException(404)
            candidate = (WEB_DIST / path).resolve()
            if path and candidate.is_file() and WEB_DIST.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(WEB_DIST / "index.html")  # "/", "/editor", ...

    return app


def start_server(
    bus: Bus,
    map_dir: Path | str,
    scenario: Scenario,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> threading.Thread:
    """Start the server on a daemon thread and return it."""
    map_dir = Path(map_dir)
    hub = Hub(bus, build_hello(map_dir, scenario))
    server = uvicorn.Server(
        uvicorn.Config(create_app(hub, map_dir), host=host, port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, name="sgr-server", daemon=True)
    thread.start()
    print(f"sgr: dashboard on http://{host}:{port}")
    return thread
