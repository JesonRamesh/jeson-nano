"""Thread-safe queues between the sim loop (main thread) and the server thread. Owned by S.

    sim loop  --publish()/publish_frame()-->  outbound  --> server (D) sends to all WS clients
    server    --send_command()-->             inbound   --> sim loop drains once per tick

Outbound is bounded and drops the OLDEST item when full, so a slow or absent browser
never blocks or bloats the sim loop. Inbound is unbounded (commands are rare and small).
"""

from __future__ import annotations

import queue
from dataclasses import dataclass

from pydantic import BaseModel

from sgr.contracts import pack_frame


@dataclass(frozen=True)
class JsonOut:
    """A JSON message for all clients (already serialised)."""

    text: str


@dataclass(frozen=True)
class FrameOut:
    """A binary camera frame for all clients: [u8 channel][u32 seq][JPEG]."""

    drone: str
    frame_id: str
    data: bytes  # packed, ready to send


Outbound = JsonOut | FrameOut


class Bus:
    def __init__(self, outbound_max: int = 2000) -> None:
        self._out: queue.Queue[Outbound] = queue.Queue(maxsize=outbound_max)
        self._in: queue.Queue[BaseModel] = queue.Queue()

    # ---- sim loop side -------------------------------------------------
    def publish(self, msg: BaseModel) -> None:
        """Queue any server->client contract message (HelloMsg, StateMsg, TraceMsg, ...)."""
        self._put_out(JsonOut(msg.model_dump_json()))

    def publish_frame(self, drone: str, channel: int, seq: int, frame_id: str, jpeg: bytes) -> None:
        self._put_out(FrameOut(drone=drone, frame_id=frame_id, data=pack_frame(channel, seq, jpeg)))

    def drain_inbound(self) -> list[BaseModel]:
        """All client commands received since the last call (non-blocking)."""
        items: list[BaseModel] = []
        while True:
            try:
                items.append(self._in.get_nowait())
            except queue.Empty:
                return items

    # ---- server side ---------------------------------------------------
    def send_command(self, msg: BaseModel) -> None:
        """Queue a parsed client->server contract message (ClickMsg, UndoMsg, ...)."""
        self._in.put(msg)

    def get_outbound(self, timeout: float | None = None) -> Outbound | None:
        """Block up to `timeout` s for the next outbound item; None if nothing arrived."""
        try:
            return self._out.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_outbound(self, max_items: int = 1000) -> list[Outbound]:
        items: list[Outbound] = []
        while len(items) < max_items:
            try:
                items.append(self._out.get_nowait())
            except queue.Empty:
                break
        return items

    # ---- internals -----------------------------------------------------
    def _put_out(self, item: Outbound) -> None:
        while True:
            try:
                self._out.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._out.get_nowait()  # drop oldest
                except queue.Empty:
                    pass
