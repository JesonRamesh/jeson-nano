"""Server tests (owned by D): hello, WS fan-out, inbound commands, frames, REST."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sgr.bus import Bus
from sgr.contracts import (
    ClickMsg,
    DroneState,
    Scenario,
    StateMsg,
    UndoMsg,
    parse_server_message,
    unpack_frame,
)
from sgr.server.server import Hub, build_hello, create_app

TEST_MAP = Path(__file__).resolve().parents[1] / "maps" / "test"


@pytest.fixture
def map_dir(tmp_path: Path) -> Path:
    d = tmp_path / "test"
    shutil.copytree(TEST_MAP, d)
    return d


@pytest.fixture
def setup(map_dir: Path):
    scenario = Scenario.load(map_dir)
    bus = Bus()
    hub = Hub(bus, build_hello(map_dir, scenario))
    with TestClient(create_app(hub, map_dir)) as client:
        yield client, bus, hub, scenario


def test_hello_has_no_hidden_fields(map_dir: Path) -> None:
    hello = build_hello(map_dir, Scenario.load(map_dir))
    text = hello.model_dump_json()
    for hidden in ("compromised", "fires", "survivor", "decoy", "flare"):
        assert hidden not in text
    assert hello.map.size[0] > 0 and hello.map.grid_url == "/maps/test/grid.png"
    assert [d.id for d in hello.drones] == ["D1", "D2", "D3", "D4"]


def test_ws_hello_state_frames_and_commands(setup) -> None:
    client, bus, hub, scenario = setup
    with client.websocket_connect("/ws") as ws:
        assert parse_server_message(ws.receive_text()).type == "hello"

        drone = DroneState(id="D1", x=1, y=2, z=1.6, yaw=0, status="EXPLORING", task="", temp=20,
                           lidar=[6.0] * 36, alive=True)
        bus.publish(StateMsg(t=1.5, speed=1.0, drones=[drone]))
        bus.publish_frame("D1", 1, 7, "D1-7", b"\xff\xd8jpeg")
        msg = parse_server_message(ws.receive_text())
        assert msg.type == "state" and msg.t == 1.5
        ch, seq, jpeg = unpack_frame(ws.receive_bytes())
        assert (ch, seq, jpeg) == (1, 7, b"\xff\xd8jpeg")

        ws.send_text(json.dumps({"type": "click", "mode": "debris", "x": 3.0, "y": 4.0}))
        ws.send_text(json.dumps({"type": "undo"}))
        ws.send_text("not json")  # ignored, must not kill the socket
        ws.send_text(json.dumps({"type": "control", "action": "pause"}))
        # The pause arriving proves the bad message didn't drop the connection.
        got: list = []
        for _ in range(100):
            if len(got) >= 3:
                break
            time.sleep(0.02)
            got += bus.drain_inbound()
        assert isinstance(got[0], ClickMsg) and got[0].x == 3.0
        assert isinstance(got[1], UndoMsg)
        assert got[2].type == "control"

    r = client.get("/api/frames/D1-7")
    assert r.status_code == 200 and r.content == b"\xff\xd8jpeg"
    assert client.get("/api/frames/D1-999").status_code == 404


def test_map_files_and_scenario_rest(setup, map_dir: Path) -> None:
    client, *_ = setup
    r = client.get("/maps/test/grid.png")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert client.get("/maps/test/scenario.json").status_code == 404  # not a public map file
    for bad in ("/maps/..%2F..%2Fetc/grid.png", "/maps/test/..%2Fscenario.json"):
        assert client.get(bad).status_code in (400, 404)

    sc = client.get("/api/maps/test/scenario").json()
    sc["flare"]["t"] = 42.0
    assert client.put("/api/maps/test/scenario", json=sc).status_code == 200
    assert Scenario.load(map_dir).flare.t == 42.0
    assert client.put("/api/maps/test/scenario", json={"name": "broken"}).status_code == 422
