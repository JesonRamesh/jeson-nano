"""Every contract model survives a JSON round-trip; WS unions dispatch on `type`.

When you add a model to sgr/contracts.py, add an example here (test_every_model_has_an_example
fails until you do).
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel, ValidationError

from sgr import contracts as c

POSE = c.Pose(x=1.0, y=2.0, z=1.6, yaw=0.5)
GRAPH = c.Graph(
    nodes=[c.GraphNode(id="n1", x=2, y=3, kind="entrance"), c.GraphNode(id="n2", x=8, y=3, kind="junction")],
    edges=[c.GraphEdge(id="e1", a="n1", b="n2", width=1.8)],
)
ENTRANCE = c.Entrance(x=2, y=3, yaw=0)
EVIDENCE = c.Evidence(sensor="lidar", value=0.8, scan=[6.0] * 36)
ROW = dict(id="tr-1", t=12.3, drone="D2", action="MARK_BLOCKED", reason="LiDAR 0.8 m < 3.1 m expected", evidence=EVIDENCE)
BOX = c.Box(x=10, y=20, w=30, h=40, label="fire", score=0.62)
GEMMA = c.GemmaResult(fire=False, person=False, confidence=0.31, caption="orange panel on wall", latency_s=1.2)
ROUTE = c.Route(points=[(2, 3), (8, 3), (8, 10)], status="ok", version=3)
SURV = c.SurvivorState(status="confirmed", x=26.5, y=11.0, conf=0.9)
DRONE = c.DroneState(id="D1", x=1, y=2, z=1.6, yaw=0, status="EXPLORING", task="area n4", temp=21.0, lidar=[6.0] * 36, alive=True)
HANDLE = c.Handle(id="debris-1", kind="debris", x=14, y=3)
TRANSFORM = c.Transform(M=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], grid_size=(300, 200))
EVENT = c.EventMsg(kind="replan", drone="D1", xy=(14, 3))
SCENARIO = c.Scenario(
    name="t",
    scan=c.ScanRef(),
    entrance=ENTRANCE,
    graph=GRAPH,
    compromised=[c.Compromise(edge="e1", at=0.5)],
    fires=[c.Fire(x=5, y=3)],
    survivor=c.Survivor(x=7, y=3, yaw=1.0),
    decoy=c.Decoy(x=4, y=3),
    drones=c.DroneSetup(start=[(2, 2.5), (2, 3.5), (2.5, 2.5), (2.5, 3.5)]),
    flare=c.Flare(t=65, target="D3"),
    beats=c.Beats(decoy_drone="D2", d3_area_node="n2"),
    checkpoints=[c.Checkpoint(x=8, y=3, node="n2")],
)

EXAMPLES: dict[type[BaseModel], BaseModel] = {
    c.XY: c.XY(x=1, y=2),
    c.Pose: POSE,
    c.CamInfo: c.CamInfo(fx=120, fy=120, cx=160, cy=120, pose=POSE),
    c.Handle: HANDLE,
    c.VelCmd: c.VelCmd(vx=0.5, vy=-0.2, wz=0.1),
    c.Readings: c.Readings(t=1.0, pose=POSE, lidar=[6.0] * 36, temperature=22.5),
    c.GridRef: c.GridRef(),
    c.ScanRef: c.ScanRef(),
    c.Entrance: ENTRANCE,
    c.GraphNode: GRAPH.nodes[0],
    c.GraphEdge: GRAPH.edges[0],
    c.Graph: GRAPH,
    c.Compromise: c.Compromise(edge="e1"),
    c.Fire: c.Fire(x=1, y=2, size=1.5),
    c.Survivor: c.Survivor(x=1, y=2),
    c.Decoy: c.Decoy(x=1, y=2),
    c.DroneSetup: SCENARIO.drones,
    c.Flare: c.Flare(t=65, target="D3"),
    c.Heat: c.Heat(),
    c.Beats: c.Beats(decoy_drone="D2"),
    c.Scenario: SCENARIO,
    c.Transform: TRANSFORM,
    c.GroundTruth: c.GroundTruth(
        fires=[c.Fire(x=5, y=3)], survivor=c.Survivor(x=7, y=3), decoy=None, debris=[HANDLE], compromised=[], dead_drones=["D3"]
    ),
    c.Box: BOX,
    c.GemmaResult: GEMMA,
    c.Detection: c.Detection(
        id="det-1", drone="D2", t=25.0, label="fire", world_xy=c.XY(x=8.7, y=6.5), score_fast=0.62,
        gemma=GEMMA, combined=0.19, status="rejected", frame_id="D2-77", bbox=BOX, relooks=2,
    ),
    c.PerceptionMetrics: c.PerceptionMetrics(mode="local", model_tag="gemma4:e4b", p50_s=1.1, p95_s=2.0, queue_len=1),
    c.Evidence: EVIDENCE,
    c.TraceRow: c.TraceRow(**ROW),
    c.Route: ROUTE,
    c.OldRoute: c.OldRoute(points=[(2, 3), (14, 3)], until=80.0),
    c.Checkpoint: c.Checkpoint(x=8, y=10, drone="D4", node="n4"),
    c.SurvivorState: SURV,
    c.DroneState: DRONE,
    c.FFState: c.FFState(x=2, y=3, yaw=0),
    c.MapMeta: c.MapMeta(name="test", grid_url="/maps/test/grid.png", size=(300, 200)),
    c.DroneMeta: c.DroneMeta(id="D1", colour="#4fc3f7"),
    c.GemmaView: c.GemmaView(caption="fire in corridor", conf=0.9, pending=False),
    # server -> client
    c.HelloMsg: c.HelloMsg(
        map=c.MapMeta(name="test", grid_url="/maps/test/grid.png", size=(300, 200)), graph=GRAPH,
        entrance=ENTRANCE, drones=[c.DroneMeta(id="D1", colour="#4fc3f7")], transform=TRANSFORM,
    ),
    c.StateMsg: c.StateMsg(
        t=10, speed=2.5, act=2, phase="act2", drones=[DRONE], ff=c.FFState(x=2, y=3, yaw=0), route=ROUTE,
        old_routes=[c.OldRoute(points=[(2, 3)], until=5)], checkpoints=[c.Checkpoint(x=8, y=10)],
        survivor=SURV, click_handles=[HANDLE], voice="online",
    ),
    c.MapDeltaMsg: c.MapDeltaMsg(cells=[(70, 15, "blocked_new"), (1, 2, "fire")], searched=[(3, 4)], map_full="iVBORw0="),
    c.TraceMsg: c.TraceMsg(**ROW),
    c.DetectionMsg: c.DetectionMsg(drone="D2", frame_id="D2-77", boxes=[BOX], gemma=c.GemmaView(pending=True), status="pending"),
    c.EventMsg: EVENT,
    c.GuidanceMsg: c.GuidanceMsg(arrow_deg=-30, distance=4.2, subtitle="Bear left.", say="Bear left.", alarm=False, severity="warn"),
    c.BudgetMsg: c.BudgetMsg(network="offline", model_tag="gemma4:e4b", gemma_p50_s=1.1, gemma_p95_s=2.3, vram_used_mb=6600, vram_total_mb=8192, sim_rtf=1.0, speed=1.0, control_hz=10, camera_fps=10),
    c.SummaryMsg: c.SummaryMsg(time_s=95, distance_m=48.2, hazards_avoided=3, corrections=2, replans=2, trace_url="/api/trace"),
    c.CalloutMsg: c.CalloutMsg(speaker="D2", kind="fire_side", text="Careful, fire on your right.", severity="warn",
                               source="gemini", audio_b64="UklGRg==", audio_mime="audio/wav"),
    # client -> server
    c.ClickMsg: c.ClickMsg(mode="fire", x=14, y=17),
    c.UndoMsg: c.UndoMsg(),
    c.ControlMsg: c.ControlMsg(action="speed", value=2.5),
    c.Act2StartMsg: c.Act2StartMsg(),
    c.SnapshotMsg: c.SnapshotMsg(op="save", name="act2_start"),
    c.FFInputMsg: c.FFInputMsg(fwd=1.0, strafe=-0.5, yaw=1.57),
    c.VoiceModeMsg: c.VoiceModeMsg(mode="offline"),
    c.FailDroneMsg: c.FailDroneMsg(drone="D2"),
    # swarm
    c.SwarmDrone: c.SwarmDrone(id="D1", status="EXPLORING", task="area n4"),
    c.SwarmState: c.SwarmState(
        t=10, phase="explore", drones=[c.SwarmDrone(id="D1", status="EXPLORING")], route=ROUTE,
        survivor=SURV, chain_formed=True, lead="D1", map_delta=[(1, 2, "unsafe")], searched_delta=[(1, 2)], events=[EVENT],
        relook_requests=["det0001"],
    ),
    c.PriorMap: c.PriorMap(
        name="test", resolution=0.1, origin=(0, 0), size=(300, 200), entrance=ENTRANCE, graph=GRAPH,
        drone_ids=["D1", "D2", "D3", "D4"], altitude=1.6,
    ),
}

SERVER_TYPES = [c.HelloMsg, c.StateMsg, c.MapDeltaMsg, c.TraceMsg, c.DetectionMsg, c.EventMsg, c.GuidanceMsg, c.BudgetMsg, c.SummaryMsg, c.CalloutMsg]
CLIENT_TYPES = [c.ClickMsg, c.UndoMsg, c.ControlMsg, c.Act2StartMsg, c.SnapshotMsg, c.FFInputMsg, c.VoiceModeMsg, c.FailDroneMsg]


def _contract_models() -> set[type[BaseModel]]:
    return {
        obj
        for name, obj in inspect.getmembers(c, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == c.__name__ and not name.startswith("_")
    }


def test_contract_version() -> None:
    assert c.CONTRACT_VERSION == 5


def test_every_model_has_an_example() -> None:
    missing = {m.__name__ for m in _contract_models()} - {m.__name__ for m in EXAMPLES}
    assert not missing, f"add examples for: {sorted(missing)}"


@pytest.mark.parametrize("model", list(EXAMPLES), ids=lambda m: m.__name__)
def test_json_round_trip(model: type[BaseModel]) -> None:
    obj = EXAMPLES[model]
    again = model.model_validate_json(obj.model_dump_json())
    assert again.model_dump() == obj.model_dump()


@pytest.mark.parametrize("model", SERVER_TYPES, ids=lambda m: m.__name__)
def test_server_union_dispatch(model: type[BaseModel]) -> None:
    parsed = c.parse_server_message(EXAMPLES[model].model_dump_json())
    assert type(parsed) is model


@pytest.mark.parametrize("model", CLIENT_TYPES, ids=lambda m: m.__name__)
def test_client_union_dispatch(model: type[BaseModel]) -> None:
    parsed = c.parse_client_message(EXAMPLES[model].model_dump_json())
    assert type(parsed) is model


def test_client_message_from_raw_browser_json() -> None:
    msg = c.parse_client_message('{"type": "click", "mode": "debris", "x": 14.0, "y": 3.0}')
    assert isinstance(msg, c.ClickMsg) and msg.mode == "debris"
    with pytest.raises(ValidationError):
        c.parse_client_message('{"type": "teleport"}')


def test_frame_header_round_trip() -> None:
    buf = c.pack_frame(3, 123456, b"\xff\xd8jpeg")
    assert buf[:5] == bytes([3, 0, 1, 0xE2, 0x40])  # big-endian u8 + u32
    assert c.unpack_frame(buf) == (3, 123456, b"\xff\xd8jpeg")


def test_test_map_scenario_loads(test_map) -> None:
    sc = c.Scenario.load(test_map)
    assert sc.name == "test" and len(sc.drones.ids) == 4
    prior = c.PriorMap.from_map(test_map, sc)
    assert prior.free.shape == (200, 300)
    assert prior.free[int(sc.entrance.y / 0.1), int(sc.entrance.x / 0.1)]
    assert not hasattr(prior, "compromised") and not hasattr(prior, "survivor")
    assert not hasattr(prior, "checkpoints")  # authored checkpoints sit on the survivor: never in the prior


def test_marshgate_checkpoints_load() -> None:
    """W's real map: the three authored checkpoints survive Scenario.load (contract v2)."""
    from pathlib import Path

    sc = c.Scenario.load(Path(__file__).resolve().parents[1] / "maps" / "marshgate-route")
    assert [cp.node for cp in sc.checkpoints] == ["n4", "n13", "n16"]
    assert all(cp.drone is None for cp in sc.checkpoints)


def test_old_scenario_without_checkpoints_still_loads(test_map) -> None:
    assert c.Scenario.load(test_map).checkpoints == []
