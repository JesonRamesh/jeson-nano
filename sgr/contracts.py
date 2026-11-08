"""Shared data contracts, v1. Owned by S.

This is the ONLY shared surface between team members (spec §1.2). Everything that
crosses a module boundary or the WebSocket is defined here.

Rules after H1: only ADD fields (with defaults). Never rename or remove. Every change
bumps CONTRACT_VERSION and D mirrors it in web/src/contracts.ts within 30 min.

Conventions (spec §3): metres, z up, x east, y north, angles in radians (CCW from +x)
unless the field name says `_deg`. Grid PNG is 0.1 m/px, bottom-left pixel = world (0, 0),
image rows are flipped. Belief grid (map_delta cells) is 0.2 m: i = column (x), j = row
counted from the bottom (y).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter, field_validator

CONTRACT_VERSION = 3
# v2: Scenario.checkpoints, Checkpoint.node (W's authored checkpoints), SwarmState.relook_requests (§8.4)
# v3: CalloutMsg (Act 2 drone voices)


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------

Label = Literal["fire", "person", "decoy"]
DetectionStatus = Literal["pending", "confirmed", "low_conf", "rejected", "unconfirmed"]
Sensor = Literal["lidar", "camera", "gemma", "temp", "heartbeat", "planner"]
Action = Literal[
    "ASSIGN_AREA",
    "EXPLORE",
    "INSPECT",
    "CONFIRM_HAZARD",
    "REJECT_DETECTION",
    "UNCONFIRMED",
    "MARK_BLOCKED",
    "REPLAN",
    "RETREAT",
    "DRONE_LOST",
    "REASSIGN",
    "SURVIVOR_CONFIRMED",
    "CHAIN_POSITION",
    "NO_ROUTE",
    "ESCORT",
    "FF_CORRECTION",
    "MISSION_COMPLETE",
]
NodeKind = Literal["entrance", "junction", "door", "room", "stairs"]
CellState = Literal["wall_prior", "free", "blocked_new", "fire", "danger", "unsafe"]
# idle = no route requested yet; none = no safe route exists ("No safe route. Holding.")
RouteStatus = Literal["idle", "ok", "replanning", "none"]
EventKind = Literal[
    "fire_confirmed",
    "false_alarm",
    "drone_lost",
    "blocked",
    "replan",
    "survivor_found",
    "chain_formed",
    "no_route",
    "mission_complete",
]
Severity = Literal["info", "warn", "danger"]
PerceptionMode = Literal["mock", "remote", "local"]

XYPoint = tuple[float, float]


# ---------------------------------------------------------------------------
# Geometry / sensing primitives (§4)
# ---------------------------------------------------------------------------


class XY(_Base):
    x: float
    y: float


class Pose(_Base):
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0


class CamInfo(_Base):
    """Pinhole intrinsics + camera pose for one rendered frame."""

    width: int = 320
    height: int = 240
    fovy_deg: float = 90.0
    fx: float
    fy: float
    cx: float
    cy: float
    pose: Pose
    pitch_deg: float = -15.0  # tilted down


class Handle(_Base):
    """A placed object (judge click or flare-up) that can be removed with Undo."""

    id: str
    kind: Literal["debris", "fire"]
    x: float
    y: float


class VelCmd(_Base):
    """World-frame velocity command (m/s, rad/s)."""

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wz: float = 0.0


class Readings(_Base):
    """One drone's sensor readings for one control tick. A lost drone has no entry."""

    t: float
    pose: Pose
    lidar: list[float]  # 36 rays, ray k at yaw + k*10deg, metres, 6.0 = no hit
    temperature: float  # degC


# ---------------------------------------------------------------------------
# Scenario (§5) and map files (§6.3)
# ---------------------------------------------------------------------------


class GridRef(_Base):
    file: str = "grid.png"
    resolution: float = 0.1
    origin: XYPoint = (0.0, 0.0)


class ScanRef(_Base):
    file: str = "scan.glb"
    transform: str = "transform.json"


class Entrance(_Base):
    x: float
    y: float
    yaw: float = 0.0


class GraphNode(_Base):
    id: str
    x: float
    y: float
    kind: NodeKind


class GraphEdge(_Base):
    id: str
    a: str
    b: str
    width: float = 1.8


class Graph(_Base):
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []


class Compromise(_Base):
    edge: str
    at: float = 0.5  # 0..1 along a -> b
    kind: Literal["debris"] = "debris"


class Fire(_Base):
    x: float
    y: float
    size: float = 1.5


class Survivor(_Base):
    x: float
    y: float
    yaw: float = 0.0


class Decoy(_Base):
    x: float
    y: float
    kind: str = "orange_panel"


class DroneSetup(_Base):
    ids: list[str] = ["D1", "D2", "D3", "D4"]
    start: list[XYPoint]
    altitude: float = 1.6


class Flare(_Base):
    t: float
    target: str


class Heat(_Base):
    limit_c: float = 70.0
    ambient_c: float = 20.0


class Beats(_Base):
    decoy_drone: str | None = None
    d3_area_node: str | None = None


class Scenario(_Base):
    version: int = 1
    name: str
    grid: GridRef = GridRef()
    scan: ScanRef | None = None  # None = no scan.glb (e.g. maps/test); D extrudes walls from grid
    entrance: Entrance
    graph: Graph
    compromised: list[Compromise] = []
    fires: list[Fire] = []
    survivor: Survivor
    decoy: Decoy | None = None
    drones: DroneSetup
    flare: Flare | None = None
    heat: Heat = Heat()
    beats: Beats = Beats()
    # Authored by W in the route editor: expected chain positions, for the editor overlay and tests.
    # NEVER copied into PriorMap: the last one sits on the survivor (honesty rule).
    checkpoints: list[Checkpoint] = []

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        """Load from a map directory or a scenario.json path."""
        p = Path(path)
        if p.is_dir():
            p = p / "scenario.json"
        return cls.model_validate_json(p.read_text(encoding="utf-8"))


class Transform(_Base):
    """transform.json (§6.3): raw scan.glb -> sim frame."""

    M: list[list[float]]  # 4x4 row-major
    pitch: float = 0.1
    grid_origin: XYPoint = (0.0, 0.0)
    grid_size: tuple[int, int]  # (w, h) px


class GroundTruth(_Base):
    """ONLY for mock perception and the mission summary. Never for sgr/swarm or sgr/guidance."""

    fires: list[Fire]
    survivor: Survivor
    decoy: Decoy | None
    debris: list[Handle]
    compromised: list[Compromise]
    dead_drones: list[str] = []


# ---------------------------------------------------------------------------
# Perception (§4, §7)
# ---------------------------------------------------------------------------


class Box(_Base):
    """Pixel bbox in a 320x240 frame."""

    x: float
    y: float
    w: float
    h: float
    label: Label
    score: float


class GemmaResult(_Base):
    fire: bool
    person: bool
    confidence: float
    caption: str
    latency_s: float | None = None
    model_tag: str | None = None


class Detection(_Base):
    id: str
    drone: str
    t: float
    label: Label
    world_xy: XY | None
    score_fast: float
    gemma: GemmaResult | None = None
    combined: float | None = None
    status: DetectionStatus
    frame_id: str
    bbox: Box | None = None
    relooks: int = 0


class PerceptionMetrics(_Base):
    mode: PerceptionMode
    model_tag: str | None = None
    p50_s: float | None = None
    p95_s: float | None = None
    queue_len: int = 0
    failures: int = 0
    timeouts: int = 0


# ---------------------------------------------------------------------------
# Decision trace (§8.5)
# ---------------------------------------------------------------------------


class Evidence(_Base):
    sensor: Sensor
    value: Any  # number, string, list or dict; must not be None
    frame_id: str | None = None
    scan: list[float] | None = None  # LiDAR ranges, when sensor == "lidar"

    @field_validator("value")
    @classmethod
    def _value_present(cls, v: Any) -> Any:
        if v is None:
            raise ValueError("evidence.value is required: no action without evidence")
        return v


class TraceRow(_Base):
    id: str
    t: float
    drone: str  # drone id, or "swarm" for swarm-wide decisions
    action: Action
    reason: str
    evidence: Evidence  # required: no action without evidence


# ---------------------------------------------------------------------------
# Pieces shared by WS messages and SwarmState (§10.2)
# ---------------------------------------------------------------------------


class Route(_Base):
    points: list[XYPoint] = []
    status: RouteStatus = "idle"
    version: int = 0


class OldRoute(_Base):
    points: list[XYPoint]
    until: float  # sim time when it stops being drawn


class Checkpoint(_Base):
    x: float
    y: float
    drone: str | None = None
    node: str | None = None  # route-graph node id, when the checkpoint sits on a node


Scenario.model_rebuild()  # Scenario.checkpoints refers to Checkpoint, defined just above


class SurvivorState(_Base):
    status: Literal["unknown", "suspected", "confirmed"]
    x: float | None = None
    y: float | None = None
    conf: float | None = None


class DroneState(_Base):
    id: str
    x: float
    y: float
    z: float
    yaw: float
    status: str  # chip text, e.g. EXPLORING / INSPECTING / RETREAT / LOST
    task: str  # short task text, e.g. "area n4"
    temp: float
    lidar: list[float]
    alive: bool


class FFState(_Base):
    x: float
    y: float
    yaw: float


class MapMeta(_Base):
    name: str
    grid_url: str
    resolution: float = 0.1
    size: tuple[int, int]  # (w, h) px of grid.png
    origin: XYPoint = (0.0, 0.0)
    belief_resolution: float = 0.2


class DroneMeta(_Base):
    id: str
    colour: str  # "#rrggbb"


class GemmaView(_Base):
    caption: str | None = None
    conf: float | None = None
    pending: bool = False


# ---------------------------------------------------------------------------
# WebSocket: server -> client JSON (§10.2)
# ---------------------------------------------------------------------------


class HelloMsg(_Base):
    type: Literal["hello"] = "hello"
    contract_version: int = CONTRACT_VERSION
    map: MapMeta
    graph: Graph  # structure only, no hidden fields
    entrance: Entrance
    drones: list[DroneMeta]
    transform: Transform | None = None


class StateMsg(_Base):
    type: Literal["state"] = "state"
    t: float
    speed: float
    act: Literal[1, 2] = 1
    phase: str = ""
    drones: list[DroneState]
    ff: FFState | None = None
    route: Route = Route()
    old_routes: list[OldRoute] = []
    checkpoints: list[Checkpoint] = []
    survivor: SurvivorState | None = None
    click_handles: list[Handle] = []


class MapDeltaMsg(_Base):
    type: Literal["map_delta"] = "map_delta"
    cells: list[tuple[int, int, CellState]] = []  # [i, j, state] at 0.2 m
    searched: list[tuple[int, int]] = []  # newly searched cells (flag, independent of state)
    map_full: str | None = None  # base64 PNG every 5 s for resync


class TraceMsg(TraceRow):
    type: Literal["trace"] = "trace"


class DetectionMsg(_Base):
    type: Literal["detection"] = "detection"
    drone: str
    frame_id: str
    boxes: list[Box] = []
    gemma: GemmaView | None = None
    status: DetectionStatus


class EventMsg(_Base):
    type: Literal["event"] = "event"
    kind: EventKind
    drone: str | None = None
    xy: XYPoint | None = None


class GuidanceMsg(_Base):
    """Also the return type of Guidance.tick (§4)."""

    type: Literal["guidance"] = "guidance"
    arrow_deg: float = 0.0
    distance: float = 0.0
    subtitle: str = ""
    say: str | None = None
    alarm: bool = False
    severity: Severity = "info"


class CalloutMsg(_Base):
    """Act 2: one drone talks to the firefighter ("Careful, fire on your right."), spoken in that drone's voice.

    Guidance decides what is said (facts from the swarm's beliefs only); the optional voice agent rewords it
    (Gemini) and voices it (Gemini TTS). Danger lines never come here: GuidanceMsg.say stays instant.
    """

    type: Literal["callout"] = "callout"
    speaker: str  # "D1".."D4"
    kind: str  # intro | handover | fire_side | survivor_close | progress | lead_wait | lead_return
    text: str  # what is said; also the subtitle
    severity: Severity = "info"
    source: Literal["gemini", "template"] = "template"  # who worded it
    audio_b64: str | None = None  # base64 audio (Gemini TTS); None -> the browser speaks `text` itself
    audio_mime: str | None = None  # e.g. "audio/wav"


class BudgetMsg(_Base):
    type: Literal["budget"] = "budget"
    network: Literal["offline", "online", "unknown"] = "unknown"
    model_tag: str | None = None
    gemma_p50_s: float | None = None
    gemma_p95_s: float | None = None
    gemma_queue: int = 0
    vram_used_mb: float | None = None  # None = N/A on this machine
    vram_total_mb: float | None = None
    sim_rtf: float | None = None
    speed: float = 1.0
    control_hz: float | None = None
    camera_fps: float | None = None


class SummaryMsg(_Base):
    type: Literal["summary"] = "summary"
    time_s: float
    distance_m: float
    hazards_avoided: int
    corrections: int
    replans: int
    trace_url: str | None = None


ServerMessage = Annotated[
    Union[
        HelloMsg,
        StateMsg,
        MapDeltaMsg,
        TraceMsg,
        DetectionMsg,
        EventMsg,
        GuidanceMsg,
        BudgetMsg,
        SummaryMsg,
        CalloutMsg,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# WebSocket: client -> server JSON (§10.2). `camera{mode}` is client-side only.
# ---------------------------------------------------------------------------


class ClickMsg(_Base):
    type: Literal["click"] = "click"
    mode: Literal["debris", "fire"]
    x: float
    y: float


class UndoMsg(_Base):
    type: Literal["undo"] = "undo"


class ControlMsg(_Base):
    type: Literal["control"] = "control"
    action: Literal["start", "pause", "reset", "speed"]
    value: float | None = None


class Act2StartMsg(_Base):
    type: Literal["act2_start"] = "act2_start"


class SnapshotMsg(_Base):
    type: Literal["snapshot"] = "snapshot"
    op: Literal["save", "load"]
    name: str


class FFInputMsg(_Base):
    type: Literal["ff_input"] = "ff_input"
    fwd: float  # -1..1
    strafe: float  # -1..1, + = right
    yaw: float  # absolute heading, radians


ClientMessage = Annotated[
    Union[ClickMsg, UndoMsg, ControlMsg, Act2StartMsg, SnapshotMsg, FFInputMsg],
    Field(discriminator="type"),
]

server_message_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)
client_message_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


def parse_server_message(raw: str | bytes) -> BaseModel:
    return server_message_adapter.validate_json(raw)


def parse_client_message(raw: str | bytes) -> BaseModel:
    return client_message_adapter.validate_json(raw)


# ---------------------------------------------------------------------------
# WebSocket: server -> client binary frames: [u8 channel][u32 frame_seq][JPEG]
# Big-endian (network order; JS DataView default). channel 1-4 = drone cams D1-D4.
# ---------------------------------------------------------------------------

FRAME_HEADER = struct.Struct(">BI")


def pack_frame(channel: int, seq: int, jpeg: bytes) -> bytes:
    return FRAME_HEADER.pack(channel, seq) + jpeg


def unpack_frame(buf: bytes) -> tuple[int, int, bytes]:
    channel, seq = FRAME_HEADER.unpack_from(buf)
    return channel, seq, buf[FRAME_HEADER.size :]


# ---------------------------------------------------------------------------
# Swarm output (§4) and the swarm's prior map
# ---------------------------------------------------------------------------


class SwarmDrone(_Base):
    id: str
    status: str
    task: str = ""
    alive: bool = True


class SwarmState(_Base):
    t: float
    phase: str = ""
    drones: list[SwarmDrone] = []
    route: Route = Route()
    old_routes: list[OldRoute] = []
    checkpoints: list[Checkpoint] = []
    survivor: SurvivorState | None = None
    chain_formed: bool = False
    lead: str | None = None
    map_delta: list[tuple[int, int, CellState]] = []
    searched_delta: list[tuple[int, int]] = []
    events: list[EventMsg] = []
    # Detection ids to re-look at this tick. The swarm only asks once a drone is ~2 m in front of the
    # target and facing it (§8.4); the loop forwards each one to Perception.request_relook().
    relook_requests: list[str] = []


class PriorMap(_Base):
    """Pre-incident plan: grid.png + route graph with ALL edges open.

    Deliberately has no compromised edges, fires, survivor or decoy.
    """

    name: str
    resolution: float
    origin: XYPoint
    size: tuple[int, int]  # (w, h) px
    entrance: Entrance
    graph: Graph
    drone_ids: list[str]
    altitude: float
    _free: np.ndarray | None = PrivateAttr(default=None)

    @property
    def free(self) -> np.ndarray:
        """bool array [iy, ix]; row 0 is the bottom of the map (y ~ 0)."""
        if self._free is None:
            raise ValueError("PriorMap has no grid loaded; build it with PriorMap.from_map()")
        return self._free

    @staticmethod
    def load_free_grid(png_path: str | Path) -> np.ndarray:
        """Read grid.png (0 = wall, 255 = free) -> bool [iy, ix] with rows flipped so iy=0 is y=0."""
        from PIL import Image

        img = np.asarray(Image.open(png_path).convert("L"))
        return img[::-1] > 127

    @classmethod
    def from_map(cls, map_dir: str | Path, scenario: Scenario) -> PriorMap:
        free = cls.load_free_grid(Path(map_dir) / scenario.grid.file)
        pm = cls(
            name=scenario.name,
            resolution=scenario.grid.resolution,
            origin=scenario.grid.origin,
            size=(free.shape[1], free.shape[0]),
            entrance=scenario.entrance,
            graph=scenario.graph,
            drone_ids=list(scenario.drones.ids),
            altitude=scenario.drones.altitude,
        )
        pm._free = free
        return pm
