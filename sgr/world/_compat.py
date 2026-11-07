"""The ONE place sgr.world gets its contract types from.

S owns sgr/contracts.py (pydantic, spec §4/§5). Until it lands, minimal
stand-ins with the same field names live here. When contracts.py exists,
the import below wins. Then delete the fallback and fix any field-name
mismatches in this file only.

Every World method builds these types with keyword arguments only, so
pydantic models and these dataclasses are interchangeable.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

try:  # pragma: no cover - depends on S's skeleton
    from sgr.contracts import CamInfo, GroundTruth, Handle, Pose, Scenario, XY  # type: ignore

    HAVE_CONTRACTS = True
except ImportError:  # ------------------------------------------- fallback
    from dataclasses import dataclass, field

    HAVE_CONTRACTS = False

    @dataclass
    class Pose:
        x: float
        y: float
        z: float
        yaw: float

    @dataclass
    class XY:
        x: float
        y: float

    @dataclass
    class CamInfo:
        drone: str
        width: int
        height: int
        fx: float
        fy: float
        cx: float
        cy: float
        fovy_deg: float
        pos: tuple[float, float, float]  # camera centre, world frame
        rot: tuple[float, ...]  # 3x3 row-major, columns = camera x(right), y(up), z(back) in world

    @dataclass(frozen=True)
    class Handle:
        kind: str  # "debris" | "fire"
        idx: int  # pool slot

    @dataclass
    class GroundTruth:
        fires: list[XY] = field(default_factory=list)
        debris: list[XY] = field(default_factory=list)
        survivor: XY | None = None
        decoy: XY | None = None
        dead_drones: list[str] = field(default_factory=list)

    Scenario = SimpleNamespace  # attribute access like the pydantic model


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def load_scenario(path: str | Path):
    """Load scenario.json into whatever Scenario type is active."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if HAVE_CONTRACTS:  # pragma: no cover
        return Scenario.model_validate(raw)
    return _ns(raw)


__all__ = ["CamInfo", "GroundTruth", "Handle", "Pose", "Scenario", "XY", "HAVE_CONTRACTS", "load_scenario"]
