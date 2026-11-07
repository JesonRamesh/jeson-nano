"""Honesty rules (§1.2 rule 4, §12): agents learn only through sensors.

Show this file to judges who ask "is it scripted?".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from sgr.contracts import Evidence, TraceRow

ROOT = Path(__file__).resolve().parents[1]
NO_GROUND_TRUTH_DIRS = [ROOT / "sgr" / "swarm", ROOT / "sgr" / "guidance"]


def _py_files(dirs: list[Path]) -> list[Path]:
    return [p for d in dirs for p in d.rglob("*.py")]


def test_dirs_exist() -> None:
    for d in NO_GROUND_TRUTH_DIRS:
        assert d.is_dir(), d


def test_no_ground_truth_in_swarm_or_guidance() -> None:
    offenders = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in _py_files(NO_GROUND_TRUTH_DIRS)
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if "ground_truth" in line or "GroundTruth" in line
    ]
    assert not offenders, f"ground truth used in: {offenders}"


def test_swarm_does_not_import_world_or_perception_internals() -> None:
    pat = re.compile(r"^\s*(from|import)\s+sgr\.(world|perception)\b", re.M)
    offenders = [str(p.relative_to(ROOT)) for p in _py_files([ROOT / "sgr" / "swarm"]) if pat.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"sgr/swarm must only use sgr.contracts types: {offenders}"


def test_trace_row_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        TraceRow(id="1", t=0.0, drone="D1", action="REPLAN", reason="no evidence")  # type: ignore[call-arg]


def test_evidence_requires_a_value() -> None:
    with pytest.raises(ValidationError):
        Evidence(sensor="lidar", value=None)


def test_swarm_stub_rows_have_evidence(test_map) -> None:
    from sgr.contracts import PriorMap, Readings, Pose, Scenario
    from sgr.swarm import Swarm

    sc = Scenario.load(test_map)
    swarm = Swarm(PriorMap.from_map(test_map, sc), sc.drones.ids)
    readings = {d: Readings(t=0, pose=Pose(x=2, y=3), lidar=[6.0] * 36, temperature=20) for d in sc.drones.ids}
    _, rows, _ = swarm.tick(0.0, readings, [])
    assert all(isinstance(r.evidence, Evidence) for r in rows)
