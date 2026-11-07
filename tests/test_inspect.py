"""Rules when unsure (§8.4): inspect / re-look from ~2 m, confirm / reject, fire cells, heat retreat.

Drives the Swarm facade directly with scripted Readings + Detections (no World), so each rule is
checked in isolation.
"""

from __future__ import annotations

import math

import pytest

from sgr import config
from sgr.contracts import XY, Detection, GemmaResult, Pose, PriorMap, Readings, Scenario
from sgr.swarm import Swarm
from sgr.swarm.belief import FIRE

DECOY = (8.7, 6.5)  # maps/test: on the n2-n4 corridor wall


@pytest.fixture
def swarm(test_map) -> Swarm:
    sc = Scenario.load(test_map)
    return Swarm(PriorMap.from_map(test_map, sc), list(sc.drones.ids))


def _readings(t: float, poses: dict[str, Pose], temp: float = 20.0) -> dict[str, Readings]:
    return {d: Readings(t=t, pose=p, lidar=[6.0] * 36, temperature=temp) for d, p in poses.items()}


def _det(status: str, relooks: int = 0, gemma: float | None = None, label: str = "fire", xy=DECOY) -> Detection:
    g = None if gemma is None else GemmaResult(fire=gemma > 0.5, person=label == "person", confidence=gemma,
                                               caption="orange glow")
    return Detection(id="det0001", drone="D2", t=0.0, label=label, world_xy=XY(x=xy[0], y=xy[1]), score_fast=0.6,
                     gemma=g, combined=gemma, status=status, frame_id="D2-7", relooks=relooks)


def _actions(rows) -> list[str]:
    return [r.action for r in rows]


def test_pending_holds_low_conf_approaches_then_relooks_from_2m(swarm: Swarm) -> None:
    poses = {"D1": Pose(x=2, y=3, z=1.6), "D2": Pose(x=8.0, y=3.0, z=1.6, yaw=math.pi / 2)}
    _, rows, st = swarm.tick(0.0, _readings(0.0, poses), [_det("pending")])
    assert "INSPECT" in _actions(rows)
    assert {d.id: d.status for d in st.drones}["D2"] == "INSPECTING"
    assert st.relook_requests == []

    _, rows, st = swarm.tick(0.1, _readings(0.1, poses), [_det("low_conf", gemma=0.35)])
    assert "INSPECT" in _actions(rows) and "moving closer" in rows[-1].reason
    assert st.relook_requests == []  # not yet: D2 is not in position
    ins = swarm.hazards.active["det0001"]
    sx, sy = ins.standoff
    assert math.hypot(sx - DECOY[0], sy - DECOY[1]) == pytest.approx(config.INSPECT_STANDOFF_M, abs=0.05)

    # put D2 at the standoff point, facing the target -> the swarm asks for the re-look
    face = math.atan2(DECOY[1] - sy, DECOY[0] - sx)
    poses["D2"] = Pose(x=sx, y=sy, z=1.6, yaw=face)
    asked = []
    for k in range(3):
        _, _, st = swarm.tick(0.2 + 0.1 * k, _readings(0.2 + 0.1 * k, poses), [])
        asked += st.relook_requests
    assert asked == ["det0001"]  # exactly once

    _, rows, st = swarm.tick(0.6, _readings(0.6, poses), [_det("rejected", relooks=1, gemma=0.8)])
    assert "REJECT_DETECTION" in _actions(rows)
    assert any(e.kind == "false_alarm" for e in st.events)
    assert "det0001" not in swarm.hazards.active
    assert {d.id: d.status for d in st.drones}["D2"] != "INSPECTING"


def test_still_unsure_after_two_relooks_is_rejected(swarm: Swarm) -> None:
    poses = {"D2": Pose(x=8.0, y=3.0, z=1.6)}
    swarm.tick(0.0, _readings(0.0, poses), [_det("pending")])
    _, rows, _ = swarm.tick(0.1, _readings(0.1, poses), [_det("low_conf", relooks=2, gemma=0.4)])
    assert "REJECT_DETECTION" in _actions(rows) and "after 2 closer looks" in rows[-1].reason


def test_confirmed_fire_becomes_no_go_cells(swarm: Swarm) -> None:
    poses = {"D1": Pose(x=20.0, y=3.0, z=1.6)}
    fire_xy = (24.0, 3.0)
    _, rows, st = swarm.tick(0.0, _readings(0.0, poses), [_det("confirmed", gemma=0.88, xy=fire_xy)])
    assert "CONFIRM_HAZARD" in _actions(rows)
    i, j = swarm.belief.world_to_cell(*fire_xy)
    assert swarm.belief.state[j, i] == FIRE
    assert (i, j, "fire") in st.map_delta
    assert any(e.kind == "fire_confirmed" for e in st.events)


def test_confirmed_person_sets_the_survivor(swarm: Swarm) -> None:
    poses = {"D1": Pose(x=24.0, y=10.0, z=1.6)}
    swarm.tick(0.0, _readings(0.0, poses), [_det("pending", label="person", xy=(26.5, 11.0))])
    assert swarm.bb.survivor.status == "suspected"
    _, rows, st = swarm.tick(0.1, _readings(0.1, poses),
                             [_det("confirmed", gemma=0.93, label="person", xy=(26.5, 11.0))])
    assert "SURVIVOR_CONFIRMED" in _actions(rows)
    assert st.survivor is not None and st.survivor.status == "confirmed"


def test_hot_drone_retreats_once_and_marks_danger(swarm: Swarm) -> None:
    poses = {"D1": Pose(x=8.0, y=10.0, z=1.6)}
    _, rows, st = swarm.tick(0.0, _readings(0.0, poses, temp=80.0), [])
    assert _actions(rows).count("RETREAT") == 1
    assert {d.id: d.status for d in st.drones}["D1"] == "RETREAT"
    assert any(s == "danger" for _, _, s in st.map_delta)
    _, rows, _ = swarm.tick(0.1, _readings(0.1, poses, temp=85.0), [])
    assert "RETREAT" not in _actions(rows)  # still retreating: no repeat row
