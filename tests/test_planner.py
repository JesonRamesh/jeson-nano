"""Planner + swarm-core unit tests (§12): A* on small grids, blocked -> re-route, disconnected -> none,
fire margin, checkpoint turn extraction; plus trace evidence and blackboard upsert-by-id."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sgr.contracts import XY, Detection, Evidence, Pose
from sgr.swarm.belief import BeliefGrid
from sgr.swarm.blackboard import Blackboard
from sgr.swarm.coverage import camera_footprint
from sgr.swarm.planner import astar, path_hits, path_length, rdp, turn_points
from sgr.swarm.trace import Trace, TraceError

RES = 0.2


def _grid(w: int = 40, h: int = 20) -> np.ndarray:
    """Open room with a 1-cell wall border (bool wall [j, i])."""
    wall = np.zeros((h, w), bool)
    wall[0, :] = wall[-1, :] = wall[:, 0] = wall[:, -1] = True
    return wall


def _belief(wall: np.ndarray, entrance=(1.0, 2.0)) -> BeliefGrid:
    return BeliefGrid(wall, RES, (0.0, 0.0), entrance)


def test_astar_open_room_is_straight() -> None:
    b = _belief(_grid())
    p = astar(b, (1.5, 2.0), (6.5, 2.0))
    assert p is not None
    assert math.dist(p[0], (1.5, 2.0)) < 0.2 and math.dist(p[-1], (6.5, 2.0)) < 0.2
    assert path_length(p) == pytest.approx(5.0, abs=0.25)


def test_astar_goes_through_the_gap() -> None:
    wall = _grid()
    wall[:, 20] = True  # wall at x = 4.0..4.2
    wall[12:16, 20] = False  # gap at y = 2.4..3.2 (0.8 m: fits the 0.3 m inflation)
    b = _belief(wall)
    p = astar(b, (1.5, 1.0), (6.5, 1.0))
    assert p is not None
    crossing = [y for x, y in p if 4.0 <= x <= 4.2]
    assert crossing and all(2.4 <= y <= 3.2 for y in crossing)


def test_blocked_cell_forces_a_reroute() -> None:
    wall = _grid()
    b = _belief(wall)
    p1 = astar(b, (1.5, 2.0), (6.5, 2.0))
    assert p1 is not None
    # LiDAR finds debris across the straight line: a 1.6 m tall bar at x = 4.0
    for j in range(5, 13):
        b.set_state(20, j, "blocked_new")
    assert path_hits(p1, [(20, 10)], b)
    p2 = astar(b, (1.5, 2.0), (6.5, 2.0))
    assert p2 is not None
    assert not path_hits(p2, [(20, j) for j in range(5, 13)], b, margin=0.2)
    assert path_length(p2) > path_length(p1)
    assert b.pop_map_delta()[0][2] == "blocked_new"


def test_disconnected_returns_none() -> None:
    wall = _grid()
    wall[:, 20] = True
    b = _belief(wall)
    assert astar(b, (1.5, 2.0), (6.5, 2.0)) is None


def test_fire_margin_is_impassable() -> None:
    wall = _grid(60, 20)
    b = _belief(wall)
    b.set_state(30, 10, "fire")  # fire at (6.1, 2.1), right on the straight line
    p = astar(b, (1.5, 2.0), (10.5, 2.0))
    if p is not None:  # with a 1 m margin the 3.6 m tall room may or may not leave a gap
        assert min(math.dist(q, (6.1, 2.1)) for q in p) > 1.0
    wall2 = _grid(60, 12)  # 2.0 m tall room: the 1 m margin closes it completely
    b2 = _belief(wall2)
    b2.set_state(30, 6, "fire")
    assert astar(b2, (1.5, 1.2), (10.5, 1.2)) is None


def test_turn_points_and_rdp() -> None:
    l_path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (3.0, 2.0)]
    simp = rdp(l_path, 0.1)
    assert simp == [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0)]
    turns = turn_points(simp, 35.0)
    assert len(turns) == 1
    assert turns[0][0] == (3.0, 0.0) and turns[0][1] == pytest.approx(90.0)
    assert turn_points([(0, 0), (1, 0), (2, 0.2)], 35.0) == []  # gentle bend is not a turn


def test_camera_footprint_respects_walls_and_fov() -> None:
    wall = _grid()
    wall[:, 20] = True  # wall at x = 4.0
    b = _belief(wall)
    m = camera_footprint(b, Pose(x=2.0, y=2.0, yaw=0.0))
    assert m[10, 15]  # 1 m ahead
    assert not m[10, 25]  # behind the wall
    assert not m[10, 3]  # behind the camera


def test_lidar_change_detection_needs_two_scans() -> None:
    b = _belief(_grid())
    pose = Pose(x=1.5, y=2.0, yaw=0.0)
    scan = list(b.expected_ranges(pose.x, pose.y, np.radians(np.arange(36) * 10.0)))
    scan[0] = 1.0  # something 1 m ahead the plan doesn't know about
    blocked, _ = b.update_lidar("D1", pose, scan)
    assert blocked == []
    blocked, _ = b.update_lidar("D1", pose, scan)
    assert blocked == [b.world_to_cell(2.55, 2.0)]


def test_trace_raises_without_evidence() -> None:
    tr = Trace()
    with pytest.raises(TraceError):
        tr.log(0.0, "D1", "REPLAN", "no evidence", None)
    with pytest.raises(TraceError):
        tr.log(0.0, "D1", "REPLAN", "empty evidence", {"sensor": "planner", "value": None})
    row = tr.log(1.0, "D1", "MARK_BLOCKED", "debris", Evidence(sensor="lidar", value=1.2))
    assert row.evidence.sensor == "lidar"
    assert tr.log_change("area", "n2", 2.0, "D1", "ASSIGN_AREA", "a", Evidence(sensor="planner", value="n2"))
    assert tr.log_change("area", "n2", 3.0, "D1", "ASSIGN_AREA", "a", Evidence(sensor="planner", value="n2")) is None
    assert len(tr.drain()) == 2


def test_blackboard_upserts_detections_by_id() -> None:
    bb = Blackboard()
    d = Detection(id="det0001", drone="D2", t=1.0, label="fire", world_xy=XY(x=1, y=2), score_fast=0.6,
                  status="pending", frame_id="D2-1")
    bb.upsert_detection(d)
    prev, cur = bb.upsert_detection(d.model_copy(update={"status": "low_conf"}))
    assert prev is not None and prev.status == "pending"
    assert len(bb.detections) == 1 and bb.detections["det0001"].status == "low_conf"
