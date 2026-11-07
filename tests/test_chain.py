"""Rescue route + chain (§8.3): plan on SURVIVOR_CONFIRMED, chain roles, REPLAN when the route is
blocked (old route kept 6 s), NO_ROUTE + hold + re-check, chain rebuilt when a drone is lost, fire watch.

Drives the Swarm facade with scripted Readings + Detections on maps/test (no World).
"""

from __future__ import annotations

import math

import pytest

from sgr.contracts import XY, Detection, GemmaResult, Pose, PriorMap, Readings, Scenario
from sgr.swarm import Swarm

SURVIVOR = (26.5, 11.0)
POSES = {"D1": Pose(x=24.0, y=11.0, z=1.6), "D2": Pose(x=2.5, y=3.0, z=1.6),
         "D3": Pose(x=20.0, y=8.0, z=1.6), "D4": Pose(x=8.0, y=12.0, z=1.6)}


@pytest.fixture
def swarm(test_map) -> Swarm:
    sc = Scenario.load(test_map)
    return Swarm(PriorMap.from_map(test_map, sc), list(sc.drones.ids))


def _r(t: float, poses=POSES) -> dict[str, Readings]:
    return {d: Readings(t=t, pose=p, lidar=[6.0] * 36, temperature=20.0) for d, p in poses.items()}


def _found(label: str = "person", xy=SURVIVOR, det_id: str = "det0009") -> Detection:
    return Detection(id=det_id, drone="D1", t=0.0, label=label, world_xy=XY(x=xy[0], y=xy[1]), score_fast=0.85,
                     gemma=GemmaResult(fire=label == "fire", person=label == "person", confidence=0.93, caption="x"),
                     combined=0.9, status="confirmed", frame_id="D1-3")


def _block(swarm: Swarm, x: float, y0: float, y1: float) -> list[tuple[int, int]]:
    """Wall of blocked_new cells across a corridor at x (as LiDAR change detection would)."""
    b = swarm.belief
    i = b.world_to_cell(x, 0)[0]
    cells = [(i, j) for j in range(b.world_to_cell(0, y0)[1], b.world_to_cell(0, y1)[1] + 1)]
    for c in cells:
        b.set_state(*c, "blocked_new")
    return cells


def test_survivor_confirmed_plans_route_and_forms_chain(swarm: Swarm) -> None:
    _, rows, st = swarm.tick(0.0, _r(0.0), [_found()])
    acts = [r.action for r in rows]
    assert "SURVIVOR_CONFIRMED" in acts and acts.count("CHAIN_POSITION") == 3
    assert st.route.status == "ok" and len(st.route.points) >= 2
    assert st.chain_formed and st.phase == "chain"
    assert any(e.kind == "chain_formed" for e in st.events)
    roles = swarm.rescue.roles
    assert roles["D1"] == "survivor"  # nearest to the survivor
    assert roles["D2"] == "lead"  # nearest to the entrance
    assert list(roles.values()).count("sentinel") == 1 and len(roles) == 3  # 4th drone keeps exploring
    assert st.lead == "D2" and len(st.checkpoints) == 3
    assert "REPLAN" not in acts  # the first plan is not a re-plan


def test_blocked_route_replans_and_keeps_old_route(swarm: Swarm) -> None:
    _, _, st = swarm.tick(0.0, _r(0.0), [_found()])
    v0, old = st.route.version, st.route.points
    assert any(abs(y - 3.0) < 0.5 and 12 < x < 16 for x, y in swarm.rescue.path)  # first route: along e2
    cells = _block(swarm, 14.0, 2.0, 4.0)
    swarm.rescue.on_new_cells("lidar", "D3", cells)
    _, rows, st = swarm.tick(0.1, _r(0.1), [])
    assert "REPLAN" in [r.action for r in rows]
    assert st.route.status == "ok" and st.route.version == v0 + 1
    assert not any(abs(y - 3.0) < 0.5 and 13 < x < 15 for x, y in swarm.rescue.path)  # around via n4-n6-n7
    assert st.old_routes and st.old_routes[0].points == old and st.old_routes[0].until == pytest.approx(6.1)
    _, _, st = swarm.tick(6.2, _r(6.2), [])
    assert st.old_routes == []  # drawn red for 6 s only


def test_no_route_holds_then_recovers(swarm: Swarm) -> None:
    swarm.tick(0.0, _r(0.0), [_found()])
    cells = _block(swarm, 22.0, 9.0, 11.0)  # door corridor of the survivor room (e9)
    swarm.rescue.on_new_cells("lidar", "D3", cells)
    _, rows, st = swarm.tick(0.1, _r(0.1), [])
    assert "NO_ROUTE" in [r.action for r in rows]
    assert st.route.status == "none" and st.phase == "no_route"
    assert any(e.kind == "no_route" for e in st.events)
    for i, j in cells:  # Undo: the debris is gone
        swarm.belief.set_state(i, j, "free")
    _, rows, st = swarm.tick(1.0, _r(1.0), [])
    assert st.route.status == "none"  # re-checks every 2 s, not every tick
    _, rows, st = swarm.tick(2.2, _r(2.2), [])
    assert st.route.status == "ok" and any("found again" in r.reason for r in rows if r.action == "REPLAN")


def test_lost_chain_drone_is_replaced(swarm: Swarm) -> None:
    swarm.tick(0.0, _r(0.0), [_found()])
    sentinel = next(d for d, r in swarm.rescue.roles.items() if r == "sentinel")
    alive = {d: p for d, p in POSES.items() if d != sentinel}
    _, rows, st = swarm.tick(3.5, _r(3.5, alive), [])  # sentinel silent > 3 s
    assert any(r.action == "DRONE_LOST" and r.drone == sentinel for r in rows)
    assert sentinel not in swarm.rescue.roles and len(swarm.rescue.roles) == 3  # the explorer joined
    assert sentinel not in {c.drone for c in st.checkpoints}


def test_confirmed_fire_gets_a_watcher_away_from_corridors(swarm: Swarm) -> None:
    fire = (1.6, 12.0)  # maps/test: far corner of the left dead-end room
    _, rows, _ = swarm.tick(0.0, _r(0.0), [_found("fire", fire, "det0002")])
    assert swarm.rescue.watcher is not None
    post = swarm.agents[swarm.rescue.watcher]._station
    assert post is not None
    d = ((post[0] - fire[0]) ** 2 + (post[1] - fire[1]) ** 2) ** 0.5
    assert 2.9 <= d <= 3.8  # warm but under the 70 °C retreat limit
    assert any(r.action == "INSPECT" and "keeps its camera on the fire" in r.reason for r in rows)


# ---- Act 2 (§9 "Route becomes unsafe"): the firefighter's tracked position drives the re-route ----


def _act2(swarm: Swarm, ff: Pose):
    """Survivor confirmed (route along e2, y = 3), then Act 2 with the firefighter at `ff`."""
    swarm.tick(0.0, _r(0.0), [_found()])
    return swarm.tick(0.1, _r(0.1), [], ff=ff)


def test_act2_blockage_behind_the_firefighter_is_ignored(swarm: Swarm) -> None:
    _, _, st = _act2(swarm, Pose(x=16.0, y=3.0))
    v = st.route.version
    cells = _block(swarm, 12.0, 2.0, 4.0)  # e2, behind the firefighter at x = 16
    swarm.rescue.on_new_cells("lidar", "D3", cells)
    _, rows, st = swarm.tick(0.2, _r(0.2), [], ff=Pose(x=16.0, y=3.0))
    assert "REPLAN" not in [r.action for r in rows] and st.route.version == v


def test_act2_blockage_ahead_replans_from_the_firefighter(swarm: Swarm) -> None:
    ff = Pose(x=10.0, y=3.0)
    _act2(swarm, ff)
    cells = _block(swarm, 14.0, 2.0, 4.0)  # e2, ahead of the firefighter at x = 10
    swarm.rescue.on_new_cells("lidar", "D3", cells)
    _, rows, st = swarm.tick(0.2, _r(0.2), [], ff=ff)
    assert "REPLAN" in [r.action for r in rows]
    assert st.route.status == "ok"
    x0, y0 = st.route.points[0]
    assert abs(x0 - ff.x) < 0.5 and abs(y0 - ff.y) < 0.5  # the new route starts at the firefighter
    assert not any(abs(y - 3.0) < 0.5 and 13 < x < 15 for x, y in swarm.rescue.path)  # around via n4-n6-n7
    lead = next(d for d, r in swarm.rescue.roles.items() if r == "lead")
    near_ff = min(swarm.rescue.roles, key=lambda d: (POSES[d].x - ff.x) ** 2 + (POSES[d].y - ff.y) ** 2)
    assert lead == near_ff or swarm.rescue.roles[near_ff] == "survivor"
    assert any(r.action == "CHAIN_POSITION" and "guiding the firefighter" in r.reason for r in rows)


# ---- Act 1 patrol: once the chain is ready, the lead + sentinel sweep the route (judge clicks get seen) ----


def test_patrol_starts_when_chain_ready_and_stops_in_act2(swarm: Swarm) -> None:
    swarm.tick(0.0, _r(0.0), [_found()])
    roles = swarm.rescue.roles
    walkers = [d for d, r in roles.items() if r != "survivor"]
    assert not any(swarm.agents[d].patrolling for d in walkers)  # still flying to their posts
    _, _, st = swarm.tick(41.0, _r(41.0), [])  # 40 s after forming: ready (latched)
    assert st.phase == "chain_ready"
    assert all(swarm.agents[d].patrolling for d in walkers)
    survivor = next(d for d, r in roles.items() if r == "survivor")
    assert not swarm.agents[survivor].patrolling  # stays over the survivor
    lead = next(d for d, r in roles.items() if r == "lead")
    lead_pts, sent_pts = swarm.agents[lead]._patrol, swarm.agents[next(d for d in walkers if d != lead)]._patrol
    ent = swarm.rescue.entrance
    assert min(math.dist(p, ent) for p in lead_pts) < min(math.dist(p, ent) for p in sent_pts)  # lead: first half
    _, _, st = swarm.tick(41.1, _r(41.1), [], ff=Pose(x=2.0, y=3.0))  # Act 2: back to the posts
    assert not any(swarm.agents[d].patrolling for d in walkers)
    assert st.phase == "chain_ready"
