"""Exploration: drones spread out, recognise dead ends, share them, and all form the path to the survivor.

maps/test: the closet n11 (x 0.8-3.8, y 4.4-6.4) is reached from the entrance by one short corridor
(x 1.2-2.8 at y 4.0-4.6). Debris across that corridor makes the closet a dead end.
"""

from __future__ import annotations

import pytest

from sgr.contracts import Pose, PriorMap, Readings, Scenario
from sgr.swarm import Swarm


@pytest.fixture
def swarm(test_map) -> Swarm:
    sc = Scenario.load(test_map)
    return Swarm(PriorMap.from_map(test_map, sc), list(sc.drones.ids))


def _block_closet(sw: Swarm) -> None:
    b = sw.belief
    j = b.world_to_cell(0, 4.1)[1]
    for i in range(b.world_to_cell(1.2, 0)[0], b.world_to_cell(2.8, 0)[0] + 1):
        b.set_state(i, j, "blocked_new")


def _r(t, poses):
    return {d: Readings(t=t, pose=p, lidar=[6.0] * 36, temperature=20.0) for d, p in poses.items()}


def test_debris_makes_the_closet_a_dead_end_for_everyone(swarm: Swarm) -> None:
    closet = swarm.areas["n11"]
    assert closet.unsearched(swarm.belief).sum() > 50
    _block_closet(swarm)
    assert closet.unsearched(swarm.belief).sum() == 0  # cut off: nothing left anyone can search there
    assert closet.cut_off(swarm.belief) > 50


def test_a_drone_in_a_dead_end_says_so_and_turns_back(swarm: Swarm) -> None:
    poses = {"D1": Pose(x=2.0, y=3.0, z=1.6, yaw=1.57)}
    swarm.tick(0.0, _r(0.0, poses), [])
    swarm.agents["D1"].set_area("n11")
    swarm.bb.claims["n11"] = "D1"
    _block_closet(swarm)
    _, rows, _ = swarm.tick(0.1, _r(0.1, poses), [])
    row = next(r for r in rows if r.action == "EXPLORE")
    assert "area n11 is a dead end: blocked by debris" in row.reason and "turning back" in row.reason
    assert swarm.agents["D1"].area != "n11"  # moved on to another region


def test_dead_ends_are_shared_and_kept_until_the_map_opens(swarm: Swarm) -> None:
    b = swarm.belief
    closet = swarm.areas["n11"]
    ag = swarm.agents["D1"]
    ag.goal = b.world_to_cell(2.2, 5.4)
    ag._give_up_goal(b, closet)  # D1 swept here and still couldn't see these cells
    assert b.dead.any()
    left = closet.unsearched(b).sum()
    # another drone's view of the closet: the dead-end cells are gone for it too
    assert (closet.mask & b.dead).sum() > 0 and left < closet.n_cells
    b.set_state(*b.world_to_cell(14.0, 3.0), "blocked_new")  # the map CLOSES somewhere else: still remembered
    swarm.tick(0.0, _r(0.0, {"D2": Pose(x=8, y=3, z=1.6)}), [])
    assert b.dead.any()
    b.set_state(*b.world_to_cell(14.0, 3.0), "free")  # debris removed (Undo): the map OPENS, re-try dead ends
    swarm.tick(0.1, _r(0.1, {"D2": Pose(x=8, y=3, z=1.6)}), [])
    assert not b.dead.any()


def test_no_drone_is_sent_into_a_known_dead_end(swarm: Swarm) -> None:
    _block_closet(swarm)
    poses = {d: Pose(x=2.0 + 0.5 * k, y=3.0, z=1.6) for k, d in enumerate(swarm.drones)}
    for k in range(3):
        swarm.tick(0.1 * k, _r(0.1 * k, poses), [])
    assert all(swarm.agents[d].area != "n11" for d in swarm.drones)


def test_drones_start_in_different_regions(swarm: Swarm) -> None:
    """They fan out to explore, they don't all head the same way (nobody knows where the survivor is)."""
    poses = {d: Pose(x=2.0 + 0.3 * k, y=3.0, z=1.6) for k, d in enumerate(swarm.drones)}
    swarm.tick(0.0, _r(0.0, poses), [])
    areas = [swarm.agents[d].area for d in swarm.drones]
    assert None not in areas and len(set(areas)) == len(areas)
