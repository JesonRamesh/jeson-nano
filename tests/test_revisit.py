"""Nothing reachable left to search (W's Act 1 livelock): no area hand-out loop, and a "person" Gemma ruled out
gets another look once its verdict is 60 s old. Also: a blockage re-check really clears an undone click."""

from __future__ import annotations

import collections

import numpy as np
import pytest

import sgr.perception.mock as mock
import sgr.swarm.swarm as swarm_mod
from sgr.contracts import XY, Detection, GemmaResult, Pose, PriorMap, Readings, Scenario
from sgr.perception import Perception
from sgr.loop import SimLoop
from sgr.swarm import Swarm
from sgr.swarm.hazards import REVISIT_AFTER_S
from sgr.world import World


@pytest.fixture
def swarm(test_map) -> Swarm:
    sc = Scenario.load(test_map)
    return Swarm(PriorMap.from_map(test_map, sc), list(sc.drones.ids))


def _readings(t: float, poses: dict[str, Pose]) -> dict[str, Readings]:
    return {d: Readings(t=t, pose=p, lidar=[6.0] * 36, temperature=20.0) for d, p in poses.items()}


def _person(status: str, gemma: float, det_id: str = "det0001", relooks: int = 0) -> Detection:
    g = GemmaResult(fire=False, person=status == "confirmed", confidence=gemma, caption="debris")
    return Detection(id=det_id, drone="D1", t=0.0, label="person", world_xy=XY(x=26.5, y=11.0), score_fast=0.85,
                     gemma=g, combined=gemma, status=status, frame_id="D1-3", relooks=relooks)


def test_an_area_with_no_drone_sized_way_in_is_not_handed_out(swarm: Swarm, monkeypatch) -> None:
    """Its cells can be connected through a gap too narrow for a drone: offering it made the drone give it up at
    once and get it offered again, every tick."""
    monkeypatch.setattr(swarm_mod, "goal_mask", lambda belief, area: np.zeros_like(area.mask))
    poses = {"D1": Pose(x=2.0, y=3.0, z=1.2), "D2": Pose(x=2.5, y=3.0, z=1.2)}
    rows = []
    for k in range(30):
        _, r, _ = swarm.tick(k * 0.1, _readings(k * 0.1, poses), [])
        rows += r
    assert not swarm.bb.claims
    assert not [r for r in rows if r.action == "ASSIGN_AREA"]


def test_a_ruled_out_person_blocks_new_sightings_only_for_a_while(swarm: Swarm) -> None:
    h = swarm.hazards
    poses = {"D1": Pose(x=24.0, y=11.0, z=1.2)}
    h.on_detection(10.0, _person("pending", 0.0), swarm.agents, poses)
    h.on_detection(12.0, _person("rejected", 0.26), swarm.agents, poses)
    assert h.revisit_due(12.0 + REVISIT_AFTER_S - 1) is None  # still trusted
    again = _person("pending", 0.0, det_id="det0002")
    h.on_detection(30.0, again, swarm.agents, poses)
    assert "det0002" in h.done and "det0002" not in h.active  # ignored: same spot, verdict still fresh
    due = h.revisit_due(12.0 + REVISIT_AFTER_S + 1)
    assert due is not None and abs(due[0] - 26.5) < 0.01
    later = _person("pending", 0.0, det_id="det0003")
    h.on_detection(12.0 + REVISIT_AFTER_S + 2, later, swarm.agents, poses)
    assert "det0003" in h.active  # an old verdict no longer blocks a fresh look


def _run_until_chain(test_map, max_t: float, n_wrong: int, monkeypatch):
    """Full Act 1 on maps/test, mock Gemma saying "no person" to the survivor's first `n_wrong` checks."""
    calls = collections.Counter()
    orig = mock._gemma

    def gemma(kind, relooks, latency=mock.LATENCY_S):
        if kind == "survivor":
            calls[kind] += 1
            if calls[kind] <= n_wrong:
                return GemmaResult(fire=False, person=False, confidence=0.8, caption="A pile of debris.",
                                   latency_s=latency, model_tag="mock")
        return orig(kind, relooks, latency)

    monkeypatch.setattr(mock, "_gemma", gemma)
    sc = Scenario.load(test_map)
    w = World(test_map, sc)
    loop = SimLoop(w, Perception(mode="mock", ground_truth=w.ground_truth, line_of_sight=w.line_of_sight,
                                 pixel_to_world=w.pixel_to_world, mock_latency_s=1.0),
                   Swarm(PriorMap.from_map(test_map, sc), list(sc.drones.ids)), sc, render=False, realtime=False)
    while loop.t < max_t and not loop.swarm.bb.chain_formed:
        loop.tick()
    return loop


def test_wrongly_rejected_survivor_is_found_on_a_revisit(test_map, monkeypatch) -> None:
    loop = _run_until_chain(test_map, 260.0, n_wrong=3, monkeypatch=monkeypatch)
    acts = [r.action for r in loop.trace]
    assert "REJECT_DETECTION" in acts
    assert loop.swarm.bb.survivor.status == "confirmed" and loop.swarm.bb.chain_formed
    rejected_t = next(r.t for r in loop.trace if r.action == "REJECT_DETECTION" and "person" in r.reason)
    confirmed_t = next(r.t for r in loop.trace if r.action == "SURVIVOR_CONFIRMED")
    assert confirmed_t - rejected_t >= REVISIT_AFTER_S - 1  # found again by the revisit, not by chance
    per_s = collections.Counter(int(r.t) for r in loop.trace if r.action in ("ASSIGN_AREA", "EXPLORE"))
    assert max(per_s.values()) <= 6  # no hand-out / give-up loop while nothing is left
