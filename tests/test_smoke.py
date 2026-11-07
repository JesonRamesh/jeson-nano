"""Smoke test = the merge gate (§12). Headless, mock perception, test map, max speed.

`test_act1_beats` checks the Act 1 beats end to end on maps/test (decoy rejected, re-plan, flare-up
victim lost, survivor confirmed, chain formed).
"""

from __future__ import annotations

import math


from sgr.contracts import PriorMap, Scenario
from sgr.guidance import Guidance
from sgr.loop import SimLoop
from sgr.perception import Perception
from sgr.swarm import Swarm
from sgr.world import FakeWorld, World

TICKS = 1500


def _loop(test_map, world_cls=World) -> SimLoop:
    sc = Scenario.load(test_map)
    world = world_cls(test_map, sc)
    return SimLoop(
        world,
        Perception(
            mode="mock", ground_truth=world.ground_truth, line_of_sight=world.line_of_sight
        ),
        Swarm(PriorMap.from_map(test_map, sc), list(sc.drones.ids)),
        sc,
        guidance=Guidance(),
        bus=None,
        render=False,
        realtime=False,
    )


def test_loop_runs_headless(test_map) -> None:
    loop = _loop(test_map)
    loop.run(max_ticks=TICKS)
    assert loop.ticks == TICKS
    assert math.isclose(loop.t, TICKS * 0.1, rel_tol=1e-6)
    # flare-up at t=65 s hits the drone nearest a fire (the fire watcher), else scenario.flare.target
    assert loop.flare_victim is not None and loop.flare_victim in loop.dead
    # ...and the flare-up really placed a fire (regression: clearance check refused it)
    h = loop.flare_fire
    assert h is not None, "flare-up fire was not placed"
    assert any(math.hypot(f.x - h.x, f.y - h.y) < 0.1 for f in loop.world.ground_truth().fires)


def test_act1_beats(test_map) -> None:
    loop = _loop(test_map)
    loop.run(max_ticks=TICKS)
    actions = {r.action for r in loop.trace}
    assert "SURVIVOR_CONFIRMED" in actions, "survivor never confirmed"
    assert "REPLAN" in actions, "no REPLAN"
    # decoy beat dropped (team decision): the survivor is checked before it is confirmed
    inspect_survivor = [r.t for r in loop.trace if r.action == "INSPECT" and "person" in r.reason]
    confirmed = [r.t for r in loop.trace if r.action == "SURVIVOR_CONFIRMED"]
    assert inspect_survivor and confirmed and inspect_survivor[0] <= confirmed[0], "survivor not inspected first"
    assert "CONFIRM_HAZARD" in actions, "fire never confirmed"
    victim = loop.flare_victim
    assert any(r.action == "DRONE_LOST" and r.drone == victim for r in loop.trace), f"{victim} never declared lost"
    st = loop.swarm_state
    assert st.chain_formed and st.route.status == "ok", "no chain / no safe route at the end"
    assert victim not in {c.drone for c in st.checkpoints}, "a lost drone still holds a checkpoint"


def test_swarm_covers_test_map(test_map) -> None:
    """H1-H4 done criterion: 4 drones search >= 90 % of maps/test in under 120 sim-seconds."""
    loop = _loop(test_map)
    t90 = None
    for _ in range(1200):  # 120 s
        loop.tick()
        if t90 is None and loop.swarm.coverage() >= 0.90:
            t90 = loop.t
    assert t90 is not None, f"coverage only {loop.swarm.coverage():.1%} after 120 s"
    actions = [r.action for r in loop.trace]
    assert actions.count("ASSIGN_AREA") >= 4
    assert "MARK_BLOCKED" in actions  # e2 debris seen by LiDAR
    assert all(r.evidence.value is not None for r in loop.trace)


# ---- FakeWorld sanity (keeps the stub honest for everyone developing against it) ----


def test_fake_world_lidar_and_motion(test_map) -> None:
    sc = Scenario.load(test_map)
    w = FakeWorld(test_map, sc)
    scan = w.lidar("D1")  # D1 at (2, 2.5) facing east, corridor y in [2.1, 3.9]
    assert scan.shape == (36,)
    assert scan[0] > 4.0  # long way east down the corridor
    assert scan[27] < 1.0  # ray 27 = 270 deg = south wall ~0.4 m away
    w.set_drone_cmd("D1", 1.0, 0.0, 0.0, 0.0)
    w.step(200)  # 1 s
    assert math.isclose(w.drone_pose("D1").x, 3.0, abs_tol=0.06)
    w.set_drone_cmd("D1", 0.0, -1.0, 0.0, 0.0)  # into the south wall
    w.step(400)
    assert w.drone_pose("D1").y > 2.1  # stopped by the wall


def test_fake_world_compromised_edge_blocks(test_map) -> None:
    sc = Scenario.load(test_map)
    w = FakeWorld(test_map, sc)
    w.set_drone_cmd("D2", 1.2, 0.0, 0.0, 0.0)
    w.step(200 * 20)  # 20 s east along e2; debris at x=14
    assert w.drone_pose("D2").x < 14.0


def test_fake_world_change_tool_and_heat(test_map) -> None:
    sc = Scenario.load(test_map)
    w = FakeWorld(test_map, sc)
    assert w.place_debris(2.0, 3.0) is None  # too close to drones
    h = w.place_debris(6.0, 3.0)
    assert h is not None
    w.set_drone_cmd("D1", 1.0, 0.0, 0.0, 0.0)
    w.step(200 * 6)
    assert w.drone_pose("D1").x < 5.6
    w.remove(h)
    w.step(200 * 3)
    assert w.drone_pose("D1").x > 6.0
    f = w.place_fire(10.0, 3.0)
    assert f is not None
    w.set_drone_cmd("D1", 0.0, 0.0, 0.0, 0.0)
    assert w.temperature("D1") > 20.0
    state = w.get_state()
    w.remove(f)
    w.set_state(state)
    assert len(w.ground_truth().fires) == len(sc.fires) + 1


# ---- loop wiring (S): re-look forwarding, Guidance in Act 1, Budget publishing ----


def test_loop_wiring_relook_guidance_budget(test_map) -> None:
    from sgr.bus import Bus
    from sgr.contracts import BudgetMsg, SwarmState, parse_server_message
    from sgr.perception.budget import Budget

    sc = Scenario.load(test_map)
    world = FakeWorld(test_map, sc)
    prior = PriorMap.from_map(test_map, sc)
    perception = Perception(mode="mock", ground_truth=world.ground_truth, line_of_sight=world.line_of_sight)
    asked: list[str] = []
    perception.request_relook = asked.append  # type: ignore[method-assign]

    class AskingSwarm(Swarm):
        def tick(self, t, readings, detections, **kw):
            cmds, rows, state = super().tick(t, readings, detections, **kw)
            return cmds, rows, state.model_copy(update={"relook_requests": ["det0001"]})

    observed: list[SwarmState] = []
    guidance = Guidance(prior=prior, world=world)
    guidance.observe = observed.append  # type: ignore[method-assign]
    bus = Bus()
    budget = Budget(net_probe=lambda: False, vram_probe=lambda: None, start=False)
    loop = SimLoop(world, perception, AskingSwarm(prior, list(sc.drones.ids)), sc, guidance=guidance, bus=bus,
                   budget=budget, render=False, realtime=False)
    loop.run(max_ticks=5)
    assert asked == ["det0001"] * 5  # every request reaches Perception.request_relook
    assert len(observed) == 5  # Act 1: guidance sees every tick's map_delta
    msgs = [parse_server_message(o.text) for o in bus.drain_outbound() if hasattr(o, "text")]
    budgets = [m for m in msgs if isinstance(m, BudgetMsg)]
    assert budgets and budgets[0].network == "unknown"  # probes not started; message still flows
