"""Snapshots (§8.6), dashboard buttons, belief-map / detection publishing, unsafe zones (§8.4)."""

from __future__ import annotations

import math

from sgr import snapshot
from sgr.bus import Bus
from sgr.contracts import (
    Act2StartMsg,
    ControlMsg,
    DetectionMsg,
    HelloMsg,
    MapDeltaMsg,
    Pose,
    PriorMap,
    Readings,
    Scenario,
    SnapshotMsg,
    parse_server_message,
)
from sgr.guidance import Guidance
from sgr.loop import SimLoop
from sgr.perception import Perception
from sgr.swarm import Swarm
from sgr.world import FakeWorld, World


def _loop(test_map, world_cls=World, bus=None, snapshot_dir=None) -> SimLoop:
    sc = Scenario.load(test_map)
    world = world_cls(test_map, sc)
    prior = PriorMap.from_map(test_map, sc)
    return SimLoop(world, Perception(mode="mock", ground_truth=world.ground_truth, line_of_sight=world.line_of_sight),
                   Swarm(prior, list(sc.drones.ids)), sc, guidance=Guidance(prior=prior, world=world), bus=bus,
                   render=False, realtime=False, map_dir=test_map, snapshot_dir=snapshot_dir)


def _poses(loop: SimLoop) -> dict[str, tuple[float, float]]:
    return {d: (round(p.x, 4), round(p.y, 4)) for d in loop.scenario.drones.ids for p in [loop.world.drone_pose(d)]}


def _msgs(bus: Bus) -> list:
    return [parse_server_message(o.text) for o in bus.drain_outbound(100000) if hasattr(o, "text")]


def test_snapshot_round_trip_resumes_exactly(test_map, tmp_path) -> None:
    loop = _loop(test_map)
    loop.run(max_ticks=150)  # 15 s: areas assigned, decoy inspected
    path = snapshot.save("mid", snapshot.capture(loop), tmp_path)
    loop.run(max_ticks=300)
    after_a, trace_a = _poses(loop), [(r.action, r.drone) for r in loop.trace]

    other = _loop(test_map)  # a fresh process would do exactly this
    snapshot.restore(other, snapshot.load_path(path))
    assert abs(other.t - 15.0) < 1e-6 and other.ticks == 150
    other.run(max_ticks=300)
    pa = _poses(other)  # MuJoCo warm-start isn't in mj state: allow sub-cm drift, nothing more
    assert all(math.dist(pa[d], after_a[d]) < 0.01 for d in after_a), (pa, after_a)
    assert [(r.action, r.drone) for r in other.trace] == trace_a


def test_auto_snapshots_and_jump_to_act2(test_map, tmp_path) -> None:
    loop = _loop(test_map, world_cls=FakeWorld, snapshot_dir=tmp_path)
    loop.run(max_ticks=900)  # chain forms ~55 s on maps/test
    assert (tmp_path / "act1_start.pkl").is_file()
    assert (tmp_path / "act2_start.pkl").is_file(), "act2_start not saved when the chain formed"
    t_chain = snapshot.load_path(tmp_path / "act2_start.pkl")["loop"]["t"]

    bus = Bus()
    fresh = _loop(test_map, world_cls=FakeWorld, bus=bus, snapshot_dir=tmp_path)
    bus.send_command(Act2StartMsg())
    fresh.tick()  # loads the snapshot instead of ticking
    assert fresh.act == 2 and fresh.t == t_chain
    assert fresh.swarm.bb.chain_formed
    msgs = _msgs(bus)
    assert any(isinstance(m, HelloMsg) for m in msgs)  # the browser resets before the restored run
    fresh.tick()
    assert any(isinstance(m, MapDeltaMsg) and m.searched for m in _msgs(bus))  # full map resync

    bus.send_command(ControlMsg(action="reset"))  # Restart Act 1
    fresh.tick()
    assert fresh.act == 1 and fresh.t == 0.0 and not fresh.swarm.bb.chain_formed


def test_realtime_pacing_survives_a_time_jump(test_map, tmp_path) -> None:
    """Jump to Act 2 moves sim time forward ~55 s: the loop must keep pacing, not sleep it off.

    (Found by driving the real app over /ws: the dashboard froze for seconds after Jump to Act 2.)
    """
    import time

    src = _loop(test_map, world_cls=FakeWorld, snapshot_dir=tmp_path)
    src.run(max_ticks=900)  # saves act2_start when the chain forms
    snap_ticks = snapshot.load("act2_start", tmp_path)["loop"]["ticks"]

    bus = Bus()
    loop = _loop(test_map, world_cls=FakeWorld, bus=bus, snapshot_dir=tmp_path)
    loop.realtime, loop.speed = True, 20.0  # 20x: 20 ticks = 2 s sim = 0.1 s wall
    loop.run(max_ticks=10)
    bus.send_command(Act2StartMsg())
    wall = time.perf_counter()
    loop.run(max_ticks=snap_ticks + 20)  # the first tick loads the snapshot, then 20 paced ticks
    took = time.perf_counter() - wall
    assert loop.act == 2 and loop.ticks == snap_ticks + 20
    assert took < 1.0, f"loop stalled {took:.1f} s after the time jump"


def test_named_snapshot_save_and_load(test_map, tmp_path) -> None:
    bus = Bus()
    loop = _loop(test_map, world_cls=FakeWorld, bus=bus, snapshot_dir=tmp_path)
    loop.run(max_ticks=20)
    bus.send_command(SnapshotMsg(op="save", name="judge"))
    loop.tick()
    t_saved = snapshot.load("judge", tmp_path)["loop"]["t"]
    loop.run(max_ticks=60)
    bus.send_command(SnapshotMsg(op="load", name="judge"))
    loop.tick()
    assert loop.t == t_saved


def test_missing_snapshot_is_a_warning_not_a_crash(test_map, tmp_path) -> None:
    bus = Bus()
    loop = _loop(test_map, world_cls=FakeWorld, bus=bus, snapshot_dir=tmp_path / "none")
    bus.send_command(Act2StartMsg())
    loop.tick()
    assert loop.act == 1 and loop.ticks == 1


def test_loop_publishes_belief_map_and_detections(test_map) -> None:
    bus = Bus()
    loop = _loop(test_map, world_cls=FakeWorld, bus=bus)
    loop.run(max_ticks=120)  # the decoy is seen at ~5 s
    msgs = _msgs(bus)
    deltas = [m for m in msgs if isinstance(m, MapDeltaMsg)]
    assert deltas and sum(len(m.searched) for m in deltas) > 100
    dets = [m for m in msgs if isinstance(m, DetectionMsg)]
    assert dets and {m.status for m in dets} >= {"pending"}
    assert any(m.gemma is not None and m.gemma.caption for m in dets)


def test_lost_drone_area_becomes_unsafe_until_reobserved(test_map) -> None:
    sc = Scenario.load(test_map)
    swarm = Swarm(PriorMap.from_map(test_map, sc), list(sc.drones.ids))
    poses = {"D1": Pose(x=2.0, y=3.0, z=1.6), "D2": Pose(x=8.0, y=12.0, z=1.6, yaw=math.pi / 2)}

    def r(t, ps):
        return {d: Readings(t=t, pose=p, lidar=[6.0] * 36, temperature=20.0) for d, p in ps.items()}

    swarm.tick(0.0, r(0.0, poses), [])
    _, rows, st = swarm.tick(3.5, r(3.5, {"D1": poses["D1"]}), [])  # D2 silent > 3 s
    assert any(row.action == "REASSIGN" and "unsafe" in row.reason for row in rows)
    unsafe = [(i, j) for i, j, s in st.map_delta if s == "unsafe"]
    assert unsafe
    # D1 flies in and looks up the n4-n6 corridor: the cells it sees are safe again
    _, _, st = swarm.tick(3.6, r(3.6, {"D1": Pose(x=8.0, y=10.0, z=1.6, yaw=math.pi / 2)}), [])
    assert any(s == "free" for _, _, s in st.map_delta)
