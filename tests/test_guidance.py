"""Guidance tests (P): §9 rules on a scripted SwarmState."""

from __future__ import annotations

import math

import pytest

from sgr.contracts import (
    Checkpoint,
    Entrance,
    Graph,
    GraphEdge,
    GraphNode,
    Pose,
    PriorMap,
    Route,
    SurvivorState,
    SwarmState,
)
from sgr.guidance import Guidance
from sgr.guidance.geometry import point_at, project

ROUTE = [(2.0, 3.0), (16.0, 3.0), (16.0, 15.0)]  # 26 m, one turn at (16, 3)


class StubWorld:
    def __init__(self, poses):
        self.poses = dict(poses)
        self.pushed = 0

    def drone_pose(self, d):
        x, y = self.poses[d]
        return Pose(x=x, y=y, z=1.6, yaw=0.0)

    def push_ff_back(self):
        self.pushed += 1


def _swarm(t=0.0, version=1, status="ok", points=ROUTE, delta=(), survivor=None) -> SwarmState:
    return SwarmState(
        t=t, route=Route(points=list(points), status=status, version=version),
        checkpoints=[Checkpoint(x=16.0, y=3.0, drone="D2")], lead="D1", chain_formed=True,
        map_delta=list(delta), survivor=survivor,
    )


def _prior() -> PriorMap:
    nodes = [GraphNode(id="n1", x=2, y=3, kind="entrance"), GraphNode(id="n2", x=16, y=3, kind="junction"),
             GraphNode(id="n3", x=16, y=15, kind="room"), GraphNode(id="n4", x=26, y=3, kind="junction")]
    edges = [GraphEdge(id="e1", a="n1", b="n2"), GraphEdge(id="e2", a="n2", b="n3"),
             GraphEdge(id="e3", a="n2", b="n4")]
    return PriorMap(name="t", resolution=0.1, origin=(0, 0), size=(300, 200), entrance=Entrance(x=2, y=3),
                    graph=Graph(nodes=nodes, edges=edges), drone_ids=["D1", "D2", "D3", "D4"], altitude=1.6)


def _run(g: Guidance, ff: Pose, seconds: float, t0: float = 0.0, **sw):
    msg = None
    for k in range(int(round(seconds / 0.1)) + 1):
        msg, cmds = g.tick(t0 + k * 0.1, ff, _swarm(t=t0 + k * 0.1, **sw))
    return msg, cmds


# ------------------------------------------------------------------ geometry
def test_projection_and_side():
    pr = project(ROUTE, (8.0, 5.0))
    assert pr.s == pytest.approx(6.0) and pr.dist == pytest.approx(2.0) and pr.side > 0  # left of travel
    assert project(ROUTE, (17.0, 10.0)).side < 0  # east of the northbound leg = right of travel
    assert point_at(ROUTE, 20.0) == pytest.approx((16.0, 9.0))


# ------------------------------------------------------------------ rules
def test_on_route_go_says_once():
    g = Guidance()
    m1, _ = g.tick(0.0, Pose(x=4, y=3, yaw=0.0), _swarm())
    m2, _ = g.tick(0.1, Pose(x=4, y=3, yaw=0.0), _swarm())
    assert m1.severity == "info" and m1.say == m1.subtitle and m2.say is None
    assert m1.distance == pytest.approx(24.0) and abs(m1.arrow_deg) < 1.0


def test_arrow_does_not_cut_the_corner():
    """1 m before a 90 deg turn the arrow aims at the corner, not 3 m round it (that line crosses the wall)."""
    m, _ = Guidance().tick(0.0, Pose(x=15, y=3, yaw=0.0), _swarm())
    assert 0 < m.arrow_deg < 30
    m, _ = Guidance().tick(0.0, Pose(x=16, y=3.6, yaw=math.pi / 2), _swarm())  # round it: straight on
    assert abs(m.arrow_deg) < 5


def test_arrow_threads_a_narrow_gap():
    """S's replay: a 3 m lookahead pointed into a debris block at the 1.2 m rubble gap (a dog-leg)."""
    jog = [(2.0, 3.0), (10.0, 3.0), (10.6, 4.2), (16.0, 4.2)]  # 1.2 m sideways jog through a gap
    g = Guidance()
    m, _ = g.tick(0.0, Pose(x=9.0, y=3.0, yaw=0.0), _swarm(points=jog))
    from sgr.guidance.geometry import project, seg_dist

    tgt = g._target(jog, project(jog, (9.0, 3.0)), (9.0, 3.0))
    s0 = project(jog, (9.0, 3.0)).s
    for k in range(1, 10):  # the straight line to the target stays on the route
        q = point_at(jog, s0 + (project(jog, tgt).s - s0) * k / 10)
        assert seg_dist(q, (9.0, 3.0), tgt) <= 0.25 + 1e-9


def test_off_route_arrow_returns_to_route():
    m, _ = Guidance().tick(0.0, Pose(x=8, y=4.5, yaw=0.0), _swarm())  # 1.5 m north of the route, facing east
    assert -90 < m.arrow_deg < -20  # back down towards the route (right), not 3 m ahead


def test_drift_after_two_seconds():
    g = Guidance()
    m, _ = _run(g, Pose(x=8, y=5.5, yaw=0.0), 1.5)
    assert m.subtitle.startswith("Follow")
    m, _ = _run(g, Pose(x=8, y=5.5, yaw=0.0), 1.0, t0=1.6)
    assert m.subtitle == "Bear right." and m.severity == "warn"
    assert g.corrections == 1


def test_wrong_heading():
    g = Guidance()
    m, _ = _run(g, Pose(x=8, y=3, yaw=math.pi), 2.1)
    assert m.subtitle == "Wrong way. Turn around."


def test_wrong_branch_edge():
    g = Guidance(prior=_prior())
    m, _ = _run(g, Pose(x=20, y=3, yaw=math.pi), 2.1)  # on e3 (off route), even facing back
    assert m.subtitle == "Wrong way. Turn around."


def test_junction_is_not_wrong_way():
    g = Guidance(prior=_prior())
    m, _ = _run(g, Pose(x=16.3, y=3.4, yaw=math.pi / 2), 3.0)  # standing in the junction, facing the route
    assert m.subtitle.startswith("Follow")


def test_danger_near_fire_and_push_back_inside():
    w = StubWorld({"D1": (6, 3), "D2": (16, 3)})
    g = Guidance(world=w)
    fire = [(30, 15, "fire")]  # 0.2 m cell centred at (6.1, 3.1)
    m, _ = g.tick(0.0, Pose(x=5.4, y=3.0, yaw=0.0), _swarm(delta=fire))
    assert m.severity == "danger" and m.alarm and m.say == "Stop. Fire ahead. Turn back."
    assert w.pushed == 0
    m, _ = g.tick(0.1, Pose(x=6.1, y=3.1, yaw=0.0), _swarm())  # the hazard is remembered
    assert m.alarm and w.pushed == 1


def test_no_route():
    m, _ = Guidance().tick(0.0, Pose(x=4, y=3), _swarm(status="none", points=[]))
    assert m.subtitle == "No safe route. Holding."


def test_route_replaced_under_firefighter():
    g = Guidance()
    _run(g, Pose(x=6, y=3, yaw=0.0), 1.0)
    new = [(2.0, 3.0), (6.0, 3.0), (6.0, 15.0), (16.0, 15.0)]
    m, _ = g.tick(1.1, Pose(x=6, y=3, yaw=0.0), _swarm(t=1.1, version=2, points=new))
    assert m.subtitle == "Stop. Route blocked." and m.alarm and m.say
    m, _ = _run(g, Pose(x=6, y=3, yaw=math.pi / 2), 1.0, t0=3.2, version=2, points=new)
    assert m.subtitle == "New route. Follow me."
    m, _ = _run(g, Pose(x=6, y=3, yaw=math.pi / 2), 0.5, t0=7.5, version=2, points=new)
    assert m.subtitle.startswith("Follow the drone")


def test_mission_complete_logs_trace():
    g = Guidance()
    sv = SurvivorState(status="confirmed", x=16.0, y=15.0, conf=0.9)
    m, _ = g.tick(0.0, Pose(x=16, y=14, yaw=math.pi / 2), _swarm(survivor=sv))
    assert m.subtitle == "Survivor reached. Mission complete." and g.complete
    rows = g.drain_trace()
    assert [r.action for r in rows] == ["MISSION_COMPLETE"] and rows[0].evidence.value["ff"] == [16.0, 14.0]


def test_corrections_are_traced_with_evidence():
    g = Guidance()
    _run(g, Pose(x=8, y=5.5, yaw=0.0), 2.5)
    rows = g.drain_trace()
    assert [r.action for r in rows] == ["FF_CORRECTION"]
    assert rows[0].evidence.sensor == "planner" and rows[0].evidence.value["rule"] == "drift"


# ------------------------------------------------------------------ lead + handover
def test_lead_moves_ahead_of_firefighter():
    w = StubWorld({"D1": (3.0, 3.0), "D2": (16, 3)})
    _, cmds = Guidance(world=w).tick(0.0, Pose(x=2.5, y=3, yaw=0.0), _swarm())
    c = cmds["D1"]
    assert c.vx > 0.5 and abs(c.vy) < 0.1 and math.hypot(c.vx, c.vy) <= 1.2 + 1e-9


def test_waiting_lead_round_a_corner_arrow_follows_route():
    """Regression (replay deadlock): the lead holds ahead round the corner; the arrow must follow the
    route (east, to the corner), not point straight at the lead through the wall (north-east)."""
    w = StubWorld({"D1": (16.0, 4.5), "D2": (26, 3)})  # 9.5 m ahead along the route: holds (6 < gap <= 10)
    m, cmds = Guidance(world=w).tick(0.0, Pose(x=8.0, y=3.0, yaw=0.0), _swarm())
    assert m.subtitle == "Follow me." and cmds["D1"].vx == 0 and cmds["D1"].vy == 0
    assert abs(m.arrow_deg) < 5


def test_far_lead_round_a_corner_comes_back_along_the_route():
    """A lead > 10 m ahead flies back to the firefighter, along the route (south to the corner), not
    through the wall; the arrow still follows the route."""
    w = StubWorld({"D1": (16.0, 11.0), "D2": (26, 3)})  # 16 m ahead along the route
    m, cmds = Guidance(world=w).tick(0.0, Pose(x=8.0, y=3.0, yaw=0.0), _swarm())
    assert m.subtitle.startswith("Your drone is coming back")
    assert cmds["D1"].vy < 0 and abs(cmds["D1"].vx) < 0.05  # back down the corridor, not diagonally
    assert abs(m.arrow_deg) < 5


def test_lead_steps_along_the_route():
    """The lead flies along the route (round the corner), never straight at its goal through the wall."""
    w = StubWorld({"D1": (15.5, 3.0), "D2": (16, 12)})  # D2 away from its post, so no handover
    _, cmds = Guidance(world=w).tick(0.0, Pose(x=14.5, y=3, yaw=0.0), _swarm())  # goal 3 m ahead = (16, 4.5)
    c = cmds["D1"]
    # carrot 1 m along the route = (16, 3.5): vy/vx = 1. Straight at the goal (16, 4.5) would be vy/vx = 3.
    assert c.vx > 0 and c.vy <= c.vx + 1e-9


def test_lead_waits_when_firefighter_falls_behind():
    w = StubWorld({"D1": (12.0, 3.0), "D2": (16, 3)})
    m, cmds = Guidance(world=w).tick(0.0, Pose(x=3.0, y=3, yaw=0.0), _swarm())
    assert cmds["D1"].vx == 0 and cmds["D1"].vy == 0
    assert m.subtitle == "Follow me." and abs(m.arrow_deg) < 1.0  # arrow points at the waiting lead


def test_handover_to_next_chain_drone():
    w = StubWorld({"D1": (17.0, 6.0), "D2": (16.0, 3.5)})
    g = Guidance(world=w)
    g.tick(0.0, Pose(x=15.0, y=3.0, yaw=0.0), _swarm())
    assert g.lead == "D2"


def test_hazards_survive_snapshot():
    g = Guidance()
    g.observe(_swarm(delta=[(30, 15, "fire")]))  # seen during Act 1
    h = Guidance()
    h.set_state(g.get_state())
    m, _ = h.tick(0.0, Pose(x=5.6, y=3.0, yaw=0.0), _swarm())
    assert m.severity == "danger"


def test_cleared_hazard_is_forgotten():
    g = Guidance()
    g.observe(_swarm(delta=[(30, 15, "fire")]))
    g.observe(_swarm(delta=[(30, 15, "free")]))
    m, _ = g.tick(0.0, Pose(x=5.6, y=3.0, yaw=0.0), _swarm())
    assert m.severity == "info"


# ------------------------------------------------------------------ with the real World
def test_walking_into_fire_pushes_real_firefighter_back(tmp_path):
    pytest.importorskip("mujoco")
    from sgr.contracts import Scenario
    from sgr.world import testmap
    from sgr.world.world import World

    d = testmap.write(tmp_path / "map")
    w = World(d, Scenario.load(d))
    w.set_ff_input(1.0, 0.0, 0.0)
    w.step(400)  # walk 2 s east along the corridor
    f0 = w.ff_pose()
    i, j = int(f0.x / 0.2), int(f0.y / 0.2)
    g = Guidance(world=w)
    m, _ = g.tick(0.0, f0, _swarm(delta=[(i, j, "fire")]))
    assert m.alarm and w.ff_pose().x < f0.x - 0.4  # pushed back against its direction of travel


# ------------------------------------------------------------------ one drone left (no chain checkpoints)
class MovingWorld(StubWorld):
    """The lead actually flies (its velocity command is integrated), for whole-walk tests."""

    def fly(self, d, cmd, dt=0.1):
        x, y = self.poses[d]
        self.poses[d] = (x + cmd.vx * dt, y + cmd.vy * dt)


def _walk_alone(lead_start, seconds=40.0):
    """One drone, no checkpoints; the firefighter walks wherever the arrow points at 1.4 m/s."""
    w = MovingWorld({"D1": lead_start})
    g = Guidance(world=w)
    sv = SurvivorState(status="confirmed", x=16.0, y=15.0, conf=0.9)
    ff, t, subs, said = [2.0, 3.0], 0.0, [], []
    while t < seconds and not g.complete:
        sw = SwarmState(t=t, route=Route(points=ROUTE, status="ok", version=1), checkpoints=[], lead="D1",
                        chain_formed=True, survivor=sv)
        m, cmds = g.tick(t, Pose(x=ff[0], y=ff[1], yaw=0.0), sw)
        subs.append(m.subtitle)
        said += [c.kind for c in g.drain_callouts()]
        if "D1" in cmds:
            w.fly("D1", cmds["D1"])
        h = math.radians(m.arrow_deg)
        ff = [ff[0] + 0.14 * math.cos(h), ff[1] + 0.14 * math.sin(h)]
        t += 0.1
    return g, w, subs, said


def test_single_drone_leads_without_checkpoints():
    g, w, subs, said = _walk_alone((3.5, 3.0))
    assert g.complete and g.lead == "D1"
    assert not any(s.startswith(("Follow me", "Your drone")) for s in subs)
    assert said[0] == "intro" and "handover_out" not in said


def test_far_lead_flies_back_to_the_firefighter():
    """The only drone is still over the survivor when Act 2 starts: it comes back instead of waiting there."""
    g, w, subs, said = _walk_alone((16.0, 14.0))
    assert g.complete
    assert subs[0].startswith("Your drone is coming back")
    assert not any(s.startswith("Follow me") for s in subs)  # never "waits" 20 m away
    assert any(s.startswith("Follow the drone") for s in subs)  # then leads as usual
    assert said[0] == "intro" and "lead_wait" not in said
