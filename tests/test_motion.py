"""Swarm motion regressions (debris gaps, stuck / circling drones, heat): one test per bug found by driving the
real World on the demo map and scoring every drone, every tick.
"""

from __future__ import annotations

import math

import numpy as np

from sgr.contracts import GuidanceMsg, Pose, PriorMap, Readings, Scenario, VelCmd
from sgr.loop import SimLoop, _WAIT
from sgr.perception import Perception
from sgr.swarm import Swarm
from sgr.swarm.agent import Agent
from sgr.swarm.belief import BeliefGrid
from sgr.swarm.hazards import HEAT_PEAK_C, HEAT_SIGMA_M
from sgr.swarm.planner import astar, cost_map, plan_path
from sgr.world import FakeWorld

RES = 0.2


def _room(w: int = 60, h: int = 40) -> np.ndarray:
    wall = np.zeros((h, w), bool)
    wall[0, :] = wall[-1, :] = wall[:, 0] = wall[:, -1] = True
    return wall


def test_astar_takes_a_diagonal_gap_between_debris_rows() -> None:
    """Debris rows LiDAR sees at an angle leave a diagonal channel only one cell wide after inflation. A*
    refused it ('no corner cutting' next to inflated cells) while exploration used it: drones flew straight
    at walls or went round in circles. Here the channel is the ONLY way through."""
    n = 40
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    wall = np.abs(i - j) > 2  # walls everywhere except a diagonal band 5 cells wide
    wall[0, :] = wall[-1, :] = wall[:, 0] = wall[:, -1] = True
    b = BeliefGrid(wall, RES, (0.0, 0.0), (1.1, 1.1))
    cm = cost_map(b)
    trav = np.argwhere(cm.traversable)
    assert len(trav) and all(abs(int(c[0]) - int(c[1])) == 0 for c in trav)  # only the centre diagonal
    p = astar(b, (1.1, 1.1), (6.9, 6.9))
    assert p is not None, "A* must take the one-cell diagonal channel"
    for x, y in p:  # never inside an obstacle
        ci, cj = b.world_to_cell(x, y)
        assert not cm.obstacle[cj, ci]


def test_plan_path_to_an_unreachable_goal_stops_at_the_closest_point() -> None:
    """Unreachable post (inside a fire's hot zone / walled off): no straight line into the wall any more."""
    wall = _room()
    wall[15:25, 30] = wall[15:25, 40] = True  # a closed box
    wall[15, 30:41] = wall[25, 30:41] = True
    b = BeliefGrid(wall, RES, (0.0, 0.0), (1.0, 1.0))
    goal = (7.0, 4.0)  # inside the box
    assert astar(b, (2.0, 2.0), goal) is None
    p = plan_path(b, (2.0, 2.0), goal)
    assert p is not None and len(p) > 1
    end = p[-1]
    assert math.dist(end, goal) < 3.0  # as close as it can get, outside the box


def _thermo(source: tuple[float, float], track: list[tuple[float, float]], t0: float = 0.0):
    out = []
    for k, (x, y) in enumerate(track):
        d2 = (x - source[0]) ** 2 + (y - source[1]) ** 2
        out.append((t0 + 0.1 * k, x, y, 20.0 + HEAT_PEAK_C * math.exp(-d2 / (2 * HEAT_SIGMA_M ** 2))))
    return out


def test_heat_source_is_located_from_the_drones_own_thermometer(test_map) -> None:
    """A flare-up fire no camera has confirmed: retreating 'the way it came' flew a drone back past it at
    360 °C. The source is now trilaterated from the drone's own readings and planned around."""
    sc = Scenario.load(test_map)
    sw = Swarm(PriorMap.from_map(test_map, sc), list(sc.drones.ids))
    ag = sw.agents["D1"]
    src = (20.0, 16.0)
    track = [(17.5 + 0.12 * k, 17.6 - 0.03 * k) for k in range(15)]  # flying past it, 2-3 m away
    ag.thermo.extend(_thermo(src, track))
    est = sw.hazards._locate_heat(1.5, ag)
    assert est is not None
    # straight flight leaves a mirror ambiguity across the track: the estimate sits on the source's side
    # or on the track, never ahead/behind it, so "away" is the right way
    along = (track[-1][0] - track[0][0], track[-1][1] - track[0][1])
    n = math.hypot(*along)
    ax, ay = along[0] / n, along[1] / n
    proj = lambda p: (p[0] - track[0][0]) * ax + (p[1] - track[0][1]) * ay  # noqa: E731
    assert abs(proj(est) - proj(src)) < 0.8
    sw.hazards._suspect_heat("D1", est)
    i, j = sw.belief.world_to_cell(*src)
    assert sw.belief.heat[j, i] > 1.0  # on the map now: drones plan round it


def test_idle_and_sentinel_drones_scan_instead_of_spinning() -> None:
    """Continuous yaw spinning read as 'moving in circles' in the 3D view: now a ±70° scan."""
    b = BeliefGrid(_room(), RES, (0.0, 0.0), (1.0, 1.0))
    ag = Agent("D1", 1.6)
    yaw, total = 0.3, 0.0
    for k in range(400):  # 40 s idle
        r = Readings(t=0.1 * k, pose=Pose(x=4, y=4, z=1.6, yaw=yaw), lidar=[6.0] * 36, temperature=20.0)
        cmd, _ = ag.step(0.1 * k, r, b, None, [])
        yaw += cmd.wz * 0.1
        total += abs(cmd.wz * 0.1)
    assert abs(yaw - 0.3) < math.radians(90)  # never wanders round
    assert total < 4 * 2 * math.pi  # a spin at 0.6 rad/s would be ~24 rad in 40 s ... a scan is ~half that


def test_guidance_never_steers_a_hot_drone(test_map) -> None:
    """The Act 2 lead was driven into heat by Guidance (234 °C) while the swarm had it in RETREAT."""
    sc = Scenario.load(test_map)
    world = FakeWorld(test_map, sc)
    p = world.drone_pose("D1")
    world.place_fire(p.x + 2.0, p.y, force=True)  # ~100 °C at the drone

    class PushGuidance:
        lead = "D1"

        def tick(self, t, ff, swarm):
            return GuidanceMsg(), {"D1": VelCmd(vx=1.0)}

        def observe(self, swarm):
            pass

        def drain_trace(self):
            return []

    sent = {}
    orig = world.set_drone_cmd
    world.set_drone_cmd = lambda d, vx, vy, vz, wz: (sent.__setitem__(d, (vx, vy)), orig(d, vx, vy, vz, wz))
    loop = SimLoop(world, Perception(mode="mock", ground_truth=world.ground_truth),
                   Swarm(PriorMap.from_map(test_map, sc), list(sc.drones.ids)), sc, guidance=PushGuidance(),
                   render=False, realtime=False)
    loop.act = 2
    loop.tick()
    assert sent["D1"] != (1.0, 0.0)  # Guidance's push was not applied to the hot drone


def test_flare_never_hits_a_random_drone(test_map) -> None:
    """After the wait, the old fallback killed scenario.flare.target wherever it was (once: on the only route)."""
    sc = Scenario.load(test_map)
    world = FakeWorld(test_map, sc)  # drones at the entrance, the fire ~9 m away
    loop = SimLoop(world, Perception(mode="mock", ground_truth=world.ground_truth),
                   Swarm(PriorMap.from_map(test_map, sc), list(sc.drones.ids)), sc, render=False, realtime=False)
    assert loop._flare_victim(waited=5.0) == _WAIT
    assert loop._flare_victim(waited=31.0) is None  # nobody near a fire: skip the flare-up
