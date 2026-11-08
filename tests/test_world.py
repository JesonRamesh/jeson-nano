"""World facade tests (P). Uses P's scratch map until S's maps/test exists."""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sgr.world import mjcf, testmap  # noqa: E402
from sgr.contracts import Scenario  # noqa: E402
from sgr.world.walls import grid_to_boxes  # noqa: E402
from sgr.world.world import LIDAR_RANGE, World  # noqa: E402


def _gl_ok() -> bool:
    try:
        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        r = mujoco.Renderer(m, 8, 8)
        r.close()
        return True
    except Exception:
        return False


needs_gl = pytest.mark.skipif(not _gl_ok(), reason="no OpenGL context available")


@pytest.fixture(scope="module")
def map_dir(tmp_path_factory):
    return testmap.write(tmp_path_factory.mktemp("map"))


@pytest.fixture
def world(map_dir):
    w = World(map_dir, Scenario.load(map_dir))
    yield w
    w.close()


# ------------------------------------------------------------------ walls
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_boxes_cover_walls_exactly(seed):
    rng = np.random.default_rng(seed)
    wall = rng.random((40, 60)) < 0.4
    cover = np.zeros(wall.shape, int)
    for r0, c0, r1, c1 in grid_to_boxes(wall):
        cover[r0:r1, c0:c1] += 1
    assert (cover == wall.astype(int)).all()  # every wall cell exactly once, no free cell


def test_nearest_wall_steps_whole_cells():
    """Regression: stepping x + k*0.1 in floats skipped the 8.9-9.0 cell and buried the decoy in the wall."""
    from sgr.world.walls import nearest_wall_normal

    free = np.zeros((40, 120), bool)
    free[:, 71:89] = True  # corridor x in [7.1, 8.9]
    d, fx, fy, nyaw = nearest_wall_normal(free, 8.7, 2.0, 0.1)
    assert fx == pytest.approx(8.9) and d == pytest.approx(0.2) and nyaw == pytest.approx(math.pi)
    d, fx, fy, nyaw = nearest_wall_normal(free, 7.3, 2.0, 0.1)
    assert fx == pytest.approx(7.1) and nyaw == pytest.approx(0.0)


def test_map_walls_merge_small(world):
    assert world.build_info.n_wall_boxes < 100
    cover = np.zeros(world.free.shape, int)
    for r0, c0, r1, c1 in grid_to_boxes(~world.free):
        cover[r0:r1, c0:c1] += 1
    assert (cover == (~world.free).astype(int)).all()


# ------------------------------------------------------------------ drones
def test_hover_holds_altitude(world):
    p0 = world.drone_pose("D1")
    world.step(400)  # 2 s
    p = world.drone_pose("D1")
    assert abs(p.z - 1.6) < 0.02
    assert math.hypot(p.x - p0.x, p.y - p0.y) < 0.01


def test_velocity_command_moves_drone(world):
    x0 = world.drone_pose("D1").x
    world.set_drone_cmd("D1", 1.0, 0.0, 0.0, 0.0)
    world.step(200)  # 1 s
    dx = world.drone_pose("D1").x - x0
    assert 0.8 < dx < 1.05


def test_yaw_command(world):
    world.set_drone_cmd("D1", 0.0, 0.0, 0.0, 1.0)
    world.step(200)
    assert 0.75 < world.drone_pose("D1").yaw < 1.05


# ------------------------------------------------------------------ lidar
def test_lidar_matches_grid_and_ignores_drones(world):
    # D1 at (2, 2.5): corridor y in [2.1, 3.9], west wall at x = 0.5; D3 sits 0.5 m east
    r = world.lidar("D1")
    assert r.shape == (36,)
    assert r[0] == pytest.approx(LIDAR_RANGE)  # east: D3 is not ray-visible, corridor is long
    assert r[9] == pytest.approx(1.4, abs=0.1)  # north
    assert r[18] == pytest.approx(1.5, abs=0.1)  # west
    assert r[27] == pytest.approx(0.4, abs=0.1)  # south


def test_lidar_rotates_with_yaw(world):
    world.teleport_drone("D1", 2.0, 2.5, math.pi / 2)
    r = world.lidar("D1")
    assert r[0] == pytest.approx(1.4, abs=0.1)  # forward is now north


def test_debris_seen_then_removed(world):
    h = world.place_debris(5.0, 2.5)
    assert h is not None
    world.step(1)
    assert 2.2 < world.lidar("D1")[0] < 2.6  # front face 0.5-0.71 m before the centre at x=5
    world.remove(h)
    world.step(1)
    assert world.lidar("D1")[0] == pytest.approx(LIDAR_RANGE)


def test_compromised_debris_independent_of_other_edges(map_dir):
    """Editing one compromised edge must not move another edge's blocks (reachability was jitter-dependent)."""
    sc = Scenario.load(map_dir)

    def e9_blocks(s):
        w = World(map_dir, s)
        out = []
        for h in w.placed_handles("debris"):
            if math.hypot(h.x - 24.0, h.y - 8.0) < 1.5:
                q = w.data.mocap_quat[w._pool_mocap["debris"][w._placed[h.id].slot]]
                out.append((round(h.x, 6), round(h.y, 6), tuple(np.round(q, 9))))
        return sorted(out)

    more = sc.model_copy(update={"compromised": [{"edge": "e1", "at": 0.5}, *sc.compromised]})
    more = Scenario.model_validate(more.model_dump())
    assert len(e9_blocks(sc)) == 2 and e9_blocks(sc) == e9_blocks(more)


def test_compromised_edge_has_debris(world):
    gt = world.ground_truth()
    assert len(world.placed_handles("debris")) == 2  # ceil(1.8 / 1.0) blocks across e9 at (24, 8)
    for d in world.placed_handles("debris"):
        assert math.hypot(d.x - 24.0, d.y - 8.0) < 1.0


# ------------------------------------------------------------------ change tool
def test_place_refusals(world):
    assert world.place_debris(0.1, 0.1) is None  # wall cell
    assert world.place_fire(2.4, 2.6) is None  # next to drones
    assert world.place_debris(27.5, 13.4) is None  # next to the survivor
    # fill the pool, then one more is refused
    while world.place_debris(16.0, 10.0, force=True) is not None:
        pass
    assert len(world.placed_handles("debris")) == mjcf.N_DEBRIS
    assert world.place_debris(16.0, 12.0) is None


# ------------------------------------------------------------------ heat
def test_heat_model(world):
    assert world.temperature_at(12.0, 16.0) == pytest.approx(400.0, abs=1.0)  # scenario fire
    assert world.temperature("D1") == pytest.approx(20.0, abs=0.01)
    p = world.drone_pose("D3")
    h = world.place_fire(p.x, p.y, force=True)  # flare-up
    assert world.temperature("D3") > 300
    world.remove(h)
    assert world.temperature("D3") == pytest.approx(20.0, abs=0.01)


def test_kill_drone_falls(world):
    world.kill_drone("D3")
    world.step(600)  # 3 s
    assert world.drone_pose("D3").z <= 0.3
    assert not world.alive("D3")
    assert "D3" in world.ground_truth().dead_drones


# ------------------------------------------------------------------ snapshots
def test_state_roundtrip_is_deterministic(world):
    s = world.get_state()
    world.set_drone_cmd("D1", 0.8, 0.2, 0.0, 0.5)
    world.step(100)
    a = world.drone_pose("D1")
    h = world.place_debris(6.0, 3.0)
    world.kill_drone("D2")

    world.set_state(s)
    assert world.alive("D2")
    assert h.id not in world._placed
    world.set_drone_cmd("D1", 0.8, 0.2, 0.0, 0.5)
    world.step(100)
    b = world.drone_pose("D1")
    assert (a.x, a.y, a.z, a.yaw) == pytest.approx((b.x, b.y, b.z, b.yaw), abs=1e-9)
    assert world.model.body_gravcomp[world.model.body("D2").id] == 1.0


# ------------------------------------------------------------------ firefighter
def test_ff_walks_and_collides(world):
    world.set_ff_input(1.0, 0.0, 0.0)
    world.step(200)
    assert world.ff_pose().x > 2.0 + 1.1
    world.set_ff_input(0.0, 1.0, 0.0)  # strafe right = -y, into the south wall
    world.step(600)
    assert world.ff_pose().y > 2.1 + 0.2  # stopped by the wall (radius 0.25)


def test_ff_can_turn_away_from_a_wall_while_walking(world):
    """Walking into a wall must not lock the heading (wall friction used to out-torque the yaw actuator)."""
    def walk(yaw, ticks):  # like the sim loop: one control update per 20 substeps
        for _ in range(ticks):
            world.set_ff_input(1.0, 0.0, yaw)
            world.step(20)

    walk(-math.pi / 2, 30)  # walk south, pressing into the corridor wall
    y_wall = world.ff_pose().y
    walk(math.pi / 2, 20)  # keep walking, now asking for north
    f = world.ff_pose()
    assert abs(f.yaw - math.pi / 2) < 0.15
    assert f.y > y_wall + 0.8


def test_push_ff_back(world):
    f0 = world.ff_pose()
    world.push_ff_back()  # standing still, heading +x -> pushed to -x
    assert world.ff_pose().x == pytest.approx(f0.x - 1.0, abs=0.01)


# ------------------------------------------------------------------ rendering
@needs_gl
def test_render_and_pixel_ray(world):
    img, cam = world.render_drone("D1")
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8
    assert img.std() > 5
    assert cam.fx == pytest.approx(120.0)
    hit = world.pixel_to_world("D1", 160, 120)
    # optical axis 15 deg down from 1.6 m hits the floor ~5.97 m ahead of the camera
    assert hit is not None
    assert hit.x == pytest.approx(2.27 + 5.97, abs=0.25) and hit.y == pytest.approx(2.5, abs=0.1)


@needs_gl
def test_camera_does_not_see_through_nearby_walls(world):
    """Regression: the near clip plane was 0.7 m (relative to the inflated model extent), so a drone
    0.23 m from a wall saw the survivor in the room behind it."""
    assert world.model.vis.map.znear * world.model.stat.extent == pytest.approx(0.02, abs=1e-6)
    from sgr.perception.detector import detect

    world.teleport_drone("D1", 24.4, 11.0, math.atan2(2.0, 3.1))  # wall at x=24.9, survivor room behind
    img, _ = world.render_drone("D1")
    assert [b for b in detect(img) if b.label == "person"] == []


def test_scan_replica_is_visual_only(world):
    """W's textured scan (maps/<name>/scan_vis.*) is camera-only: same physics state and LiDAR, walls hidden."""
    from pathlib import Path

    m = Path(__file__).resolve().parents[1] / "maps" / "marshgate-route"
    if not (m / "scan_vis.obj").is_file():
        pytest.skip("no scan replica in this checkout")
    sc = Scenario.load(m)
    plain, vis = World(m, sc, scan_vis=False), World(m, sc, scan_vis=True)
    assert vis.build_info.scan_vis and not plain.build_info.scan_vis
    assert (vis.lidar("D1") == plain.lidar("D1")).all()
    assert len(vis.get_state()) == len(plain.get_state())  # snapshots interchangeable
    vis.set_state(plain.get_state())
    walls = [g for g in range(vis.model.ngeom) if vis.model.geom(g).name.startswith("wall")]
    assert all(vis.model.geom_rgba[g, 3] == 0 for g in walls)
    assert World(m, sc, scan_vis=None).build_info.scan_vis  # on by default when the files exist


@needs_gl
def test_dead_drone_camera_is_static(world):
    world.kill_drone("D4")
    img, _ = world.render_drone("D4")
    assert img.std() > 50  # uniform noise


def test_render_refuses_other_threads(world):
    err = []
    t = threading.Thread(target=lambda: err.append(_try(world.render_drone, "D1")))
    t.start()
    t.join()
    assert isinstance(err[0], RuntimeError)


def _try(f, *a):
    try:
        f(*a)
    except Exception as e:
        return e
    return None


# ------------------------------------------------------------------ W's assets in the cameras
def test_rubble_rng_matches_web_hazards():
    """rubble.rng is W's mulberry32 (web/src/scene3d/dressing/hazards.ts), so piles match the 3D view."""
    from sgr.world.rubble import rng

    r = rng(1)
    assert r() == pytest.approx(0.627073940, abs=1e-8)


def test_debris_is_rubble_on_an_invisible_block(world):
    """Physics/LiDAR keep the 2 m block (group 4, never drawn); the cameras see W's rubble pile (group 1)."""
    from sgr.world import shapes

    m = world.model
    b = m.body("debris00").id
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
    block = [g for g in geoms if m.geom_group[g] == shapes.DEBRIS_COLLISION_GROUP]
    rubble = [g for g in geoms if m.geom_group[g] == 1]
    assert len(block) == 1 and m.geom_contype[block[0]] == 1
    assert np.allclose(m.geom_size[block[0]] * 2, shapes.DEBRIS_SIZE)
    assert rubble and all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in rubble)


def test_fire_particles_animate_and_stay_visual(world):
    """W-style flame particles move each step; no fire geom collides or blocks LiDAR."""
    fire = [g for g in range(world.model.ngeom) if world.model.geom(g).name.startswith("fire00_")]
    assert all(world.model.geom_contype[g] == 0 and world.model.geom_group[g] == 1 for g in fire)
    p = world._fire_fx[0]["p"]
    before = world.model.geom_pos[p].copy()
    world.step(20)
    assert not np.allclose(before, world.model.geom_pos[p])


def test_drone_model_is_visual_only(map_dir, monkeypatch):
    """The shared drone.glb is drawn in the cameras, but drone physics (mass, inertia) is unchanged."""
    from sgr.world import shapes

    if not shapes.drone_model_parts():
        pytest.skip("drone.glb not available in this checkout")
    with_model = World(map_dir, Scenario.load(map_dir), scan_vis=False)
    b = with_model.model.body("D1").id
    model_geoms = [g for g in range(with_model.model.ngeom)
                   if with_model.model.geom_bodyid[g] == b and with_model.model.geom(g).name.startswith("D1_m_")]
    assert len(model_geoms) == len(shapes.drone_model_parts())
    assert all(with_model.model.geom_group[g] == 2 and with_model.model.geom_contype[g] == 0 for g in model_geoms)

    monkeypatch.setattr(shapes, "drone_model_parts", lambda: ())  # the primitive quadcopter only
    plain = World(map_dir, Scenario.load(map_dir), scan_vis=False)
    pb = plain.model.body("D1").id
    assert with_model.model.body_mass[b] == pytest.approx(plain.model.body_mass[pb])
    assert np.allclose(with_model.model.body_inertia[b], plain.model.body_inertia[pb])


# ------------------------------------------------------------------ dashboard fog (UI only)
def test_apply_fog_fades_with_distance():
    from sgr.world.fog import FOG_RGB, apply_fog

    rgb = np.full((2, 2, 3), 200, np.uint8)
    depth = np.array([[0.0, 2.0], [6.0, 60.0]])
    out = apply_fog(rgb, depth, 0.2)
    assert (out[0, 0] == 200).all()  # at the lens: untouched
    assert out[0, 1, 0] > out[1, 0, 0] > out[1, 1, 0]  # further = foggier
    assert np.allclose(out[1, 1], FOG_RGB, atol=1)  # far: all fog
    assert apply_fog(rgb, depth, 0.0) is rgb  # off


@needs_gl
def test_dashboard_frame_fogs_a_copy_only(world):
    img, _ = world.render_drone("D1")
    clean = img.copy()
    ui = world.dashboard_frame("D1", img)
    assert (img == clean).all()  # perception's frame is untouched
    assert ui.shape == img.shape and ui.dtype == np.uint8 and not (ui == img).all()
    depth = world.render_depth("D1")
    assert depth.shape == img.shape[:2] and depth.min() > 0
    world.kill_drone("D2")
    assert world.render_depth("D2") is None  # dead: static, not fogged


# ------------------------------------------------------------------ the firefighter figure (dashboard only)
def test_ff_figure_is_visual_only_and_never_in_perception(map_dir, monkeypatch):
    """W's firefighter.glb rides on the FF body: no mass (physics identical), and only in its own render group,
    so the perception frame never contains it (its hi-vis stripes are in the detector's 'person' range)."""
    from sgr.world import shapes

    if not shapes.ff_model_parts():
        pytest.skip("firefighter.glb not available in this checkout")
    w = World(map_dir, Scenario.load(map_dir), scan_vis=False)
    m = w.model
    b = m.body("FF").id
    fig = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom(g).name.startswith("FF_m_")]
    assert len(fig) == len(shapes.ff_model_parts())
    assert all(m.geom_group[g] == shapes.FF_FIGURE_GROUP and m.geom_contype[g] == 0 for g in fig)
    from sgr.world.world import RENDER_GROUPS

    assert RENDER_GROUPS[shapes.FF_FIGURE_GROUP] == 0  # the drone (perception) render never draws it
    assert m.geom_group[m.geom("FF_body").id] == 3  # the collision capsule is hidden too (no more blob)
    monkeypatch.setattr(shapes, "ff_model_parts", lambda: ())
    plain = World(map_dir, Scenario.load(map_dir), scan_vis=False)
    pb = plain.model.body("FF").id
    assert m.body_mass[b] == pytest.approx(plain.model.body_mass[pb])
    assert np.allclose(m.body_inertia[b], plain.model.body_inertia[pb])


@needs_gl
def test_ff_figure_shows_on_the_dashboard_once_steered(world):
    from sgr.world import shapes

    if not shapes.ff_model_parts():
        pytest.skip("firefighter.glb not available in this checkout")
    f = world.ff_pose()
    world.teleport_drone("D1", f.x + 3.0, f.y, math.pi)  # 3 m in front of the firefighter, looking at it
    world.step(2)
    img, _ = world.render_drone("D1")
    before = world.dashboard_frame("D1", img)
    world.set_ff_input(0.0, 0.0, 0.0)  # Act 2: being steered
    img2, _ = world.render_drone("D1")
    after = world.dashboard_frame("D1", img2)
    assert np.abs(img.astype(int) - img2).mean() < 1.0  # perception: unchanged
    assert np.abs(after.astype(int) - before).mean() > 0.3  # dashboard: the figure is drawn
    world.set_state(world.get_state())  # a snapshot load (new act): hidden until steered again
    assert not world._ff_figure_in_view("D1")
