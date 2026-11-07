"""Fast detector tests (P): real World renders of fire / survivor / decoy."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("cv2")

from sgr.perception.detector import FastDetector, detect  # noqa: E402
from sgr.world import testmap  # noqa: E402
from sgr.contracts import Scenario  # noqa: E402
from sgr.world.world import World  # noqa: E402

from test_world import needs_gl  # noqa: E402

pytestmark = needs_gl

# studio layout (testmap.write_studio): drone -> what it looks at from 3 m
VIEWS = {"D1": "fire", "D2": "survivor", "D3": "decoy"}


@pytest.fixture(scope="module")
def studio(tmp_path_factory):
    d = testmap.write_studio(tmp_path_factory.mktemp("studio"))
    w = World(d, Scenario.load(d))
    yield w
    w.close()


@pytest.fixture(scope="module")
def frames(studio):
    return {VIEWS[k]: studio.render_drone(k)[0] for k in VIEWS}


def _labels(boxes):
    return {b.label for b in boxes}


def test_fire_detected_confidently(frames):
    boxes = detect(frames["fire"])
    assert _labels(boxes) == {"fire"}
    assert boxes[0].score_fast >= 0.75
    assert boxes[0].area_frac >= 0.005


def test_survivor_detected_as_person(frames):
    boxes = detect(frames["survivor"])
    assert _labels(boxes) == {"person"}
    assert boxes[0].score_fast >= 0.75


def test_decoy_passes_fire_threshold_but_scores_lower(frames):
    decoy = detect(frames["decoy"])
    fire = detect(frames["fire"])
    assert _labels(decoy) == {"fire"}  # deliberately a false positive (§7.2)
    assert 0.45 <= decoy[0].score_fast <= 0.70
    assert decoy[0].score_fast < fire[0].score_fast - 0.15


@pytest.mark.parametrize("label,ty,dist", [("fire", 3.0, 5.0), ("person", 8.5, 4.5)])
def test_detects_out_to_coverage_range(studio, label, ty, dist):
    """§8.2 counts cells within 5 m as searched, so targets must be detectable about that far."""
    studio.teleport_drone("D4", 4.0 + dist, ty, math.pi)  # look west at the target
    boxes = [b for b in detect(studio.render_drone("D4")[0]) if b.label == label]
    assert len(boxes) == 1  # one object -> one box (parts are joined)


def test_empty_wall_has_no_detections(studio):
    studio.teleport_drone("D4", 9.0, 9.8, math.pi / 2)  # 0.7 m from the north wall
    img, _ = studio.render_drone("D4")
    assert detect(img) == []


def test_world_positions_dedup_and_blacklist(studio, frames):
    gt = studio.ground_truth()
    truth = {"fire": (gt.fires[0].x, gt.fires[0].y), "survivor": (gt.survivor.x, gt.survivor.y),
             "decoy": (gt.decoy.x, gt.decoy.y)}
    fd = FastDetector()
    for drone, target in VIEWS.items():
        c = fd.process(drone, frames[target], t=0.0, frame_id=f"{drone}-1", pixel_to_world=studio.pixel_to_world)
        assert len(c) == 1 and c[0].is_new
        assert math.dist(c[0].world_xy, truth[target]) < 0.8, (target, c[0].world_xy)

    # same view again -> merged, no new job
    again = fd.process("D1", frames["fire"], t=1.0, frame_id="D1-2", pixel_to_world=studio.pixel_to_world)
    assert len(again) == 1 and not again[0].is_new

    # Gemma rejects the decoy -> blacklisted for 60 s
    fd.blacklist("fire", truth["decoy"], t=2.0)
    assert fd.process("D3", frames["decoy"], t=3.0, frame_id="D3-2", pixel_to_world=studio.pixel_to_world) == []
    back = fd.process("D3", frames["decoy"], t=63.0, frame_id="D3-3", pixel_to_world=studio.pixel_to_world)
    assert len(back) == 1 and back[0].is_new


def test_detector_state_roundtrip():
    fd = FastDetector()
    fake = lambda d, u, v: SimpleNamespace(x=1.0, y=2.0)  # noqa: E731
    frame = np.zeros((240, 320, 3), np.uint8)
    frame[100:140, 140:180] = (255, 120, 0)  # orange square
    fd.process("D1", frame, 0.0, "f1", fake)
    fd.blacklist("person", (5.0, 5.0), 0.0)
    fd2 = FastDetector()
    fd2.set_state(fd.get_state())
    assert fd2.get_state() == fd.get_state()
