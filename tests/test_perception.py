"""Perception tests (P): fusion rules, mock mode on the real World, live mode with a scripted Gemma."""

from __future__ import annotations

import math
import time

import pytest

pytest.importorskip("mujoco")

from sgr.contracts import GemmaResult, Scenario  # noqa: E402
from sgr.perception import Perception  # noqa: E402
from sgr.perception.fusion import fuse  # noqa: E402
from sgr.perception.gemma import GemmaTimeout  # noqa: E402
from sgr.world import testmap  # noqa: E402
from sgr.world.world import World  # noqa: E402

from test_world import needs_gl  # noqa: E402


def _g(fire=False, person=False, conf=0.9) -> GemmaResult:
    return GemmaResult(fire=fire, person=person, confidence=conf, caption="test")


# ------------------------------------------------------------------ fusion (§7.3)
@pytest.mark.parametrize("label,fast,g,relooks,combined,status", [
    ("fire", 0.80, _g(fire=True, conf=0.88), 0, 0.848, "confirmed"),
    ("person", 0.85, _g(person=True, conf=0.93), 0, 0.898, "confirmed"),
    ("fire", 0.60, _g(fire=True, conf=0.35), 0, 0.45, "low_conf"),  # decoy, first look
    ("fire", 0.60, _g(fire=False, conf=0.8), 1, 0.18, "rejected"),  # decoy, closer look
    ("fire", 0.90, _g(fire=False), 0, 0.27, "low_conf"),  # a first look never rejects
    ("fire", 0.90, _g(fire=False), 2, 0.27, "rejected"),  # still disagrees after 2 re-looks
    ("fire", 0.50, _g(fire=True, conf=0.6), 2, 0.56, "unconfirmed"),  # agrees, but weakly
    ("fire", 0.62, None, 0, 0.62, "unconfirmed"),  # Gemma timeout / down
    ("fire", 0.62, _g(fire=True, conf=0.85), 0, 0.758, "low_conf"),  # weak colour: closer look first
    ("fire", 0.62, _g(fire=True, conf=0.85), 1, 0.758, "confirmed"),  # ...and agreed again up close
    ("fire", 0.62, _g(fire=False, conf=0.9), 1, 0.186, "rejected"),  # decoy seen head-on
])
def test_fusion(label, fast, g, relooks, combined, status):
    assert fuse(label, fast, g, relooks) == (pytest.approx(combined, abs=1e-3), status)


# ------------------------------------------------------------------ mock mode
@pytest.fixture
def studio(tmp_path):
    d = testmap.write_studio(tmp_path / "studio")
    w = World(d, Scenario.load(d))
    yield w
    w.close()


def _mock(world) -> Perception:
    return Perception(mode="mock", ground_truth=world.ground_truth, line_of_sight=world.line_of_sight)


def _look(p: Perception, world, drone: str, t: float, n: int = 1):
    p.submit_frame(drone, None, world.cam_info(drone), t, f"{drone}-{n}")
    return p.poll()


def test_mock_fire_confirmed_after_latency(studio):
    p = _mock(studio)
    first = _look(p, studio, "D1", 0.0)
    assert [(d.label, d.status) for d in first] == [("fire", "pending")]
    assert _look(p, studio, "D4", 0.5) == []  # Gemma still thinking
    done = _look(p, studio, "D4", 1.0)
    assert len(done) == 1 and done[0].id == first[0].id
    assert done[0].status == "confirmed" and done[0].combined == pytest.approx(0.848)
    assert math.dist((done[0].world_xy.x, done[0].world_xy.y), (4.0, 3.0)) < 1e-6
    assert p.metrics().p50_s == 1.0


def test_mock_survivor_confirmed(studio):
    p = _mock(studio)
    assert [d.label for d in _look(p, studio, "D2", 0.0)] == ["person"]
    done = _look(p, studio, "D4", 1.0)
    assert done[0].status == "confirmed" and done[0].combined == pytest.approx(0.898)


def test_mock_decoy_rejected_on_relook_then_blacklisted(studio):
    p = _mock(studio)
    det = _look(p, studio, "D3", 0.0)[0]
    assert det.label == "fire"  # the swarm doesn't know it's a decoy
    low = _look(p, studio, "D4", 1.0)[0]
    assert low.status == "low_conf" and low.combined == pytest.approx(0.45)
    p.request_relook(det.id)
    pend = p.poll()[0]
    assert pend.status == "pending" and pend.relooks == 1
    rej = _look(p, studio, "D4", 2.0)[0]
    assert rej.status == "rejected" and rej.combined == pytest.approx(0.18)
    assert "panel" in rej.gemma.caption
    assert _look(p, studio, "D3", 30.0) == []  # blacklisted for 60 s
    again = _look(p, studio, "D3", 63.0)
    assert len(again) == 1 and again[0].id != det.id


def test_mock_latency_is_configurable(studio):
    """Rehearse live-Gemma timing headless: answers arrive after mock_latency_s, not 1 s."""
    p = Perception(mode="mock", ground_truth=studio.ground_truth, line_of_sight=studio.line_of_sight,
                   mock_latency_s=7.0)
    assert _look(p, studio, "D1", 0.0)[0].status == "pending"
    assert _look(p, studio, "D4", 6.9) == []
    done = _look(p, studio, "D4", 7.0)
    assert done[0].status == "confirmed" and done[0].gemma.latency_s == 7.0
    assert p.metrics().p50_s == 7.0


def test_mock_dedups_repeated_views(studio):
    p = _mock(studio)
    assert len(_look(p, studio, "D1", 0.0)) == 1
    assert _look(p, studio, "D1", 0.1, 2) == []


def test_mock_sees_placed_fire(studio):
    p = _mock(studio)
    studio.teleport_drone("D4", 9.0, 5.5, -math.pi / 2)  # facing south, empty floor ahead
    assert _look(p, studio, "D4", 0.0) == []
    assert studio.place_fire(9.0, 2.0) is not None  # judge click
    assert [d.label for d in _look(p, studio, "D4", 0.1)] == ["fire"]


def test_mock_respects_walls(tmp_path):
    d = testmap.write(tmp_path / "map")
    w = World(d, Scenario.load(d))
    # in the east corridor, looking at the survivor room through its wall (door is further north)
    w.teleport_drone("D1", 24.0, 11.0, math.atan2(2.0, 3.5))
    blind = Perception(mode="mock", ground_truth=w.ground_truth, line_of_sight=w.line_of_sight)
    assert _look(blind, w, "D1", 0.0) == []
    with pytest.warns(UserWarning, match="through walls"):
        xray = Perception(mode="mock", ground_truth=w.ground_truth)
    assert [x.label for x in _look(xray, w, "D1", 0.0)] == ["person"]


def test_mock_state_roundtrip(studio):
    p = _mock(studio)
    det = _look(p, studio, "D3", 0.0)[0]
    s = p.get_state()
    q = _mock(studio)
    q.set_state(s)
    done = _look(q, studio, "D4", 1.0)
    assert done[0].id == det.id and done[0].status == "low_conf"


@needs_gl
def test_snapshot_from_live_mode_loads_in_mock(studio):
    """act2_start.pkl is recorded with live Gemma on the demo laptop but loaded by teammates in mock mode."""
    live = _live(studio, lambda f: _g(fire=True, conf=0.9))
    _frame(live, studio, "D1", 0.0)
    _settle(live)
    state = live.get_state()
    live.close()
    mock = _mock(studio)
    with pytest.warns(UserWarning, match="mode 'local'"):
        mock.set_state(state)
    assert [d.label for d in _look(mock, studio, "D1", 1.0)] == ["fire"]  # works normally afterwards


def test_mock_needs_ground_truth():
    with pytest.raises(ValueError):
        Perception(mode="mock")


# ------------------------------------------------------------------ live mode (scripted Gemma, no Ollama)
class ScriptedGemma:
    """Stands in for GemmaClient. answer(frame) -> GemmaResult, or raises."""

    model_tag = "scripted"

    def __init__(self, answer, timeout_s: float = 8.0):
        self.answer = answer
        self.timeout_s = timeout_s
        self.calls = 0

    def warm(self):
        return True

    def ask(self, frame):
        self.calls += 1
        r = self.answer(frame)
        return r.model_copy(update={"latency_s": 0.01})

    def close(self):
        pass


def _live(world, answer) -> Perception:
    return Perception(mode="local", pixel_to_world=world.pixel_to_world, gemma=ScriptedGemma(answer), warm=False)


def _frame(p: Perception, world, drone: str, t: float, n: int = 1):
    img, cam = world.render_drone(drone)
    p.submit_frame(drone, img, cam, t, f"{drone}-{n}")


def _settle(p: Perception, timeout: float = 3.0):
    """Poll until no detection is pending (the worker thread answers asynchronously)."""
    out, latest = [], {}
    end = time.time() + timeout
    while time.time() < end:
        for d in p.poll():
            out.append(d)
            latest[d.id] = d
        if latest and all(d.status != "pending" for d in latest.values()):
            return out
        time.sleep(0.01)
    raise AssertionError(f"still pending: {latest}")


@needs_gl
def test_live_fire_confirmed(studio):
    p = _live(studio, lambda f: _g(fire=True, conf=0.9))
    _frame(p, studio, "D1", 0.0)
    out = _settle(p)
    assert [d.status for d in out] == ["pending", "confirmed"]
    det = out[-1]
    assert det.label == "fire" and det.combined >= 0.9 and det.bbox is not None
    assert math.dist((det.world_xy.x, det.world_xy.y), (4.0, 3.0)) < 0.8
    m = p.metrics()
    assert m.mode == "local" and m.model_tag == "scripted" and m.p50_s is not None
    p.close()


@needs_gl
def test_live_timeout_is_unconfirmed(studio):
    def slow(frame):
        raise GemmaTimeout("6 s")

    p = _live(studio, slow)
    _frame(p, studio, "D2", 0.0)
    det = _settle(p)[-1]
    assert det.label == "person" and det.status == "unconfirmed" and det.gemma is None
    assert p.metrics().timeouts == 1
    p.close()


@needs_gl
def test_live_hard_deadline_drops_late_answer(studio):
    def slow(frame):
        time.sleep(0.6)
        return _g(person=True, conf=0.95)

    client = ScriptedGemma(slow, timeout_s=0.2)
    p = Perception(mode="local", pixel_to_world=studio.pixel_to_world, gemma=client, warm=False)
    _frame(p, studio, "D2", 0.0)
    det = _settle(p)[-1]
    assert det.status == "unconfirmed" and det.gemma is None  # deadline hit, fast-only
    time.sleep(0.6)
    assert p.poll() == []  # the late answer is dropped
    assert p.metrics().timeouts == 1
    p.close()


@needs_gl
def test_live_decoy_relook_rejected_and_blacklisted(studio):
    client = ScriptedGemma(lambda f: _g(fire=False, conf=0.8))
    p = Perception(mode="local", pixel_to_world=studio.pixel_to_world, gemma=client, warm=False)
    _frame(p, studio, "D3", 0.0)
    low = _settle(p)[-1]
    assert low.label == "fire" and low.status == "low_conf"  # first look never rejects
    p.request_relook(low.id)
    assert p.poll()[0].relooks == 1
    _frame(p, studio, "D3", 1.0, 2)  # the closer view arrives
    rej = _settle(p)[-1]
    assert rej.status == "rejected" and client.calls == 2
    _frame(p, studio, "D3", 2.0, 3)
    assert p.poll() == []  # blacklisted: no new job
    assert client.calls == 2
    p.close()


def test_live_needs_pixel_to_world():
    with pytest.raises(ValueError):
        Perception(mode="local", gemma=ScriptedGemma(lambda f: _g()), warm=False)
