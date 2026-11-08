"""Act 2 drone voices (P): when drones speak (Guidance callouts) and the Gemini voice agent's safety net.

All offline: the Gemini API is faked with httpx.MockTransport.
"""

from __future__ import annotations

import base64
import json
import time

import httpx
import pytest

from sgr.contracts import Checkpoint, Pose, Route, SurvivorState, SwarmState
from sgr.guidance import Guidance
from sgr.guidance.voice import Callout, VoiceAgent, check_line, load_api_key

ROUTE = [(2.0, 3.0), (16.0, 3.0), (16.0, 15.0)]  # 26 m


class StubWorld:
    def __init__(self, poses):
        self.poses = dict(poses)

    def drone_pose(self, d):
        x, y = self.poses[d]
        return Pose(x=x, y=y, z=1.6, yaw=0.0)

    def push_ff_back(self):
        pass


def _swarm(t=0.0, delta=(), survivor=None, lead="D1") -> SwarmState:
    return SwarmState(t=t, route=Route(points=ROUTE, status="ok", version=1),
                      checkpoints=[Checkpoint(x=16.0, y=3.0, drone="D2")], lead=lead, chain_formed=True,
                      map_delta=list(delta), survivor=survivor)


def _kinds(g: Guidance) -> list[tuple[str, str, str]]:
    return [(m.speaker, m.kind, m.text) for m in g.drain_callouts()]


# ------------------------------------------------------------------ when drones speak
def test_intro_once_from_the_lead():
    g = Guidance(world=StubWorld({"D1": (5, 3), "D2": (16, 3)}))
    said = []
    for k in range(30):
        g.tick(k * 0.1, Pose(x=3.0, y=3.0, yaw=0.0), _swarm(t=k * 0.1))
        said += _kinds(g)
    assert [s[:2] for s in said] == [("D1", "intro")]
    assert "survivor" in said[0][2]


def test_fire_on_the_right_comes_from_the_drone_watching_it():
    w = StubWorld({"D1": (7, 3), "D2": (16, 3), "D3": (9, 1)})
    g = Guidance(world=w)
    g.tick(0.0, Pose(x=4.0, y=3.0, yaw=0.0), _swarm())
    _kinds(g)  # intro
    fire = [(37, 2, "fire")]  # cell centre (7.5, 0.5): 2.9 m to the right of an FF at (6, 3) heading +x
    out = []
    for k in range(1, 120):  # past the 7 s gap after the intro
        g.tick(k * 0.1, Pose(x=6.0, y=3.0, yaw=0.0), _swarm(t=k * 0.1, delta=fire if k == 1 else ()))
        out += _kinds(g)
    fires = [o for o in out if o[1] == "fire_side"]
    assert len(fires) == 1  # said once, not every tick
    speaker, _, text = fires[0]
    assert "right" in text and "3 metres" in text
    assert speaker == "D1"  # D3 is not in the chain; the lead is the nearest chain drone within 8 m


def test_handover_is_one_line_from_the_new_lead():
    w = StubWorld({"D1": (17.0, 6.0), "D2": (16.0, 3.5)})
    g = Guidance(world=w)
    g.tick(0.0, Pose(x=10.0, y=3.0, yaw=0.0), _swarm())
    g.lead = "D1"
    _kinds(g)
    g.tick(0.1, Pose(x=15.0, y=3.0, yaw=0.0), _swarm(t=0.1))  # within 2 m of D2 -> D2 leads (even inside the gap)
    said = _kinds(g)
    assert said == [("D2", "handover", "D2 here, taking over from D1. Follow me.")]


def test_quiet_while_the_hud_warns():
    g = Guidance(world=StubWorld({"D1": (6, 3), "D2": (16, 3)}))
    fire = [(30, 15, "fire")]
    g.tick(0.0, Pose(x=5.4, y=3.0, yaw=0.0), _swarm(delta=fire))  # within 1 m of fire: danger
    assert _kinds(g) == []


def test_near_the_survivor_says_so_not_follow_me():
    g = Guidance(world=StubWorld({"D1": (20, 3), "D2": (16, 3)}))
    sv = SurvivorState(status="confirmed", x=16.0, y=15.0, conf=0.9)
    said, t, m = [], 0.0, None
    for x, y in [(2.0, 3.0)] * 2 + [(15.5, 3.0)] * 50 + [(16.0, 9.0)] * 50:  # 6 m from the survivor
        m, _ = g.tick(t, Pose(x=x, y=y, yaw=1.57), _swarm(t=t, survivor=sv))
        said += _kinds(g)
        t += 0.1
    assert m.subtitle == "Survivor ahead. 6 metres."  # the HUD, instead of "Follow me." / "Follow the drone."
    assert [s[1] for s in said] == ["intro", "handover"]  # no extra drone line repeating it


def test_handover_to_the_survivor_drone_is_not_follow_me():
    sv = SurvivorState(status="confirmed", x=16.0, y=15.0, conf=0.9)
    sw = lambda t: SwarmState(  # noqa: E731
        t=t, route=Route(points=ROUTE, status="ok", version=1), lead="D1", chain_formed=True, survivor=sv,
        checkpoints=[Checkpoint(x=16.0, y=15.0, drone="D3")])  # 2 drones left: lead + survivor drone
    g = Guidance(world=StubWorld({"D1": (17.0, 9.0), "D3": (16.0, 13.0)}))
    g.tick(0.0, Pose(x=16.0, y=7.0, yaw=1.57), sw(0.0))
    g.lead = "D1"
    _kinds(g)
    g.tick(0.1, Pose(x=16.0, y=11.5, yaw=1.57), sw(0.1))  # within 2 m of D3 -> D3 leads
    assert ("D3", "handover", "D3 here, I'm with the survivor. Nearly there.") in _kinds(g)


def test_progress_every_20_m_and_halfway():
    g = Guidance()
    g._cl_total, g.lead = 60.0, "D1"
    assert g._progress_callout(55) is None  # still in the first 20 m band
    assert g._progress_callout(39).text == "39 metres to go."
    assert g._progress_callout(38) is None  # once per band
    assert g._progress_callout(29).text == "Halfway there. 29 metres to go."
    assert g._progress_callout(19).text == "19 metres to go."  # past the 20 m mark
    assert g._progress_callout(10) is None  # the end is the survivor line's job


def test_snapshot_load_resets_the_conversation():
    g = Guidance(world=StubWorld({"D1": (5, 3), "D2": (16, 3)}))
    g.tick(0.0, Pose(x=3.0, y=3.0, yaw=0.0), _swarm())
    assert _kinds(g)
    g.set_state(g.get_state())
    g.tick(10.0, Pose(x=3.0, y=3.0, yaw=0.0), _swarm(t=10.0))
    assert [s[1] for s in _kinds(g)] == ["intro"]


def test_quiet_after_mission_complete():
    g = Guidance(world=StubWorld({"D1": (17.0, 6.0), "D2": (16.0, 14.0)}))
    sv = SurvivorState(status="confirmed", x=16.0, y=15.0, conf=0.9)
    g.tick(0.0, Pose(x=16.0, y=14.0, yaw=1.57), _swarm(survivor=sv))  # at the survivor: complete
    assert g.complete
    g.lead = "D1"
    g.tick(0.1, Pose(x=16.0, y=14.0, yaw=1.57), _swarm(t=0.1, survivor=sv))  # D2 is within 2 m: would hand over
    assert _kinds(g) == []


def test_callouts_off():
    g = Guidance(world=StubWorld({"D1": (5, 3), "D2": (16, 3)}), callouts=False)
    g.tick(0.0, Pose(x=3.0, y=3.0, yaw=0.0), _swarm())
    assert g.drain_callouts() == []


# ------------------------------------------------------------------ fact check on Gemini's wording
def fire() -> Callout:  # fresh each time: a callout older than STALE_S skips Gemini
    return Callout("D2", "fire_side", "Careful, fire on your right, about 3 metres.", {"side": "right", "distance_m": 3},
                   severity="warn")


FIRE = fire()  # for the fact-check tests (no clock involved)


@pytest.mark.parametrize("line", [
    "Heads up, there's fire on your right, three metres out.",
    "Watch your right side, flames about 3 metres away.",
])
def test_check_line_accepts_a_faithful_rewording(line):
    assert check_line(line, FIRE) == line


@pytest.mark.parametrize("line", [
    "Fire on your left, three metres.",  # wrong side
    "Fire on your right, about five metres.",  # wrong distance
    "Fire on the right, turn back now.",  # invented direction
    "Fire on your right, and debris ahead too.",  # invented hazard
    "Fire on your right, D4 is coming to help.",  # invented drone
    "Fire!",  # too short
])
def test_check_line_rejects_invented_facts(line):
    assert check_line(line, FIRE) is None


# ------------------------------------------------------------------ the agent, against a fake Gemini
WAV = base64.b64encode(b"RIFF....WAVEfmt ").decode()


def _agent(text_reply: str | None, tts_ok: bool = True, delay: float = 0.0) -> tuple[VoiceAgent, list]:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if delay:
            time.sleep(delay)
        if "tts" in request.url.path:
            if not tts_ok:
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [
                {"inlineData": {"mimeType": "audio/wav", "data": WAV}}]}}]})
        if text_reply is None:
            return httpx.Response(503, json={"error": "overloaded"})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text_reply}]}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return VoiceAgent("test-key", client=client, log=False), calls


def _wait(agent: VoiceAgent, n: int = 1, timeout: float = 5.0):
    out, end = [], time.monotonic() + timeout
    while len(out) < n and time.monotonic() < end:
        out += agent.poll()
        time.sleep(0.01)
    return out


def progress() -> Callout:
    return Callout("D2", "progress", "Halfway there. 30 metres to go.", {"metres_left": 30})


def test_agent_rewords_and_voices():
    agent, calls = _agent("Halfway now, thirty metres to go.")
    agent.submit(progress())
    (m,) = _wait(agent)
    agent.close()
    assert m.source == "gemini" and m.text == "Halfway now, thirty metres to go."
    assert m.audio_b64 == WAV and m.audio_mime == "audio/wav" and m.speaker == "D2"
    tts_body = calls[-1][1]
    assert tts_body["contents"][0]["parts"][0]["text"] == m.text  # just the line: no style prompt read aloud
    assert tts_body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"


def test_urgent_lines_skip_the_rewording():
    """A fire warning goes straight to the voice (the fixed line): ~0.5 s sooner."""
    agent, calls = _agent("Heads up, fire on your right, three metres out.")
    agent.submit(fire())
    (m,) = _wait(agent)
    agent.close()
    assert m.source == "template" and m.text == fire().text and m.audio_b64 == WAV and m.severity == "warn"
    assert [c[0] for c in calls] == [c[0] for c in calls if "tts" in c[0]]  # only the TTS model was called


def test_agent_falls_back_to_the_fixed_line_when_gemini_invents_things():
    agent, _ = _agent("Halfway there, fire on your left, 30 metres.")
    agent.submit(progress())
    (m,) = _wait(agent)
    agent.close()
    assert m.source == "template" and m.text == progress().text and m.audio_b64 == WAV  # fixed line, still voiced
    assert agent.stats["rejected"] == 1


def test_agent_survives_api_errors():
    agent, _ = _agent(None, tts_ok=False)
    agent.submit(fire())
    (m,) = _wait(agent)
    agent.close()
    assert m.source == "template" and m.text == FIRE.text and m.audio_b64 is None  # browser speaks it locally


def test_stale_callouts_skip_the_network():
    agent, calls = _agent("Heads up, fire on your right.")
    old = Callout(**{**FIRE.__dict__, "created": time.monotonic() - 10})
    agent.submit(old)
    (m,) = _wait(agent)
    agent.close()
    assert m.source == "template" and calls == []


def test_clear_drops_queued_lines():
    agent, _ = _agent("Heads up, fire on your right, three metres out.", delay=0.3)
    agent.submit(fire())
    agent.submit(fire())
    time.sleep(0.05)
    agent.clear()  # e.g. a snapshot load: lines from before must not play afterwards
    assert _wait(agent, n=1, timeout=1.5) == []
    agent.close()


def test_load_api_key_from_env_file(tmp_path, monkeypatch):
    for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "google_api"):
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / ".env"
    f.write_text('OTHER=1\ngoogle_api = "abc123"\n', encoding="utf-8")
    assert load_api_key(f) == "abc123"
    monkeypatch.setenv("GOOGLE_API_KEY", "fromenv")
    assert load_api_key(f) == "fromenv"


# ------------------------------------------------------------------ the online / offline switch (Act 2)
class FakeAgent:
    """Records what Guidance hands to the voice agent."""

    def __init__(self):
        self.submitted, self.prewarmed, self.cleared = [], [], 0

    def submit(self, c):
        self.submitted.append(c)

    def poll(self):
        return []

    def clear(self):
        self.cleared += 1

    def prewarm(self, lines):
        self.prewarmed += lines


def test_online_voices_the_hud_too_and_prewarms_fixed_lines():
    agent = FakeAgent()
    g = Guidance(world=StubWorld({"D1": (5, 3), "D2": (16, 3)}), voice=agent)
    assert g.voice_mode == "online"
    assert ("HUD", "Stop. Fire ahead. Turn back.") in agent.prewarmed
    assert ("D2", "D2 here, taking over from D1. Follow me.") in agent.prewarmed
    g.tick(0.0, Pose(x=3.0, y=3.0, yaw=0.0), _swarm())
    kinds = [(c.speaker, c.kind, c.text) for c in agent.submitted]
    assert ("HUD", "hud", "Follow the drone.") in kinds  # the HUD line, without the changing distance
    assert ("D1", "intro") in [k[:2] for k in kinds]


def test_offline_never_touches_gemini():
    agent = FakeAgent()
    g = Guidance(world=StubWorld({"D1": (5, 3), "D2": (16, 3)}), voice=agent)
    before = agent.cleared
    g.set_voice_mode("offline")
    assert g.voice_mode == "offline" and agent.cleared == before + 1  # lines voiced online are dropped
    g.tick(0.0, Pose(x=3.0, y=3.0, yaw=0.0), _swarm())
    assert agent.submitted == []  # nothing goes to the network
    said = g.drain_callouts()
    assert [m.kind for m in said] == ["intro"] and said[0].audio_b64 is None  # fixed line, local voice
    g.set_voice_mode("online")
    assert g.voice_mode == "online"


def test_no_key_means_unavailable():
    g = Guidance(world=StubWorld({"D1": (5, 3), "D2": (16, 3)}))
    assert g.voice_mode == "unavailable"
    g.set_voice_mode("online")  # can't go online without a key
    assert g.voice_mode == "unavailable"


def test_turn_warning_once_per_corner():
    g = Guidance(world=StubWorld({"D1": (14, 3), "D2": (16, 3)}))
    g.tick(0.0, Pose(x=3.0, y=3.0, yaw=0.0), _swarm())
    _kinds(g)  # intro
    out = []
    for k in range(1, 80):  # walking towards the corner at (16, 3)
        x = min(10.0 + k * 0.05, 12.5)
        g.tick(k * 0.1, Pose(x=x, y=3.0, yaw=0.0), _swarm(t=k * 0.1))
        out += _kinds(g)
    turns = [o for o in out if o[1] == "turn"]
    assert len(turns) == 1 and turns[0][2].startswith("Left turn in")  # north = CCW = left


def test_cached_lines_play_instantly_and_uncached_hud_warnings_dont_wait():
    agent, calls = _agent("unused")
    agent._cache[("D2", "D2 here, taking over from D1. Follow me.")] = (WAV, "audio/wav")
    agent.submit(Callout("D2", "handover", "D2 here, taking over from D1. Follow me."))
    (m,) = agent.poll()  # no waiting: straight out of the cache
    assert m.audio_b64 == WAV and agent.stats["cached"] == 1
    agent.submit(Callout("HUD", "hud", "Wrong way. Turn around.", severity="warn"))
    (w,) = agent.poll()  # a warning can't wait for TTS: local voice now ...
    assert w.audio_b64 is None and w.text == "Wrong way. Turn around."
    end = time.monotonic() + 3
    while ("HUD", "Wrong way. Turn around.") not in agent._cache and time.monotonic() < end:
        time.sleep(0.01)
    assert ("HUD", "Wrong way. Turn around.") in agent._cache  # ... and voiced for next time
    agent.close()


def test_one_warning_per_fire_even_if_it_spans_cells():
    g = Guidance(world=StubWorld({"D1": (7, 3), "D2": (16, 3)}))
    g.tick(0.0, Pose(x=4.0, y=3.0, yaw=0.0), _swarm())
    _kinds(g)
    fire = [(i, j, "fire") for i in range(35, 41) for j in range(0, 4)]  # a 1.2 x 0.8 m fire south of the route
    out = []
    for k in range(1, 400):
        x = 4.0 + k * 0.02  # walking past it slowly
        g.tick(k * 0.1, Pose(x=x, y=3.0, yaw=0.0), _swarm(t=k * 0.1, delta=fire if k == 1 else ()))
        out += _kinds(g)
    assert len([o for o in out if o[1] == "fire_side"]) == 1
