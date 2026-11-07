"""tools/replay.py (§12): the whole demo hands-free, with scripted judge actions.

FakeWorld on maps/test keeps this fast (~5 s): Act 1 -> judge debris click on the route -> Undo ->
Jump to Act 2 (from the snapshot saved once the chain is at its posts) -> the autopilot firefighter
reaches the survivor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import replay  # noqa: E402


def test_default_demo_script_runs_end_to_end(test_map, tmp_path, capsys) -> None:
    script = json.loads((ROOT / "tools" / "replay_demo.json").read_text(encoding="utf-8"))
    act1_click = next(s for s in script["steps"] if s["do"] == "click" and "x" in s)
    act1_click.pop("x"), act1_click.pop("y")  # the demo pins a marshgate spot; on maps/test click past the
    act1_click["on_route"] = 0.08  # entrance, where the lead drone sees it (e1 is the only way in: NO_ROUTE)
    act1_click["after"], act1_click["wait"] = "chain_ready", 2  # the lead is at the entrance by then
    path = tmp_path / "demo_test_map.json"
    path.write_text(json.dumps(script), encoding="utf-8")
    code = replay.main([str(path), "--map", str(test_map), "--fake", "-q", "--snapshots", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    for beat in ("SURVIVOR_CONFIRMED", "chain_ready", "MISSION_COMPLETE", "summary"):
        assert f"ok   {beat}" in out
    assert "REPLAY click" in out and "REPLAY undo" in out and "REPLAY act2" in out
    assert out.count("click noticed") == 2 and "WARNING" not in out  # both judge clicks got a reaction
    assert (tmp_path / "act2_start.pkl").is_file()


def test_missing_beat_fails_the_replay(test_map, tmp_path, capsys) -> None:
    script = {"max_t": 5, "expect": ["MISSION_COMPLETE"], "steps": []}
    path = tmp_path / "s.json"
    path.write_text(json.dumps(script), encoding="utf-8")
    code = replay.main([str(path), "--map", str(test_map), "--fake", "-q", "--snapshots", str(tmp_path)])
    assert code == 1 and "MISS MISSION_COMPLETE" in capsys.readouterr().out


def test_unnoticed_click_fails_the_replay(test_map, tmp_path, capsys) -> None:
    """A click no drone can see is a silent beat live: the replay must fail loudly (W's report)."""
    script = {"max_t": 30, "expect": [], "steps": [{"t": 2, "do": "click", "mode": "debris", "x": 26.0, "y": 13.0}]}
    path = tmp_path / "s.json"
    path.write_text(json.dumps(script), encoding="utf-8")
    code = replay.main([str(path), "--map", str(test_map), "--fake", "-q", "--snapshots", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1 and "not noticed" in out


def test_along_and_lookahead() -> None:
    pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    assert replay._along(pts, 0.5) == (10.0, 0.0)
    assert replay._along(pts, 0.25) == (5.0, 0.0)
    dx, dy = replay._lookahead(pts, (9.5, 0.2), 1.2)  # just before the corner: aim round it, not across
    assert dx > 0 and dy > 0
