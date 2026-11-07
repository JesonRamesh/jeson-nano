"""Snapshots (owned by S). §8.6.

A snapshot is a pickle of: World.get_state() (full physics + pools), Perception.get_state()
(detections, blacklist), Swarm.get_state() (blackboard, belief, agents), Guidance.get_state(),
the decision trace, the loop's own state (time, act, dead drones, flare, clicks) and the scenario name.
The renderer and threads are never pickled.

Files: snapshots/<name>.pkl. `act2_start` can also come from the committed maps/<name>/act2_start.pkl
(produced at the Day-1 gate), so "Jump to Act 2" works on a fresh checkout.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_VERSION = 1


class SnapshotError(RuntimeError):
    pass


# ---------------------------------------------------------------------- files
def save(name: str, payload: dict[str, Any], directory: Path = SNAPSHOT_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pkl"
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(path)  # atomic: a crash mid-write never leaves a half snapshot
    return path


def load(name: str, directory: Path = SNAPSHOT_DIR) -> dict[str, Any]:
    return load_path(Path(directory) / f"{name}.pkl")


def load_path(path: Path) -> dict[str, Any]:
    payload = pickle.loads(Path(path).read_bytes())
    if not isinstance(payload, dict) or payload.get("version") != SNAPSHOT_VERSION:
        raise SnapshotError(f"{path}: not a v{SNAPSHOT_VERSION} snapshot")
    return payload


def find(name: str, directory: Path | None, map_dir: Path | None) -> Path | None:
    """snapshots/<name>.pkl first, then the committed maps/<map>/<name>.pkl."""
    for d in (directory, map_dir):
        if d is not None and (Path(d) / f"{name}.pkl").is_file():
            return Path(d) / f"{name}.pkl"
    return None


# ---------------------------------------------------------------------- loop state
def capture(loop: Any) -> dict[str, Any]:
    """Everything needed to resume `loop` exactly where it is (between ticks)."""
    g = loop.guidance
    return {
        "version": SNAPSHOT_VERSION,
        "scenario": loop.scenario.name,
        "loop": {
            "t": loop.t,
            "ticks": loop.ticks,
            "act": loop.act,
            "dead": sorted(loop.dead),
            "flared": loop._flared,
            "flare_victim": loop.flare_victim,
            "flare_fire": loop.flare_fire.model_dump() if loop.flare_fire else None,
            "click_stack": [h.model_dump() for h in loop.click_stack],
            "frame_seq": loop._frame_seq,
            "cam_rr": loop._cam_rr,
        },
        "trace": [r.model_dump() for r in loop.trace],
        "world": loop.world.get_state(),
        "perception": loop.perception.get_state(),
        "swarm": loop.swarm.get_state(),
        "guidance": g.get_state() if g is not None and hasattr(g, "get_state") else None,
    }


def restore(loop: Any, payload: dict[str, Any]) -> None:
    from sgr.contracts import Handle, TraceRow

    if payload.get("scenario") != loop.scenario.name:
        raise SnapshotError(f"snapshot is for map '{payload.get('scenario')}', running '{loop.scenario.name}'")
    loop.world.set_state(payload["world"])
    loop.perception.set_state(payload["perception"])
    loop.swarm.set_state(payload["swarm"])
    if payload.get("guidance") is not None and loop.guidance is not None and hasattr(loop.guidance, "set_state"):
        loop.guidance.set_state(payload["guidance"])
    s = payload["loop"]
    loop.t, loop.ticks, loop.act = s["t"], s["ticks"], s["act"]
    loop.dead = set(s["dead"])
    loop._flared, loop.flare_victim = s["flared"], s["flare_victim"]
    loop.flare_fire = Handle(**s["flare_fire"]) if s["flare_fire"] else None
    loop.click_stack = [Handle(**h) for h in s["click_stack"]]
    loop._frame_seq, loop._cam_rr = s["frame_seq"], s["cam_rr"]
    loop.trace = [TraceRow(**r) for r in payload["trace"]]
    loop.swarm_state = None
