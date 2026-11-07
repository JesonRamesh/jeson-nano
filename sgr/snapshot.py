"""Snapshots (owned by S). §8.6.

STUB: save/load a dict of pickled parts. The full version stores World.get_state(),
Perception state, Swarm blackboard + agents, trace and scenario time.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

SNAPSHOT_DIR = Path("snapshots")


def save(name: str, payload: dict[str, Any], directory: Path = SNAPSHOT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pkl"
    path.write_bytes(pickle.dumps(payload))
    return path


def load(name: str, directory: Path = SNAPSHOT_DIR) -> dict[str, Any]:
    return pickle.loads((directory / f"{name}.pkl").read_bytes())
