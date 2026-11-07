"""P-local scratch map so World can be built and tested before S's maps/test exists.

30 x 20 m, 0.1 m/px, 1.8 m corridors, 3 branches off the entrance corridor,
one loop, one dead-end room. Written to a directory of your choice; never into
maps/ (W/S own it).

    python -m sgr.world.testmap OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RES = 0.1
W_M, H_M = 30.0, 20.0


def _grid() -> np.ndarray:
    h, w = int(H_M / RES), int(W_M / RES)
    g = np.zeros((h, w), np.uint8)  # all wall

    def carve(x0, y0, x1, y1):
        c0, c1 = int(round(x0 / RES)), int(round(x1 / RES))
        r0, r1 = h - int(round(y1 / RES)), h - int(round(y0 / RES))
        g[r0:r1, c0:c1] = 255

    carve(0.5, 2.1, 25.0, 3.9)  # entrance corridor, y = 3
    for x in (8.0, 16.0, 24.0):  # three branches north
        carve(x - 0.9, 2.1, x + 0.9, 16.9)
    carve(7.1, 15.1, 24.9, 16.9)  # top corridor -> loop
    carve(1.0, 10.0, 6.0, 16.0)  # dead-end room (west)
    carve(6.0, 12.4, 7.1, 13.6)  # its door
    carve(26.0, 10.0, 29.5, 16.0)  # survivor room (east)
    carve(24.9, 12.4, 26.0, 13.6)  # its door
    return g


def _scenario() -> dict:
    nodes = [
        ("n1", 2.0, 3.0, "entrance"), ("n2", 8.0, 3.0, "junction"), ("n3", 16.0, 3.0, "junction"),
        ("n4", 24.0, 3.0, "junction"), ("n5", 8.0, 13.0, "junction"), ("n6", 8.0, 16.0, "junction"),
        ("n7", 16.0, 16.0, "junction"), ("n8", 24.0, 16.0, "junction"), ("n9", 24.0, 13.0, "junction"),
        ("n10", 3.5, 13.0, "room"), ("n11", 27.5, 13.0, "room"),
    ]
    edges = [("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n2", "n5"), ("n5", "n6"), ("n6", "n7"),
             ("n3", "n7"), ("n7", "n8"), ("n4", "n9"), ("n9", "n8"), ("n5", "n10"), ("n9", "n11")]
    return {
        "version": 1,
        "name": "p-scratch",
        "grid": {"file": "grid.png", "resolution": RES, "origin": [0, 0]},
        "scan": {"file": "scan.glb", "transform": "transform.json"},
        "entrance": {"x": 2.0, "y": 3.0, "yaw": 0.0},
        "graph": {
            "nodes": [{"id": i, "x": x, "y": y, "kind": k} for i, x, y, k in nodes],
            "edges": [{"id": f"e{k + 1}", "a": a, "b": b, "width": 1.8} for k, (a, b) in enumerate(edges)],
        },
        "compromised": [{"edge": "e9", "at": 0.5, "kind": "debris"}],  # n4-n9, on the naive route
        "fires": [{"x": 12.0, "y": 16.0, "size": 1.5}],
        "survivor": {"x": 27.5, "y": 13.0, "yaw": 1.2},
        "decoy": {"x": 8.6, "y": 6.2, "kind": "orange_panel"},
        "drones": {"ids": ["D1", "D2", "D3", "D4"],
                   "start": [[2, 2.5], [2, 3.5], [2.5, 2.5], [2.5, 3.5]], "altitude": 1.6},
        "flare": {"t": 65.0, "target": "D3"},
        "heat": {"limit_c": 70.0, "ambient_c": 20.0},
        "beats": {"decoy_drone": "D2", "d3_area_node": "n7"},
    }


def write(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_grid()).save(out / "grid.png")
    (out / "scenario.json").write_text(json.dumps(_scenario(), indent=2), encoding="utf-8")
    return out


def write_studio(out_dir: str | Path) -> Path:
    """One open 15 x 10 m room for detector tests.

    D1 looks at the fire, D2 at the survivor (broadside) and D3 at the decoy
    on the east wall, each from 3 m. D4 (yaw set by the test) faces the north wall.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    h, w = int(11 / RES), int(16 / RES)
    g = np.zeros((h, w), np.uint8)
    g[int(0.5 / RES):h - int(0.5 / RES), int(0.5 / RES):w - int(0.5 / RES)] = 255
    Image.fromarray(g).save(out / "grid.png")
    sc = _scenario()
    sc.update({
        "name": "p-studio",
        "entrance": {"x": 8.0, "y": 1.5, "yaw": 0.0},
        "graph": {"nodes": [{"id": "n1", "x": 8.0, "y": 1.5, "kind": "entrance"},
                            {"id": "n2", "x": 8.0, "y": 9.0, "kind": "room"}],
                  "edges": [{"id": "e1", "a": "n1", "b": "n2", "width": 1.8}]},
        "compromised": [],
        "fires": [{"x": 4.0, "y": 3.0, "size": 1.5}],
        "survivor": {"x": 4.0, "y": 8.5, "yaw": 1.5708},
        "decoy": {"x": 15.3, "y": 5.5, "kind": "orange_panel"},
        "drones": {"ids": ["D1", "D2", "D3", "D4"],
                   "start": [[1.0, 3.0], [1.0, 8.5], [12.5, 5.5], [9.0, 9.8]], "altitude": 1.6},
    })
    (out / "scenario.json").write_text(json.dumps(sc, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(write(sys.argv[1] if len(sys.argv) > 1 else "p_scratch_map"))
