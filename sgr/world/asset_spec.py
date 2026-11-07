"""One definition of what the drone cameras see, shared with the dashboard's 3D view.

The simulation (sgr/world/shapes.py) is the reference: this module parses the exact MJCF
snippets MuJoCo renders and writes them as plain JSON primitives, so the 3D view draws the
same fire, debris and survivor. tests/test_asset_spec.py fails if the committed JSON drifts.

    uv run python -m sgr.world.asset_spec            # rewrite web/src/scene3d/simAssets.json

Units: metres, sim frame (z up, x forward). Primitive types, MuJoCo semantics:
  ellipsoid size = radii; sphere size = [r]; cylinder size = [r, half-height] along z;
  box size = half-extents; capsule = fromto [x1 y1 z1 x2 y2 z2] + size [r].
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from . import shapes

OUT = Path(__file__).resolve().parents[2] / "web" / "src" / "scene3d" / "simAssets.json"
_EMISSION = {"mat_fire": shapes.FIRE_EMISSION, "mat_decoy": shapes.DECOY_EMISSION, "mat_matte": 0.0}


def _floats(s: str | None) -> list[float] | None:
    return None if s is None else [round(float(v), 4) for v in s.split()]


def _parts(xml: str) -> list[dict]:
    root = ET.fromstring(f"<root>{xml}</root>")
    out = []
    for g in root.iter("geom"):
        part = {"type": g.get("type", "sphere"), "size": _floats(g.get("size")),
                "rgba": _floats(g.get("rgba")) or [0.5, 0.5, 0.5, 1.0],
                "emission": _EMISSION.get(g.get("material", ""), 0.0)}
        if g.get("fromto"):
            part["fromto"] = _floats(g.get("fromto"))
        else:
            part["pos"] = _floats(g.get("pos")) or [0.0, 0.0, 0.0]
        out.append(part)
    return out


def spec() -> dict:
    return {
        "version": 1,
        "source": "sgr/world/shapes.py (what the MuJoCo drone cameras render)",
        "units": "metres, sim frame: z up, x forward",
        "fire": {"parts": _parts(shapes.fire_cluster_geoms("fire"))},
        "debris": {"parts": _parts(shapes.debris_geom("debris"))},
        "survivor": {"parts": _parts(shapes.survivor("survivor", 0.0, 0.0, 0.0))},
        "drone": {"model": "dressing/drone.glb", "scale": shapes.DRONE_MODEL_SCALE},
    }


def dumps() -> str:
    return json.dumps(spec(), indent=1) + "\n"


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    out.write_text(dumps(), encoding="utf-8", newline="\n")
    print(f"wrote {out}")
