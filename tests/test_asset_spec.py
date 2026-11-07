"""The dashboard's 3D view draws the same objects as the MuJoCo drone cameras (P)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sgr.world import asset_spec, shapes


def test_committed_sim_assets_match_the_simulation():
    """web/src/scene3d/simAssets.json is generated from shapes.py; regenerate it after changing a shape:
    uv run python -m sgr.world.asset_spec"""
    assert asset_spec.OUT.is_file(), "missing web/src/scene3d/simAssets.json"
    committed = json.loads(asset_spec.OUT.read_text(encoding="utf-8"))
    assert committed == json.loads(asset_spec.dumps()), "simAssets.json is stale: run `uv run python -m sgr.world.asset_spec`"


def test_spec_has_every_object():
    s = asset_spec.spec()
    assert {p["type"] for p in s["fire"]["parts"]} >= {"ellipsoid", "cylinder"}
    assert s["debris"]["parts"][0]["size"] == [0.5, 0.5, 1.0]  # 1 x 1 x 2 m block
    assert any(p["type"] == "capsule" and "fromto" in p for p in s["survivor"]["parts"])
    assert s["drone"]["scale"] == shapes.DRONE_MODEL_SCALE


def test_drone_model_is_visual_only(monkeypatch):
    """The shared drone.glb is drawn in the cameras, but drone physics (mass, inertia) is unchanged."""
    pytest.importorskip("mujoco")
    if not shapes.drone_model_parts():
        pytest.skip("drone.glb not available in this checkout")
    from sgr.contracts import Scenario
    from sgr.world import testmap
    from sgr.world.world import World

    import tempfile
    from pathlib import Path

    d = testmap.write(Path(tempfile.mkdtemp()) / "map")
    with_model = World(d, Scenario.load(d), scan_vis=False)
    b = with_model.model.body("D1").id
    model_geoms = [g for g in range(with_model.model.ngeom)
                   if with_model.model.geom_bodyid[g] == b and with_model.model.geom(g).name.startswith("D1_m_")]
    assert len(model_geoms) == len(shapes.drone_model_parts())
    assert all(with_model.model.geom_group[g] == 2 and with_model.model.geom_contype[g] == 0 for g in model_geoms)

    monkeypatch.setattr(shapes, "drone_model_parts", lambda: ())  # the primitive quadcopter only
    plain = World(d, Scenario.load(d), scan_vis=False)
    pb = plain.model.body("D1").id
    assert with_model.model.body_mass[b] == pytest.approx(plain.model.body_mass[pb])
    assert np.allclose(with_model.model.body_inertia[b], plain.model.body_inertia[pb])
