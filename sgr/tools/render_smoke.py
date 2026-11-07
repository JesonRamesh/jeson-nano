"""Render smoke test: `uv run python -m sgr.tools.render_smoke` -> render_smoke.png.

Checks that offscreen MuJoCo rendering works on this machine (spec §1.3/§1.4).
Linux: set MUJOCO_GL=egl (GPU) or osmesa first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

# Camera looks along +x, tilted 15 deg down (like a drone cam, §7.1).
XML = """
<mujoco model="render_smoke">
  <visual>
    <global offwidth="320" offheight="240"/>
    <quality shadowsize="0"/>
  </visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom type="plane" size="4 4 0.1" rgba="0.25 0.25 0.25 1"/>
    <geom type="box" pos="2.5 0 1.3" size="0.1 2 1.3" rgba="0.75 0.7 0.6 1"/>
    <geom type="cylinder" pos="1.6 -0.6 0.3" size="0.2 0.3" rgba="1 0.55 0.1 1"/>
    <geom type="capsule" fromto="1.4 0.4 0.15 1.9 0.7 0.15" size="0.12" rgba="0.7 1 0.2 1"/>
    <camera name="cam" pos="0 0 1.6" xyaxes="0 -1 0 0.259 0 0.966" fovy="90"/>
  </worldbody>
</mujoco>
"""


def main(out: str = "render_smoke.png") -> int:
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=240, width=320)
    try:
        renderer.update_scene(data, camera="cam")
        img = renderer.render()
    finally:
        renderer.close()
    if img.shape != (240, 320, 3) or float(np.mean(img)) < 5.0:
        print(f"render_smoke: FAILED (shape={img.shape}, mean={np.mean(img):.1f})")
        return 1
    Image.fromarray(img).save(out)
    print(f"render_smoke: OK mujoco {mujoco.__version__}, 320x240, mean={np.mean(img):.1f} -> {Path(out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
