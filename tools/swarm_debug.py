"""Swarm debug plot: `uv run python tools/swarm_debug.py --map maps/test --seconds 120 [--world fake|mujoco]`.

Runs the sim loop headless with mock perception, then plots the belief map (walls / free / searched /
blocked_new), the area split, every drone's path and the coverage-over-time curve. Owned by S.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sgr import config  # noqa: E402
from sgr.contracts import PriorMap, Scenario  # noqa: E402
from sgr.loop import SimLoop  # noqa: E402
from sgr.perception import Perception  # noqa: E402
from sgr.swarm import Swarm  # noqa: E402
from sgr.swarm.belief import BLOCKED, WALL  # noqa: E402


def run(map_dir: Path, seconds: float, world_kind: str):
    sc = Scenario.load(map_dir)
    if world_kind == "fake":
        from sgr.world import FakeWorld as W
    else:
        from sgr.world import World as W
    world = W(map_dir, sc)
    swarm = Swarm(PriorMap.from_map(map_dir, sc), list(sc.drones.ids))
    perception = Perception(mode="mock", ground_truth=world.ground_truth, line_of_sight=world.line_of_sight)
    loop = SimLoop(world, perception, swarm, sc, bus=None, render=False, realtime=False)
    paths: dict[str, list[tuple[float, float]]] = {d: [] for d in sc.drones.ids}
    cov: list[tuple[float, float]] = []
    t90 = None
    wall0 = time.perf_counter()
    for _ in range(int(round(seconds / config.CONTROL_DT))):
        loop.tick()
        for d in sc.drones.ids:
            if d not in loop.dead:
                p = world.drone_pose(d)
                paths[d].append((p.x, p.y))
        c = swarm.coverage()
        cov.append((loop.t, c))
        if t90 is None and c >= 0.9:
            t90 = loop.t
    return sc, swarm, loop, paths, cov, t90, time.perf_counter() - wall0


def plot(sc, swarm, paths, cov, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    b = swarm.belief
    img = np.full(b.state.shape, 1, np.uint8)  # 1 free (unsearched)
    img[b.searched] = 2
    img[b.state == WALL] = 0
    img[b.state == BLOCKED] = 3
    cmap = ListedColormap(["#2b2b2b", "#f2f2f2", "#bfe3c4", "#e8443a"])
    ext = (b.origin[0], b.origin[0] + b.w * b.res, b.origin[1], b.origin[1] + b.h * b.res)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1]})
    ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    labels = np.full(b.state.shape, np.nan)
    for k, a in enumerate(swarm.areas.values()):
        labels[a.mask] = k
    ax.contour(np.arange(b.w) * b.res + b.res / 2, np.arange(b.h) * b.res + b.res / 2, labels,
               levels=np.arange(len(swarm.areas)) + 0.5, colors="#7a7a7a", linewidths=0.5)
    for n in sc.graph.nodes:
        ax.annotate(n.id, (n.x, n.y), fontsize=8, color="#555555", ha="center")
    for d, pts in paths.items():
        if pts:
            p = np.asarray(pts)
            c = config.DRONE_COLOURS.get(d, "#000000")
            ax.plot(p[:, 0], p[:, 1], color=c, lw=1.4, label=d, path_effects=None)
            ax.plot(*p[-1], "o", color=c, mec="black", ms=6)
    ax.set_title(f"Belief map, coverage {swarm.coverage():.1%} (green = searched, red = blocked_new)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    t, c = np.asarray(cov).T
    ax2.plot(t, c * 100, color="#3373ff")
    ax2.axhline(90, color="#999999", lw=0.8, ls="--")
    ax2.set_xlabel("sim time (s)")
    ax2.set_ylabel("searched (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Coverage")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=Path, default=ROOT / "maps" / "test")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--world", choices=["fake", "mujoco"], default="mujoco")
    ap.add_argument("--out", type=Path, default=Path("swarm_debug.png"))
    ap.add_argument("--trace", action="store_true", help="print the decision trace")
    a = ap.parse_args()
    sc, swarm, loop, paths, cov, t90, wall = run(a.map, a.seconds, a.world)
    if a.trace:
        for r in loop.trace:
            print(f"{r.t:7.1f} {r.drone:5} {r.action:16} {r.reason}")
    plot(sc, swarm, paths, cov, a.out)
    print(f"coverage {swarm.coverage():.1%} at t={loop.t:.1f}s; t(90%)={t90}; wall {wall:.1f}s; plot -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
