#!/usr/bin/env python3
"""Check a scenario with the real swarm planner (sgr.swarm.planner A* on the 0.2 m belief grid).

    uv run python tools/scan_check_scenario.py maps/<name> [--png out.png]

Places debris the way sgr.world does (ceil(edge.width / 1 m) 1x1 m blocks across the edge at
`at`) and fires as discs (size/2 radius), then reports:
  * the route entrance -> survivor with nothing known (what the swarm plans first),
  * the route once every debris pile and fire is known (must exist: survivor reachable),
  * for each debris pile: whether it sits on the first route (= forces a re-plan), the detour
    length, and how close a drone can get before it (LiDAR range 6 m),
  * fire: distance from the route, temperature on the route (heat model), D3-area hint,
  * decoy: distance to the nearest wall (must be <= 3 m),
  * flare target / time present.
Writes an optional PNG with both routes. Exit code 1 if a hard requirement fails.

--act1 also runs Act 1 headless (real World + mock perception + real Swarm, like the smoke
test, ~10 s): key trace actions with their first time, the flare target's temperature
before the flare, where it is at flare.t, and whether the survivor is still reachable with
the flare-up fire placed there (that fire lands ON the drone, wherever it happens to be).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgr import config  # noqa: E402
from sgr.swarm.belief import BLOCKED, FIRE, BeliefGrid  # noqa: E402
from sgr.swarm.planner import astar  # noqa: E402
from sgr.world.heat import temperature_at  # noqa: E402

DEBRIS = 1.0


def run_act1(map_dir: Path, sd: dict, ticks: int = 1500, flare: bool = True):
    """Headless Act 1. Returns (loop, samples) with samples[t] = {drone: (x, y, temp_c)} every 1 s."""
    from sgr.contracts import PriorMap, Scenario
    from sgr.guidance import Guidance
    from sgr.loop import SimLoop
    from sgr.perception import Perception
    from sgr.swarm import Swarm
    from sgr.world import World
    sd = dict(sd)
    if not flare:
        sd["flare"] = None
    sc = Scenario.model_validate(sd)
    w = World(map_dir, sc)
    loop = SimLoop(w, Perception(mode="mock", ground_truth=w.ground_truth, line_of_sight=w.line_of_sight),
                   Swarm(PriorMap.from_map(map_dir, sc), list(sc.drones.ids)), sc, guidance=Guidance(),
                   bus=None, render=False, realtime=False)
    samples = {}
    for k in range(ticks):
        loop.tick()
        if k % 10 == 9:
            samples[round(loop.t)] = {d: (w.drone_pose(d).x, w.drone_pose(d).y, w.temperature(d))
                                      for d in sc.drones.ids}
    return loop, samples


def plen(path) -> float:
    return sum(math.dist(a, b) for a, b in zip(path, path[1:])) if path else math.inf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir", type=Path)
    ap.add_argument("--png", type=Path)
    ap.add_argument("--act1", action="store_true", help="also run Act 1 headless with mock perception (~10 s)")
    a = ap.parse_args()
    sc = json.loads((a.map_dir / "scenario.json").read_text())
    grid = cv2.imread(str(a.map_dir / "grid.png"), cv2.IMREAD_GRAYSCALE)
    pitch = float(sc["grid"]["resolution"])
    free = (grid > 127)[::-1]                       # iy = 0 at y = 0
    res = config.BELIEF_RES
    k = max(1, int(round(res / pitch)))
    h, w = free.shape[0] // k, free.shape[1] // k
    wall = ~free[: h * k, : w * k].reshape(h, k, w, k).all(axis=(1, 3))
    ent = (sc["entrance"]["x"], sc["entrance"]["y"])
    sv = (sc["survivor"]["x"], sc["survivor"]["y"])

    def fresh():
        return BeliefGrid(wall.copy(), k * pitch, (0.0, 0.0), ent)

    nodes = {n["id"]: (n["x"], n["y"]) for n in sc["graph"]["nodes"]}
    edges = {e["id"]: e for e in sc["graph"]["edges"]}
    jj, ii = np.mgrid[0:h, 0:w]
    cx, cy = (ii + 0.5) * k * pitch, (jj + 0.5) * k * pitch

    def debris_mask(c):
        e = edges[c["edge"]]
        (ax, ay), (bx, by) = nodes[e["a"]], nodes[e["b"]]
        px, py = ax + c["at"] * (bx - ax), ay + c["at"] * (by - ay)
        ang = math.atan2(by - ay, bx - ax)
        ux, uy, nx, ny = math.cos(ang), math.sin(ang), -math.sin(ang), math.cos(ang)
        n = max(1, math.ceil(e.get("width", 1.8) / DEBRIS))
        m = np.zeros((h, w), bool)
        for j in range(n):
            off = (j - (n - 1) / 2) * DEBRIS
            qx, qy = px + off * nx, py + off * ny
            du, dv = (cx - qx) * ux + (cy - qy) * uy, (cx - qx) * nx + (cy - qy) * ny
            m |= (np.abs(du) <= DEBRIS / 2) & (np.abs(dv) <= DEBRIS / 2)
        return m, (px, py)

    def plan(marks):
        b = fresh()
        for mask, state in marks:
            b.state[mask & (b.state != 0)] = state
        b.version += 1
        return astar(b, ent, sv)

    fails = []
    base = plan([])
    print(f"first route (nothing known): {plen(base):.1f} m" if base else "first route: NONE")
    if not base:
        fails.append("survivor unreachable even with no debris/fire")
    fire_marks = [(((cx - f["x"]) ** 2 + (cy - f["y"]) ** 2) <= (f.get("size", 1.5) / 2) ** 2, FIRE)
                  for f in sc.get("fires", [])]
    deb = [(c, *debris_mask(c)) for c in sc.get("compromised", [])]
    full = plan(fire_marks + [(m, BLOCKED) for _, m, _ in deb])
    print(f"route with all debris + fire known: {plen(full):.1f} m" if full else
          "route with all debris + fire known: NONE  <-- survivor unreachable")
    if not full:
        fails.append("survivor unreachable once debris/fire are known")

    base_arr = np.array(base) if base else np.zeros((0, 2))
    on_route = []
    for c, m, (px, py) in deb:
        e = edges[c["edge"]]
        hits = bool(base) and any(m[int(y / (k * pitch)), int(x / (k * pitch))] for x, y in base)
        alone = plan(fire_marks + [(m, BLOCKED)])
        dmin = float(np.min(np.hypot(base_arr[:, 0] - px, base_arr[:, 1] - py))) if len(base_arr) else math.inf
        tag = (f"ON the first route -> re-plan, detour {plen(alone) - plen(base):+.1f} m" if hits and alone else
               "ON the first route and NO way around -> 'No safe route'" if hits else
               f"off route ({dmin:.1f} m from it; a drone exploring {e['a']}-{e['b']} finds it)")
        print(f"  debris {c['edge']} ({e['a']}-{e['b']}) at ({px:.1f}, {py:.1f}): {tag}")
        if hits:
            on_route.append(c["edge"])
        if hits and not alone:
            fails.append(f"debris {c['edge']} blocks the only way")
    if not on_route:
        print("  WARNING: no debris on the first route -> no LiDAR re-plan beat (spec §5)")

    fires_xy = [(f["x"], f["y"]) for f in sc.get("fires", [])]
    amb = sc.get("heat", {}).get("ambient_c", 20)
    lim = sc.get("heat", {}).get("limit_c", 70)
    route = full or base
    for f in sc.get("fires", []):
        d = min(math.dist((f["x"], f["y"]), p) for p in route) if route else math.inf
        print(f"  fire ({f['x']:.1f}, {f['y']:.1f}) size {f.get('size', 1.5)}: route passes {d:.1f} m away")
    if fires_xy and route:
        # heat adds up over all fires (sgr.world.heat sums them)
        tmax, at = max((temperature_at(x, y, fires_xy, amb), (x, y)) for x, y in route)
        print(f"  heat on the route (all fires together): max {tmax:.0f} C at ({at[0]:.1f}, {at[1]:.1f}) "
              f"(retreat at {lim} C)" + ("  <-- drones on the route will overheat" if tmax >= lim else ""))
        if tmax >= lim:
            fails.append(f"route heat {tmax:.0f} C >= {lim} C")
    if not sc.get("fires"):
        print("  WARNING: no fires")
    dc = sc.get("decoy")
    if dc:
        dist = cv2.distanceTransform((grid > 127).astype(np.uint8), cv2.DIST_L2, 5) * pitch
        r, c_ = grid.shape[0] - 1 - int(dc["y"] / pitch), int(dc["x"] / pitch)
        dw = float(dist[r, c_])
        print(f"  decoy ({dc['x']:.1f}, {dc['y']:.1f}): {dw:.1f} m from the nearest wall" +
              ("" if dw <= 3 else "  <-- must be <= 3 m"))
        if dw > 3:
            fails.append("decoy > 3 m from a wall")
    else:
        print("  WARNING: no decoy")
    fl = sc.get("flare")
    print(f"  flare: {fl['target']} at t={fl['t']} s, beats.d3_area_node={sc.get('beats', {}).get('d3_area_node')}"
          if fl else "  WARNING: no flare")

    if a.png:
        td = a.map_dir / "plan_sheet.png"
        img = cv2.imread(str(td)) if td.exists() else cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
        H = img.shape[0]
        s = img.shape[1] / (grid.shape[1] * pitch)
        P = lambda x, y: (int(x * s), int(H - y * s))
        for _, m, _ in deb:
            img[cv2.resize(m[::-1].astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0] = (40, 40, 230)
        for route, col, th in ((base, (255, 255, 255), 10), (full, (0, 200, 255), 6)):
            if route:
                cv2.polylines(img, [np.array([P(*p) for p in route], np.int32)], False, col, th, cv2.LINE_AA)
        cv2.imwrite(str(a.png), img)
        print(f"wrote {a.png} (white = first route, amber = route with everything known)")
    if a.act1:
        print("Act 1 (headless, mock perception, 150 s):")
        loop, smp = run_act1(a.map_dir, sc)
        first = {}
        for r in loop.trace:
            first.setdefault(r.action, (r.t, r.drone, r.reason))
        for act in ("ASSIGN_AREA", "MARK_BLOCKED", "REPLAN", "CONFIRM_HAZARD", "REJECT_DETECTION", "RETREAT",
                    "DRONE_LOST", "SURVIVOR_CONFIRMED", "CHAIN_POSITION"):
            n = sum(r.action == act for r in loop.trace)
            if act in first:
                t, d, why = first[act]
                print(f"  {act:20s} x{n:<4d} first at {t:5.1f} s ({d}: {why[:70]})")
            else:
                print(f"  {act:20s} --")
        others = sorted({r.action for r in loop.trace} - set(first) | (set(first) - {
            "ASSIGN_AREA", "MARK_BLOCKED", "REPLAN", "CONFIRM_HAZARD", "REJECT_DETECTION", "RETREAT",
            "DRONE_LOST", "SURVIVOR_CONFIRMED", "CHAIN_POSITION"}))
        if others:
            print("  other actions:", ", ".join(others))
        fl = sc.get("flare")
        if fl:
            tgt, tf = fl["target"], fl["t"]
            pre = [smp[t][tgt] for t in sorted(smp) if tf - 20 <= t < tf]
            peak = max(p[2] for p in pre) if pre else float("nan")
            at = smp.get(round(tf)) or smp[min(smp, key=lambda t: abs(t - tf))]
            print(f"  flare: {tgt} peaks at {peak:.0f} C in the 20 s before t={tf} (retreat at "
                  f"{sc.get('heat', {}).get('limit_c', 70)} C); at the flare it is at ({at[tgt][0]:.1f}, {at[tgt][1]:.1f})")
            if loop.flare_fire is not None:
                ff = {"x": loop.flare_fire.x, "y": loop.flare_fire.y, "size": 1.5}
                fm = [(((cx - f["x"]) ** 2 + (cy - f["y"]) ** 2) <= (f.get("size", 1.5) / 2) ** 2, FIRE)
                      for f in sc.get("fires", []) + [ff]]
                after = plan(fm + [(m, BLOCKED) for _, m, _ in deb])
                print(f"  survivor route after the flare-up fire at ({ff['x']:.1f}, {ff['y']:.1f}): "
                      + (f"{plen(after):.1f} m" if after else "NONE  <-- flare fire cuts off the survivor"))
                if not after:
                    fails.append("flare-up fire cuts off the survivor")
            if peak < 60:
                print("  WARNING: no heat lead-up before the flare (target never gets close to a fire)")
        for need in ("SURVIVOR_CONFIRMED", "REPLAN", "DRONE_LOST"):
            if need not in first:
                fails.append(f"Act 1: no {need}")
    for f in fails:
        print("FAIL:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
