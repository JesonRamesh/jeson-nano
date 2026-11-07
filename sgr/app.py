"""Entry point (owned by S): `uv run sgr --map maps/test --perception mock --speed 1`."""

from __future__ import annotations

import argparse
from pathlib import Path

from sgr.bus import Bus
from sgr.contracts import PriorMap, Scenario
from sgr.guidance import Guidance
from sgr.loop import SimLoop
from sgr.perception import Perception
from sgr.swarm import Swarm
from sgr.world import World


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sgr", description="Swarm-Guided Rescue simulator")
    p.add_argument("--map", type=Path, default=Path("maps/test"), help="map directory")
    p.add_argument("--perception", choices=["mock", "remote", "local"], default="mock")
    p.add_argument("--speed", type=float, default=1.0, help="sim speed multiplier (2.5 for the video)")
    p.add_argument("--no-render", action="store_true", help="skip camera rendering (mock frames)")
    p.add_argument("--max-ticks", type=int, default=None, help="stop after N control ticks (testing)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = Scenario.load(args.map)
    world = World(args.map, scenario)
    perception = Perception(mode=args.perception, ground_truth=world.ground_truth)
    swarm = Swarm(PriorMap.from_map(args.map, scenario), list(scenario.drones.ids))
    bus = Bus()
    # TODO(D): start sgr.server on a background thread here, passing `bus` (§3).
    loop = SimLoop(
        world,
        perception,
        swarm,
        scenario,
        guidance=Guidance(),
        bus=bus,
        speed=args.speed,
        render=not args.no_render,
    )
    print(f"sgr: map={args.map} perception={args.perception} speed={args.speed} render={not args.no_render}")
    try:
        loop.run(max_ticks=args.max_ticks)
    except KeyboardInterrupt:
        pass
    print(f"sgr: stopped at t={loop.t:.1f}s after {loop.ticks} ticks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
