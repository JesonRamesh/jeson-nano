"""Dashboard dev runner (owned by D). Not used in the demo.

    uv run python -m sgr.server.dev            # fake publisher: every widget has data
    uv run python -m sgr.server.dev --sim      # the real sim stack (same as `uv run sgr`)
    cd web && npm run dev                      # then open http://localhost:5173

`--sim` exists only until S starts the server from sgr/app.py (the TODO(D) there).
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from sgr.bus import Bus
from sgr.contracts import Scenario
from sgr.server import fake_publisher
from sgr.server.server import start_server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=Path, default=Path("maps/test"))
    ap.add_argument("--sim", action="store_true", help="run the real sim loop instead of fake data")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    scenario = Scenario.load(args.map)
    bus = Bus()
    start_server(bus, args.map, scenario, args.host, args.port)

    if args.sim:
        from sgr.contracts import PriorMap
        from sgr.guidance import Guidance
        from sgr.loop import SimLoop
        from sgr.perception import Perception
        from sgr.swarm import Swarm
        from sgr.world import World

        world = World(args.map, scenario)
        loop = SimLoop(
            world,
            Perception(mode="mock", ground_truth=world.ground_truth),
            Swarm(PriorMap.from_map(args.map, scenario), list(scenario.drones.ids)),
            scenario,
            guidance=Guidance(),
            bus=bus,
            render=not args.no_render,
        )
        try:
            loop.run()
        except KeyboardInterrupt:
            pass
        return

    stop = threading.Event()
    threading.Thread(target=fake_publisher.run, args=(bus, scenario, stop), daemon=True).start()
    try:
        while True:  # print browser commands so the round trip is visible
            for msg in bus.drain_inbound():
                print("client ->", msg.model_dump())
            stop.wait(0.2)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
