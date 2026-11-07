"""Dashboard dev runner (owned by D). Not used in the demo.

    uv run python -m sgr.server.dev            # fake publisher: every widget has data
    uv run python -m sgr.server.dev --sim      # S's real `uv run sgr` (sgr.app.main) + dashboard
    cd web && npm run dev                      # then open http://localhost:5173

`--sim` runs sgr.app.main itself, so it always matches the real demo command.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from sgr.bus import Bus
from sgr.contracts import Scenario
from sgr.server import fake_publisher
from sgr.server.server import start_server


def run_real_app(args: argparse.Namespace) -> None:
    """Run S's real entry point (sgr.app.main) with the dashboard attached.

    Until S starts the server in sgr/app.py (its TODO(D)), we wrap the Bus it
    creates so the server starts with it. Once app.py calls start_server itself,
    this just runs app.main unchanged (no double start).
    """
    import inspect

    import sgr.app as app

    argv = ["--map", str(args.map), "--perception", args.perception, "--speed", str(args.speed)]
    if args.no_render:
        argv.append("--no-render")
    if "start_server(" not in inspect.getsource(app.main):
        real_bus = app.Bus
        scenario = Scenario.load(args.map)

        def bus_with_server() -> Bus:
            bus = real_bus()
            start_server(bus, args.map, scenario, args.host, args.port)
            return bus

        app.Bus = bus_with_server  # type: ignore[misc]
    app.main(argv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=Path, default=Path("maps/test"))
    ap.add_argument("--sim", action="store_true", help="run the real sim loop instead of fake data")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--perception", choices=["mock", "remote", "local"], default="mock", help="--sim only")
    ap.add_argument("--speed", type=float, default=1.0, help="--sim only")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.sim:
        run_real_app(args)
        return

    scenario = Scenario.load(args.map)
    bus = Bus()
    start_server(bus, args.map, scenario, args.host, args.port)

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
