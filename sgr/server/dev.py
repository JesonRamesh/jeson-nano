"""Standalone dev server: real server + fake publisher, no sim. Owned by D.

    python -m sgr.server.dev            # then: cd web && npm run dev
    open http://localhost:5173          # (or :8000 after `npm run build`)
"""

from __future__ import annotations

import argparse
import queue
import threading

from sgr.server import fake_publisher
from sgr.server.server import start_server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    outbound: queue.Queue = queue.Queue()
    inbound: queue.Queue = queue.Queue()
    start_server(outbound, inbound, args.host, args.port)
    print(f"sgr dev server on http://{args.host}:{args.port}  (Ctrl+C to stop)")

    stop = threading.Event()
    pub = threading.Thread(target=fake_publisher.run, args=(outbound, stop), daemon=True)
    pub.start()
    try:
        while True:  # main thread: print client commands so the round trip is visible
            try:
                print("client ->", inbound.get(timeout=0.5))
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
