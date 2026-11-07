"""Server (owned by D): FastAPI + WS /ws. Talks to the sim loop only via sgr.bus."""

from sgr.server.server import build_hello, start_server

__all__ = ["build_hello", "start_server"]
