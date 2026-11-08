"""Entry point (owned by S): `uv run sgr --map maps/test --perception mock --speed 1`."""

from __future__ import annotations

import argparse
from pathlib import Path

from sgr.bus import Bus
from sgr.contracts import PriorMap, Scenario
from sgr.guidance import Guidance
from sgr.loop import SimLoop
from sgr.perception import Perception
from sgr.perception.budget import Budget
from sgr.swarm import Swarm
from sgr.world import World


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sgr", description="Swarm-Guided Rescue simulator")
    p.add_argument("--map", type=Path, default=Path("maps/test"), help="map directory")
    p.add_argument("--perception", choices=["mock", "remote", "local"], default="mock")
    p.add_argument("--speed", type=float, default=1.0, help="sim speed multiplier (2.5 for the video)")
    p.add_argument("--no-render", action="store_true", help="skip camera rendering (mock frames)")
    p.add_argument("--max-ticks", type=int, default=None, help="stop after N control ticks (testing)")
    p.add_argument("--no-server", action="store_true", help="don't start the dashboard server (headless runs)")
    p.add_argument("--host", default="127.0.0.1", help="dashboard host")
    p.add_argument("--port", type=int, default=8000, help="dashboard port")
    p.add_argument("--snapshots", type=Path, default=Path("snapshots"),
                   help="where act1_start / act2_start / named snapshots are saved (§8.6)")
    p.add_argument("--voice", choices=["auto", "gemini", "template", "off"], default="auto",
                   help="Act 2 drone voices: gemini (needs an API key in .env), template (fixed lines, offline), "
                        "off; auto = gemini if a key is found")
    return p


def make_voice(mode: str):
    """(VoiceAgent | None, callouts on?). The agent only rewords/voices lines; the loop never waits for it."""
    if mode == "off":
        return None, False
    if mode == "template":
        return None, True
    from sgr.guidance.voice import VoiceAgent

    agent = VoiceAgent.from_env()
    if agent is None:
        print("sgr: no Google API key (GOOGLE_API_KEY or google_api in .env): drone voices use fixed lines")
        if mode == "gemini":
            print("sgr: WARNING --voice gemini requested but no key found")
    else:
        print(f"sgr: drone voices: Gemini ({agent.text_model} + {agent.tts_model})")
    return agent, True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = Scenario.load(args.map)
    world = World(args.map, scenario)
    perception = Perception(
        mode=args.perception,
        ground_truth=world.ground_truth,  # mock mode only
        line_of_sight=world.line_of_sight,  # mock occlusion
        pixel_to_world=world.pixel_to_world,  # required by local/remote
    )
    prior = PriorMap.from_map(args.map, scenario)
    swarm = Swarm(prior, list(scenario.drones.ids))
    bus = Bus()
    budget = Budget()  # network / VRAM probes on a daemon thread; rates from the loop hooks
    if not args.no_server:
        from sgr.server.server import start_server

        start_server(bus, args.map, scenario, args.host, args.port)  # daemon thread (§3)
    voice, callouts = make_voice(args.voice)
    loop = SimLoop(
        world,
        perception,
        swarm,
        scenario,
        guidance=Guidance(prior=prior, world=world, voice=voice, callouts=callouts),  # world: drone_pose + push_ff_back only (§9)
        bus=bus,
        budget=budget,
        speed=args.speed,
        render=not args.no_render,
        map_dir=args.map,
        snapshot_dir=args.snapshots,
    )
    print(f"sgr: map={args.map} perception={args.perception} speed={args.speed} render={not args.no_render}")
    try:
        if args.perception != "mock":
            print("sgr: waiting for Gemma to warm up (~30 s)...")
        if not perception.wait_ready(timeout=120.0):  # don't hang forever if Ollama is down
            print("sgr: WARNING perception not ready, starting anyway")
        loop.run(max_ticks=args.max_ticks)
    except KeyboardInterrupt:
        pass
    finally:
        budget.close()
        if voice is not None:
            voice.close()
    print(f"sgr: stopped at t={loop.t:.1f}s after {loop.ticks} ticks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
