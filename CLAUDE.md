# Swarm-Guided Rescue: rules for every Claude Code session

Source of truth: `.plans/swarm-guided-rescue-build-spec.md`. It overrides the project description.

## Ownership (spec §1.1): only edit your own directories

| Code | Role | Owns |
| --- | --- | --- |
| **W** | World / Scan | `tools/scan_*`, `tools/propose_graph.py`, `maps/**`, `web/src/editor/**`, `web/src/scene3d/dressing/**` |
| **S** | Swarm + Integrator | `pyproject.toml`, `CLAUDE.md`, `sgr/app.py`, `sgr/loop.py`, `sgr/bus.py`, `sgr/contracts.py`, `sgr/config.py`, `sgr/snapshot.py`, `sgr/swarm/**`, `tools/replay.py`, `tools/make_test_map.py`, `tests/test_smoke.py`, `tests/test_contracts.py`, `tests/test_honesty.py` |
| **P** | Perception + Sim core + Demo laptop | `sgr/world/**`, `sgr/perception/**`, `sgr/guidance/**` (Day 2) |
| **D** | Dashboard + 3D + Firefighter view | `sgr/server/**`, `web/**` except `web/src/editor/` and `web/src/scene3d/dressing/` |

**If a task needs a change to a file you don't own: STOP. Don't edit it.** Write the request
(file, what to change, why) for the owner, or for S if it's a shared file.

## Rules (spec §1.2)

1. **Only edit your own directories** (table above).
2. **Contracts are the only shared surface.** Python: `sgr/contracts.py` (S). TypeScript mirror: `web/src/contracts.ts` (D).
   - To change a contract: S edits it, bumps `CONTRACT_VERSION`, and posts in the team chat. D mirrors it within 30 min.
   - Only add fields (with defaults). Never rename or remove fields after H1.
3. **Modules talk through facades only:** `World`, `Perception`, `Swarm`, `Guidance` (spec §4). Never import another package's internals.
4. **Never call `World.ground_truth()` from `sgr/swarm` or `sgr/guidance`.** `tests/test_honesty.py` greps those folders for `ground_truth` / `GroundTruth`, even in comments. Ground truth is only for mock perception and the mission summary. The swarm learns only through `Readings` and `Detection`s.
5. **Git:** trunk-based.
   - Branches `w/…`, `s/…`, `p/…`, `d/…`, each living at most 2 h.
   - Before merging: `uv run pytest -q` must pass.
   - Everyone pulls `main` at the top of every hour.
   - Never force-push `main`.
6. **Large binaries:** `scan.glb` < 50 MB (else Git LFS for `maps/**/*.glb`). `snapshots/` and `maps/*/raw/` are gitignored. Pass raw scans by USB or Drive.
7. **Feature freeze is H15.** After that, bug fixes only.

## Coordinates (spec §3)

- Metres, **z up, x east, y north**. Angles in radians, CCW from +x, unless a field ends in `_deg`.
- `grid.png`: 8-bit, **0 = wall, 255 = free**, **0.1 m/px**. The bottom-left pixel is world (0, 0). Image rows are flipped: use `PriorMap.load_free_grid()`, which returns `free[iy, ix]` with `iy = 0` at y = 0.
- The swarm belief grid is **0.2 m**. In `map_delta` cells, `i` = column (x) and `j` = row counted from the bottom (y).
- `transform.json` maps scan.glb to world (spec §6.3).
- LiDAR: 36 rays, ray k at `yaw + k·10°`, 6.0 m = no hit.
- `World.set_drone_cmd` velocities are **world frame**.
- Binary WS frames: `[u8 channel][u32 frame_seq][JPEG]`, **big-endian**. Channel 1–4 = D1–D4. Use `pack_frame` / `unpack_frame`.

## Layout

```
sgr/
  app.py         entry point: `uv run sgr --map maps/test --perception mock --speed 1 [--no-render]`
  loop.py        sim loop on the main thread (spec §3 tick order)
  bus.py         thread-safe queues: sim loop <-> server
  contracts.py   all shared pydantic models, CONTRACT_VERSION
  config.py      constants
  snapshot.py    snapshots (spec §8.6)
  world/         World facade; fake_world.py = kinematic FakeWorld (keep it: the swarm is developed on it)
  perception/    Perception facade (mock | remote | local)
  swarm/         Swarm facade
  guidance/      Guidance facade (Act 2)
  server/        FastAPI + WS /ws (D)
  tools/         render_smoke.py
tools/           offline scripts (make_test_map.py, scan_*, propose_graph.py, replay.py)
maps/test/       synthetic test map (grid.png + scenario.json), made by tools/make_test_map.py
tests/           test_smoke.py (merge gate), test_honesty.py, test_contracts.py
```

## Commands

- Setup: `uv sync`
- Tests (must pass before every merge): `uv run pytest -q`
- Render check: `uv run python -m sgr.tools.render_smoke` (writes `render_smoke.png`)
- Regenerate the test map: `uv run python tools/make_test_map.py`
- Run headless: `uv run sgr --map maps/test --no-render --no-server --max-ticks 600`
- Run with the dashboard: `uv run sgr --map maps/marshgate-route --perception mock` -> http://127.0.0.1:8000 (snapshots in `snapshots/`)
- Act 2 drone voices: `--voice auto` (default: Gemini if `google_api` / `GOOGLE_API_KEY` is in `.env`), `gemini`, `template` (offline fixed lines), `off`

## Working habits (spec §15.1)

- Start each block in plan mode. One branch per block. `/clear` between blocks.
- Every prompt: who I am · what to read · the task with exact files · how we know it's done.
