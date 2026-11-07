# marshgate-route: MAP FROZEN

Frozen by W on 2026-11-07 (team decision). The map content last changed in 18cd066
(merged in a9800e6): 4 fires, flare 55 s, e6 debris width 2.0 at 0.95.

**Do not change these files** without pinging W and P first. Any change to
`scenario.json` or `grid.png` makes `act2_start.pkl` stale, and P must re-record it on the
demo laptop.

Frozen files (checksums in `FROZEN.sha256`):
`scenario.json`, `grid.png`, `transform.json`, `scan.glb`, `scan_vis.obj`, `scan_vis.png`, `replay_marshgate.json`

Verify (from this folder): `shasum -a 256 -c FROZEN.sha256` (all must say OK).

Not frozen: the 3D dressing (`web/src/scene3d/dressing/`, looks only; it doesn't affect the
sim or the snapshot), `plan_sheet.png`, `preview.png`, `topdown.*`, `route_preview.jpg`.

Snapshot: `act2_start.pkl` must be recorded on main at or after this freeze commit.
Verified at freeze: `uv run python tools/replay.py maps/marshgate-route/replay_marshgate.json`
passes 6/6 in mock and `--mock-latency 7`.
