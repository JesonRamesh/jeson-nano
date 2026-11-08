# Swarm-Guided Rescue — Build Spec (team of 4, 18 build hours)

Status: agreed in interview, 7 Nov 2026. This spec **overrides** `Swarm-Guided Rescue — Project Description.md` wherever they differ. The project description stays the pitch narrative; this file is the build plan.

---

## 0. What changed from the project description

| Topic | Project description | This spec (decided) |
| --- | --- | --- |
| Time | One 10 h day | **Two days: Day 1 = 10 h (H0–H10), Day 2 = 8 h (H10–H18)**, judging at the end of Day 2. Overnight: optional bug fixes only, never planned work |
| 3D visuals | MuJoCo viewer + scan mesh in MuJoCo via Blender | **No Blender, no MuJoCo viewer.** The 3D view and the firefighter view are rendered in the **browser with React Three Fiber (R3F)**, which loads the phone scan (`scan.glb`) directly |
| MuJoCo's job | Everything | **Physics, LiDAR (ray casting), heat and the 4 drone cameras only.** The MuJoCo world is simple: box walls from the grid, plus simple shapes for fire, debris, decoy and survivor |
| Sensors | Camera, LiDAR, microphone, temperature | **Camera, LiDAR, temperature.** The microphone has been cut |
| Fire | Spreads | **Static.** Placed by `scenario.json` or by a judge's click. The only "new" fire is drone 3's timed flare-up |
| Compromised building | Debris anywhere | **Route-graph edges marked "compromised" in an editor** get debris (solid, LiDAR-detectable). The drones are **not told**; they discover it |
| Scenario | Somewhat fixed | **Fully fixed** (`scenario.json`: fires, survivor, decoy, compromised edges, flare-up). Only the judge's clicks and the firefighter's controls vary |
| Change tool | Debris or fire | Judge clicks drop **debris or fire** (two buttons) + **Undo**. Live in both acts |
| "Uncertain, then corrected" | Orange glow | A **decoy** orange object that the fast detector flags. Gemma rejects it after a closer look |
| Drone loss | Timed flare-up | Same: **timed flare-up, with real heat readings before it** |
| Gemma | Strong tier | **Core**, verified in hour 1. Fallback: fast detector + second-look rule |
| Video timing | — | **Global sim speed multiplier** (`--speed 2.5` for the 1-minute video, 1.0 live) |
| Act 2 | — | Starts from a **snapshot** at "chain formed". **WASD + mouse-look** (pointer lock). HUD = arrow + distance + subtitles only |
| Cut entirely, or post-base stretch only | — | Microphone/sound search (cut). Stretch, only after the base works: the unverified-stretch (>30 s) rule, ask-another-drone second view, Gemini after-action report |

---

## 1. Team, ownership and hard rules

### 1.1 Roles (people already map to roles)

| Code | Role | Owns (directories) | Load |
| --- | --- | --- | --- |
| **W** | World / Scan | `tools/scan_*`, `tools/propose_graph.py`, `maps/**`, `web/src/editor/**`, `web/src/scene3d/dressing/**` | ~11 h |
| **S** | Swarm + **Integrator** | `pyproject.toml`, `CLAUDE.md`, `sgr/app.py`, `sgr/loop.py`, `sgr/bus.py`, `sgr/contracts.py`, `sgr/config.py`, `sgr/snapshot.py`, `sgr/swarm/**`, `tools/replay.py`, `tests/test_smoke.py` | ~16 h |
| **P** | Perception + Sim core + **Demo laptop** | `sgr/world/**`, `sgr/perception/**`, `sgr/guidance/**` (Day 2) | ~15 h |
| **D** | Dashboard + 3D + Firefighter view | `sgr/server/**`, `web/**` except `editor/` and `scene3d/dressing/` | ~15 h |

**Demo laptop** (Windows 11, RTX 5060 8 GB) belongs to **P**. It is the only machine that runs real Gemma locally. P pulls `main` and runs the full offline check at every integration checkpoint.

### 1.2 Rules that prevent conflicts (put these in `CLAUDE.md`)

1. **Only edit your own directories.** If you need a change somewhere else, ask the owner (or ask S for shared files). Claude Code sessions must be told this in their first prompt.
2. **Contracts are the only shared surface.** Python: `sgr/contracts.py` (pydantic models, owned by S). TypeScript mirror: `web/src/contracts.ts` (owned by D).
   - To change a contract: S edits it, bumps `CONTRACT_VERSION`, and posts in the team chat.
   - D mirrors the change within 30 min.
   - Only add fields; never rename or remove them after H1.
3. **Modules talk through facades only**: `World`, `Perception`, `Swarm`, `Guidance` (§4). Never import another package's internals.
4. **`sgr/swarm/` must never call `World.ground_truth()`.** `tests/test_honesty.py` greps for it. Ground truth is only for mock perception and the mission summary.
5. **Git:** trunk-based.
   - Branches `w/…`, `s/…`, `p/…`, `d/…`, each living at most 2 h.
   - Before merging: `uv run pytest -q` must pass (smoke + honesty + planner tests).
   - Everyone pulls `main` at the top of every hour.
   - Never force-push `main`.
6. **Large binaries:**
   - `scan.glb` must be compressed below 50 MB (§6.4). If it's bigger, use Git LFS for `maps/**/*.glb`.
   - `snapshots/` and raw scans are gitignored. Pass raw scans by USB or Drive.
7. **Feature freeze is H15.** After that, bug fixes only.

### 1.3 Environment (identical on all machines)

- Python **3.12**, managed with **uv** (`pyproject.toml` + `uv.lock`). Setup: `uv sync`, then run with `uv run sgr …`.
- **MuJoCo pinned to `3.17.0`.** 4.0.0 shipped on 6 Nov 2026 with a package split, so do not upgrade during the event.
- Node 22 LTS. Web stack: **Vite + React + TypeScript**, `@react-three/fiber`, `@react-three/drei`, `three`. The demo serves `web/dist` from FastAPI (built offline).
- Other Python packages: `fastapi`, `uvicorn`, `pydantic`, `numpy`, `opencv-python-headless`, `trimesh`, `scikit-image`, `httpx`, `pillow`, `pytest`.
- Per-OS rendering:
  - Windows: works out of the box.
  - macOS: offscreen `mujoco.Renderer` works without `mjpython` (we don't use the passive viewer).
  - Linux: `MUJOCO_GL=egl` (with a GPU) or `osmesa`.
- Ollama on the demo laptop. Gemma tag chosen in H0–H1 (§7.3).

### 1.4 Pre-event checklist (the night before, at home; the venue Wi-Fi may be slow)

- [ ] **Everyone:** clone the repo, run `uv sync` and `cd web && npm ci`, then run `uv run python -m sgr.tools.render_smoke` (it writes `render_smoke.png`).
- [ ] **P:** install Ollama on the demo laptop and run `ollama pull gemma4:e4b`, plus `gemma4:e2b` as the fallback.
  - Check free VRAM with `nvidia-smi`.
  - Run one image query with Wi-Fi **off**.
- [ ] **W:** install **Scaniverse** (+ the fallback app from §6.1). Do a practice scan of a room at home and export it in the §6 format.
  - Run `tools/scan_to_grid.py` on it if S's skeleton already includes the script stub.
- [ ] **W:** ask the venue for permission to scan the route.
- [ ] **S:** push the repo skeleton: directories, `CLAUDE.md`, `contracts.py` v1, test map `maps/test/` (`grid.png` + `scenario.json`), facade stubs, smoke test. *(Claude can scaffold this now; see the end of this file.)*
- [ ] **D:** `npm create vite` with React/TS + R3F deps; the page connects to `/ws` and shows "connected".

---

## 2. Scoring map (unchanged targets, updated mechanisms)

| Criterion | Weight | What judges see in our build |
| --- | --- | --- |
| Perception-driven action | 25% | **Decision trace**: every action has evidence (sensor, value, frame). Decoy flagged → closer look → Gemma rejects → "false alarm corrected" |
| Adaptation to change | 25% | Compromised corridors found by LiDAR → re-plan. Judge clicks debris/fire anywhere → detected → old route red, new route drawn. In Act 2 the route is blocked while the firefighter is walking it |
| Relevance and impact | 20% | Firefighter-disorientation statistics. The critical failure (a safe route turning unsafe) is caught live in Act 2 |
| Operational autonomy | 15% | Live budget panel: **NETWORK OFFLINE ✓** probe, Gemma p50/p95, VRAM, sim and FPS rates. Explicit rules for uncertainty (§8.4) |
| Ambition / quality | 15% | iPhone-LiDAR digital twin rendered in 3D, a coordinating swarm, a human guided through fog |

Track rules still met: simulated sensors (camera, LiDAR, temperature) → decisions → movement and communication. No cloud in the loop.

---

## 3. System architecture

```
iPhone LiDAR scan ──(tools, offline, W)──► maps/<name>/ scan.glb · grid.png · transform.json · scenario.json
                                                      │                 │
                         ┌────────────────────────────┘                 │
                         ▼                                              ▼
 ┌──────────── Python process: `uv run sgr --map maps/<name> --perception local --speed 1` ─────────────┐
 │ MAIN THREAD (GLFW/GL must live here)                                                                 │
 │   loop.py  ── control tick every 0.1 s sim-time ──►                                                  │
 │     1 drain inbound commands (clicks, ff input, buttons)                                             │
 │     2 scenario timed events (flare-up)                                                               │
 │     3 World: lidar(), temperature(), render drone cams (round-robin, wall-clock 10 fps each)         │
 │     4 Perception: fast detector on new frames → Gemma jobs → poll results                            │
 │     5 Swarm.tick() → velocity commands + trace rows                                                  │
 │     6 Guidance.tick() (Act 2)                                                                        │
 │     7 World.step(20 × 0.005 s)                                                                       │
 │     8 bus.publish(state @10 Hz wall, frames, trace, events)                                          │
 │ GEMMA WORKER THREAD: one Ollama request at a time (localhost:11434)                                  │
 │ SERVER THREAD: uvicorn/FastAPI — WS /ws (JSON + binary JPEG), REST /api/*, static web/dist          │
 └──────────────────────────────────────────────────────────────────────────────────────────────────────┘
                         │ WebSocket (localhost)
                         ▼
 Browser (Chrome, fullscreen 1920×1080): React dashboard · R3F 3D view (scan.glb + live overlays) ·
 2D belief map + click tool · drone panels · trace · budget · Act 2 firefighter view · route editor (/editor)
```

- **Speed multiplier.** The loop runs control ticks so that `sim_time ≈ speed × wall_time`. Camera rendering is paced by wall-clock time, so a higher speed doesn't multiply the render cost. If the loop lags, it skips renders, never ticks.
- **Gemma is asynchronous.** A drone waiting for a check hovers ("INSPECTING"); the other drones carry on. At speed 2.5 this looks fine.
- **Coordinates everywhere:** metres, z up, x east, y north. Grid PNG at **0.1 m/px**; the bottom-left pixel is world (0, 0); image rows are flipped. The swarm's belief grid is **0.2 m** (half resolution). `transform.json` maps scan.glb → world (§6.3).

---

## 4. Module facades (the code contract between people)

All types live in `sgr/contracts.py`. Signatures are v1. Owners implement them, and everyone else codes against them from H1 using stubs.

```python
# sgr/world/world.py  (P)
class World:
    def __init__(self, map_dir: Path, scenario: Scenario): ...
    def step(self, n: int) -> None
    def set_drone_cmd(self, drone: str, vx: float, vy: float, vz: float, wz: float) -> None
    def drone_pose(self, drone: str) -> Pose                     # x, y, z, yaw
    def lidar(self, drone: str) -> np.ndarray                    # (36,) metres, 6.0 = no hit; mj_ray incl. debris
    def temperature(self, drone: str) -> float                   # °C, from heat model
    def render_drone(self, drone: str) -> tuple[np.ndarray, CamInfo]  # 320×240 RGB + intrinsics/pose
    def pixel_to_world(self, drone: str, u: int, v: int) -> XY | None  # mj_ray through pixel
    def place_debris(self, x: float, y: float) -> Handle | None  # None = pool empty / too close to agent
    def place_fire(self, x: float, y: float) -> Handle | None
    def remove(self, h: Handle) -> None                          # for Undo
    def kill_drone(self, drone: str) -> None                     # flare-up victim: falls, goes dark
    def set_ff_input(self, fwd: float, strafe: float, yaw: float) -> None
    def ff_pose(self) -> Pose
    def push_ff_back(self) -> None                               # walked into fire
    def ground_truth(self) -> GroundTruth                        # ONLY mock perception + summary
    def get_state(self) -> bytes; def set_state(self, b: bytes) -> None   # snapshots

# sgr/perception/perception.py  (P)
class Perception:  # mode: "mock" | "remote" | "local"
    def submit_frame(self, drone: str, frame: np.ndarray, cam: CamInfo, t: float, frame_id: str) -> None
    def poll(self) -> list[Detection]     # Detection{id, drone, t, label: fire|person|decoy?, world_xy,
                                          #   score_fast, gemma: GemmaResult|None, combined, status:
                                          #   pending|confirmed|low_conf|rejected|unconfirmed, frame_id}
    def request_relook(self, detection_id: str) -> None
    def metrics(self) -> PerceptionMetrics  # p50/p95 latency, queue, model tag, failures

# sgr/swarm/swarm.py  (S)
class Swarm:
    def __init__(self, scenario_prior: PriorMap, drones: list[str]): ...   # prior = pre-incident plan ONLY
    def tick(self, t: float, readings: dict[str, Readings], detections: list[Detection]
             ) -> tuple[dict[str, VelCmd], list[TraceRow], SwarmState]

# sgr/guidance/guidance.py  (P, Day 2)
class Guidance:
    def tick(self, t: float, ff: Pose, swarm: SwarmState) -> tuple[GuidanceMsg, dict[str, VelCmd]]
```

`PriorMap` is built from `grid.png` + the route graph **with all edges open**. The swarm never sees `compromised`, `fires`, `survivor` or `decoy`.

---

## 5. The fixed scenario — `maps/<name>/scenario.json`

Authored by W in the route editor (§10), reviewed by the whole team for 15 min at IC2.

```json
{
  "version": 1,
  "name": "marshgate-route",
  "grid": { "file": "grid.png", "resolution": 0.1, "origin": [0, 0] },
  "scan": { "file": "scan.glb", "transform": "transform.json" },
  "entrance": { "x": 2.0, "y": 3.0, "yaw": 0.0 },
  "graph": {
    "nodes": [ { "id": "n1", "x": 2.0, "y": 3.0, "kind": "entrance" },
               { "id": "n2", "x": 12.0, "y": 3.0, "kind": "junction" } ],
    "edges": [ { "id": "e1", "a": "n1", "b": "n2", "width": 1.8 } ]
  },
  "compromised": [ { "edge": "e4", "at": 0.5, "kind": "debris" } ],
  "fires": [ { "x": 18.0, "y": 9.0, "size": 1.5 } ],
  "survivor": { "x": 25.0, "y": 14.0, "yaw": 1.2 },
  "decoy": { "x": 9.5, "y": 6.2, "kind": "orange_panel" },
  "drones": { "ids": ["D1", "D2", "D3", "D4"], "start": [[2,2.5],[2,3.5],[2.5,2.5],[2.5,3.5]], "altitude": 1.6 },
  "flare": { "t": 65.0, "target": "D3" },
  "heat": { "limit_c": 70.0, "ambient_c": 20.0 },
  "beats": { "decoy_drone": "D2", "d3_area_node": "n7" }
}
```

**Scenario requirements** (the editor shows a warning if any is broken):
- At least 3 branches from the entrance region.
- At least one **wrong-way branch** next to the final route.
- At least one compromised edge on the *naive* shortest route, so the LiDAR re-plan happens without any judge.
- The survivor must stay reachable after the compromised edges are applied.
- All nodes and placements must be on free cells.

**Act 1 beat targets (sim seconds at 1×; about 60 s of video at 2.5×):**

| t (s) | Beat | Mechanism |
| --- | --- | --- |
| 0–20 | Fan-out, coverage fills | Area assignment |
| 20–40 | D2 sees the decoy → closer look → Gemma: not fire → REJECTED | Decoy on D2's first area |
| 40–55 | Real fire seen → confirmed → zone marked; routes avoid it | Fire near D3's area |
| 50–65 | D3 temperature climbs → RETREAT → flare-up at `flare.t` → heartbeat lost after 3 s → area unsafe, reassigned | Timed flare-up, real heat readings |
| 70–100 | LiDAR hits the compromised-edge debris → MARK_BLOCKED → REPLAN | Compromised edge |
| 100–130 | Survivor seen → Gemma confirms → safest route → chain forms → snapshot `act2_start` saved | |
| any | Judge click (debris or fire) → detected → old route red → new route | Change tool |

---

## 6. iPhone LiDAR → map pipeline (W) — no Blender

**Verdict: feasible without Blender.** The Python/Node pipeline below was tested end to end on a synthetic GLB: trimesh load → slice → grid, gltf-transform simplify/compress, and MuJoCo loading the OBJ + PNG. The axis convention, triangle count and atlas count of a *real* Scaniverse file are **not yet verified**, so W does a one-room test scan in H0. Accuracy reference: one continuous Scaniverse scan of a 40 m corridor plus 4 rooms had a mean error of 7 cm (Treccani et al. 2024). That is fine for a 0.1 m grid.

### 6.1 App choice
- **Primary: Scaniverse → Mesh → *Area* mode** (40 mm LiDAR resolution). Free; exports **GLB**.
- **Fallback: 3D Scanner App (Laan Labs), LiDAR mode** (GLB/OBJ). Which exports are free is unverified; check it the night before.
- **Not used:**
  - Polycam: the free tier exports glTF only, and floor plans are paid.
  - Apple RoomPlan apps: cleaner walls, but best for single rooms of about 9×9 m, and they don't share a coordinate frame with the 3D model.
- **Do ONE continuous scan.** Several scans means several coordinate frames, and lining them up without Blender is too hard on the day.

### 6.2 Procedure (W), with time estimates

1. **Prep (15 min).**
   - All lights on.
   - Avoid or cover glass and mirrors: glass disappears, and mirrors create "false rooms".
   - Clear people from the route. Phone fully charged.
   - Pick a route with **≥ 3 branches, one loop and one dead end/side room** (§5).
2. **Test scan (10 min, H0).** Scan one room, export GLB, run `tools/scan_to_grid.py` on it, and confirm the axes and scale before the real scan.
3. **Real scan (20–30 min + one retake).**
   - Walk at a steady ~0.6 m/s, zigzagging gently between the walls.
   - Keep door frames and corners in view, never just floor.
   - At each room: frame the door, circle it clockwise, leave by the same door.
   - **End where you started** (loop closure).
   - Export GLB as `maps/<name>/raw/scan_raw.glb` (gitignored).
4. **Make two GLBs (Node only, 15 min).**
   - **Web:** `npx @gltf-transform/cli optimize scan_raw.glb ../scan.glb --compress meshopt --texture-compress webp --texture-size 2048`
   - **Sim/analysis (uncompressed, because trimesh can't read meshopt):**
     - `npx @gltf-transform/cli simplify scan_raw.glb sim.glb --ratio 0.25 --error 0.001`
     - `npx @gltf-transform/cli resize sim.glb sim.glb --width 2048 --height 2048`
   - Don't use trimesh's own decimation: it drops UVs/textures.
5. **`tools/scan_to_grid.py sim.glb` (Claude writes it; about 2–3 h including tuning).**
   1. `trimesh.load` → rotate **+90° about X** (glTF is Y-up; trimesh doesn't convert it).
   2. Floor z = the area-weighted peak height of the upward-facing faces. Shift so floor = 0.
   3. `mesh.section(plane_origin=[0,0,1.2], plane_normal=[0,0,1])`. Draw `section.discrete` segments in **world XY** with `cv2.polylines` at 0.1 m/px. (Don't use `Path2D.to_2D()`, which changes the frame.)
   4. Apply `MORPH_CLOSE`, then flood-fill free space from the entrance seed. Anything not reached becomes wall.
   5. Write `grid.png`, `transform.json` and `preview.png`.
6. **Manual cleanup (20 min).** Open `grid.png` in any paint app (Paint is fine):
   - Erase furniture and people that show up as walls.
   - Paint in glass walls.
   - Make sure doors are open.
7. **`tools/propose_graph.py` (Claude writes it).** `skimage.morphology.skeletonize` on free space → junctions and endpoints become nodes, skeleton branches become edges, saved into `scenario.json`. Then W fixes it by hand in `/editor`.
8. **Alignment check (10 min).** Click one known doorway in the editor (grid), in the R3F view (`scan.glb`) and in MuJoCo (a debris placed there). All three must agree within about 0.2 m.
9. **Strong tier only: `tools/scan_to_mujoco.py`.**
   - Exports one OBJ + PNG per geometry (`export_obj(..., include_texture=True, return_texture=True)`).
   - Rename each `material_0.png` to a unique name per chunk.
   - Writes `visual.xml` (one `<texture>` + `<material>` + `<mesh>` + `<geom contype=0 conaffinity=0 group=1>` per chunk).
   - Result: the drone cameras see the real building.
   - Only if Act 1 is green at the Day-1 gate.

A reference test script from the feasibility check is at `C:\Users\ashra\AppData\Local\Temp\claude\d37fd1503d4476e4\68834b24-f18a-4e1b-9ead-5ed346b02e63\scratchpad\test_pipeline.py`. Copy it into the repo if useful; it is temporary.

### 6.3 Outputs W must produce (contract)

| File | What | Used by |
| --- | --- | --- |
| `maps/<name>/grid.png` | 8-bit; **0 = wall, 255 = free**; 0.1 m/px; cleaned (doors open, no specks) | P (MuJoCo walls), S (prior map), D (2D map), W (editor bg) |
| `maps/<name>/scan.glb` | Compressed scan for R3F (< 50 MB, textures ≤ 2048 px) | D (3D view, FF view) |
| `maps/<name>/transform.json` | `{ "M": 4×4 row-major (raw GLB → sim frame: Z-up, floor z=0, origin = grid bottom-left), "pitch": 0.1, "grid_origin": [0,0], "grid_size": [w,h] }`. R3F wraps the GLB in `<group matrix={C·M}>`, where C = −90° about X (Z-up back to three.js Y-up) | D (R3F), W (editor overlay check), P (strong-tier visuals) |
| `maps/<name>/scenario.json` | §5 | everyone |
| `maps/<name>/preview.png` | Grid + graph overlay, for the README/pitch | pitch |

### 6.4 Web performance check
`scan.glb` (step 4 above) must load in D's 3D view at ≥ 45 FPS on the demo laptop. drei `useGLTF` decodes meshopt/Draco by default. If it's too slow: `--texture-size 1024`, and `simplify --ratio 0.5` before `optimize`.

**Gaussian splats** (Scaniverse SPZ/PLY via Spark) look the most realistic, but there is no official R3F example. Out of scope; treat as a post-event idea.

---

## 7. Perception (P)

### 7.1 MuJoCo world look (simple, for the drone cameras only)
- **Walls:** boxes from `grid.png`. Merge neighbouring wall cells into rectangles (greedy). Height 2.6 m, flat grey-beige. Floor dark grey. No textures needed.
- **Fire:** a cluster of 3–5 emissive yellow-orange cones plus an orange point light. Flicker by changing `geom_rgba` each tick. Heat source in Python.
- **Debris:** grey-brown boxes (1.0 × 1.0 × 1.2 m), slightly rotated. Collision ON. A compromised edge gets ⌈width / 1.0⌉ blocks across it.
- **Decoy:** a flat orange rectangle (0.6 × 0.9 m) on a wall at 1.5 m height, i.e. the "orange panel".
- **Survivor:** a capsule humanoid (head sphere + torso + limbs), lying down, **hi-vis yellow-green**.
- **Drones:** sphere radius 0.25 + 4 small discs. Joints: slide x/y/z + hinge yaw. Velocity actuators. `gravcomp=1`.
- **Firefighter:** capsule, slide x/y + hinge yaw, velocity actuators. Collides with walls and debris.
- **Hidden pools (MuJoCo cannot add bodies at runtime):**
  - 40 debris and 30 fire bodies as **mocap bodies**, parked at z = −50.
  - Placing one = moving its mocap position. Mocap debris collides with drones and the firefighter, and `mj_ray` hits it.
  - Do **not** toggle `contype` at runtime (the broadphase uses compile-time `body_contype`).
- **Geom groups:** 0 = walls/debris (ray-visible), 1 = fire/decoy/survivor visuals, 2 = drones, 3 = hidden.
  - LiDAR `mj_ray` uses a geomgroup mask {0} and excludes the drone's own body.
- **Renderer:** **one** `mujoco.Renderer(240, 320)`, created on the main thread and reused for all 4 drone cameras. Shadows and reflections OFF. Drone camera: fovy 90°, tilted down 15°.

### 7.2 Fast detector (`detector.py`)
- HSV thresholds on each frame (a few ms):
  - `fire`: bright yellow-orange, high V, area ≥ 0.5%.
  - `person`: hi-vis yellow-green.
- The decoy deliberately passes the fire threshold.
- Output: bbox, area fraction, `score_fast`.
- World position = `pixel_to_world(bbox centre)` via `mj_ray`.
- **De-duplication:** within 1.0 m of an existing detection of the same label → merge, no new job.
- **Rejected locations are blacklisted for 60 s.**

### 7.3 Gemma (`gemma.py`)

**Hour-1 test (P, demo laptop, Wi-Fi off).**
- Render 3 MuJoCo frames: fire, survivor, decoy.
- Run `gemma4:e4b` (or its QAT variant), and `gemma4:e2b` if VRAM is short.
- Record: VRAM in use, p50 latency, answers.
- **Pick the tag and write it in `config.py`.** The 6.6 GB E4B leaves about 1.4 GB on 8 GB, so close other GPU apps.

**Call.**
- `POST http://localhost:11434/api/chat`
- `model`: pinned tag; `stream: false`; `keep_alive: -1` (warm the model at startup); `options: {temperature: 0, num_ctx: 2048}`
- `format`: JSON schema
- The image is a crop: bbox padded 2×, at least 224 px (upscaled).

**Prompt (fixed):**
> "You are the vision system of a search-and-rescue drone in a burning building. Look at the image. Answer only JSON: {"fire": bool, "person": bool, "confidence": 0..1, "caption": "≤12 words"}."

**Fusion.**
- If Gemma agrees with the detector label: `combined = 0.4·score_fast + 0.6·gemma.confidence`. If it disagrees: `combined = 0.3·score_fast`.
- `combined ≥ 0.7` → **confirmed**.
- Otherwise → **low_conf** → the swarm makes a closer look (≤ 2 relooks).
- Still low after that → **rejected** (if Gemma disagrees) or **unconfirmed** (if Gemma is unavailable).

**Timeouts and modes.**
- Timeout 4 s wall → fast-only, status `unconfirmed`, so a second view must agree.
- Modes:
  - **mock**: geometric. A target in the frustum with line of sight produces a scripted detection (fire 0.85; decoy 0.45 then 0.2 → rejected; survivor 0.9), with a simulated 1.0 s latency. It must reproduce every Act 1 beat. This is what S develops against.
  - **remote**: Ollama on the demo laptop over the LAN (`OLLAMA_HOST=0.0.0.0`). If the venue Wi-Fi isolates clients, use a phone hotspot.
  - **local**: the demo.

**If Gemma can't recognise the renders (decided in H1–H4):**
1. Increase emissive brightness and size.
2. Add more limbs or a head to the survivor.
3. Last resort: Gemma becomes caption-plus-confirmation for fire only, and the person is confirmed by the hi-vis colour plus a second view. Say so honestly in the pitch.

### 7.4 Heat and drone loss
- `T(x) = ambient + Σ_fires 380·exp(−d²/(2·1.5²))`. No occlusion.
- Drone retreat threshold: 70 °C.
- **Flare-up:** at `flare.t` a fire body is placed at the target drone's position. The drone's temperature jumps above 300 °C and `kill_drone` runs: the drone stops publishing and its camera panel shows static.
- The swarm declares it lost after **3 s without a heartbeat**.
- The pitch states honestly: "the heat build-up is real; only the moment of the flare-up is timed."

### 7.5 Budget metrics (`budget.py`)
- **Network probe:** TCP connect to `1.1.1.1:443`, 1 s timeout, every 5 s → `OFFLINE ✓` / `ONLINE ⚠`.
- Gemma model tag, p50/p95 latency, queue length.
- VRAM via `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` every 5 s (N/A elsewhere).
- Sim real-time factor and speed, control Hz, camera FPS.
- **Stated budget** (numbers finalised after measurement at H1): "1 laptop · RTX 5060 8 GB · 0 network calls · Gemma check ≤ X s p95 · control 10 Hz".

---

## 8. Swarm (S)

### 8.1 Belief state (per drone, merged on the blackboard)
- **Belief grid (0.2 m), cell states:** `wall_prior`, `free`, `blocked_new` (LiDAR), `fire` (confirmed camera), `danger` (heat > 50 °C), `unsafe` (lost drone's area), `searched` flag (camera coverage).
- **Blackboard** (in-process object; each drone publishes its own layer, then merged):
  - map deltas
  - hazards `{id, kind, xy, source_sensor, t, confidence}`
  - area claims, survivor estimate
  - route `{points, version, status}`, checkpoints
  - heartbeats
- **A lost drone simply stops publishing.**

### 8.2 Exploration = camera coverage
- **Searched:** a cell counts as searched when it is inside a drone camera's 90° frustum, within 5 m, with line of sight on the belief grid (DDA).
- **Areas** = a Voronoi split of free cells by route-graph nodes (from `scenario.graph`; the structure is public, compromise is not).
- **Assignment** (greedy): each idle drone claims the nearest unclaimed area with an unsearched fraction above 20%. A lost drone's claims go back to the pool.
- **Inside an area:** go to the nearest unsearched reachable cell, yaw-sweep 360° on arrival.
- **Reflex layer** (every tick):
  - separation of 2 m from other drones
  - wall repulsion from LiDAR rays under 0.6 m
  - velocity clamp 1.2 m/s, smoothing

### 8.3 Change detection, re-planning, chain
- **LiDAR vs prior:**
  - Expected range comes from ray-marching the belief grid.
  - Measured < expected − 0.3 m on **2 consecutive scans** → hit cells become `blocked_new`.
  - Measured > expected (debris removed by Undo) on 2 scans → cleared.
- **Planner:**
  - A* (heapq) on the belief grid at 0.2 m.
  - Obstacles inflated 0.3 m. Fire plus a 1.0 m margin is impassable.
  - Cost per cell = 1 + 5·heat_norm + 3·near_wall.
  - Re-plan whenever a new `blocked_new`/`fire` cell falls within 0.5 m of the current route. Target: < 1 s after detection.
  - The old route stays in `old_routes` (drawn red) for 6 s.
- **No route:** status `none` → banner and voice "No safe route. Holding." The chain holds and re-checks every 2 s. The change tool's Undo exists for this.
- **Checkpoint chain** (on `SURVIVOR_CONFIRMED`):
  - Route from entrance to survivor → RDP-simplify → turn points (> 35°).
  - Roles:
    - `D_survivor` hovers over the survivor.
    - `D_lead` flies to the entrance to meet the firefighter.
    - Any remaining drone takes the **largest turn** as a sentinel, yaw-sweeping its stretch of route with LiDAR.
  - Emit `chain_formed` → auto-save snapshot `act2_start`.

### 8.4 Rules when unsure (base)

| Situation | Action | Trace |
| --- | --- | --- |
| Detection `low_conf` | Hold, move to 2 m from the target facing it, re-look (≤ 2), then reject or confirm | INSPECT → CONFIRM / REJECT_DETECTION |
| Gemma timeout / down | Fast-only, `unconfirmed`; needs a second view (another frame ≥ 2 s later, other angle) | UNCONFIRMED |
| Heartbeat missing 3 s | Drone LOST; its last area becomes `unsafe` until re-observed; claims reassigned; chain rebuilt | DRONE_LOST, REASSIGN |
| Temp > 70 °C | Retreat 3 m away from the gradient; mark `danger` zone | RETREAT |
| LiDAR says blocked, camera sees nothing | Cautious one wins: blocked | MARK_BLOCKED |

**Stretch (only after base):** an unverified-stretch rule (>30 s unseen → re-check before the firefighter enters); a second-view request to another drone.

### 8.5 Decision trace (`trace.py`)
- **Row:** `{id, t, drone, action, reason, evidence: {sensor: lidar|camera|gemma|temp|heartbeat|planner, value, frame_id?, scan?}}`.
- `trace.log()` **raises if evidence is missing** (no action without evidence). A unit test enforces it.
- Only **decision changes** are logged, not every tick.
- **Actions:** `ASSIGN_AREA, EXPLORE, INSPECT, CONFIRM_HAZARD, REJECT_DETECTION, UNCONFIRMED, MARK_BLOCKED, REPLAN, RETREAT, DRONE_LOST, REASSIGN, SURVIVOR_CONFIRMED, CHAIN_POSITION, NO_ROUTE, ESCORT, FF_CORRECTION, MISSION_COMPLETE`.
- **Reason sentences** are short English, e.g. "D2 inspecting orange object: fast detector 0.62, Gemma 0.31 — moving closer".

### 8.6 Snapshots (`snapshot.py`)
- What is saved: `mj_getState` (full physics) plus a pickle of the World runtime (pool usage), Perception state (detections, blacklist), Swarm (blackboard, agents), trace and scenario time.
- The renderer and threads are not pickled.
- Files go in `snapshots/<name>.pkl`.
- Buttons: **Restart Act 1**, **Jump to Act 2** (loads `act2_start`, or the committed `maps/<name>/act2_start.pkl` produced at the Day-1 gate).

---

## 9. Act 2 — firefighter (P: logic + body; D: view)

- **Controls:**
  - **WASD + mouse-look** via pointer lock, 1.4 m/s walk. Esc releases the pointer.
  - The browser sends `ff_input` at 30 Hz. MuJoCo is authoritative for position (collisions).
  - The browser renders the camera at the server pose (interpolated), with local yaw/pitch for responsiveness. Yaw is sent to the server.
- **View:** R3F first-person on `scan.glb`, `FogExp2` dark grey (visibility about 4 m). Drones are emissive and their colours are visible through the fog. The lead drone pulses white-blue.
- **Guidance rules:**

| Situation | Trigger | Response |
| --- | --- | --- |
| Follow | Lead stays 3 m ahead on the route; FF > 6 m behind → lead stops | "Follow me." |
| Handover | FF within 2 m of the next chain drone | That drone becomes the lead |
| Drifting | > 2 m off the route polyline for 2 s | Arrow pulses; "Bear left/right." |
| Wrong way | Nearest graph edge not on the route for > 2 s, or heading > 120° away from the next checkpoint for > 2 s | Nearest drone flashes red; "Wrong way. Turn around." |
| Danger | Inside a hazard margin | Red screen edges + alarm tone; "Stop. Fire ahead. Turn back." |
| Route becomes unsafe | Chain/lead LiDAR detects new debris/fire on the route ahead | "Stop. Route blocked." → re-route → drones reposition → "New route. Follow me." |
| Walks into fire | Inside fire cell | Flash, `push_ff_back()`, warning (no fail state) |
| Complete | Within 1.5 m of the survivor | MISSION_COMPLETE → summary |

- **Voice:** browser `speechSynthesis`, one calm voice. Pick the first voice with `localService === true` and lang `en-GB`/`en-US` (Windows local voices are offline; avoid the "Google …" voices, which use the network). Danger messages call `cancel()` first. Subtitles always on. Alarm = WebAudio beep. The HUD guidance voice stays local.
- **Drone voices (team decision, Day 2):** the drones also talk to the firefighter (`CalloutMsg`, contract v3): intro, handovers, "fire on your right", survivor close, progress, lead waiting. Guidance decides when and the facts (swarm beliefs only); `sgr/guidance/voice.py` rewords the line with Gemini (`gemini-3.5-flash-lite`, ~0.5 s), fact-checks it (no invented directions, numbers, drones or hazards, else the fixed line) and voices it with Gemini TTS (`gemini-3.8-flash-lite-tts`, one voice per drone, ~2 s). Danger lines never go through Gemini. Firefighter-view switch (V): Online = Gemini voices; Offline = no voice-over, on-screen text only (the danger beep stays). No API key: fixed lines, local voices. `uv run sgr --voice auto|gemini|template|off`.
- **Summary screen:** time, distance walked, hazards avoided, corrections given, number of re-plans, and a link to the full trace.

---

## 10. Dashboard and 3D (D) + editor (W)

### 10.1 Act 1 layout (1920 × 1080, projector-readable: body text ≥ 16 px, trace ≥ 18 px)
```
┌──────────────────────────────┬──────────┬──────────┐
│ BELIEF MAP (canvas) + click  │   D1     │   D2     │   drone panel: JPEG + bbox/labels,
│ tool [Debris][Fire][Undo]    │          │          │   LiDAR mini-polar, temp bar,
│   ┌──────────────┐           ├──────────┼──────────┤   Gemma caption + conf, status chip.
│   │ R3F 3D VIEW  │           │   D3     │   D4     │   Lost → static + "SIGNAL LOST"
│   │ (inset, big) │           │          │          │
│   └──────────────┘           │          │          │
├──────────────────────────────┴──────────┴──────────┤
│ DECISION TRACE (scrolling; click row → evidence)   │ BUDGET │
└────────────────────────────────────────────────────┴────────┘
```

**Belief map layers:**
- prior walls
- searched shading
- `blocked_new` (red hatched), fire/danger (orange), `unsafe` (purple)
- route (green) and old route (red dashed, fades out)
- checkpoints (rings)
- drones with heading and LiDAR rays (toggle)
- survivor marker
- click mode cursor

**R3F 3D view:**
- `scan.glb` with the `transform.json` transform.
- Drones: CC0 drone GLB; fallback glowing spheres.
- Fire: emissive flickering cones + point light. Debris: boxes.
- Route: drei `Line`. Checkpoints: rings.
- If `scan.glb` is missing, extrude walls from `grid.png` (the same code also gives the test map a 3D view).

**Camera:**
- **Auto-director** (default): on each `event`, cut to the involved drone/area with a smooth transition (drei `CameraControls`).
- Buttons: Overview / D1–D4 chase / Free (orbit).

**Other UI:**
- **Banners** for key events (FIRE CONFIRMED, D3 SIGNAL LOST, ROUTE BLOCKED — RE-PLANNING, SURVIVOR FOUND, NO SAFE ROUTE).
- **Buttons:** Start / Pause / Speed (1×, 2.5×) / Restart Act 1 / Jump to Act 2 / Save snapshot. "Send in firefighter" appears after `chain_formed`.
- **Act 2 layout:** FF view full width; the map and trace shrink to a right sidebar; drone panels hidden.

### 10.2 WebSocket contract `/ws` (v1)

**Server → client JSON** (`{"type": ..., ...}`):

| type | rate | payload |
| --- | --- | --- |
| `hello` | on connect | contract_version, map meta (grid url, resolution, size, origin), scenario graph + entrance (no hidden fields), drone ids + colours, transform |
| `state` | 10 Hz (30 Hz FF pose in Act 2) | t, speed, act, phase; drones[{id, x, y, z, yaw, status, task, temp, lidar[36], alive}]; ff{x, y, yaw}\|null; route{points, status, version}; old_routes[{points, until}]; checkpoints[{x, y, drone}]; survivor{status, x, y, conf}; click_handles[] |
| `map_delta` | on change | cells[[i, j, state]] at 0.2 m; `map_full` (base64 PNG) every 5 s for resync |
| `trace` | on event | TraceRow |
| `detection` | on change | {drone, frame_id, boxes[{x, y, w, h, label, score}], gemma{caption, conf, pending}, status} |
| `event` | on event | {kind: fire_confirmed\|false_alarm\|drone_lost\|blocked\|replan\|survivor_found\|chain_formed\|no_route\|mission_complete, drone?, xy?} → banners + auto-director |
| `guidance` | Act 2, on change | {arrow_deg, distance, subtitle, say?, alarm?, severity} |
| `budget` | 1 Hz | §7.5 |
| `summary` | at end | §9 |

**Server → client binary:** `[u8 channel][u32 frame_seq][JPEG]`, where channel 1–4 = drone cams, quality 70. The server keeps a ring buffer of the last 300 frames per drone, served at `GET /api/frames/{frame_id}`, for trace evidence.

**Client → server JSON:** `click{mode: debris|fire, x, y}` · `undo` · `control{action: start|pause|reset|speed, value?}` · `act2_start` · `snapshot{op: save|load, name}` · `ff_input{fwd, strafe, yaw}` · `camera{mode}` (client-side only, no message needed).

**REST:** `GET /api/maps/{name}/scenario`, `PUT …/scenario` (editor), `GET /maps/{name}/grid.png|scan.glb|transform.json`, `GET /api/frames/{id}`.

**Click safety:**
- Refuse within 1.0 m of a drone, the firefighter or the survivor (toast).
- Pool empty → toast.
- Undo removes the last click placement.

### 10.3 Route editor `/editor` (W)
- **Background:** `grid.png` (+ optional `scan.glb` top-down overlay to check alignment).
- **Tools:**
  - Node (kind: entrance / junction / door / room / stairs) and Edge (drag node → node, width)
  - **Compromise** (toggle on an edge; `at` slider)
  - Fire, Survivor, Decoy, Drone starts
  - Flare `t` field
- **Auto-propose:** a button calls `tools/propose_graph.py` (skeletonise the free space → junction/endpoint nodes + edges) as a starting point, then edit by hand.
- **Validation panel:** the §5 requirements. Save via `PUT`.

---

## 11. Timeline (H0 = Day 1 start)

**Integration checkpoints (IC):** S merges, then P runs on the demo laptop with Wi-Fi off. 30 min each.

### Day 1 — goal: **Act 1 end to end, on the real map, offline, on the demo laptop**

| Time | W | S | P | D |
| --- | --- | --- | --- | --- |
| **H0–H1** (together) | 10-min test scan + export check; then leave to scan | Skeleton live; **contracts v1 frozen at H1**; test map | **Gemma hour-1 test** (§7.3), pick tag; render smoke on all 4 laptops | Vite app shows a test JPEG over WS |
| **H1–H4** | Scan route; `scan_to_grid.py` → first `grid.png` by H4 | Blackboard, belief grid, coverage, areas, A*, agent loop on test map with mock perception (debug matplotlib OK) | `World` on test map: walls from grid, drones, pools, LiDAR, heat, renderer; detector | `server.py` (WS/REST/frames), belief map canvas, drone panels, trace list |
| **IC1 @ H4** | | Drones explore the test map in the browser | | |
| **H4–H7** | Editor + `propose_graph.py`; real `scenario.json` + `transform.json` + compressed `scan.glb` by H6.5 | LiDAR change detection → blocked → REPLAN; inspect/relook flow; trace everywhere; survivor → chain | Gemma client + fusion + timeouts; mock parity for all beats; budget backend | Click tool + Undo; budget panel; R3F 3D view (grid-extrude fallback, then `scan.glb`) |
| **IC2 @ H7** | **Swap to real map.** 15-min team scenario review | | | |
| **H7–H9.5** | Tune placements so the beats land in order (§5) | Drone loss, reassignment, unsafe zones, no-route, snapshots | Flare-up + `kill_drone`; local Gemma end-to-end | Auto-director, event banners, trace row → evidence |
| **IC3 @ H9.5 — DAY-1 GATE** | Act 1 offline on the demo laptop; save `act2_start.pkl` and commit it to `maps/<name>/`; write the bug list | | | |

**Overnight (optional):** only items from the gate bug list. No new features.

### Day 2 — goal: **Act 2 + polish + deliverables**

| Time | W | S | P | D |
| --- | --- | --- | --- | --- |
| **H10–H13** | **Disaster dressing** in R3F (`scene3d/dressing/`): fog colour, red emergency lights, smoke (drei `Cloud`/`Sparkles`), CC0 debris GLBs at compromised edges & clicks, scorch decals, drone GLB | Act 2 swarm: escort lead, sentinel scanning, re-route with FF on route; summary data; `tools/replay.py` (scripted clicks for rehearsal) | FF body + `ff_input`; `Guidance` (§9 table) | FF view (pointer lock, fog, HUD, subtitles, voice, alarm), Act 2 layout switch |
| **IC4 @ H13** | Act 2 end to end from the snapshot | | | |
| **H13–H15** | Visual tuning on the demo laptop; screenshots for slides | Bug bash with `replay.py`; stretch rules only if all green | Budget numbers final; Gemma tuning | Summary screen; projector readability pass |
| **H15** | **FEATURE FREEZE** | | | |
| **H15–H16** | Record the 1-min video: several takes at `--speed 2.5`, dashboard full screen, OBS; a teammate who hasn't rehearsed does the click | | | |
| **H16–H17** | One-page PDF + slides (owners decided then; time reserved for all 4) | | | |
| **H17–H18** | 2 full rehearsals (one with a "judge" stranger), final offline run, laptop charged, buffer | | | |

### Cut ladder (cut from the top when behind)
1. Stretch rules (unverified stretch, ask-another-drone), Gemini report (already out of base)
2. Drone GLB model → glowing spheres
3. Disaster dressing beyond fog + red lights
4. Auto-director → fixed overview + manual presets
5. Summary screen → a static count overlay
6. Trace row → evidence frame click (keep the text evidence)
7. **Never cut:** decision trace with evidence, click tool + re-plan, drone loss, chain, Act 2 route-blocked correction, offline probe

---

## 12. Tests (minimal set)

- `tests/test_smoke.py` (merge gate): headless; mock perception; test map; 1500 ticks at max speed, no renderer (mock frames).
  - Asserts no exception, and that the survivor was confirmed.
  - Asserts at least one REPLAN, at least one REJECT_DETECTION, and DRONE_LOST for D3.
- `tests/test_honesty.py`: no `ground_truth` in `sgr/swarm/` or `sgr/guidance/`; every trace row has evidence.
- `tests/test_planner.py`: A* on small grids; blocked → re-route; disconnected → `none`; checkpoint turn extraction.
- `tools/replay.py`: runs a scenario with scripted clicks at set times, for rehearsing the whole demo hands-free.

---

## 13. Risks (updated)

| Risk | Mitigation | Fallback |
| --- | --- | --- |
| Scan unusable (glass, drift, long route) | Short segments; test scan in H0; route chosen for features over size | Fallback app (§6.1); hand-drawn `grid.png` of the same route + grid-extruded 3D |
| Gemma doesn't recognise simple renders | Hour-1 test; tune emissive/shape | Gemma for fire only; colour + second view for the person; say so |
| VRAM (E4B ≈ 6.6 GB of 8) | Close other GPU apps; one renderer | E4B QAT or E2B |
| `scan.glb` too heavy for the browser | gltf-transform resize/compress | Grid-extruded walls coloured to match |
| Mac/Linux rendering issues | Render smoke test the night before | Mock frames on that machine |
| Merge conflicts across 4 Claude sessions | Directory ownership, facades, contracts-only shared surface, hourly pulls | S resolves at ICs |
| Contract churn | Additive-only after H1 | — |
| Venue Wi-Fi blocks peer LAN (remote Gemma) | Mock mode for development | Phone hotspot |
| Judges think it's scripted | Agents learn only via sensors (honesty test); judge clicks; trace evidence | Show `test_honesty.py` in Q&A |
| Live demo crash | Snapshots + Restart/Jump buttons | The recorded video |

---

## 14. Deliverables (time reserved H15–H17; owners decided on Day 2)
1. **One-page PDF:** user and problem, features, architecture diagram (§3), where each model runs with **measured** latency/VRAM, limits, critical failure mode + uncertainty rules.
2. **1-min video:** one continuous take at 2.5×; beats per §5; ends on the budget panel showing OFFLINE ✓.
3. **5-min pitch:** problem 40 s · how it works 50 s · live demo 2.5 min (shortened Act 1 with a judge click, then Act 2 with route-blocked) · can/can't 30 s · why it matters 20 s.

**Limits to state:** one floor at a fixed altitude; simplified fire and smoke; simple shapes for the drone cameras; colour-assisted detection; timed flare-up; no radio limits; not yet on real drones.

---

## 15. Working with Claude Code: how-to and first prompts

### 15.1 Habits for every session
- Open Claude Code **in the repo root**, so it loads `CLAUDE.md` (ownership rules) automatically.
- **Start each block in plan mode** (Shift+Tab). Read the plan, correct it, then let it build.
- **Use one branch per block** (`s/…`, `p/…`, `w/…`, `d/…`). Merge once `uv run pytest -q` is green. Pull `main` every hour.
- **For a new block, start fresh** (`/clear`) with a new prompt built from your §11 timeline cell. Don't let one conversation run all day.
- If Claude wants to change a file you don't own, it must **stop and write the request** for the owner (or S for shared files) instead of editing.
- Every prompt has four parts: **who I am · what to read · the task with exact files · how we know it's done.**

### 15.2 S — the night before (repo skeleton everyone else depends on)
```
I am S (Swarm + Integrator) on a 4-person hackathon team. Read .plans/swarm-guided-rescue-build-spec.md fully; it is the source of truth.

Task: scaffold the repo so all four of us can start in parallel at H0. Create:
1. pyproject.toml for uv, Python 3.12, deps from §1.3, mujoco pinned ==3.17.0; console script `sgr` -> sgr.app:main with flags --map, --perception {mock,remote,local}, --speed, --no-render.
2. CLAUDE.md containing the ownership table (§1.1), the rules (§1.2), the coordinate conventions (§3), and "never call World.ground_truth() from sgr/swarm or sgr/guidance".
3. The directory layout from §1.1 with __init__.py files, plus sgr/bus.py (thread-safe queues between the sim loop and the server).
4. sgr/contracts.py: pydantic v2 models for everything in §4, §5 (Scenario), §8.5 (TraceRow) and §10.2 (every WS message, both directions), CONTRACT_VERSION = 1.
5. Facade stubs for World, Perception, Swarm, Guidance with the exact §4 signatures. The World stub is a fake that moves drones kinematically on the grid and fakes LiDAR from the grid, so nobody is blocked waiting for MuJoCo.
6. tools/make_test_map.py: draws maps/test/grid.png (30x20 m, 0.1 m/px, 1.8 m corridors, >=3 branches, one loop, one dead-end room) and writes maps/test/scenario.json meeting §5's requirements. Run it and commit the outputs.
7. sgr/tools/render_smoke.py: tiny MJCF, offscreen render 320x240, saves render_smoke.png.
8. tests: test_smoke.py (xfail until real modules exist), test_honesty.py (§12), test_contracts.py (JSON round-trip of every message model).
9. .gitignore: snapshots/, maps/*/raw/, web/node_modules, web/dist, __pycache__, .venv.

Do not write any swarm logic yet. Done when `uv sync`, `uv run pytest -q` and `uv run python -m sgr.tools.render_smoke` all pass. Plan mode first; show me the file list before writing.
```

### 15.3 S — H1 (swarm core)
```
I am S. Read CLAUDE.md and §8 of the spec. Block H1–H4, branch s/swarm-core.
Implement in sgr/swarm/: blackboard.py, belief.py (0.2 m belief grid from PriorMap), coverage.py (camera-frustum "searched" cells, §8.2), areas.py (Voronoi split by graph nodes + greedy assignment), planner.py (A*, §8.3), agent.py (reflex + tactical layers), trace.py (raises if evidence is missing), swarm.py (the Swarm facade).
Use the World stub and mock perception only; never import sgr.world internals. Add tests/test_planner.py and a debug script that plots the belief map + drone paths with matplotlib.
Done when 4 drones search >=90% of maps/test in under 120 sim-seconds, headless, and pytest is green.
```

### 15.4 P — H0 (Gemma bench on the demo laptop, Wi-Fi OFF) and H1 (sim core)
```
I am P (Perception + Sim core + demo laptop). Read CLAUDE.md, §4 and §7 of the spec. Two tasks, in order.

Task 1 (H0–H1, branch p/gemma-bench): write sgr/perception/gemma_bench.py. It builds a tiny MuJoCo scene in code with the §7.1 shapes (cluster of emissive orange-yellow cones for fire; capsule humanoid lying down in hi-vis yellow-green; flat orange decoy panel on a grey wall), renders each at 320x240 from 3 m, crops it, and sends it to Ollama (POST http://localhost:11434/api/chat, temperature 0, keep_alive -1, JSON-schema format, the prompt in §7.3) for each model tag passed on the command line. Record the latency over 5 runs, VRAM from nvidia-smi before and after, and the JSON answers in bench_results.md. I will pick the tag from that.

Task 2 (H1–H4, branch p/world): implement the World facade (§4) in sgr/world/ on maps/test: walls from grid.png merged into boxes; 4 drones (slide x/y/z + hinge yaw, velocity actuators, gravcomp); firefighter capsule; mocap pools of 40 debris + 30 fire parked at z=-50; survivor; decoy; LiDAR via mj_ray (36 rays, 6 m, geomgroup {0}, excluding the drone itself); heat model (§7.4); ONE mujoco.Renderer created on the main thread and reused for all drone cameras. Then sgr/perception/detector.py (§7.2).
Done when the smoke test runs with the real World instead of the stub. Plan mode first.
```

### 15.5 W — H0 (start this session BEFORE leaving to scan; review it when back)
```
I am W (World/Scan). Read CLAUDE.md and §6 of the spec. I'm going out to scan the building with Scaniverse. While I'm away, build the processing tools on branch w/scan-tools:
1. tools/scan_to_grid.py, exactly per §6.2 step 5: input sim.glb + entrance seed (x, y in metres) -> grid.png (0=wall, 255=free, 0.1 m/px), transform.json (§6.3 format), preview.png. Print the detected up-axis, floor height and bounding box so I can sanity-check them.
2. tools/prepare_scan.ps1 and tools/prepare_scan.sh: run the two gltf-transform commands from §6.2 step 4.
3. tools/propose_graph.py, per §6.2 step 7: skeletonise the free space -> nodes/edges written into scenario.json.
I don't have the scan yet, so generate a synthetic Y-up GLB (box rooms + corridor, textured) and test all three tools on it.
Only edit tools/ and maps/. Done when the synthetic test produces a sensible grid.png, preview.png and graph.
```
**W — H4 (editor):**
```
I am W. Read CLAUDE.md, §5 and §10.3 of the spec. Branch w/editor. Build the route editor in web/src/editor/ (D has already registered the /editor route to web/src/editor/index.tsx): grid.png background; node and edge tools; compromise toggle with an `at` slider; place fire / survivor / decoy / drone starts; flare t field; an "Auto-propose" button that calls POST /api/maps/{name}/propose_graph; a validation panel for the §5 requirements; Save via PUT /api/maps/{name}/scenario. Only edit web/src/editor/. If you need a server change, write it down as a request for D.
```

### 15.6 D — H0 (dashboard foundation)
```
I am D (Dashboard + 3D + Firefighter view). Read CLAUDE.md, §9 and §10 of the spec. Block H0–H4, branch d/dashboard.
1. web/: Vite + React + TS with @react-three/fiber, @react-three/drei and three. web/src/contracts.ts mirrors sgr/contracts.py exactly (check CONTRACT_VERSION on `hello`). A useSocket hook that handles JSON messages and binary frames ([u8 channel][u32 seq][JPEG]).
2. Routes: / (dashboard) and /editor, which only renders web/src/editor/index.tsx (W owns that folder; create an empty placeholder there once).
3. sgr/server/server.py: FastAPI run by uvicorn on a background thread; WS /ws; the REST endpoints in §10.2, including GET/PUT scenario and POST /api/maps/{name}/propose_graph (runs tools/propose_graph.py); a frame ring buffer; static web/dist. It talks to the sim loop only through the sgr/bus.py queues.
4. H1 milestone: a fake publisher sends a test JPEG + fake state at 10 Hz and the page displays it.
5. Then the Act 1 layout from §10.1: belief-map canvas, 2x2 drone panels, trace list, budget placeholder. Fonts readable on a 1080p projector.
Plan mode first.
```
