<div align="center">

# Swarm-Guided Rescue

**A phone scan of a real building becomes a live fire simulation. Four drones search it with their own sensors and an on-device Gemma model, re-plan when the building changes, then guide a firefighter to the survivor by voice and light.**

![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![MuJoCo 3.17](https://img.shields.io/badge/sim-MuJoCo%203.17-0B57D0)
![Gemma 4 E4B, local](https://img.shields.io/badge/vision-Gemma%204%20E4B%20(local%2C%20Ollama)-4CC88B)
![No cloud in the control loop](https://img.shields.io/badge/control%20loop-offline-EE5257)
![React + three.js](https://img.shields.io/badge/dashboard-React%20%2B%20three.js-61DAFB?logo=react&logoColor=black)

<img src="docs/media/act1-explore.webp" alt="Act 1: four drones explore the scanned building; camera feeds, LiDAR, decision log and comms update live" width="100%">

<sub>Act 1, 2× speed. Left: the 3D twin with an auto-director camera. Centre: each drone's camera, LiDAR and Gemma verdict. Bottom: every decision with the reading that caused it.<br>
▶ <a href="docs/media/full-run.mp4"><b>Watch the full uncut run (4 min, MP4)</b></a>: loading → Act 1 → a debris drop → Act 2 walk → mission complete.</sub>

</div>

---

## In one minute

- **The problem.** Being lost, caught or trapped is the second leading cause of firefighter deaths on the fireground (18.2%, USFA). Smoke takes away every visual reference, and nobody has a live picture of the building.
- **What we built.** A swarm goes in first. Four simulated drones explore a **digital twin made from an iPhone LiDAR scan** of the Marshgate building, find fire and a survivor **using only their own sensors**, then hold a chain of checkpoints along the safest route. A person walks that route in first person while the drones talk them through it.
- **Why you can trust what you see.** Every action is logged with the sensor reading behind it, and the code refuses to log an action without one. The swarm never sees ground truth, and a test fails the build if it tries. When the scene changes (a judge clicks to drop debris, a drone is lost to heat), the drones find out through their LiDAR and cameras and re-plan on screen.
- **Where it runs.** One laptop. MuJoCo on the CPU, Gemma 4 E4B through a local Ollama server on an 8 GB laptop GPU, and nothing in the control loop that needs the internet.

## How it scores against the judging criteria

| Criterion | Weight | What you will see | Jump to |
|---|:-:|---|---|
| **Perception-driven action** | 25% | Every row in the decision log cites its sensor and value. Uncertain detections trigger a closer look before anything is marked. | [§1](#1-perception-driven-action-25) |
| **Adaptation to change** | 25% | Debris dropped on the route: LiDAR flags it **0.2 s** later, the old route turns red and a new one appears. A drone is lost mid-mission and its area is reassigned. | [§2](#2-adaptation-to-change-25) |
| **Relevance and impact** | 20% | The deadliest failure, a safe route going unsafe while the firefighter is on it, caught live: "Stop. Route blocked." and a new route **0.5 s** later. | [§3](#3-relevance-and-impact-20) |
| **Operational autonomy** | 15% | One laptop, local Gemma, a measured edge budget, hard timeouts, and explicit rules for when the swarm is unsure. | [§4](#4-operational-autonomy-15) |
| **Ambition and technical quality** | 15% | Phone scan → physics twin → coordinating swarm → a human guided through smoke, backed by about 300 automated tests and an honesty test. | [§5](#5-ambition-and-technical-quality-15) |

---

## The demo, in two acts

<table>
<tr>
<td width="50%"><img src="docs/media/start-frame.jpg" alt="Start frame: the firefighter's view from the entrance with four drones overhead"></td>
<td width="50%"><img src="docs/media/dashboard-fire-confirmed.jpg" alt="Mission control dashboard during Act 1"></td>
</tr>
<tr>
<td><b>Deploy.</b> The mission opens from the firefighter's view at the entrance, four drones hovering overhead. Pressing <kbd>Start</kbd> sends them in.</td>
<td><b>Act 1: mission control.</b> The drones fan out, map the building, confirm four fires, lose one drone to a flare-up, find the survivor and form a relay chain.</td>
</tr>
<tr>
<td><img src="docs/media/act2-route-blocked.jpg" alt="Act 2: firefighter view, debris ahead, 'Stop. Route blocked.'"></td>
<td><img src="docs/media/mission-complete.jpg" alt="Mission complete summary card"></td>
</tr>
<tr>
<td><b>Act 2: the rescue.</b> A judge walks the route in first person through about 4 m of visibility, following the lead drone, a heads-up arrow and spoken directions.</td>
<td><b>Mission complete.</b> Time, distance, hazards avoided, corrections and re-plans, with the full decision log beside it.</td>
</tr>
</table>

---

## 1. Perception-driven action (25%)

> *"Every decision traces to what it perceived, and it corrects mid-run."*

**No action without evidence.** Every swarm decision is a `TraceRow`: time, drone, action, a plain-English reason, and an `Evidence` record (sensor, value, camera frame id or LiDAR scan). `Trace.log` raises an error if the evidence or its value is missing, and the pydantic contract requires both. The dashboard shows every row, and rows with a camera frame or LiDAR scan link to it (`img` / `scan`).

<img src="docs/media/decision-log.jpg" alt="Focused decision log: each row has a time, drone, action and the reading that caused it" width="100%">

Real rows from the recorded run:

```text
34.7  D4     INSPECT            D4 sees possible fire: fast detector 0.80, holding for Gemma
35.7  D4     CONFIRM_HAZARD     Fire confirmed at (20.5, 28.8): fast 0.80, Gemma 0.88 "Flames on the floor ahead." -> 0.85; 20 cells no-go (+1 m margin)
57.9  D4     DRONE_LOST         D4 silent for 3.0 s: declared lost
57.9  swarm  REASSIGN           D4 lost: area n9 back to the pool; area n9 unsafe until re-observed
68.1  D2     SURVIVOR_CONFIRMED Survivor confirmed at (36.6, 45.6): fast 0.85, Gemma 0.93 "A person lying on the floor." -> 0.90
109.7 D1     MARK_BLOCKED       D1 LiDAR: path blocked near (5.3, 7.9), not on the plan
109.7 swarm  REPLAN             Route blocked by debris near (5.3, 7.9) (D1): new route 67.9 m (+0.7 m)
```

**Four sensors per drone, each with a job:**

| Sensor | Simulated how | What the swarm does with it |
|---|---|---|
| **Camera** (320×240, 90° FOV) | MuJoCo offscreen render of the textured scan, with fire composited in | Fast colour detector → Gemma check → fire or survivor marked; also decides which cells count as *searched* (in view, ≤5 m, line of sight) |
| **2D LiDAR** (36 rays, 6 m) | MuJoCo `mj_multiRay` against the collision walls | Updates the belief map. A hit 0.3 m shorter than the prior map, seen twice, marks `blocked_new` and triggers a re-plan |
| **Temperature** | Heat field around each fire | Above 50 °C marks danger cells, above 70 °C the drone retreats; readings also locate heat the cameras haven't seen |
| **Heartbeat** | Each drone's link to the blackboard | 3 s of silence → `DRONE_LOST`, area reassigned and marked unsafe |

**Seeing → checking → acting.** The perception pipeline is two-stage, so the swarm never acts on one weak reading:

1. **Fast detector** (OpenCV HSV thresholds, a few ms per frame) flags a *possible* fire or hi-vis survivor. A 348-pose sweep of the map produced zero false fires or persons.
2. **Gemma 4 E4B** gets the full frame with a fixed prompt and a JSON schema: `fire`, `person`, `confidence`, and a caption of at most 12 words.
3. **Fusion**: combined = 0.4·fast + 0.6·Gemma when they agree (0.3·fast when they don't). Confirmed at **≥ 0.7**.
4. **Correcting itself mid-run**: below 0.7 the drone holds, moves to about 2 m from the target and asks again (up to 2 re-looks). If Gemma doesn't answer in time, the finding stays `UNCONFIRMED` and the swarm gets a second view instead of guessing. A clear disagreement on a re-look rejects the detection (`REJECT_DETECTION`).

---

## 2. Adaptation to change (25%)

> *"The scene changes mid-sequence and the agent visibly re-plans."*

**Anyone can change the world mid-run.** Clicking the 3D map drops debris or starts a fire in the MuJoCo world, not in the swarm's memory. The drones find out the honest way: LiDAR change detection for debris, camera + Gemma + heat for fire.

<table>
<tr>
<td width="55%"><img src="docs/media/act1-replan.webp" alt="Debris dropped on the route; LiDAR detects it; old route turns red and dashed; new green route appears"></td>
<td>

**Recorded in Act 1** (real time):

- `109.5 s` debris dropped on the route, 3.2 m from D1
- `109.7 s` **D1 LiDAR**: path blocked, not on the plan
- `109.7 s` **REPLAN**: new route 67.9 m (+0.7 m)
- old route drawn **red and dashed**, new route green, checkpoints re-posted

Across a scripted sweep, 8 of 9 judge-style clicks were noticed **0.2–12 s** after placement. The one miss was debris under 0.3 m from a wall, which LiDAR can't tell apart from the wall itself.

</td>
</tr>
</table>

**Losing a drone, live.** Every run loses exactly one drone during exploration. Usually it's the drone watching the fire, caught by a scripted flare-up; the heat build-up beforehand is real. If no flare-up fits, a drone loses its signal instead. The swarm:

- declares it lost after **3 s** without a heartbeat (`DRONE_LOST`),
- puts its unexplored area back in the pool (`REASSIGN`),
- marks that area **unsafe until someone re-observes it**,
- rebuilds the checkpoint chain for three drones.

The lost drone's feed turns to *NO SIGNAL*, its node is struck through on the comms graph, and the planner routes around it. The dashboard also has a per-drone stop button for failing a drone on demand.

**Re-planning everywhere.** A* runs on a 0.2 m belief grid: walls inflated 0.3 m, fire plus a 1 m margin impassable, heat and wall proximity cost extra. When one drone's LiDAR sees debris on another's path, the re-plan cites `seen_by`. The comms view shows the message travelling between them.

<table>
<tr>
<td width="50%"><img src="docs/media/comms-messages.jpg" alt="Comms graph focused: glowing links labelled with what each drone shared"></td>
<td width="50%"><img src="docs/media/survivor-found-chain.jpg" alt="Survivor confirmed by D2; relay chain formed; D4 offline"></td>
</tr>
<tr>
<td><b>Comms, focused.</b> Each shared decision lights the link from sender to receivers, labelled with what was said ("taking n11", "n6 done", "fire confirmed").</td>
<td><b>Survivor found, chain formed.</b> D2 confirms the survivor (fast 0.85, Gemma 0.93) and hovers over them. D1 goes back to meet the firefighter, D3 takes the midpoint, D4 is offline.</td>
</tr>
</table>

---

## 3. Relevance and impact (20%)

> *"A well-evidenced need, with the most critical failure mode handled on screen."*

**The need.**
- A US Fire Administration study of 1990–2000 found being **lost, caught or trapped** was the second leading immediate cause of fireground firefighter deaths, at **18.2%** ([Fire Engineering](https://www.fireengineering.com/firefighting/improving-situational-awareness-in-a-fire-structure/)).
- A NIOSH review found that between 1997 and 2001 **at least 20 firefighters died** and 5 were severely injured after becoming lost or disoriented inside burning structures, with poor visibility as the common thread ([NIOSH via CDC Stacks](https://stacks.cdc.gov/view/cdc/181198)).
- Large non-residential buildings, like the university building we scanned, make it worse: longer corridors, bigger floor plates, more ways to get turned around.

**The most critical failure mode: a safe route turning unsafe while the firefighter is on it.** We trigger it live in every Act 2 and the system handles it on screen:

<img src="docs/media/act2-route-blocked.webp" alt="Act 2: debris lands ahead of the firefighter; 'Stop. Route blocked.'; new route; 'New route. Follow me.'" width="100%">

| Time (sim) | What happens |
|---|---|
| 127.8 s | Debris dropped on the route about 5.5 m ahead of the walking firefighter |
| 128.0 s | Checkpoint drones D3 and D1 detect it with LiDAR (`MARK_BLOCKED`) |
| 128.3 s | `REPLAN`: route blocked by debris, new route 43.3 m |
| on screen | **"Stop. Route blocked."**, then **"New route. Follow me."** with the arrow swinging to the new heading |

Three safeguards keep the firefighter out of danger:

- Checkpoint drones keep scanning their stretch of the route.
- Any new blockage on the route triggers an immediate stop and re-route.
- Guidance can't drive a drone back into heat: a drone at 60 °C or more stays under the swarm's control until it cools below 45 °C.

**Firefighter guidance rules** (plain code, instant, offline), highest priority first:

| Situation | Response |
|---|---|
| Walks into fire | Pushed back: *"Stop. Fire ahead. Turn back."* with red screen edges and an alarm tone |
| Within 1 m of a known fire | Danger warning |
| Route blocked ahead | *"Stop. Route blocked."* → new route → *"New route. Follow me."* |
| Heat on the route itself | *"Heat on the route. Keep moving."* |
| No safe route | *"No safe route. Holding."* |
| Wrong way (on an off-route branch, or heading more than 120° away from the next checkpoint, for 2 s) | *"Wrong way. Turn around."* |
| Drifting more than 2 m off route for 2 s | *"Bear left / right"*, arrow pulses |
| Falling more than 6 m behind | The lead drone (normally 3 m ahead) stops: *"Follow me."* |
| Within 2 m of the next checkpoint drone | That drone takes over the guiding |
| Within 1.5 m of the survivor | Mission complete |

**Why it matters beyond this demo.** The twin comes from a free phone app, not specialist kit, so any building could be pre-scanned in minutes as part of fire-safety planning. The same pattern of scouting with sensors, proving a route and guiding a human along it extends to earthquakes, gas leaks, flooded buildings and assistive navigation.

---

## 4. Operational autonomy (15%)

> *"Runs within a stated edge-device budget and behaves sensibly when unsure."*

**Edge budget.** Everything in the decision loop runs on **one laptop**:

| Component | Runs on | Budget | Measured |
|---|---|---|---|
| Physics + sensors (MuJoCo 3.17) | CPU | control loop **10 Hz** (0.1 s step, 20 × 5 ms substeps) | full tick with rendering: mean 23 ms, p95 35 ms |
| Drone cameras (320×240) | integrated / discrete GPU | one camera render per tick | 18.6 ms mean, 25 ms p95 with fire in view |
| Fast detector (OpenCV) | CPU | every frame | a few ms |
| **Gemma 4 E4B** via Ollama | **RTX 5060 Laptop, 8 GB** | **≤ 12 s hard timeout**, never blocks the loop | +4.7 GB VRAM · p50 4.7 s / p95 5.3 s on the bench · p50 7.8 s / p95 8.3 s on the demo laptop with the 3D view running |
| Swarm planning + guidance | CPU, same process | every tick | inside the tick budget |
| **Network calls in the control loop** | — | **0** | the sim, swarm, perception and HUD guidance all run offline |

<sub>Model choice: the hour-one bench (`sgr/perception/bench/bench_results.md`) compared E4B (26/30 correct, p95 5.3 s) against E2B (20/30, p95 3.4 s, and it called the decoy a fire every time). We chose accuracy. With thinking turned off, Gemma answers in about 35 tokens per frame instead of about 645.</sub>

**When it is unsure, it behaves conservatively:**

| Situation | What the swarm does |
|---|---|
| Combined confidence < 0.7 | Hold, move closer (about 2 m standoff), look again, up to 2 re-looks. Nothing is marked on one weak reading |
| Gemma slow or down | Hard deadline on a separate worker thread; the result is `UNCONFIRMED` and the swarm gets a second view. The sim never waits |
| Still unsure after re-looks | `REJECT_DETECTION`: logged with its evidence, not acted on |
| Contact lost with a drone | Its last area is **unsafe until re-observed** |
| Temperature > 70 °C | Retreat at least 3.5 m to cooler cells and mark a danger zone |
| No safe route exists | `NO_ROUTE`: hold, re-check every 2 s, and send a drone to look at the blockage again |
| Drone running hot during Act 2 | At 60 °C or more, guidance stops steering it until it cools below 45 °C |

**Honest note on Act 2 voices.** The drones' spoken lines can optionally be reworded and voiced by Gemini (`--voice gemini`, needs an API key). It runs on its own thread outside the control loop: guidance never waits for it, danger lines never go through it, and every line has a fixed fallback. For a run with **no network at all**, use `--voice template`, which uses fixed lines spoken by the browser's local voices.

---

## 5. Ambition and technical quality (15%)

> *"An original approach to a demanding problem, executed to a high standard."*

**Real-to-sim in an afternoon.**

<table>
<tr>
<td width="40%"><img src="maps/marshgate-route/plan_sheet.png" alt="Marshgate route plan sheet: textured scan from above, metre grid, route graph, fires, debris, checkpoints, survivor"></td>
<td>

1. **Scan** a route through the Marshgate building with an iPhone Pro's LiDAR (Scaniverse, free).
2. **`tools/prepare_scan.sh`** turns the raw scan into `scan.glb` for the dashboard and `sim.glb` for the cameras.
3. **`tools/scan_to_grid.py`** slices it into a 0.1 m occupancy grid. **`walls.py`** greedily merges wall cells into a small set of MuJoCo collision boxes (a scanned mesh would collide as one convex lump).
4. **`tools/scan_to_mujoco.py`** builds a textured replica that only the cameras see, so the drones' images look like the real building.
5. **`tools/scan_route_draw.py` / `propose_graph.py`** build the route graph, and the editor at `/editor` places fires, debris and the survivor. `scan_check_scenario.py` checks it all with the real planner.

The result: a **39 × 52 m** twin with 13 route nodes, 4 fires, 4 debris piles, a survivor and a scripted flare-up.

</td>
</tr>
</table>

**Engineering we're proud of:**

- **Honest by construction.** `tests/test_honesty.py` greps `sgr/swarm` and `sgr/guidance` for any use of ground truth (comments included), forbids the swarm from importing the world or perception internals, and checks that every trace row carries evidence. The swarm learns only through `Readings` and `Detection`s.
- **Contracts as the only shared surface.** One pydantic file (`sgr/contracts.py`, version 5) mirrored in TypeScript (`web/src/contracts.ts`). Four people built four packages in parallel against facades (`World`, `Perception`, `Swarm`, `Guidance`).
- **Robust under stress.** Sweeps over Gemma latency (1–11 s) × start positions: **18/18** marshgate runs complete every beat, plus 9/9 on the test map. Peak drone heat fell from 409–521 °C to ≤ 162 °C after heat-aware planning.
- **Tested.** About 300 `pytest` tests (`uv run pytest -q`), including a kinematic `FakeWorld` so the swarm can be developed without MuJoCo, a replay tool that scripts judge actions, and a smoke test that gates every merge.
- **A dashboard built for judges.** A 3D twin with an auto-director camera that glides to events, live drone feeds with LiDAR minimaps and Gemma verdicts, a decision log you can expand, a swarm-links map, a comms graph, and a first-person firefighter view, all running in the browser with three.js.

<table>
<tr>
<td width="50%"><img src="docs/media/swarm-links.jpg" alt="Swarm links view: relay tree from the entrance to the drones, mapped roads"></td>
<td width="50%"><img src="docs/media/loading.jpg" alt="Loading screen"></td>
</tr>
<tr>
<td><b>Swarm links.</b> Relay tree from the entrance to each drone over the mapped route graph. It's a visual aid, not a simulated radio.</td>
<td><b>Start-up.</b> The dashboard stays behind a loading cover until the sim, scan and models are ready, then holds the sim at t = 0 for the Start button.</td>
</tr>
</table>

---

## Architecture

```mermaid
flowchart LR
    subgraph Offline["Before the demo"]
        S[iPhone LiDAR scan] --> T[tools/: grid, walls, textured replica, route graph]
    end
    T --> W
    subgraph Laptop["One laptop: the decision loop (no network)"]
        W["World (MuJoCo 3.17)<br/>walls, drones, firefighter,<br/>fire, debris, survivor"]
        W -- "camera frames, LiDAR,<br/>temperature" --> P["Perception<br/>fast detector + Gemma 4 E4B<br/>(Ollama, local GPU)"]
        W -- "LiDAR, temperature,<br/>heartbeats" --> SW["Swarm<br/>belief grid, A*, areas,<br/>hazards, chain"]
        P -- "Detections<br/>(score, caption, frame)" --> SW
        SW -- "velocity commands" --> W
        SW -- "route + checkpoints" --> G["Guidance (Act 2)<br/>arrow, subtitles,<br/>corrections"]
        G -- "lead drone commands" --> W
        SW -- "TraceRows with Evidence" --> B[(Bus)]
        G --> B
    end
    B <-- "WebSocket /ws<br/>JSON + binary JPEG frames" --> D["Dashboard<br/>React + three.js"]
    D -- "clicks: debris / fire,<br/>firefighter WASD" --> B
```

| Package | Role | Owner |
|---|---|---|
| `sgr/world/` | MuJoCo world: grid → walls, drones, firefighter body, LiDAR, cameras, fire and debris pools, heat | P |
| `sgr/perception/` | `mock` / `remote` / `local` perception, fast detector, Gemma client, fusion, budget probe | P |
| `sgr/swarm/` | Blackboard, belief grid, area assignment, A*, hazards, drone loss, rescue chain, decision trace | S |
| `sgr/guidance/` | Act 2: firefighter guidance rules, handover, corrections, summary, drone voices | P |
| `sgr/loop.py`, `sgr/app.py`, `sgr/bus.py`, `sgr/contracts.py` | 10 Hz sim loop, entry point, thread-safe queues, shared contracts | S |
| `sgr/server/`, `web/` | FastAPI + WebSocket server, React + three.js dashboard and firefighter view | D |
| `tools/`, `maps/`, `web/src/editor/` | Scan pipeline, route graph, scenario editor, disaster dressing | W |

---

## Quick start

**Requirements:** Python 3.12, [uv](https://docs.astral.sh/uv/), Node 22. For real perception: [Ollama](https://ollama.com) with `gemma4:e4b` and a GPU with about 5 GB free.

```bash
# 1. Python deps
uv sync

# 2. Build the dashboard (the server serves web/dist)
cd web && npm install && npm run build && cd ..

# 3. Run the demo map with mock perception (no GPU needed)
uv run sgr --map maps/marshgate-route --perception mock --voice template
#    → open http://127.0.0.1:8000 and press Start
```

**With real, on-device Gemma (the demo setup):**

```bash
ollama pull gemma4:e4b
uv run sgr --map maps/marshgate-route --perception local --voice template
```

**In the dashboard:**

| Key / action | What it does |
|---|---|
| <kbd>Enter</kbd> | Start the mission from the start frame. On a run that's already under way: <kbd>Enter</kbd> resumes, <kbd>R</kbd> restarts Act 1 |
| Click a tile | Focus it (<kbd>Esc</kbd> to go back) |
| Click tool on the 3D map | Drop debris or start a fire anywhere; the drones only find out through their sensors |
| ■ on a drone feed | Fail that drone (it loses its signal) |
| <kbd>G</kbd> | Send in the firefighter (Act 2), once the chain is in position |
| <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd>, arrows, mouse | Walk and look in the firefighter view |
| <kbd>F</kbd> | Full screen |

**Useful flags:** `--speed 2.5` (faster for video), `--no-render` (no camera rendering), `--no-server --max-ticks 600` (headless), `--voice auto|gemini|template|off`.

**Tests:** `uv run pytest -q`

---

## Repository layout

```text
sgr/
  app.py  loop.py  bus.py  contracts.py  config.py  snapshot.py
  world/        MuJoCo World facade (+ fake_world.py, kinematic, for tests)
  perception/   mock | remote | local, fast detector, Gemma, fusion, budget, bench/
  swarm/        blackboard, belief, areas, planner, hazards, rescue chain, trace
  guidance/     Act 2 rules, voices
  server/       FastAPI + WebSocket /ws
web/            React + three.js dashboard, firefighter view, route editor (/editor)
tools/          scan pipeline, route graph, test map, replay, debugging
maps/           marshgate-route (demo), synthetic, test
tests/          smoke, contracts, honesty, swarm, perception, guidance, replay, ...
docs/media/     README images and clips (captured from a real run)
```

---

## Limits we state openly

- **Simulation, not hardware.** One floor at a fixed flight height, drones driven by velocity actuators (no rotor physics), and no real radio limits. Not yet tested on real drones.
- **No microphone.** The original plan had drones follow the survivor's voice; it was cut for time. Survivors are found by camera.
- **Simplified fire.** Four fixed fires plus one scripted flare-up; fire doesn't spread.
- **Colour-based fast detector.** It depends on how fire and the hi-vis survivor are rendered, which is why Gemma checks every candidate.
- **Gemma latency.** Seconds per check, not milliseconds. The swarm is designed around it (hard deadlines, `UNCONFIRMED`, second views), but it slows confirmations. Gemma also misses the survivor from some angles (11 of 12 correct in our prompt tests).
- **Debris within 0.3 m of a wall** can't be told apart from the wall by LiDAR.
- **The edge budget is measured in the backend but not shown on the dashboard** in this build. The measured numbers are in the table above.
- **The firefighter's position in Act 2 comes straight from the sim** (standing in for a body-worn tracker); the drones still only learn about hazards through their sensors.
- **Screens and clips in this README** were captured with `--perception mock` (scripted Gemma answers with simulated latency), so they could be recorded on a machine without the demo GPU.

---

## Team

Built in a weekend by a team of four for the AgentVerse 2026 Robotics Track, each owning one slice of the system through shared contracts:

| Role | Owns |
|---|---|
| **W: World & Scan** | iPhone scan → twin pipeline, route graph, scenario editor, disaster dressing |
| **S: Swarm & Integration** | Swarm brain, sim loop, contracts, replay, merge gate |
| **P: Perception & Sim core** | MuJoCo world, fast detector, Gemma, fusion, Act 2 guidance and voices |
| **D: Dashboard & 3D** | Server, mission-control dashboard, 3D twin view, firefighter view |

Contributors: [@ashrayap-web](https://github.com/ashrayap-web) · [@JesonRamesh](https://github.com/JesonRamesh) · [@yash-joshi5379](https://github.com/yash-joshi5379) · [@ItzSmudge](https://github.com/ItzSmudge)

## Sources

- [Fire Engineering: Improving Situational Awareness in a Fire Structure](https://www.fireengineering.com/firefighting/improving-situational-awareness-in-a-fire-structure/) (USFA lost/caught/trapped statistic)
- [NIOSH review of firefighter structure fire fatality investigations, via CDC Stacks](https://stacks.cdc.gov/view/cdc/181198)
- Gemma bench: [`sgr/perception/bench/bench_results.md`](sgr/perception/bench/bench_results.md)
- Build spec: [`.plans/swarm-guided-rescue-build-spec.md`](.plans/swarm-guided-rescue-build-spec.md)
