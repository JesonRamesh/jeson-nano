# Swarm-Guided Rescue — Project Description

Nov 7, 2026 · @Ashraya

## Executive summary

We turn an iPhone LiDAR scan of a real building into a live fire simulation, where four drones find a trapped survivor using their own simulated sensors and an on-device Gemma model, re-plan when the building changes, and guide a firefighter to the survivor by voice and light, all on one laptop with the network switched off.

The prototype is deliberately simple: one laptop running a MuJoCo simulation, a local Python agent stack and a small web dashboard. Every drone decision traces to a sensor reading shown on screen, anyone can change the scene mid-run by clicking on the map, and nothing in the control loop needs the internet.

The working title is **Swarm-Guided Rescue**. It aims at three of the track's targets: coordinated fleets, adaptive autonomy and assistive perception.

## How we're judged, and how we score

The track scores five weighted criteria, and half the marks (50%) come from two things: decisions that visibly trace to perception, and visible re-planning when the scene changes. The design below is built around those two.

| Criterion | Weight | Top score, per the organisers | How we earn it | Where judges see it |
| --- | --- | --- | --- | --- |
| Perception-driven action | 25% | Every decision traces to what was perceived; corrects mid-run | Each drone action is logged with the sensor reading that caused it; drones take a closer look when a detection is uncertain and correct themselves | The decision trace panel beside the camera feeds |
| Adaptation to change | 25% | Scene changes mid-sequence; the agent visibly re-plans | Click anywhere to drop debris or fire; a drone is lost to heat; the old route turns red and a new one is drawn | The overview map, live |
| Relevance and impact | 20% | Well-evidenced need; most critical failure mode handled on screen | Firefighter disorientation statistics; the route going unsafe while the firefighter is on it, caught and corrected | The opening of the pitch and Act 2 |
| Operational autonomy | 15% | Runs within a stated edge-device budget; behaves sensibly when unsure | Everything local on one laptop with Wi-Fi off; a stated budget shown live; explicit rules for uncertainty | The budget display and the uncertainty behaviours |
| Ambition and technical quality | 15% | Original approach to a demanding problem, done well | Phone-scan digital twin, a coordinating swarm, and a human guided by it | The whole demo |

**Track rules we must meet.** The agent must take input from at least one simulated sensor (we use four: camera, LiDAR, microphone and temperature), choose actions from what it senses rather than a scripted path, and act by moving, navigating or communicating (the drones fly, navigate and talk). The track also warns that core decisions depending on a remote service score lowest on autonomy, so no cloud model sits in the control loop.

**Simulator.** We use MuJoCo, one of the lighter simulators the organisers list, so there is no question about whether it is accepted. It runs on CPU, so every teammate can run the full simulation on their own laptop; only the demo laptop's GPU is needed, and only for Gemma.

## The problem and its impact

Getting lost inside a burning building is one of the deadliest things that can happen to a firefighter, and the cause is nearly always the same: smoke removes every visual reference, and nobody has a live picture of the building.

- A US Fire Administration study of 1990–2000 found that being lost, caught or trapped was the second leading immediate cause of fireground firefighter deaths, at 18.2% ([Fire Engineering](https://www.fireengineering.com/firefighting/improving-situational-awareness-in-a-fire-structure/)).
- A NIOSH review found that between 1997 and 2001 at least 20 firefighters died and 5 were severely injured after becoming lost or disoriented inside burning structures, with poor visibility as the common thread ([NIOSH via CDC Stacks](https://stacks.cdc.gov/view/cdc/181198)).

The same blindness slows the search for civilians. Crews crawl room by room through smoke, not knowing where the fire is spreading, which corridors are blocked, or where people are trapped. Large non-residential buildings, like the university building we scan, make this worse: longer corridors, bigger floor plates and more ways to get turned around.

**What we change.** A drone swarm goes in first. It maps hazards, finds the survivor, and then stays in the building as a chain of living signposts, guiding the firefighter along the safest route by light and voice. The firefighter no longer navigates from memory in the dark; they follow a path the swarm has already proven.

**Why it scales.** The digital twin comes from a phone scan, not specialist equipment. Any building could be pre-scanned in minutes as part of fire safety planning, so the twin already exists before an incident. The same approach extends to earthquakes, gas leaks and flooded buildings, and to everyday assistive navigation for blind and partially sighted people.

## The solution

Swarm-Guided Rescue is a simulated drone swarm that perceives a real building's digital twin through its own sensors, decides on-device, and acts together to find a survivor and guide a human to them.

1. **A real-to-sim digital twin.** A route through the Marshgate building is scanned with an iPhone Pro's LiDAR (Scaniverse, free). In MuJoCo the mesh is used for visuals only, and a 2D occupancy grid derived from it generates the solid walls the drones collide with and sense.
2. **Drones that sense for themselves.** Each drone has a camera, a 2D LiDAR, a microphone and a temperature sensor. They start with the pre-incident floor plan but discover hazards, debris and the survivor only through those sensors.
3. **On-device decisions.** Planning, coordination and safety rules run as plain code on the laptop. A local Gemma 4 model checks what the cameras see and explains it. No decision in the loop needs the internet.
4. **A human in the loop.** Once the survivor is found, the drones form a chain of checkpoints, and a judge walks the route in first person while the drones guide them by voice and light, correcting them if they go wrong.

**What we deliberately left out to keep it buildable:** a second device, cloud AI in the control loop, a speech API, push-to-talk, Gaussian splats and realistic radio networking. All of these are possible extensions, not requirements.

## Demo flow

The live demo runs on one laptop in two acts: the swarm finds the way (about 2.5 minutes), then a judge walks it as the firefighter (about 1.5 minutes).

**Act 1: Mission control**

1. **Set the scene.** Show the iPhone scan becoming the 3D twin, and point out that Wi-Fi is off: "Everything you're about to see runs on this laptop."
2. **Explore.** Four drones fan out from the entrance. Their LiDAR maps fill in on the overview, and each camera panel shows live detections. The decision trace panel logs each move with its cause, such as "D1 heading east: unexplored area at corridor end (LiDAR)."
3. **Uncertain, then corrected.** Drone 2's camera picks up an orange glow. The fast detector flags possible fire, but Gemma's confidence is low, so Drone 2 moves closer for a second look before deciding. This is the agent behaving sensibly when unsure and correcting itself mid-run.
4. **Fire confirmed.** The café is marked blocked, and every planned route avoids it.
5. **Following a sound.** Microphone levels rise towards the stairwell, and the drones steer up the loudness gradient.
6. **A drone is lost.** Drone 3's temperature sensor climbs near the fire, and then its heartbeat stops. The swarm marks that area unsafe until someone re-observes it, and reassigns Drone 3's unexplored area.
7. **Survivor found.** Drone 4's camera detects the survivor and Gemma confirms. The planner computes the safest route, and the three remaining drones form the checkpoint chain.
8. **Unscripted change.** A judge clicks anywhere on the map to drop debris or start a fire. If it blocks the route, the nearest drone's LiDAR detects it, the old route turns red, and a new route and new checkpoint positions appear.

**Act 2: The rescue (same laptop, firefighter view)**

9. **Guided walk.** The judge walks in first person through smoke, following the glowing lead drone and its spoken directions.
10. **Going the wrong way.** If they turn towards the fire, the nearest drone flashes red and warns them back.
11. **The critical failure, handled.** While the judge walks, a teammate drops debris on the route ahead. The checkpoint drone detects it, tells the firefighter to stop, and guides them onto the new route. This is the failure that matters most in real life (a route that was safe becoming unsafe) caught on screen.
12. **Mission complete.** The summary shows time, hazards avoided, corrections given and the full decision trace.

## Frontend and views

The demo uses two windows on one laptop: MuJoCo's own 3D viewer for the scene, and a small web dashboard that puts perception next to action.

**MuJoCo viewer.** Shows the scanned building, drones, fire, debris and the survivor in 3D. Python draws extra markers into the scene each step: the planned route (red when it becomes unsafe), the checkpoints, and each drone's LiDAR rays.

**Web dashboard** (one page, fed by Python over a WebSocket)

- **Map and change tool.** A top-down map of what the swarm believes: explored areas, hazards, the route and checkpoints. Clicking the map drops debris or starts a fire at that spot in the MuJoCo world.
- **Four drone panels.** Each drone's MuJoCo camera image with detection boxes and labels, a sensor strip (LiDAR scan, microphone level, temperature), Gemma's caption with its confidence, and the drone's status. A lost drone's panel turns to static with SIGNAL LOST.
- **Decision trace.** A live list where each row reads: time, drone, action, and the sensor reading that caused it. Clicking a row shows the camera frame or LiDAR scan behind the decision. This is our main evidence for the 25% perception criterion.
- **Edge budget display.** Network off, Gemma latency, GPU memory in use, simulation rate and frame rate, updated live.

**Firefighter mode.** The firefighter is a body in MuJoCo with a head camera. The dashboard switches to a large first-person view streamed from that camera, with MuJoCo's fog limiting visibility. Keys pressed in the dashboard move the firefighter, and the heads-up arrow, distance and subtitles are drawn over the image. Spoken directions use the browser's built-in speech in the dashboard, which works offline.

**Keeping it light.** Drone cameras render offscreen at about 320 by 240 pixels, 10 to 15 times a second, and the firefighter view at about 640 by 480, around 20 times a second. This works on integrated graphics, and images travel to the dashboard as JPEGs.

## System architecture

Everything in the decision loop runs on one laptop: MuJoCo simulates the world and renders the drones' sensors, a local Python agent stack turns sensor data into actions, and a web dashboard shows both.

&#91;embedded content: system architecture · scan, MuJoCo simulation, local agent stack\]

Sensor data flows from the simulation to the agent stack, and the planner sends velocity commands back to the drones. The scan is the only step done in advance.

| Component | Technology | Runs on | Speed target (measure on day one) |
| --- | --- | --- | --- |
| Scan and twin | iPhone Pro, Scaniverse, Blender | Phone, then laptop, before the demo | Not in the loop |
| Simulation | MuJoCo, through its Python bindings | Laptop CPU | Physics faster than real time; agents updated 10 times a second |
| Sensors | MuJoCo cameras; LiDAR from MuJoCo ray casting; sound and heat computed in Python from distance and walls | Laptop CPU or integrated graphics | Cameras 10 to 15 times a second; LiDAR 10 |
| Agent stack | Python, in the same process as the simulation | Laptop CPU | 10 updates a second |
| Fast detector | Colour thresholds with OpenCV | Laptop CPU | A few milliseconds per frame |
| Gemma 4 E4B | Ollama, fully local | Laptop GPU (8 GB) | Roughly 1 to 2 seconds per check (an estimate) |
| Dashboard | Web page fed by FastAPI over a WebSocket | Browser on the same laptop | 10 updates a second |

**Choosing the Gemma size.** Gemma 4 E4B is the small edge model in Google's open Gemma 4 family, described as the sweet spot for 8 GB graphics cards ([InsiderLLM](https://insiderllm.com/guides/gemma-models-guide/)). Most RTX 5060 laptops have 8 GB of video memory, so check yours. The larger Gemma 4 12B targets laptops with 16 GB of video memory ([Google Developers Blog](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/?linkId=62164598)), so it is probably too big here.

**Gemini's place.** No cloud model sits in the loop. If the hackathon has a Gemini prize, the one safe use is after the mission: sending the decision log to Gemini for a written after-action report, clearly labelled as outside the control loop.

**MuJoCo implementation notes**

- **Walls.** Generate box-shaped walls from the occupancy grid, merging neighbouring cells into longer boxes to keep the count low. Don't use the scan mesh for collision: MuJoCo collides with a mesh as its convex hull, so a scanned room would become one solid lump. Load the mesh as visuals only, with collisions switched off.
- **Drones.** Skip rotor physics. Each drone body slides in x, y and z and turns about its vertical axis, driven by velocity actuators, and the planner sends velocity commands. A camera is attached to each drone.
- **LiDAR.** 36 horizontal rays per drone using MuJoCo's ray casting, with a 6 m range.
- **Fire, debris and the change tool.** MuJoCo can't add objects while running, so build the model with spare hidden objects, for example 200 fire tiles and 30 debris blocks parked out of sight. Spreading fire, or a click on the map, moves one into place and makes it visible (and, for debris, solid).
- **Survivor.** A simple body in a bright high-visibility colour, so the camera detector can find it.
- **Smoke.** MuJoCo's built-in fog for the firefighter's camera; drone images near the fire are darkened in Python to degrade perception.
- **Overlays.** The route, checkpoints and LiDAR rays are drawn as extra decorations in the viewer each step, not as objects in the model.

**Developing without a GPU.** Only the demo laptop has a GPU, so perception sits behind one interface with three settings: mock (instant fake detections, for most development), remote (Ollama on the demo laptop over the team's Wi-Fi, during development only), and local (the real Gemma with the network off, for the demo and the video). Everyone else needs only Python and MuJoCo.

## Agent design

Each drone is an agent with three layers, all on-device, and the decision-making is deliberately plain code so that every action can be traced back to a sensor reading.

| Layer | How often | Where | What it does |
| --- | --- | --- | --- |
| Reflex | Every simulation step | Simulation loop (Python) | Keeps about 2 m from other drones, steers away from walls using short LiDAR rays, smooths velocity commands |
| Tactical | 10 times a second | Python | Updates the drone's own map from its LiDAR, picks the next unexplored spot in its area, plans a path with A\* around known hazards, sends a heartbeat |
| Perception check | When the fast detector flags something | Python and Gemma | Confirms fire or a person, writes a short caption, and gives a confidence score |

**What each sensor drives**

| Sensor | What it detects | What the swarm does with it |
| --- | --- | --- |
| 2D LiDAR | Walls, debris, new blockages | Updates the map; a blocked route triggers a re-plan |
| Camera, checked by Gemma | Fire, the survivor | Marks hazards and the survivor's position |
| Microphone | Loudness of the survivor's voice | Steers the search towards louder areas |
| Temperature | Heat near fire | Marks danger zones; tells a drone to retreat |

**Coordination.** Drones share one state: a merged map, hazards (each with source, time and confidence), the survivor estimate, and the plan. Areas are assigned greedily: each drone takes the nearest unexplored area nobody else holds. When a drone is lost, its area goes back into the pool.

**The checkpoint chain.** Once the survivor is confirmed, A\* finds the safest route from the entrance on the merged map. Checkpoints go at the route's major turns. One drone hovers over the survivor, the nearest drones take the checkpoints, and one returns to the entrance to meet the firefighter.

**The decision trace.** Every action is written as time, drone, action, reason, and evidence (which sensor, what value, which camera frame). The rule in the code is simple: no action without evidence. This is what makes the 25% perception criterion easy to show.

**The Gemma check.** The fast detector sends the cropped camera frame to Gemma with a fixed prompt asking for JSON: whether there is fire, whether there is a person, a confidence from 0 to 1, and a caption of at most 12 words. Code combines this with the detector's own score before anything is marked.

## Uncertainty, change and failure

The swarm handles three kinds of trouble on screen: not being sure, the world changing, and losing a drone. The most critical failure, a safe route becoming unsafe while the firefighter is on it, is caught live.

**When unsure (operational autonomy)**

| Situation | What the swarm does |
| --- | --- |
| A detection with low combined confidence (below 0.7) | Holds position, moves closer and looks again, up to twice; if still unsure, asks another drone for a second view. Nothing is marked on one weak reading |
| Gemma is slow or unavailable | Uses the fast detector alone, but marks findings unconfirmed until a second view agrees |
| A stretch of route not seen for over 30 seconds | Marks it unverified; a drone re-checks before the firefighter enters it |
| Contact lost with a drone | Treats its last area as unsafe until someone re-observes it |
| Temperature above the limit | The drone retreats and marks a danger zone |
| Two sensors disagree | The more cautious reading wins: assume blocked |

**When the world changes (adaptation).** Fire spreads by itself, and anyone can click the map to drop debris or start a fire. The agents are never told about these changes: they find out only when a drone's LiDAR or camera perceives them. Within about a second of detection the plan is re-run, the old route turns red, the new route appears, and checkpoints move. Keeping this honest (changes go into the world, not into the agents' memory) is what makes the re-planning count as perception-driven.

**Losing a drone.** Drone 3's temperature climbs near the fire; it is ordered to retreat but is caught by a flare-up. After three seconds without a heartbeat it is declared lost. Its last area is marked unsafe, its unexplored area is reassigned, and the checkpoint chain is rebuilt for three drones. The heat build-up is real; only the moment of the flare-up is timed, so the demo is reliable.

**The critical failure mode.** The worst outcome is guiding the firefighter into danger along a route that was safe when it was planned. Three safeguards prevent it: checkpoint drones keep scanning their stretch of the route; any new blockage triggers an immediate "Stop" and a re-route; and the firefighter is never sent into a stretch marked unverified. Act 2 shows this happening live.

**Limits we will state openly.** One floor at a fixed flight height; simplified fire and smoke; colour-based detection that depends on how things are rendered; no real radio limits; not yet tested on real drones.

## Firefighter mode

In Act 2 a judge becomes the firefighter on the same laptop: they walk the building in first person through smoke while the drones guide them, wait for them, and talk them back when they go wrong.

**Controls.** Keys pressed in the dashboard move the firefighter's body in MuJoCo: WASD to move and the arrow keys to turn, at walking pace, with the grid-generated walls solid. Fog limits visibility to a few metres, so the glowing drones are the clearest landmarks.

**Follow and hand over.** The lead drone hovers about 3 m ahead. If the firefighter falls more than about 6 m behind, it stops and calls back. Within about 2 m of the next checkpoint drone, that drone takes over, until they reach the drone hovering over the survivor.

**Corrections**

| Situation | Trigger | Response |
| --- | --- | --- |
| Drifting | More than about 2 m off the route for 2 seconds | The heads-up arrow pulses; "Bear left" |
| Wrong way | Heading down a branch not on the route | The nearest drone flashes red; "Wrong way. Turn around." |
| Danger | Entering the margin around a known hazard | Red screen edges, alarm tone; "Stop. Fire ahead. Turn back." |
| Route becomes unsafe | A checkpoint drone detects new debris or fire ahead | "Stop. Route blocked." Then the new route is drawn and the drones reposition |

All of this is plain code, and the voice uses the browser's built-in speech, so it responds instantly and works offline. If the firefighter walks into fire anyway, the screen flashes and they're pushed back with a warning, rather than failing the run.

**Mission complete.** Reaching the survivor ends the mission and shows the summary: time, distance, hazards avoided, corrections given, and the full decision trace.

## Risks and fallbacks

The biggest risk is still scope; with the cloud dependency gone, every remaining risk has a fallback that keeps a working demo.

| Risk | Likelihood | Mitigation | Fallback |
| --- | --- | --- | --- |
| Too much to build in the time | High | Build in tiers; freeze features an hour before judging | Cut the strong tier; the core still scores on every criterion |
| MuJoCo rendering problems on a teammate's laptop | Medium | Install and test on every laptop in the first hour | Develop with mock camera images; render cameras only on the demo laptop |
| Scan mesh collides wrongly or slows the viewer | Medium | Mesh is visual only; walls come from the grid; simplify the mesh in Blender | Show grid-generated walls only, coloured to match the building |
| Atrium scan fails (glass, high ceilings, 5 m LiDAR range) | Medium | Scan a compact route: entrance, corridor, café, stairs base | Polycam as a second app; a hand-drawn grid of the same route |
| Gemma 4 E4B too slow or too big for the GPU | Medium | Only run it on flagged frames, one at a time | The smaller E2B model, or the fast detector alone with the second-view rule |
| Detector false alarms (exit signs, orange objects) | Medium | Gemma check plus a second look | Show it as a feature: the swarm corrects itself on screen |
| Camera rendering slows the simulation | Low | Small images, 10 to 15 per second | Refresh fewer drone panels at once |
| Judges think it's scripted | Medium | Let a judge use the change tool; agents learn about changes only through sensors | The decision trace shows the evidence behind each action |
| The one-minute take fails | Medium | Record several takes | Submit the best clean continuous take |
| Overclaiming real-world readiness | Medium | State limits openly | Present real drones as the next step |
| Permission to scan the building | Low | Ask the venue first | Scan a different public route |

## Build plan

The core tier alone scores on all five criteria, so build it first, add the strong tier only once the core runs end to end, and freeze features an hour before judging.

**Tiers**

- **Core (must have).** Scanned route and grid; four drones with reflex and tactical layers; LiDAR, camera, microphone and temperature sensors; the fast detector; shared state; the decision trace panel; the change tool and live re-planning; the drone loss; the checkpoint chain; firefighter mode with corrections, including the route becoming unsafe; browser speech; the edge budget display; a full run with Wi-Fi off.
- **Strong (should have).** Gemma checks with captions and confidence; the second-look behaviour driven by Gemma's confidence; the uncertainty overlay showing unverified route stretches; the mission summary screen.
- **Stretch (nice to have).** A heat threshold that triggers the drone loss by itself; Gaussian splat visuals; an optional Gemini after-action report outside the loop.

**Team roles (four people, each with their own Claude Code session)**

| Person | Owns | Done when |
| --- | --- | --- |
| 1. World | Scan, grid, the MuJoCo model (walls, hidden fire and debris, survivor), fire spread, sound and heat, the change tool's effect on the world | The model loads, the world changes on a click, and sensors can read it |
| 2. Swarm | LiDAR maps, area assignment, A\*, unsure rules, drone loss, checkpoints, decision trace | Drones explore, re-plan and form the chain from sensor data alone |
| 3. Perception and autonomy | Camera capture, fast detector, Gemma through Ollama (mock, remote and local settings), confidence fusion, edge budget measurement | Detections with confidence arrive within the stated budget, offline |
| 4. Dashboard | The web dashboard (map, drone panels, trace, budget, change tool), viewer overlays, firefighter mode, voice | Both modes run smoothly on the demo laptop |

**First hour, together (the track says get the stream running before writing agent code)**

- [ ] MuJoCo installed and running on every laptop
- [ ] A test scene with one drone whose camera image renders
- [ ] That image showing on the dashboard over a WebSocket
- [ ] Gemma answering one frame locally through Ollama on the demo laptop, with Wi-Fi off
- [ ] The shared state format agreed

**Schedule for a one-day build (about 10 hours)**

1. **Hour 0 to 1:** the first-hour checklist; Person 1 leaves to scan.
2. **Hours 1 to 4:** in parallel, scan and grid; drones exploring with LiDAR maps on a placeholder grid; fast detector working; overview and drone panels.
3. **Hours 4 to 6:** swap in the real grid; change tool and re-planning; decision trace panel; checkpoint chain.
4. **Hours 6 to 8:** drone loss; firefighter mode with corrections; edge budget display; Gemma checks.
5. **Hours 8 to 9:** strong-tier polish; record the one-minute video takes.
6. **Hours 9 to 10:** feature freeze; write the one-page summary; rehearse the pitch twice.

## Deliverables and pitch

The track asks for three things: a one-page technical summary, a simulation video of up to one minute, and a five-minute live pitch. Each maps onto work already in this plan.

**1. Technical summary (one page, PDF).** Build it from this doc's sections:

| The organisers ask for | Take it from |
| --- | --- |
| User and problem | The problem and its impact |
| Key features | The solution |
| System architecture | The architecture diagram |
| Where each model runs, with hardware and latency | The component table, with latencies we have actually measured |
| Limits | Limits we will state openly |
| Critical failure mode and safety | The critical failure mode, and the rules for when unsure |

**2. Simulation video (up to one minute, one continuous take).** It must show perception on screen next to the actions, and one unscripted change. Screen-record the dashboard and the MuJoCo viewer side by side, so the drone panels and decision trace are always visible next to the 3D scene:

1. **0 to 10 seconds:** drones launch and fan out; maps fill in.
2. **10 to 25 seconds:** an uncertain detection, a closer look, fire confirmed; routes avoid the café.
3. **25 to 40 seconds:** someone who hasn't rehearsed it clicks anywhere to drop debris; a drone's LiDAR detects it; the route turns red and re-plans.
4. **40 to 55 seconds:** survivor found; the checkpoint chain forms.
5. **55 to 60 seconds:** the edge budget display, showing the network off.

**3. Pitch (five minutes, live).** The organisers want four answers: what problem the robot solves, how it works, what it can and cannot do, and why it matters to its user.

1. **The problem (40 seconds).** Firefighters lose their bearings in smoke, and getting lost is one of the leading causes of their deaths on the fireground.
2. **How it works (50 seconds).** Phone scan to digital twin; four drones with four sensors; on-device decisions; one architecture slide.
3. **Live demo (2.5 minutes).** A shortened Act 1, with a judge invited to use the change tool, then a judge walks Act 2, including the route becoming unsafe.
4. **What it can and cannot do (30 seconds).** Read the limits plainly; judges reward honesty here.
5. **Why it matters (20 seconds).** Any building can be pre-scanned with a phone, and the swarm gives firefighters a verified way in and out.

Have answers ready for: how much is scripted (agents only learn about changes through sensors; the trace shows the evidence), what happens with no connection (nothing changes, it's already offline), and what it would take to run on real drones.

## Sources

- AgentVerse 2026 Robotics Track slides (challenge, simulation stack, deliverables, judging), provided by the team
- [Fire Engineering: Improving Situational Awareness in a Fire Structure](https://www.fireengineering.com/firefighting/improving-situational-awareness-in-a-fire-structure/) (USFA lost, caught or trapped statistic)
- [NIOSH review of firefighter structure fire fatality investigations, via CDC Stacks](https://stacks.cdc.gov/view/cdc/181198)
- [InsiderLLM: Gemma models guide](https://insiderllm.com/guides/gemma-models-guide/) (Gemma 4 E4B on 8 GB graphics cards)
- [Google Developers Blog: Gemma 4 12B developer guide](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/?linkId=62164598) (16 GB hardware target)
