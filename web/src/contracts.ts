// TypeScript mirror of sgr/contracts.py (owned by S). This file is owned by D.
//
// Mirrors contracts.py v1 field-by-field (names, optionality, literals).
// Rules: S bumps CONTRACT_VERSION on any change; D mirrors it here within 30 min.
// After H1, fields are only ever added — never renamed or removed.
//
// Conventions: metres, z up, x east, y north, radians CCW from +x unless `_deg`.
// Belief cells (map_delta) are 0.2 m: i = column (x), j = row from the bottom (y).

export const CONTRACT_VERSION = 1

// ---- literals -----------------------------------------------------------

export type Label = 'fire' | 'person' | 'decoy'
export type DetectionStatus = 'pending' | 'confirmed' | 'low_conf' | 'rejected' | 'unconfirmed'
export type Sensor = 'lidar' | 'camera' | 'gemma' | 'temp' | 'heartbeat' | 'planner'
export type Action =
  | 'ASSIGN_AREA'
  | 'EXPLORE'
  | 'INSPECT'
  | 'CONFIRM_HAZARD'
  | 'REJECT_DETECTION'
  | 'UNCONFIRMED'
  | 'MARK_BLOCKED'
  | 'REPLAN'
  | 'RETREAT'
  | 'DRONE_LOST'
  | 'REASSIGN'
  | 'SURVIVOR_CONFIRMED'
  | 'CHAIN_POSITION'
  | 'NO_ROUTE'
  | 'ESCORT'
  | 'FF_CORRECTION'
  | 'MISSION_COMPLETE'
export type NodeKind = 'entrance' | 'junction' | 'door' | 'room' | 'stairs'
export type CellState = 'wall_prior' | 'free' | 'blocked_new' | 'fire' | 'danger' | 'unsafe'
/** idle = no route requested yet; none = no safe route exists. */
export type RouteStatus = 'idle' | 'ok' | 'replanning' | 'none'
export type EventKind =
  | 'fire_confirmed'
  | 'false_alarm'
  | 'drone_lost'
  | 'blocked'
  | 'replan'
  | 'survivor_found'
  | 'chain_formed'
  | 'no_route'
  | 'mission_complete'
export type Severity = 'info' | 'warn' | 'danger'

export type XYPoint = [number, number]

// ---- primitives ---------------------------------------------------------

export interface XY {
  x: number
  y: number
}

export interface Pose {
  x: number
  y: number
  z: number
  yaw: number
}

export interface Handle {
  id: string
  kind: 'debris' | 'fire'
  x: number
  y: number
}

// ---- scenario (§5) — used by the route editor (W) ------------------------

export interface GridRef {
  file: string
  resolution: number
  origin: XYPoint
}

export interface ScanRef {
  file: string
  transform: string
}

export interface Entrance {
  x: number
  y: number
  yaw: number
}

export interface GraphNode {
  id: string
  x: number
  y: number
  kind: NodeKind
}

export interface GraphEdge {
  id: string
  a: string
  b: string
  width: number
}

export interface Graph {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface Compromise {
  edge: string
  at: number // 0..1 along a -> b
  kind: 'debris'
}

export interface Fire {
  x: number
  y: number
  size: number
}

export interface Survivor {
  x: number
  y: number
  yaw: number
}

export interface Decoy {
  x: number
  y: number
  kind: string
}

export interface DroneSetup {
  ids: string[]
  start: XYPoint[]
  altitude: number
}

export interface Flare {
  t: number
  target: string
}

export interface Heat {
  limit_c: number
  ambient_c: number
}

export interface Beats {
  decoy_drone: string | null
  d3_area_node: string | null
}

export interface Scenario {
  version: number
  name: string
  grid: GridRef
  scan: ScanRef | null
  entrance: Entrance
  graph: Graph
  compromised: Compromise[]
  fires: Fire[]
  survivor: Survivor
  decoy: Decoy | null
  drones: DroneSetup
  flare: Flare | null
  heat: Heat
  beats: Beats
}

export interface Transform {
  M: number[][] // 4×4 row-major, raw GLB -> sim frame
  pitch: number
  grid_origin: XYPoint
  grid_size: [number, number] // (w, h) px
}

// ---- perception / trace -------------------------------------------------

/** Pixel bbox in a 320×240 frame. */
export interface Box {
  x: number
  y: number
  w: number
  h: number
  label: Label
  score: number
}

export interface Evidence {
  sensor: Sensor
  value: unknown // number, string, list or dict; never null
  frame_id: string | null
  scan: number[] | null // LiDAR ranges when sensor === 'lidar'
}

export interface TraceRow {
  id: string
  t: number
  drone: string // drone id, or "swarm"
  action: Action
  reason: string
  evidence: Evidence
}

// ---- pieces shared by WS messages ---------------------------------------

export interface Route {
  points: XYPoint[]
  status: RouteStatus
  version: number
}

export interface OldRoute {
  points: XYPoint[]
  until: number // sim time when it stops being drawn
}

export interface Checkpoint {
  x: number
  y: number
  drone: string | null
}

export interface SurvivorState {
  status: 'unknown' | 'suspected' | 'confirmed'
  x: number | null
  y: number | null
  conf: number | null
}

export interface DroneState {
  id: string
  x: number
  y: number
  z: number
  yaw: number
  status: string // chip text: EXPLORING / INSPECTING / RETREAT / LOST
  task: string
  temp: number
  lidar: number[] // 36 rays, ray k at yaw + k·10°, 6.0 = no hit
  alive: boolean
}

export interface FFState {
  x: number
  y: number
  yaw: number
}

export interface MapMeta {
  name: string
  grid_url: string
  resolution: number // grid.png metres per pixel (0.1)
  size: [number, number] // (w, h) px of grid.png
  origin: XYPoint
  belief_resolution: number // 0.2
}

export interface DroneMeta {
  id: string
  colour: string // "#rrggbb"
}

export interface GemmaView {
  caption: string | null
  conf: number | null
  pending: boolean
}

// ---- server -> client JSON ----------------------------------------------

export interface HelloMsg {
  type: 'hello'
  contract_version: number
  map: MapMeta
  graph: Graph // structure only, no hidden fields
  entrance: Entrance
  drones: DroneMeta[]
  transform: Transform | null
}

export interface StateMsg {
  type: 'state'
  t: number
  speed: number
  act: 1 | 2
  phase: string
  drones: DroneState[]
  ff: FFState | null
  route: Route
  old_routes: OldRoute[]
  checkpoints: Checkpoint[]
  survivor: SurvivorState | null
  click_handles: Handle[]
}

export interface MapDeltaMsg {
  type: 'map_delta'
  cells: [number, number, CellState][] // [i, j, state] at 0.2 m
  searched: [number, number][] // newly searched cells
  map_full: string | null // base64 PNG every 5 s for resync
}

export interface TraceMsg extends TraceRow {
  type: 'trace'
}

export interface DetectionMsg {
  type: 'detection'
  drone: string
  frame_id: string
  boxes: Box[]
  gemma: GemmaView | null
  status: DetectionStatus
}

export interface EventMsg {
  type: 'event'
  kind: EventKind
  drone: string | null
  xy: XYPoint | null
}

export interface GuidanceMsg {
  type: 'guidance'
  arrow_deg: number
  distance: number
  subtitle: string
  say: string | null
  alarm: boolean
  severity: Severity
}

export interface BudgetMsg {
  type: 'budget'
  network: 'offline' | 'online' | 'unknown'
  model_tag: string | null
  gemma_p50_s: number | null
  gemma_p95_s: number | null
  gemma_queue: number
  vram_used_mb: number | null // null = N/A on this machine
  vram_total_mb: number | null
  sim_rtf: number | null
  speed: number
  control_hz: number | null
  camera_fps: number | null
}

export interface SummaryMsg {
  type: 'summary'
  time_s: number
  distance_m: number
  hazards_avoided: number
  corrections: number
  replans: number
  trace_url: string | null
}

export type ServerMsg =
  | HelloMsg
  | StateMsg
  | MapDeltaMsg
  | TraceMsg
  | DetectionMsg
  | EventMsg
  | GuidanceMsg
  | BudgetMsg
  | SummaryMsg

// Binary frame: [u8 channel][u32 frame_seq, big-endian][JPEG bytes]
// (FRAME_HEADER = struct.Struct(">BI") in sgr/contracts.py). channel 1–4 = D1–D4.
export interface DroneFrame {
  channel: number
  seq: number
  jpeg: Blob
}

// ---- client -> server JSON ----------------------------------------------

export interface ClickMsg {
  type: 'click'
  mode: 'debris' | 'fire'
  x: number
  y: number
}
export interface UndoMsg {
  type: 'undo'
}
export interface ControlMsg {
  type: 'control'
  action: 'start' | 'pause' | 'reset' | 'speed'
  value?: number | null
}
export interface Act2StartMsg {
  type: 'act2_start'
}
export interface SnapshotMsg {
  type: 'snapshot'
  op: 'save' | 'load'
  name: string
}
export interface FFInputMsg {
  type: 'ff_input'
  fwd: number // -1..1
  strafe: number // -1..1, + = right
  yaw: number // absolute heading in radians (world frame), NOT a turn rate
}

export type ClientMsg = ClickMsg | UndoMsg | ControlMsg | Act2StartMsg | SnapshotMsg | FFInputMsg
