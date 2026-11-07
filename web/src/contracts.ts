// TypeScript mirror of sgr/contracts.py (owned by S). Owned by D.
//
// PROVISIONAL: written from spec §10.2 before S pushed contracts.py v1.
// Once it lands, reconcile field-by-field and keep CONTRACT_VERSION in sync.
// Rule: after H1, fields are only ever added — never renamed or removed.

export const CONTRACT_VERSION = 1

// ---- shared -------------------------------------------------------------

export interface XY {
  x: number
  y: number
}

export interface GraphNode {
  id: string
  x: number
  y: number
  kind: 'entrance' | 'junction' | 'door' | 'room' | 'stairs'
}

export interface GraphEdge {
  id: string
  a: string
  b: string
  width: number
}

export interface MapMeta {
  name: string
  grid_url: string
  resolution: number // metres per pixel (0.1)
  size: [number, number] // [w, h] in pixels
  origin: [number, number] // world xy of the bottom-left pixel
}

export interface Transform {
  M: number[][] // 4×4 row-major, raw GLB -> sim frame
  pitch: number
  grid_origin: [number, number]
  grid_size: [number, number]
}

export interface Route {
  points: [number, number][]
  status: 'ok' | 'none' | 'stale' | string
  version: number
}

// ---- server -> client ---------------------------------------------------

export interface HelloMsg {
  type: 'hello'
  contract_version: number
  map: MapMeta
  graph: { nodes: GraphNode[]; edges: GraphEdge[] }
  entrance: { x: number; y: number; yaw: number }
  drones: { id: string; color: string }[]
  transform: Transform | null
}

export interface DroneState {
  id: string
  x: number
  y: number
  z: number
  yaw: number
  status: string
  task: string
  temp: number
  lidar: number[] // 36 ranges, 6.0 = no hit
  alive: boolean
}

export interface StateMsg {
  type: 'state'
  t: number
  speed: number
  act: 1 | 2
  phase: string
  drones: DroneState[]
  ff: { x: number; y: number; yaw: number } | null
  route: Route | null
  old_routes: { points: [number, number][]; until: number }[]
  checkpoints: { x: number; y: number; drone: string }[]
  survivor: { status: string; x: number; y: number; conf: number } | null
  click_handles: string[]
}

export type CellState = string // belief cell state name, e.g. "blocked_new"

export interface MapDeltaMsg {
  type: 'map_delta'
  cells: [number, number, CellState][]
  map_full?: string // base64 PNG, every 5 s for resync
}

export type Sensor = 'lidar' | 'camera' | 'gemma' | 'temp' | 'heartbeat' | 'planner'

export interface TraceRow {
  id: string
  t: number
  drone: string | null
  action: string
  reason: string
  evidence: { sensor: Sensor; value: unknown; frame_id?: string; scan?: number[] }
}

export interface TraceMsg extends TraceRow {
  type: 'trace'
}

export interface DetectionMsg {
  type: 'detection'
  drone: string
  frame_id: string
  boxes: { x: number; y: number; w: number; h: number; label: string; score: number }[]
  gemma: { caption: string; conf: number; pending: boolean } | null
  status: 'pending' | 'confirmed' | 'low_conf' | 'rejected' | 'unconfirmed'
}

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

export interface EventMsg {
  type: 'event'
  kind: EventKind
  drone?: string
  xy?: [number, number]
}

export interface GuidanceMsg {
  type: 'guidance'
  arrow_deg: number
  distance: number
  subtitle: string
  say?: string
  alarm?: boolean
  severity: 'info' | 'warn' | 'danger'
}

export interface BudgetMsg {
  type: 'budget'
  network: 'offline' | 'online'
  model: string
  gemma_p50: number | null
  gemma_p95: number | null
  queue: number
  vram_used_mb: number | null
  vram_total_mb: number | null
  rtf: number
  speed: number
  control_hz: number
  camera_fps: number
}

export interface SummaryMsg {
  type: 'summary'
  time: number
  distance: number
  hazards_avoided: number
  corrections: number
  replans: number
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

// Binary frame: [u8 channel][u32 frame_seq, little-endian][JPEG bytes].
// channel 1–4 = drone cams D1–D4.
export interface DroneFrame {
  channel: number
  seq: number
  jpeg: Blob
}

// ---- client -> server ---------------------------------------------------

export type ClientMsg =
  | { type: 'click'; mode: 'debris' | 'fire'; x: number; y: number }
  | { type: 'undo' }
  | { type: 'control'; action: 'start' | 'pause' | 'reset' | 'speed'; value?: number }
  | { type: 'act2_start' }
  | { type: 'snapshot'; op: 'save' | 'load'; name: string }
  | { type: 'ff_input'; fwd: number; strafe: number; yaw: number }
