import { useSyncExternalStore } from 'react'
import type {
  BudgetMsg,
  CellState,
  DetectionMsg,
  EventMsg,
  HelloMsg,
  ServerMsg,
  StateMsg,
  TraceRow,
} from '../contracts'

/** Belief cell codes in BeliefGrid.cells (0 = no info from the swarm yet). */
export const CELL_CODE: Record<CellState, number> = {
  wall_prior: 1,
  free: 2,
  blocked_new: 3,
  fire: 4,
  danger: 5,
  unsafe: 6,
}

/** The swarm's 0.2 m belief grid, as received through map_delta. Mutated in place. */
export interface BeliefGrid {
  w: number
  h: number
  res: number
  cells: Uint8Array // [j * w + i], see CELL_CODE
  searched: Uint8Array // [j * w + i], 1 = searched
  version: number // bumped on every change (canvas redraw trigger)
}

export interface Snapshot {
  hello: HelloMsg | null
  state: StateMsg | null
  trace: TraceRow[]
  detections: Record<string, DetectionMsg>
  budget: BudgetMsg | null
  budgetAt: number // performance.now() when the last budget arrived
  lastEvent: EventMsg | null
}

const TRACE_CAP = 1000

/**
 * Dashboard state, fed by useSocket. Components read slices with useSim(selector).
 * The belief grid lives outside React state (it changes many times a second);
 * the map canvas reads it directly and watches `belief.version`.
 */
class SimStore {
  snap: Snapshot = { hello: null, state: null, trace: [], detections: {}, budget: null, budgetAt: 0, lastEvent: null }
  belief: BeliefGrid | null = null
  private listeners = new Set<() => void>()

  subscribe = (fn: () => void) => {
    this.listeners.add(fn)
    return () => {
      this.listeners.delete(fn)
    }
  }

  private set(patch: Partial<Snapshot>) {
    this.snap = { ...this.snap, ...patch }
    for (const fn of this.listeners) fn()
  }

  handle(msg: ServerMsg) {
    switch (msg.type) {
      case 'hello': {
        const m = msg.map
        const w = Math.ceil((m.size[0] * m.resolution) / m.belief_resolution)
        const h = Math.ceil((m.size[1] * m.resolution) / m.belief_resolution)
        this.belief = {
          w,
          h,
          res: m.belief_resolution,
          cells: new Uint8Array(w * h),
          searched: new Uint8Array(w * h),
          version: 0,
        }
        // New session (server restart / reconnect): drop old per-run data.
        this.set({ hello: msg, state: null, trace: [], detections: {}, lastEvent: null })
        break
      }
      case 'state':
        this.set({ state: msg })
        break
      case 'map_delta':
        this.applyDelta(msg.cells, msg.searched)
        break
      case 'trace': {
        const { type: _type, ...row } = msg
        const trace = this.snap.trace.length >= TRACE_CAP ? this.snap.trace.slice(-TRACE_CAP + 1) : this.snap.trace
        this.set({ trace: [...trace, row] })
        break
      }
      case 'detection':
        this.set({ detections: { ...this.snap.detections, [msg.drone]: msg } })
        break
      case 'budget':
        this.set({ budget: msg, budgetAt: performance.now() })
        break
      case 'event':
        this.set({ lastEvent: msg })
        break
      default:
        break // guidance / summary: Act 2 (Day 2)
    }
  }

  private applyDelta(cells: [number, number, CellState][], searched: [number, number][]) {
    const b = this.belief
    if (!b || (cells.length === 0 && searched.length === 0)) return
    for (const [i, j, s] of cells) {
      if (i >= 0 && j >= 0 && i < b.w && j < b.h) b.cells[j * b.w + i] = CELL_CODE[s] ?? 0
    }
    for (const [i, j] of searched) {
      if (i >= 0 && j >= 0 && i < b.w && j < b.h) b.searched[j * b.w + i] = 1
    }
    b.version++
  }
}

export const sim = new SimStore()

export function useSim<T>(select: (s: Snapshot) => T): T {
  return useSyncExternalStore(sim.subscribe, () => select(sim.snap))
}
