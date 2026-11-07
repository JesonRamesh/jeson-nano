// Where the disaster is: fires and debris piles, in sim metres (z up).
//
// Sources:
//  - default ('belief'): the swarm's belief grid. Fire cells -> fires, blocked_new cells
//    (LiDAR-detected debris) -> rubble. Hazards appear as the drones discover them, and
//    the dashboard never sees hidden scenario fields (D's rule for /api/.../scenario).
//  - optional ('world', e.g. the firefighter view if the team wants it): scenario.json via
//    GET /api/maps/{name}/scenario. Debris blocks are placed exactly like sgr/world does:
//    ceil(edge.width / 1 m) 1×1 m blocks across the edge at `at`.
import { useEffect, useState } from 'react'
import type { Fire, Graph, Scenario } from '../../contracts'
import { CELL_CODE, type BeliefGrid } from '../../state/store'
import { DEBRIS_SIZE } from '../constants'

export interface DebrisBlock {
  x: number
  y: number
  yaw: number // edge direction (rad); blocks sit across it
  seed: number
}

export interface Hazards {
  fires: Fire[]
  piles: DebrisBlock[][] // one pile per compromised edge, one block per 1 m of width
}

export function debrisFromScenario(graph: Graph, compromised: Scenario['compromised']): DebrisBlock[][] {
  const nodes = new Map(graph.nodes.map((n) => [n.id, n]))
  const edges = new Map(graph.edges.map((e) => [e.id, e]))
  const piles: DebrisBlock[][] = []
  compromised.forEach((c, ci) => {
    const e = edges.get(c.edge)
    const a = e && nodes.get(e.a)
    const b = e && nodes.get(e.b)
    if (!e || !a || !b) return
    const px = a.x + c.at * (b.x - a.x)
    const py = a.y + c.at * (b.y - a.y)
    const ang = Math.atan2(b.y - a.y, b.x - a.x)
    const nx = -Math.sin(ang)
    const ny = Math.cos(ang)
    const size = DEBRIS_SIZE.x
    const count = Math.max(1, Math.ceil(e.width / size))
    const pile: DebrisBlock[] = []
    for (let k = 0; k < count; k++) {
      const off = (k - (count - 1) / 2) * size
      pile.push({ x: px + off * nx, y: py + off * ny, yaw: ang, seed: ci * 97 + k * 13 + 1 })
    }
    piles.push(pile)
  })
  return piles
}

/** Fetch the scenario once per map name. null until loaded (or if the server has none). */
export function useScenarioHazards(mapName: string | undefined): Hazards | null {
  const [hz, setHz] = useState<Hazards | null>(null)
  useEffect(() => {
    if (!mapName) return
    let live = true
    fetch(`/api/maps/${encodeURIComponent(mapName)}/scenario`)
      .then((r) => (r.ok ? (r.json() as Promise<Scenario>) : null))
      .then((sc) => {
        if (!live || !sc) return
        setHz({ fires: sc.fires ?? [], piles: debrisFromScenario(sc.graph, sc.compromised ?? []) })
      })
      .catch((e) => console.warn('dressing: no scenario hazards', e))
    return () => {
      live = false
    }
  }, [mapName])
  return hz
}

/**
 * Fire clusters in the belief grid (4-connected fire cells), as centres + radius.
 * Cheap enough to call when belief.version changes (a few thousand cells).
 */
export function beliefFires(b: BeliefGrid): Fire[] {
  const { w, h, cells, res } = b
  const seen = new Uint8Array(cells.length)
  const out: Fire[] = []
  const stack: number[] = []
  for (let k = 0; k < cells.length; k++) {
    if (cells[k] !== CELL_CODE.fire || seen[k]) continue
    let n = 0
    let sx = 0
    let sy = 0
    stack.push(k)
    seen[k] = 1
    while (stack.length) {
      const q = stack.pop()!
      const i = q % w
      const j = (q - i) / w
      n++
      sx += i
      sy += j
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const ii = i + di
        const jj = j + dj
        if (ii < 0 || jj < 0 || ii >= w || jj >= h) continue
        const r = jj * w + ii
        if (!seen[r] && cells[r] === CELL_CODE.fire) {
          seen[r] = 1
          stack.push(r)
        }
      }
    }
    const area = n * res * res
    out.push({ x: (sx / n + 0.5) * res, y: (sy / n + 0.5) * res, size: Math.max(1, 2 * Math.sqrt(area / Math.PI)) })
  }
  return out
}

/**
 * Rubble spots from blocked_new cells: greedy picks at least `spacing` m apart, so a
 * 2 m wide detected debris face gets two piles. Each spot is pushed `back` m away from
 * the nearest free cell, because LiDAR marks the debris FACE, not its centre.
 */
export function beliefDebris(b: BeliefGrid, spacing = 0.9, back = 0.35): DebrisBlock[] {
  const { w, h, cells, res } = b
  const pts: [number, number][] = []
  for (let k = 0; k < cells.length; k++) if (cells[k] === CELL_CODE.blocked_new) pts.push([k % w, Math.floor(k / w)])
  const out: DebrisBlock[] = []
  for (const [i, j] of pts) {
    const x = (i + 0.5) * res
    const y = (j + 0.5) * res
    if (out.some((o) => Math.hypot(o.x - x, o.y - y) < spacing)) continue
    // direction away from free space (average of free 8-neighbours, reversed)
    let fx = 0
    let fy = 0
    for (let dj = -2; dj <= 2; dj++)
      for (let di = -2; di <= 2; di++) {
        const ii = i + di
        const jj = j + dj
        if (ii < 0 || jj < 0 || ii >= w || jj >= h) continue
        const c = cells[jj * w + ii]
        if (c === CELL_CODE.free) {
          fx += di
          fy += dj
        }
      }
    const n = Math.hypot(fx, fy)
    const ox = n > 0 ? (-fx / n) * back : 0
    const oy = n > 0 ? (-fy / n) * back : 0
    out.push({ x: x + ox, y: y + oy, yaw: Math.atan2(fy, fx), seed: i * 7919 + j * 104729 })
  }
  return out
}

/** Merge fire lists, dropping any fire within `minD` m of one already kept. */
export function mergeFires(lists: Fire[][], minD = 1.5): Fire[] {
  const out: Fire[] = []
  for (const l of lists)
    for (const f of l) if (!out.some((o) => Math.hypot(o.x - f.x, o.y - f.y) < minD)) out.push(f)
  return out
}

/** Deterministic PRNG (mulberry32) so a pile looks the same on every reload. */
export function rng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Sticky rubble. The swarm keeps re-checking debris with LiDAR, so blocked_new cells at the
 * edges flicker (cleared, re-marked). Rebuilding piles from scratch made them jump, vanish
 * and change shape. Instead: a pile, once placed, keeps its position and shape (seed from
 * its quantised position); new piles only cover cells away from existing ones; a pile is
 * dropped only after no blocked cell has been near it for `holdS` seconds (judge Undo).
 * Returns the same array object when nothing changed (cheap React updates).
 */
export class RubbleTracker {
  piles: (DebrisBlock & { seen: number })[] = []
  holdS: number
  near: number
  constructor(holdS = 3, near = 1.0) {
    this.holdS = holdS
    this.near = near
  }

  update(b: BeliefGrid, t: number): DebrisBlock[] | null {
    const { w, cells, res } = b
    const pts: [number, number][] = []
    for (let k = 0; k < cells.length; k++)
      if (cells[k] === CELL_CODE.blocked_new) pts.push([((k % w) + 0.5) * res, (Math.floor(k / w) + 0.5) * res])
    let changed = false
    for (const p of this.piles) if (pts.some(([x, y]) => Math.hypot(x - p.x, y - p.y) < this.near)) p.seen = t
    const kept = this.piles.filter((p) => t - p.seen <= this.holdS)
    if (kept.length !== this.piles.length) changed = true
    // new spots only for cells not already covered by a pile
    const fresh = beliefDebris(b).filter((s) => !kept.some((p) => Math.hypot(p.x - s.x, p.y - s.y) < this.near))
    for (const s of fresh) {
      if (kept.some((p) => Math.hypot(p.x - s.x, p.y - s.y) < this.near)) continue
      kept.push({ ...s, seed: Math.round(s.x * 10) * 7919 + Math.round(s.y * 10) * 104729, seen: t })
      changed = true
    }
    this.piles = kept
    return changed ? kept.map(({ seen: _seen, ...p }) => p) : null
  }
}
