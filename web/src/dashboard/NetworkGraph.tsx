import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { HelloMsg } from '../contracts'
import { sim } from '../state/store'
import { CommsGraph } from './CommsGraph'

// Live link graph: where the drones are and how they connect back to the base, drawn like a
// SLAM pose graph on a quiet map (north up, true scale). The building's route graph is the
// faint "road" layer; each live drone is a node; links form a relay tree back to BASE
// (the entrance): every drone hangs off its nearest neighbour that is already connected
// (Prim's minimum spanning tree). A lost drone turns grey and drops out of the tree.
// This is a visual model of a mesh, not a simulated radio (the sim has no radio limits).
//
// The roads are not shown up front: they are revealed as the swarm maps the floor. A stretch
// of road appears once its cell is in the swarm's searched mask (map_delta) or a live drone has
// been next to it, so the network grows out from BASE behind the drones, in real time. Fresh
// stretches glow, then settle into the road colour. The searched area itself is a faint wash.

type Pt = [number, number]

interface View {
  scale: number // px per metre
  ox: number // px of world x = 0
  oy: number // px of world y = 0
}

/** Canvas text uses the dashboard's mono face (index.css --mono). */
const MONO = `'Google Sans Code Variable', 'Google Sans Code', ui-monospace, monospace`

const TRAIL_MAX = 160 // samples per drone
const TRAIL_STEP_M = 0.35 // record a trail point after moving this far
const ROAD_STEP_M = 0.2 // road reveal resolution (= belief cell)
const REVEAL_NEAR_M = 1.2 // a live drone this close reveals the road even before the searched mask says so
const FADE_MS = 900 // fresh road glow -> settled road
const EXPLORED_REBUILD_MS = 200 // repaint the explored wash at most this often

const C = {
  road: 'rgba(213, 218, 224, 0.16)',
  roadNode: 'rgba(213, 218, 224, 0.30)',
  fresh: [140, 230, 255] as const, // rgb of a just-mapped road stretch
  explored: [213, 218, 224, 14] as const, // rgba (0-255) of a searched cell
  grid: 'rgba(213, 218, 224, 0.045)',
  route: 'rgba(240, 178, 59, 0.75)',
  trail: 'rgba(213, 218, 224, 0.20)',
  link: 'rgba(235, 238, 242, 0.62)',
  node: '#eef1f4',
  lost: '#5a626d',
  label: 'rgba(213, 218, 224, 0.85)',
  dim: 'rgba(133, 144, 160, 0.9)',
}

interface RoadEdge {
  a: Pt
  b: Pt
  n: number // samples, ends included
  t: Float64Array // per sample: performance.now() when revealed, 0 = not yet
}

/** What the swarm has mapped so far: road samples + nodes, revealed over time. Per session. */
interface Discovery {
  hello: HelloMsg
  edges: RoadEdge[]
  nodes: { p: Pt; t: number }[]
  samples: number
  revealed: number
  beliefVer: number
  explored: HTMLCanvasElement | null
  exploredVer: number
  exploredAt: number
}

function newDiscovery(hello: HelloMsg): Discovery {
  const xy = new Map(hello.graph.nodes.map((n) => [n.id, [n.x, n.y] as Pt]))
  const edges: RoadEdge[] = []
  let samples = 0
  for (const e of hello.graph.edges) {
    const a = xy.get(e.a)
    const b = xy.get(e.b)
    if (!a || !b) continue
    const n = Math.max(2, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / ROAD_STEP_M) + 1)
    edges.push({ a, b, n, t: new Float64Array(n) })
    samples += n
  }
  return {
    hello,
    edges,
    nodes: hello.graph.nodes.map((n) => ({ p: [n.x, n.y] as Pt, t: 0 })),
    samples,
    revealed: 0,
    beliefVer: -1,
    explored: null,
    exploredVer: -1,
    exploredAt: 0,
  }
}

/** Reveal road samples / nodes the swarm has searched, or that a live drone is next to. */
function updateDiscovery(d: Discovery, drones: Pt[], now: number) {
  const b = sim.belief
  const o = d.hello.map.origin
  const searched = (x: number, y: number) => {
    if (!b) return false
    const i = Math.floor((x - o[0]) / b.res)
    const j = Math.floor((y - o[1]) / b.res)
    return i >= 0 && j >= 0 && i < b.w && j < b.h && b.searched[j * b.w + i] === 1
  }
  const r2 = REVEAL_NEAR_M * REVEAL_NEAR_M
  const near = (x: number, y: number) => drones.some(([dx, dy]) => (dx - x) ** 2 + (dy - y) ** 2 < r2)
  for (const e of d.edges) {
    for (let k = 0; k < e.n; k++) {
      if (e.t[k]) continue
      const f = k / (e.n - 1)
      const x = e.a[0] + (e.b[0] - e.a[0]) * f
      const y = e.a[1] + (e.b[1] - e.a[1]) * f
      if (searched(x, y) || near(x, y)) {
        e.t[k] = now
        d.revealed++
      }
    }
  }
  for (const n of d.nodes) if (!n.t && (searched(n.p[0], n.p[1]) || near(n.p[0], n.p[1]))) n.t = now
}

/** The searched mask as a 1 px-per-cell image (row 0 = top = highest y). */
function paintExplored(d: Discovery) {
  const b = sim.belief
  if (!b) return
  if (!d.explored || d.explored.width !== b.w || d.explored.height !== b.h) {
    d.explored = document.createElement('canvas')
    d.explored.width = b.w
    d.explored.height = b.h
  }
  const ctx = d.explored.getContext('2d')!
  const img = ctx.createImageData(b.w, b.h)
  const [r, g, bl, a] = C.explored
  for (let j = 0; j < b.h; j++) {
    const row = (b.h - 1 - j) * b.w
    for (let i = 0; i < b.w; i++) {
      if (!b.searched[j * b.w + i]) continue
      const q = (row + i) * 4
      img.data[q] = r
      img.data[q + 1] = g
      img.data[q + 2] = bl
      img.data[q + 3] = a
    }
  }
  ctx.putImageData(img, 0, 0)
  d.exploredVer = b.version
}

/** Prim's MST rooted at index 0 (BASE). Returns [parent, child] index pairs. */
function relayTree(pts: Pt[]): [number, number][] {
  const n = pts.length
  if (n < 2) return []
  const inTree = new Array(n).fill(false)
  const best = new Array(n).fill(Infinity)
  const parent = new Array(n).fill(-1)
  best[0] = 0
  const edges: [number, number][] = []
  for (let it = 0; it < n; it++) {
    let u = -1
    for (let i = 0; i < n; i++) if (!inTree[i] && (u < 0 || best[i] < best[u])) u = i
    inTree[u] = true
    if (parent[u] >= 0) edges.push([parent[u], u])
    for (let v = 0; v < n; v++) {
      if (inTree[v]) continue
      const d = Math.hypot(pts[u][0] - pts[v][0], pts[u][1] - pts[v][1])
      if (d < best[v]) {
        best[v] = d
        parent[v] = u
      }
    }
  }
  return edges
}

type Tab = 'map' | 'comms'

export function NetworkGraph({ detailed, actions }: { detailed: boolean; actions?: ReactNode }) {
  const [tab, setTab] = useState<Tab>('map')
  const [commsInfo, setCommsInfo] = useState('')
  const tabRef = useRef<Tab>(tab)
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const infoRef = useRef<HTMLDivElement>(null)
  const detailedRef = useRef(detailed)
  const dirtyRef = useRef(true)
  // pan / zoom (detailed view only), relative to the fitted view
  const camRef = useRef({ zoom: 1, panX: 0, panY: 0 })

  useEffect(() => {
    tabRef.current = tab
    dirtyRef.current = true
  }, [tab])

  useEffect(() => {
    detailedRef.current = detailed
    if (!detailed) camRef.current = { zoom: 1, panX: 0, panY: 0 }
    dirtyRef.current = true
  }, [detailed])

  useEffect(() => {
    const canvas = canvasRef.current!
    const wrap = wrapRef.current!
    const trails = new Map<string, Pt[]>()
    let lastState: unknown = null
    let disc: Discovery | null = null
    let size = { w: 0, h: 0 }
    let raf = 0

    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1
      size = { w: wrap.clientWidth, h: wrap.clientHeight }
      canvas.width = Math.round(size.w * dpr)
      canvas.height = Math.round(size.h * dpr)
      dirtyRef.current = true
    })
    ro.observe(wrap)

    const draw = () => {
      raf = requestAnimationFrame(draw)
      const { hello, state } = sim.snap
      const now = performance.now()
      if (hello && disc?.hello !== hello) {
        disc = newDiscovery(hello) // new session: nothing mapped yet
        trails.clear()
        lastState = null
      }
      const beliefVer = sim.belief?.version ?? -1
      if (disc && (state !== lastState || beliefVer !== disc.beliefVer)) {
        disc.beliefVer = beliefVer
        const live = (state?.drones ?? []).filter((d) => d.alive).map((d) => [d.x, d.y] as Pt)
        updateDiscovery(disc, live, now)
        dirtyRef.current = true
      }
      if (disc && beliefVer !== disc.exploredVer && now - disc.exploredAt > EXPLORED_REBUILD_MS) {
        disc.exploredAt = now
        paintExplored(disc)
        dirtyRef.current = true
      }
      if (state !== lastState) {
        lastState = state
        dirtyRef.current = true
        for (const d of state?.drones ?? []) {
          if (!d.alive) continue
          const t = trails.get(d.id) ?? []
          const last = t[t.length - 1]
          if (!last || Math.hypot(last[0] - d.x, last[1] - d.y) > TRAIL_STEP_M) {
            t.push([d.x, d.y])
            if (t.length > TRAIL_MAX) t.shift()
            trails.set(d.id, t)
          }
        }
      }
      // hidden behind the comms tab: keep recording (trails, discovery), draw on return
      if (!dirtyRef.current || !hello || !disc || size.w === 0 || tabRef.current !== 'map') return
      dirtyRef.current = false
      const det = detailedRef.current

      const dpr = window.devicePixelRatio || 1
      const ctx = canvas.getContext('2d')!
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, size.w, size.h)

      // ---- fit: the route graph + drones, north up, true scale ----
      const pts: Pt[] = hello.graph.nodes.map((n) => [n.x, n.y])
      for (const d of state?.drones ?? []) pts.push([d.x, d.y])
      pts.push([hello.entrance.x, hello.entrance.y])
      let x0 = Math.min(...pts.map((p) => p[0]))
      let x1 = Math.max(...pts.map((p) => p[0]))
      let y0 = Math.min(...pts.map((p) => p[1]))
      let y1 = Math.max(...pts.map((p) => p[1]))
      const padM = 2
      x0 -= padM
      x1 += padM
      y0 -= padM
      y1 += padM
      const padPx = det ? 48 : 18
      const fit = Math.min((size.w - 2 * padPx) / Math.max(1, x1 - x0), (size.h - 2 * padPx) / Math.max(1, y1 - y0))
      const cam = camRef.current
      const scale = fit * cam.zoom
      const v: View = {
        scale,
        ox: size.w / 2 - ((x0 + x1) / 2) * scale + cam.panX,
        oy: size.h / 2 + ((y0 + y1) / 2) * scale + cam.panY,
      }
      const P = (x: number, y: number): Pt => [v.ox + x * v.scale, v.oy - y * v.scale]

      // ---- metric grid (detailed) ----
      if (det) {
        const step = v.scale > 18 ? 2 : v.scale > 7 ? 5 : 10
        ctx.fillStyle = C.grid
        const gx0 = Math.floor((0 - v.ox) / v.scale / step) * step
        const gx1 = Math.ceil((size.w - v.ox) / v.scale / step) * step
        const gy0 = Math.floor((v.oy - size.h) / v.scale / step) * step
        const gy1 = Math.ceil(v.oy / v.scale / step) * step
        for (let gx = gx0; gx <= gx1; gx += step)
          for (let gy = gy0; gy <= gy1; gy += step) {
            const [px, py] = P(gx, gy)
            ctx.fillRect(px - 1, py - 1, 2, 2)
          }
      }

      // ---- explored area: what the swarm has searched so far ----
      const b = sim.belief
      if (disc.explored && b) {
        const o = hello.map.origin
        const [ex, ey] = P(o[0], o[1] + b.h * b.res)
        ctx.imageSmoothingEnabled = true
        ctx.drawImage(disc.explored, ex, ey, b.w * b.res * v.scale, b.h * b.res * v.scale)
      }

      // ---- roads: the route graph, only as far as the swarm has mapped it ----
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      const roadW = det ? Math.max(4, 1.2 * v.scale) : 3
      let fading = false
      const fresh: [Pt, Pt, number][] = [] // [from, to, age 0..1]
      ctx.strokeStyle = C.road
      ctx.lineWidth = roadW
      ctx.beginPath()
      for (const e of disc.edges) {
        const at = (k: number): Pt => {
          const f = k / (e.n - 1)
          return P(e.a[0] + (e.b[0] - e.a[0]) * f, e.a[1] + (e.b[1] - e.a[1]) * f)
        }
        for (let k = 0; k + 1 < e.n; k++) {
          const t0 = e.t[k]
          const t1 = e.t[k + 1]
          if (!t0 || !t1) continue
          const age = (now - Math.max(t0, t1)) / FADE_MS
          const p0 = at(k)
          const p1 = at(k + 1)
          ctx.moveTo(p0[0], p0[1])
          ctx.lineTo(p1[0], p1[1])
          if (age < 1) fresh.push([p0, p1, age])
        }
      }
      ctx.stroke()
      if (fresh.length) {
        fading = true
        const [r, g, bl] = C.fresh
        ctx.lineWidth = Math.max(2, roadW * 0.6)
        for (const [p0, p1, age] of fresh) {
          ctx.strokeStyle = `rgba(${r}, ${g}, ${bl}, ${(0.75 * (1 - age)).toFixed(3)})`
          ctx.beginPath()
          ctx.moveTo(p0[0], p0[1])
          ctx.lineTo(p1[0], p1[1])
          ctx.stroke()
        }
      }
      for (const n of disc.nodes) {
        if (!n.t) continue
        const age = (now - n.t) / FADE_MS
        const [px, py] = P(n.p[0], n.p[1])
        const r = det ? 2.2 : 1.4
        ctx.fillStyle = C.roadNode
        ctx.beginPath()
        ctx.arc(px, py, r, 0, Math.PI * 2)
        ctx.fill()
        if (age < 1) {
          // a node pops in: a ring that expands and fades
          fading = true
          const [cr, cg, cb] = C.fresh
          ctx.strokeStyle = `rgba(${cr}, ${cg}, ${cb}, ${(0.9 * (1 - age)).toFixed(3)})`
          ctx.lineWidth = 1.5
          ctx.beginPath()
          ctx.arc(px, py, r + age * (det ? 10 : 6), 0, Math.PI * 2)
          ctx.stroke()
        }
      }
      if (fading) dirtyRef.current = true // keep animating until the glow settles

      // ---- planned route (the one the firefighter will walk) ----
      const route = state?.route.points ?? []
      if (route.length > 1) {
        ctx.strokeStyle = C.route
        ctx.lineWidth = det ? 2.5 : 1.75
        ctx.setLineDash(det ? [] : [])
        ctx.beginPath()
        route.forEach(([x, y], i) => {
          const [px, py] = P(x, y)
          if (i === 0) ctx.moveTo(px, py)
          else ctx.lineTo(px, py)
        })
        ctx.stroke()
      }

      // ---- trails (SLAM-style breadcrumbs) ----
      ctx.strokeStyle = C.trail
      ctx.lineWidth = 1
      for (const t of trails.values()) {
        if (t.length < 2) continue
        ctx.beginPath()
        t.forEach(([x, y], i) => {
          const [px, py] = P(x, y)
          if (i === 0) ctx.moveTo(px, py)
          else ctx.lineTo(px, py)
        })
        ctx.stroke()
      }

      // ---- relay tree: BASE + live drones ----
      const drones = state?.drones ?? []
      const live = drones.filter((d) => d.alive)
      const treePts: Pt[] = [[hello.entrance.x, hello.entrance.y], ...live.map((d) => [d.x, d.y] as Pt)]
      const tree = relayTree(treePts)
      ctx.strokeStyle = C.link
      ctx.lineWidth = 1.25
      ctx.font = `500 10px ${MONO}`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      for (const [a, b] of tree) {
        const [ax, ay] = P(...treePts[a])
        const [bx, by] = P(...treePts[b])
        ctx.beginPath()
        ctx.moveTo(ax, ay)
        ctx.lineTo(bx, by)
        ctx.stroke()
        if (det) {
          const dm = Math.hypot(treePts[a][0] - treePts[b][0], treePts[a][1] - treePts[b][1])
          const mx = (ax + bx) / 2
          const my = (ay + by) / 2
          const label = `${dm.toFixed(1)} m`
          const w = ctx.measureText(label).width + 8
          ctx.fillStyle = 'rgba(8, 10, 13, 0.75)'
          ctx.fillRect(mx - w / 2, my - 8, w, 16)
          ctx.fillStyle = C.dim
          ctx.fillText(label, mx, my)
        }
      }

      // ---- BASE (entrance) ----
      {
        const [bx, by] = P(hello.entrance.x, hello.entrance.y)
        const r = det ? 6 : 4.5
        ctx.strokeStyle = C.node
        ctx.lineWidth = 1.5
        ctx.strokeRect(bx - r, by - r, 2 * r, 2 * r)
        ctx.fillStyle = 'rgba(238, 241, 244, 0.18)'
        ctx.fillRect(bx - r, by - r, 2 * r, 2 * r)
        if (det) {
          ctx.fillStyle = C.label
          ctx.textAlign = 'left'
          ctx.font = `500 11px ${MONO}`
          ctx.fillText('BASE', bx + r + 6, by)
        }
      }

      // ---- survivor (once the swarm has a fix) ----
      const s = state?.survivor
      if (s && s.x != null && s.y != null && s.status !== 'unknown') {
        const [sx, sy] = P(s.x, s.y)
        const r = det ? 6 : 4.5
        ctx.strokeStyle = s.status === 'confirmed' ? C.node : C.dim
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.moveTo(sx, sy - r)
        ctx.lineTo(sx + r, sy)
        ctx.lineTo(sx, sy + r)
        ctx.lineTo(sx - r, sy)
        ctx.closePath()
        ctx.stroke()
        if (det) {
          ctx.fillStyle = C.label
          ctx.textAlign = 'left'
          ctx.font = `500 11px ${MONO}`
          ctx.fillText(s.status === 'confirmed' ? 'SURVIVOR' : 'SURVIVOR?', sx + r + 6, sy)
        }
      }

      // ---- drones ----
      for (const d of drones) {
        const [px, py] = P(d.x, d.y)
        const r = det ? 6.5 : 4.5
        const col = d.alive ? C.node : C.lost
        if (d.alive) {
          // soft halo + heading tick
          ctx.fillStyle = 'rgba(238, 241, 244, 0.10)'
          ctx.beginPath()
          ctx.arc(px, py, r * 2.4, 0, Math.PI * 2)
          ctx.fill()
          ctx.strokeStyle = col
          ctx.lineWidth = 1.5
          ctx.beginPath()
          ctx.moveTo(px + Math.cos(d.yaw) * r, py - Math.sin(d.yaw) * r)
          ctx.lineTo(px + Math.cos(d.yaw) * (r + (det ? 9 : 6)), py - Math.sin(d.yaw) * (r + (det ? 9 : 6)))
          ctx.stroke()
        }
        ctx.fillStyle = d.alive ? col : 'rgba(90, 98, 109, 0.35)'
        ctx.strokeStyle = col
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.arc(px, py, r, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()
        if (!d.alive) {
          ctx.beginPath()
          ctx.moveTo(px - r * 0.6, py - r * 0.6)
          ctx.lineTo(px + r * 0.6, py + r * 0.6)
          ctx.moveTo(px + r * 0.6, py - r * 0.6)
          ctx.lineTo(px - r * 0.6, py + r * 0.6)
          ctx.stroke()
        }
        ctx.textAlign = 'left'
        ctx.fillStyle = d.alive ? C.label : C.lost
        ctx.font = `${det ? 600 : 500} ${det ? 12 : 10}px ${MONO}`
        ctx.fillText(d.id, px + r + 5, py - (det ? 6 : 0))
        if (det) {
          ctx.font = `400 10px ${MONO}`
          ctx.fillStyle = d.alive ? C.dim : C.lost
          ctx.fillText(d.alive ? d.status.toLowerCase() : 'offline', px + r + 5, py + 8)
        }
      }

      // ---- scale bar + north (detailed) ----
      if (det) {
        const target = 80 // px
        const raw = target / v.scale
        const nice = [1, 2, 5, 10, 20, 50].find((m) => m >= raw) ?? 50
        const len = nice * v.scale
        const bx = 16
        const by = size.h - 18
        ctx.strokeStyle = C.label
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(bx, by - 4)
        ctx.lineTo(bx, by)
        ctx.lineTo(bx + len, by)
        ctx.lineTo(bx + len, by - 4)
        ctx.stroke()
        ctx.fillStyle = C.label
        ctx.textAlign = 'left'
        ctx.font = `400 10px ${MONO}`
        ctx.fillText(`${nice} m`, bx + len + 6, by - 2)
        // north arrow
        const nx = size.w - 22
        const ny = size.h - 30
        ctx.beginPath()
        ctx.moveTo(nx, ny - 9)
        ctx.lineTo(nx + 5, ny + 5)
        ctx.lineTo(nx, ny + 2)
        ctx.lineTo(nx - 5, ny + 5)
        ctx.closePath()
        ctx.fill()
        ctx.textAlign = 'center'
        ctx.fillText('N', nx, ny - 17)
      }

      // ---- summary line (DOM, so it stays crisp and glassy) ----
      if (infoRef.current) {
        const lost = drones.length - live.length
        infoRef.current.textContent =
          `${live.length} live` + (lost ? ` · ${lost} lost` : '') + ` · ${tree.length} links` +
          (disc.samples ? ` · ${Math.round((100 * disc.revealed) / disc.samples)}% roads mapped` : '') +
          (route.length > 1 ? ' · route set' : '')
      }
    }
    raf = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [])

  // ---- pan / zoom like a map (detailed view only) ----
  const drag = useRef<{ x: number; y: number } | null>(null)
  const mapNav = detailed && tab === 'map'
  return (
    <div
      className={`netgraph ${mapNav ? 'detailed' : ''} tab-${tab}`}
      ref={wrapRef}
      onWheel={(e) => {
        if (!mapNav) return
        const cam = camRef.current
        const rect = e.currentTarget.getBoundingClientRect()
        const mx = e.clientX - rect.left - rect.width / 2
        const my = e.clientY - rect.top - rect.height / 2
        const k = Math.exp(-e.deltaY * 0.0015)
        const zoom = Math.min(8, Math.max(0.5, cam.zoom * k))
        const f = zoom / cam.zoom
        cam.panX = mx - (mx - cam.panX) * f
        cam.panY = my - (my - cam.panY) * f
        cam.zoom = zoom
        dirtyRef.current = true
      }}
      onPointerDown={(e) => {
        if (!mapNav || (e.target as HTMLElement).closest('button')) return
        drag.current = { x: e.clientX, y: e.clientY }
        e.currentTarget.setPointerCapture(e.pointerId)
      }}
      onPointerMove={(e) => {
        const d = drag.current
        if (!d) return
        camRef.current.panX += e.clientX - d.x
        camRef.current.panY += e.clientY - d.y
        drag.current = { x: e.clientX, y: e.clientY }
        dirtyRef.current = true
      }}
      onPointerUp={() => (drag.current = null)}
      onDoubleClick={() => {
        if (!mapNav) return
        camRef.current = { zoom: 1, panX: 0, panY: 0 }
        dirtyRef.current = true
      }}
    >
      <canvas ref={canvasRef} className="netgraph-map" />
      {tab === 'comms' && <CommsGraph detailed={detailed} onInfo={setCommsInfo} />}
      <div className="netgraph-head">
        <h2>Swarm links</h2>
        <div className="netgraph-tabs" role="tablist">
          {(['map', 'comms'] as const).map((t) => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              className={tab === t ? 'on' : ''}
              onClick={() => setTab(t)}
              title={t === 'map' ? 'Where the drones are, and the routes they have mapped' : 'What the drones tell each other'}
            >
              {t === 'map' ? 'Map' : 'Comms'}
            </button>
          ))}
        </div>
        <span ref={infoRef} className="netgraph-info" hidden={tab !== 'map'} />
        {tab === 'comms' && <span className="netgraph-info">{commsInfo}</span>}
        {actions && <div className="netgraph-actions">{actions}</div>}
      </div>
      {mapNav && <div className="netgraph-hint">Scroll to zoom · drag to pan · double-click to reset</div>}
    </div>
  )
}
