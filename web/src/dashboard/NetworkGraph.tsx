import { useEffect, useRef, type ReactNode } from 'react'
import { sim } from '../state/store'

// Live link graph: where the drones are and how they connect back to the base, drawn like a
// SLAM pose graph on a quiet map (north up, true scale). The building's route graph is the
// faint "road" layer; each live drone is a node; links form a relay tree back to BASE
// (the entrance): every drone hangs off its nearest neighbour that is already connected
// (Prim's minimum spanning tree). A lost drone turns grey and drops out of the tree.
// This is a visual model of a mesh, not a simulated radio (the sim has no radio limits).

type Pt = [number, number]

interface View {
  scale: number // px per metre
  ox: number // px of world x = 0
  oy: number // px of world y = 0
}

const TRAIL_MAX = 160 // samples per drone
const TRAIL_STEP_M = 0.35 // record a trail point after moving this far

const C = {
  road: 'rgba(213, 218, 224, 0.10)',
  roadNode: 'rgba(213, 218, 224, 0.22)',
  grid: 'rgba(213, 218, 224, 0.045)',
  route: 'rgba(240, 178, 59, 0.75)',
  trail: 'rgba(213, 218, 224, 0.20)',
  link: 'rgba(235, 238, 242, 0.62)',
  node: '#eef1f4',
  lost: '#5a626d',
  label: 'rgba(213, 218, 224, 0.85)',
  dim: 'rgba(133, 144, 160, 0.9)',
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

export function NetworkGraph({ detailed, actions }: { detailed: boolean; actions?: ReactNode }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const infoRef = useRef<HTMLDivElement>(null)
  const detailedRef = useRef(detailed)
  const dirtyRef = useRef(true)
  // pan / zoom (detailed view only), relative to the fitted view
  const camRef = useRef({ zoom: 1, panX: 0, panY: 0 })

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
      if (!dirtyRef.current || !hello || size.w === 0) return
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

      // ---- roads: the building's route graph ----
      const nodeXY = new Map(hello.graph.nodes.map((n) => [n.id, [n.x, n.y] as Pt]))
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.strokeStyle = C.road
      ctx.lineWidth = det ? Math.max(4, 1.2 * v.scale) : 3
      ctx.beginPath()
      for (const e of hello.graph.edges) {
        const a = nodeXY.get(e.a)
        const b = nodeXY.get(e.b)
        if (!a || !b) continue
        const [ax, ay] = P(a[0], a[1])
        const [bx, by] = P(b[0], b[1])
        ctx.moveTo(ax, ay)
        ctx.lineTo(bx, by)
      }
      ctx.stroke()
      ctx.fillStyle = C.roadNode
      for (const n of hello.graph.nodes) {
        const [px, py] = P(n.x, n.y)
        ctx.beginPath()
        ctx.arc(px, py, det ? 2.2 : 1.4, 0, Math.PI * 2)
        ctx.fill()
      }

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
      ctx.font = '500 10px "IBM Plex Mono", monospace'
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
          ctx.font = '500 11px "IBM Plex Mono", monospace'
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
          ctx.font = '500 11px "IBM Plex Mono", monospace'
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
        ctx.font = `${det ? 600 : 500} ${det ? 12 : 10}px "IBM Plex Mono", monospace`
        ctx.fillText(d.id, px + r + 5, py - (det ? 6 : 0))
        if (det) {
          ctx.font = '400 10px "IBM Plex Mono", monospace'
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
        ctx.font = '400 10px "IBM Plex Mono", monospace'
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
          `${live.length} live` + (lost ? ` · ${lost} lost` : '') + ` · ${tree.length} links` + (route.length > 1 ? ' · route set' : '')
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
  return (
    <div
      className={`netgraph ${detailed ? 'detailed' : ''}`}
      ref={wrapRef}
      onWheel={(e) => {
        if (!detailed) return
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
        if (!detailed) return
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
        camRef.current = { zoom: 1, panX: 0, panY: 0 }
        dirtyRef.current = true
      }}
    >
      <canvas ref={canvasRef} />
      <div className="netgraph-head">
        <h2>Swarm links</h2>
        <span ref={infoRef} className="netgraph-info" />
        {actions && <div className="netgraph-actions">{actions}</div>}
      </div>
      {detailed && <div className="netgraph-hint">Scroll to zoom · drag to pan · double-click to reset</div>}
    </div>
  )
}
