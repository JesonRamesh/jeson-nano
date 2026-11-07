import { useEffect, useMemo, useRef, useState } from 'react'
import type { ClientMsg } from '../contracts'
import { useFreeGrid, type FreeGrid } from '../state/grid'
import { sim, useSim, type BeliefGrid } from '../state/store'
import { placeClick, undoClick, type ClickMode } from './clickTool'
import { fitView, toCanvas, toWorld, type MapView } from './mapView'

// Layer colours (projector-friendly on a dark background).
const C = {
  free: '#1b1f27',
  wall: '#474e5b',
  searched: 'rgba(79, 195, 247, 0.16)',
  blocked: '#ff4d4f',
  fire: '#ff8a1f',
  danger: 'rgba(255, 176, 32, 0.55)',
  unsafe: 'rgba(171, 71, 188, 0.55)',
  route: '#3ddc84',
  oldRoute: '#ff4d4f',
  checkpoint: '#e6f7ff',
  survivor: '#d4ff3d',
  entrance: '#ffffff',
  lidar: 'rgba(255, 255, 255, 0.18)',
}

/** FreeGrid -> coloured canvas at grid resolution (row 0 at the top). */
function gridCanvas(g: FreeGrid): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = g.w
  c.height = g.h
  const ctx = c.getContext('2d')!
  const img = ctx.createImageData(g.w, g.h)
  const wall = [0x47, 0x4e, 0x5b]
  const free = [0x1b, 0x1f, 0x27]
  for (let row = 0; row < g.h; row++) {
    const iy = g.h - 1 - row
    for (let ix = 0; ix < g.w; ix++) {
      const col = g.free[iy * g.w + ix] ? free : wall
      const p = (row * g.w + ix) * 4
      img.data[p] = col[0]
      img.data[p + 1] = col[1]
      img.data[p + 2] = col[2]
      img.data[p + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
  return c
}

/** Belief layers -> canvas at belief resolution (1 px per 0.2 m cell, row 0 at the top). */
function paintBelief(b: BeliefGrid, target: HTMLCanvasElement) {
  if (target.width !== b.w || target.height !== b.h) {
    target.width = b.w
    target.height = b.h
  }
  const ctx = target.getContext('2d')!
  const img = ctx.createImageData(b.w, b.h)
  const rgba = (p: number, r: number, g: number, bl: number, a: number) => {
    img.data[p] = r
    img.data[p + 1] = g
    img.data[p + 2] = bl
    img.data[p + 3] = a
  }
  for (let j = 0; j < b.h; j++) {
    const row = b.h - 1 - j // flip: belief j = 0 is the bottom of the map
    for (let i = 0; i < b.w; i++) {
      const k = j * b.w + i
      const p = (row * b.w + i) * 4
      switch (b.cells[k]) {
        case 3: // blocked_new: red hatch (diagonal stripes, 3 cells = 0.6 m period)
          if ((i + j) % 3 === 0) rgba(p, 255, 77, 79, 255)
          else rgba(p, 120, 30, 32, 230)
          break
        case 4: // fire
          rgba(p, 255, 138, 31, 255)
          break
        case 5: // danger
          rgba(p, 255, 176, 32, 140)
          break
        case 6: // unsafe
          rgba(p, 171, 71, 188, 140)
          break
        default:
          if (b.searched[k]) rgba(p, 79, 195, 247, 42)
      }
    }
  }
  ctx.putImageData(img, 0, 0)
}

function polyline(ctx: CanvasRenderingContext2D, v: MapView, pts: [number, number][]) {
  ctx.beginPath()
  pts.forEach(([x, y], n) => {
    const [cx, cy] = toCanvas(v, x, y)
    if (n === 0) ctx.moveTo(cx, cy)
    else ctx.lineTo(cx, cy)
  })
  ctx.stroke()
}

export function BeliefMap({ send }: { send: (m: ClientMsg) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const viewRef = useRef<MapView | null>(null)
  const [showLidar, setShowLidar] = useState(true)
  const [mode, setMode] = useState<ClickMode | null>(null)
  const gridUrl = useSim((s) => s.hello?.map.grid_url)
  const gridRes = useSim((s) => s.hello?.map.resolution)
  const freeGrid = useFreeGrid(gridUrl, gridRes)
  const grid = useMemo(() => (freeGrid ? gridCanvas(freeGrid) : null), [freeGrid])
  const handleCount = useSim((s) => s.state?.click_handles.length ?? 0)

  // Esc leaves click mode.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setMode(null)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const onCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const v = viewRef.current
    if (!mode || !v) return
    const r = e.currentTarget.getBoundingClientRect()
    const [x, y] = toWorld(v, e.clientX - r.left, e.clientY - r.top)
    placeClick(send, freeGrid, mode, Math.round(x * 100) / 100, Math.round(y * 100) / 100)
  }

  // Draw loop: repaint on animation frames when something changed.
  useEffect(() => {
    const canvas = canvasRef.current!
    const wrap = wrapRef.current!
    const beliefCanvas = document.createElement('canvas')
    let beliefVersion = -1
    let lastState: unknown = null
    let size = { w: 0, h: 0 }
    let dirty = true
    let raf = 0

    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1
      size = { w: wrap.clientWidth, h: wrap.clientHeight }
      canvas.width = Math.round(size.w * dpr)
      canvas.height = Math.round(size.h * dpr)
      canvas.style.width = `${size.w}px`
      canvas.style.height = `${size.h}px`
      dirty = true
    })
    ro.observe(wrap)

    const draw = () => {
      raf = requestAnimationFrame(draw)
      const { hello, state } = sim.snap
      const b = sim.belief
      if (b && b.version !== beliefVersion) {
        paintBelief(b, beliefCanvas)
        beliefVersion = b.version
        dirty = true
      }
      if (state !== lastState) {
        lastState = state
        dirty = true
      }
      if (!dirty || !hello || size.w === 0) return
      dirty = false

      const dpr = window.devicePixelRatio || 1
      const ctx = canvas.getContext('2d')!
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, size.w, size.h)
      const m = hello.map
      const worldW = m.size[0] * m.resolution
      const worldH = m.size[1] * m.resolution
      const v = fitView(worldW, worldH, size.w, size.h, 12, 44, 34) // room for toolbar + legend
      viewRef.current = v
      const [x0, y0] = toCanvas(v, m.origin[0], m.origin[1] + worldH) // top-left corner

      // 1 prior walls
      ctx.imageSmoothingEnabled = false
      if (grid) ctx.drawImage(grid, x0, y0, worldW * v.scale, worldH * v.scale)
      else {
        ctx.fillStyle = C.free
        ctx.fillRect(x0, y0, worldW * v.scale, worldH * v.scale)
      }
      // 2 searched + belief cells
      if (b) ctx.drawImage(beliefCanvas, x0, y0, b.w * b.res * v.scale, b.h * b.res * v.scale)

      // judge placements (click tool): debris = 1 x 1 m block, fire = disc
      for (const h of state?.click_handles ?? []) {
        const [hx, hy] = toCanvas(v, h.x, h.y)
        if (h.kind === 'debris') {
          ctx.fillStyle = '#8d6e63'
          ctx.strokeStyle = '#d7ccc8'
          ctx.lineWidth = 2
          const s = Math.max(8, v.scale)
          ctx.fillRect(hx - s / 2, hy - s / 2, s, s)
          ctx.strokeRect(hx - s / 2, hy - s / 2, s, s)
        } else {
          ctx.fillStyle = '#ff6d00'
          ctx.beginPath()
          ctx.arc(hx, hy, Math.max(6, 0.6 * v.scale), 0, Math.PI * 2)
          ctx.fill()
        }
      }

      // entrance
      const [ex, ey] = toCanvas(v, hello.entrance.x, hello.entrance.y)
      ctx.strokeStyle = C.entrance
      ctx.lineWidth = 2
      ctx.strokeRect(ex - 6, ey - 6, 12, 12)

      if (!state) return
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'

      // 3 old routes: red dashed, fading out until `until`
      for (const r of state.old_routes) {
        const left = r.until - state.t
        if (left <= 0) continue
        ctx.globalAlpha = Math.min(1, left / 6)
        ctx.strokeStyle = C.oldRoute
        ctx.lineWidth = 3
        ctx.setLineDash([10, 8])
        polyline(ctx, v, r.points)
      }
      ctx.globalAlpha = 1
      ctx.setLineDash([])
      // 4 current route
      if (state.route.points.length > 1) {
        ctx.strokeStyle = C.route
        ctx.lineWidth = 4
        polyline(ctx, v, state.route.points)
      }
      // 5 checkpoints
      ctx.strokeStyle = C.checkpoint
      ctx.lineWidth = 2
      for (const cp of state.checkpoints) {
        const [cx, cy] = toCanvas(v, cp.x, cp.y)
        ctx.beginPath()
        ctx.arc(cx, cy, Math.max(6, 0.5 * v.scale), 0, Math.PI * 2)
        ctx.stroke()
      }
      // 6 survivor
      const s = state.survivor
      if (s && s.x != null && s.y != null && s.status !== 'unknown') {
        const [sx, sy] = toCanvas(v, s.x, s.y)
        ctx.fillStyle = C.survivor
        ctx.globalAlpha = s.status === 'confirmed' ? 1 : 0.55
        ctx.beginPath()
        ctx.moveTo(sx, sy - 10)
        ctx.lineTo(sx + 9, sy + 7)
        ctx.lineTo(sx - 9, sy + 7)
        ctx.closePath()
        ctx.fill()
        ctx.globalAlpha = 1
      }
      // 7 drones (+ LiDAR rays)
      const colours = new Map(hello.drones.map((d) => [d.id, d.colour]))
      for (const d of state.drones) {
        const [dx, dy] = toCanvas(v, d.x, d.y)
        if (showLidar && d.alive && d.lidar.length) {
          ctx.strokeStyle = C.lidar
          ctx.lineWidth = 1
          ctx.beginPath()
          d.lidar.forEach((r, k) => {
            const a = d.yaw + (k * Math.PI) / 18
            const [hx, hy] = toCanvas(v, d.x + r * Math.cos(a), d.y + r * Math.sin(a))
            ctx.moveTo(dx, dy)
            ctx.lineTo(hx, hy)
          })
          ctx.stroke()
        }
        const col = d.alive ? (colours.get(d.id) ?? '#ccc') : '#666'
        const rad = Math.max(6, 0.3 * v.scale)
        ctx.fillStyle = col
        ctx.beginPath()
        ctx.arc(dx, dy, rad, 0, Math.PI * 2)
        ctx.fill()
        // heading tick (canvas y is flipped, so negate sin)
        ctx.strokeStyle = '#000'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(dx, dy)
        ctx.lineTo(dx + Math.cos(d.yaw) * rad * 1.6, dy - Math.sin(d.yaw) * rad * 1.6)
        ctx.stroke()
        ctx.fillStyle = '#fff'
        ctx.font = '600 16px system-ui, sans-serif'
        ctx.fillText(d.alive ? d.id : `${d.id} ✕`, dx + rad + 3, dy - rad)
      }
    }
    raf = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [grid, showLidar])

  return (
    <div className="belief-map" ref={wrapRef}>
      <canvas ref={canvasRef} onClick={onCanvasClick} className={mode ? `click-mode click-${mode}` : ''} />
      <div className="map-toolbar">
        <div className="click-tool">
          <button className={mode === 'debris' ? 'active' : ''} onClick={() => setMode(mode === 'debris' ? null : 'debris')}>
            Debris
          </button>
          <button className={mode === 'fire' ? 'active fire' : 'fire'} onClick={() => setMode(mode === 'fire' ? null : 'fire')}>
            Fire
          </button>
          <button onClick={() => undoClick(send)} disabled={handleCount === 0}>
            Undo{handleCount ? ` (${handleCount})` : ''}
          </button>
        </div>
        <label>
          <input type="checkbox" checked={showLidar} onChange={(e) => setShowLidar(e.target.checked)} /> LiDAR rays
        </label>
      </div>
      <div className="map-legend">
        <span style={{ color: C.route }}>━ route</span>
        <span style={{ color: C.oldRoute }}>┅ old route</span>
        <span style={{ color: C.blocked }}>▨ blocked</span>
        <span style={{ color: C.fire }}>■ fire / danger</span>
        <span style={{ color: '#ce93d8' }}>■ unsafe</span>
        <span style={{ color: '#4fc3f7' }}>■ searched</span>
      </div>
    </div>
  )
}
