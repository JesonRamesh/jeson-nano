import { useEffect, useRef } from 'react'
import type { TraceRow } from '../contracts'
import { sim } from '../state/store'

// Comms view of the swarm: the drones as a fully connected diamond (every drone can hear every
// other), and each decision a drone shares shown as a packet flying along the link from sender to
// receiver, with a very short summary on the line.
//
// The swarm shares state through its blackboard, so most decisions are broadcasts: the packet
// goes down every link from the sender to each live drone. One case has a known pair:
// a REPLAN that cites another drone's LiDAR (evidence.value.seen_by) is drawn as that drone ->
// the re-planning drone. Rows replayed after a snapshot load (older than the sim clock) are skipped.

type Pt = [number, number]

const TRAVEL_MS = 750 // packet flight along the link
const LABEL_MS = 2300 // label fully visible (from packet launch)
const FADE_MS = 400 // label fade-out
const MAX_LIVE = 10 // transmissions on screen at once (oldest dropped)
const LIVE_WINDOW_S = 2 // ignore trace rows older than this vs the sim clock

interface Tx {
  from: string
  to: string[] // receivers
  label: string
  t0: number // performance.now() at launch
  colour: string
}

/** Very short summary of a trace row, or null if it isn't something a drone shares. */
function summarise(r: TraceRow): string | null {
  const v = (r.evidence.value ?? {}) as Record<string, unknown>
  const reason = r.reason.toLowerCase()
  switch (r.action) {
    case 'ASSIGN_AREA':
      return typeof v.area === 'string' ? `taking ${v.area}` : 'taking an area'
    case 'EXPLORE':
      return reason.includes('searched') && typeof v.area === 'string' ? `${v.area} searched` : 'exploring'
    case 'INSPECT':
      if (reason.includes('keeps its camera')) return 'watching fire'
      if (reason.includes('fire')) return 'possible fire'
      if (reason.includes('person') || reason.includes('survivor')) return 'possible person'
      return 'inspecting'
    case 'CONFIRM_HAZARD':
      return reason.startsWith('fire') ? 'fire confirmed' : 'hazard confirmed'
    case 'REJECT_DETECTION':
      return 'false alarm'
    case 'UNCONFIRMED':
      return 'unconfirmed'
    case 'MARK_BLOCKED':
      return 'debris spotted'
    case 'REPLAN':
      return 'debris on your path'
    case 'RETREAT':
      return 'too hot, backing off'
    case 'SURVIVOR_CONFIRMED':
      return 'survivor found'
    case 'CHAIN_POSITION': {
      const role = typeof v.role === 'string' ? v.role : ''
      if (role === 'survivor') return 'holding over survivor'
      if (role === 'lead') return 'chain: lead'
      if (role === 'sentinel') return 'chain: checkpoint'
      return role ? `chain: ${role}` : 'chain position'
    }
    case 'NO_ROUTE':
      return 'no safe route'
    case 'ESCORT':
      return 'escorting firefighter'
    case 'FF_CORRECTION':
      return 'firefighter off route'
    case 'MISSION_COMPLETE':
      return 'mission complete'
    default:
      return null // DRONE_LOST / REASSIGN: shown on the node, not as a message
  }
}

/** Node positions: 4 drones as a diamond (1 top, 2 right, 3 bottom, 4 left); otherwise a ring. */
function layout(n: number, cx: number, cy: number, rx: number, ry: number): Pt[] {
  return Array.from({ length: n }, (_, i) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / n
    return [cx + rx * Math.cos(a), cy + ry * Math.sin(a)] as Pt
  })
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

const MONO = `'Google Sans Code Variable', 'Google Sans Code', ui-monospace, monospace`
const SANS = `'Google Sans Flex Variable', 'Google Sans Flex', system-ui, sans-serif`

export function CommsGraph({ detailed, onInfo }: { detailed: boolean; onInfo?: (text: string) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const detailedRef = useRef(detailed)
  const onInfoRef = useRef(onInfo)
  useEffect(() => {
    detailedRef.current = detailed
    onInfoRef.current = onInfo
  })

  useEffect(() => {
    const canvas = canvasRef.current!
    const wrap = wrapRef.current!
    let size = { w: 0, h: 0 }
    let raf = 0
    let txs: Tx[] = []
    const seen = new Set<string>()
    let lastTrace: TraceRow[] | null = null
    let lastHello: unknown = null
    let sent = 0
    let lastInfo = ''

    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1
      size = { w: wrap.clientWidth, h: wrap.clientHeight }
      canvas.width = Math.round(size.w * dpr)
      canvas.height = Math.round(size.h * dpr)
    })
    ro.observe(wrap)

    const ingest = (now: number) => {
      const { trace, state, hello } = sim.snap
      if (hello !== lastHello) {
        lastHello = hello
        txs = []
        seen.clear()
        sent = 0
        // everything already in the log belongs to the past: don't animate it
        for (const r of trace) seen.add(r.id)
      }
      if (trace === lastTrace) return
      lastTrace = trace
      const colours = new Map((hello?.drones ?? []).map((d) => [d.id, d.colour]))
      const live = new Set((state?.drones ?? []).filter((d) => d.alive).map((d) => d.id))
      const simT = state?.t ?? 0
      for (let i = trace.length - 1; i >= 0 && !seen.has(trace[i].id); i--) {
        const r = trace[i]
        seen.add(r.id)
        if (r.t < simT - LIVE_WINDOW_S || r.drone === 'swarm') continue
        const label = summarise(r)
        if (!label) continue
        const v = (r.evidence.value ?? {}) as Record<string, unknown>
        let from = r.drone
        let to = [...live].filter((d) => d !== r.drone)
        if (r.action === 'REPLAN' && typeof v.seen_by === 'string' && v.seen_by !== r.drone) {
          from = v.seen_by // the drone whose LiDAR saw the debris tells the one that re-plans
          to = [r.drone]
        }
        if (!to.length) continue
        txs.push({ from, to, label, t0: now, colour: colours.get(from) ?? '#eef1f4' })
        sent++
      }
      if (seen.size > 4000) {
        seen.clear()
        for (const r of trace) seen.add(r.id)
      }
      if (txs.length > MAX_LIVE) txs = txs.slice(-MAX_LIVE)
    }

    const draw = () => {
      raf = requestAnimationFrame(draw)
      const now = performance.now()
      ingest(now)
      txs = txs.filter((t) => now - t.t0 < LABEL_MS + FADE_MS)
      const { hello, state } = sim.snap
      if (!hello || size.w === 0) return
      const det = detailedRef.current
      const dpr = window.devicePixelRatio || 1
      const ctx = canvas.getContext('2d')!
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, size.w, size.h)

      const ids = hello.drones.map((d) => d.id)
      const colour = new Map(hello.drones.map((d) => [d.id, d.colour]))
      const st = new Map((state?.drones ?? []).map((d) => [d.id, d]))
      const top = det ? 70 : 40
      const bottom = det ? 40 : 12
      const nodeR = det ? 20 : 13
      const cy = top + (size.h - top - bottom) / 2
      const ry = Math.max(20, (size.h - top - bottom) / 2 - nodeR - (det ? 18 : 6))
      const rx = Math.min(size.w / 2 - nodeR - (det ? 120 : 56), ry * (det ? 1.5 : 1.7))
      const pts = layout(ids.length, size.w / 2, cy, Math.max(30, rx), ry)
      const at = new Map(ids.map((id, i) => [id, pts[i]]))
      const alive = (id: string) => st.get(id)?.alive ?? true

      // ---- links: every pair ----
      ctx.lineCap = 'round'
      for (let i = 0; i < ids.length; i++)
        for (let j = i + 1; j < ids.length; j++) {
          const a = pts[i]
          const b = pts[j]
          const ok = alive(ids[i]) && alive(ids[j])
          ctx.setLineDash(ok ? [] : [3, 5])
          ctx.strokeStyle = ok ? 'rgba(213, 218, 224, 0.22)' : 'rgba(90, 98, 109, 0.35)'
          ctx.lineWidth = det ? 1.5 : 1.2
          ctx.beginPath()
          ctx.moveTo(a[0], a[1])
          ctx.lineTo(b[0], b[1])
          ctx.stroke()
        }
      ctx.setLineDash([])

      // ---- packets: the link lights up in the sender's colour while the packet flies ----
      for (const t of txs) {
        const a = at.get(t.from)
        if (!a) continue
        const age = (now - t.t0) / TRAVEL_MS
        if (age >= 1) continue
        for (const rcv of t.to) {
          const b = at.get(rcv)
          if (!b) continue
          const e = 1 - (1 - age) ** 3 // ease-out
          const px = a[0] + (b[0] - a[0]) * e
          const py = a[1] + (b[1] - a[1]) * e
          const tail = Math.max(0, e - 0.28)
          const g = ctx.createLinearGradient(a[0] + (b[0] - a[0]) * tail, a[1] + (b[1] - a[1]) * tail, px, py)
          g.addColorStop(0, 'rgba(255,255,255,0)')
          g.addColorStop(1, t.colour)
          ctx.strokeStyle = g
          ctx.lineWidth = det ? 4 : 3
          ctx.beginPath()
          ctx.moveTo(a[0] + (b[0] - a[0]) * tail, a[1] + (b[1] - a[1]) * tail)
          ctx.lineTo(px, py)
          ctx.stroke()
          ctx.fillStyle = '#ffffff'
          ctx.shadowColor = t.colour
          ctx.shadowBlur = 10
          ctx.beginPath()
          ctx.arc(px, py, det ? 4 : 3, 0, Math.PI * 2)
          ctx.fill()
          ctx.shadowBlur = 0
        }
      }

      // ---- nodes ----
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ids.forEach((id, i) => {
        const [x, y] = pts[i]
        const s = st.get(id)
        const ok = s?.alive ?? true
        const col = ok ? (colour.get(id) ?? '#eef1f4') : '#5a626d'
        const talking = txs.some((t) => t.from === id && now - t.t0 < TRAVEL_MS)
        if (ok) {
          ctx.fillStyle = talking ? `${col}55` : 'rgba(238, 241, 244, 0.07)'
          ctx.beginPath()
          ctx.arc(x, y, nodeR * (talking ? 1.9 : 1.6), 0, Math.PI * 2)
          ctx.fill()
        }
        ctx.fillStyle = 'rgba(10, 13, 17, 0.92)'
        ctx.strokeStyle = col
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(x, y, nodeR, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()
        ctx.fillStyle = ok ? '#eef1f4' : '#5a626d'
        ctx.font = `700 ${det ? 14 : 11}px ${MONO}`
        ctx.fillText(id, x, y + 0.5)
        if (!ok) {
          ctx.strokeStyle = 'rgba(255, 90, 78, 0.8)'
          ctx.lineWidth = 1.5
          const k = nodeR * 0.75
          ctx.beginPath()
          ctx.moveTo(x - k, y - k)
          ctx.lineTo(x + k, y + k)
          ctx.stroke()
        }
        // status beside the node, on the outside of the diamond
        const status = ok ? (s?.status ?? '').toLowerCase() : 'lost'
        if (status) {
          ctx.font = `500 ${det ? 12 : 10}px ${MONO}`
          ctx.fillStyle = ok ? 'rgba(170, 180, 192, 0.9)' : '#5a626d'
          const dx = x - size.w / 2
          const dy = y - cy
          if (Math.abs(dx) > Math.abs(dy)) {
            ctx.textAlign = dx > 0 ? 'left' : 'right'
            ctx.fillText(status, x + Math.sign(dx) * (nodeR + 7), y)
          } else {
            ctx.textAlign = 'left'
            ctx.fillText(status, x + nodeR + 7, y)
          }
          ctx.textAlign = 'center'
        }
      })

      // ---- labels on the line (broadcasts: on the line to the drone opposite the sender) ----
      const placed: [number, number, number, number][] = []
      ctx.font = `600 ${det ? 13 : 11}px ${SANS}`
      for (const t of txs) {
        const a = at.get(t.from)
        if (!a) continue
        const age = now - t.t0
        const alpha = age < 120 ? age / 120 : age > LABEL_MS ? 1 - (age - LABEL_MS) / FADE_MS : 1
        if (alpha <= 0) continue
        // the line: the one receiver, or the receiver farthest from the sender (the diagonal)
        let b: Pt | undefined
        let best = -1
        for (const r of t.to) {
          const p = at.get(r)
          if (!p) continue
          const d = Math.hypot(p[0] - a[0], p[1] - a[1])
          if (d > best) {
            best = d
            b = p
          }
        }
        if (!b) continue
        const text = t.to.length > 1 ? `${t.from}: ${t.label}` : t.label
        const w = ctx.measureText(text).width + (det ? 16 : 12)
        const h = det ? 22 : 18
        let x = a[0] + (b[0] - a[0]) * 0.5 - w / 2
        let y = a[1] + (b[1] - a[1]) * 0.5 - h / 2
        // don't stack on another label: step down (then up) until free
        for (let k = 1; k < 8 && placed.some(([px, py, pw, ph]) => x < px + pw && x + w > px && y < py + ph && y + h > py); k++) {
          y += (k % 2 ? 1 : -1) * k * (h + 3)
        }
        x = Math.max(4, Math.min(size.w - w - 4, x))
        y = Math.max(top - 8, Math.min(size.h - h - 4, y))
        placed.push([x, y, w, h])
        ctx.globalAlpha = Math.max(0, Math.min(1, alpha))
        roundRect(ctx, x, y, w, h, h / 2)
        ctx.fillStyle = 'rgba(10, 13, 17, 0.88)'
        ctx.fill()
        ctx.strokeStyle = t.colour
        ctx.lineWidth = 1.25
        ctx.stroke()
        ctx.fillStyle = '#f2f4f7'
        ctx.textAlign = 'center'
        ctx.fillText(text, x + w / 2, y + h / 2 + 0.5)
        ctx.globalAlpha = 1
      }

      const info = `${sent} message${sent === 1 ? '' : 's'} shared`
      if (info !== lastInfo) {
        lastInfo = info
        onInfoRef.current?.(info)
      }
    }
    raf = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [])

  return (
    <div className="comms" ref={wrapRef}>
      <canvas ref={canvasRef} />
    </div>
  )
}
