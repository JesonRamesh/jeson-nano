import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { TraceRow } from '../contracts'
import { sim } from '../state/store'

// Comms view of the swarm: the drones as a fully connected diamond (every drone can hear every
// other). Each decision a drone shares lights up the link from sender to receiver in the sender's
// colour: a bright pulse runs along it towards the receiver, then the whole link glows and fades.
//
// The swarm shares state through its blackboard, so most decisions are broadcasts: the packet
// goes down every link from the sender to each live drone. One case has a known pair:
// a REPLAN that cites another drone's LiDAR (evidence.value.seen_by) is drawn as that drone ->
// the re-planning drone. Rows replayed after a snapshot load (older than the sim clock) are skipped.

type Pt = [number, number]

const TRAVEL_MS = 750 // pulse runs along the link
const GLOW_MS = 1500 // the link keeps glowing, fading out, until here (from launch)
const MAX_LIVE = 10 // transmissions on screen at once (oldest dropped)
const LIVE_WINDOW_S = 2 // ignore trace rows older than this vs the sim clock

interface Tx {
  from: string
  to: string[] // receivers
  t0: number // performance.now() at launch
  colour: string
}

/** Decisions a drone shares with the others (DRONE_LOST / REASSIGN show on the node instead). */
const SHARED = new Set<TraceRow['action']>([
  'ASSIGN_AREA',
  'EXPLORE',
  'INSPECT',
  'CONFIRM_HAZARD',
  'REJECT_DETECTION',
  'UNCONFIRMED',
  'MARK_BLOCKED',
  'REPLAN',
  'RETREAT',
  'SURVIVOR_CONFIRMED',
  'CHAIN_POSITION',
  'NO_ROUTE',
  'ESCORT',
  'FF_CORRECTION',
  'MISSION_COMPLETE',
])

/** Node positions: 4 drones as a diamond (1 top, 2 right, 3 bottom, 4 left); otherwise a ring. */
function layout(n: number, cx: number, cy: number, rx: number, ry: number): Pt[] {
  return Array.from({ length: n }, (_, i) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / n
    return [cx + rx * Math.cos(a), cy + ry * Math.sin(a)] as Pt
  })
}

const MONO = `'Google Sans Code Variable', 'Google Sans Code', ui-monospace, monospace`

export function CommsGraph({ detailed, actions }: { detailed: boolean; actions?: ReactNode }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const detailedRef = useRef(detailed)
  const [info, setInfo] = useState('')
  useEffect(() => {
    detailedRef.current = detailed
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
        if (!SHARED.has(r.action)) continue
        const v = (r.evidence.value ?? {}) as Record<string, unknown>
        let from = r.drone
        let to = [...live].filter((d) => d !== r.drone)
        if (r.action === 'REPLAN' && typeof v.seen_by === 'string' && v.seen_by !== r.drone) {
          from = v.seen_by // the drone whose LiDAR saw the debris tells the one that re-plans
          to = [r.drone]
        }
        if (!to.length) continue
        txs.push({ from, to, t0: now, colour: colours.get(from) ?? '#eef1f4' })
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
      txs = txs.filter((t) => now - t.t0 < GLOW_MS)
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

      // ---- transmissions: pulse runs sender -> receiver, then the link glows and fades ----
      ctx.lineCap = 'round'
      for (const t of txs) {
        const a = at.get(t.from)
        if (!a) continue
        const ms = now - t.t0
        const run = Math.min(1, ms / TRAVEL_MS)
        const e = 1 - (1 - run) ** 3 // ease-out
        const glow = ms < TRAVEL_MS ? 1 : 1 - (ms - TRAVEL_MS) / (GLOW_MS - TRAVEL_MS)
        for (const rcv of t.to) {
          const b = at.get(rcv)
          if (!b) continue
          const hx = a[0] + (b[0] - a[0]) * e
          const hy = a[1] + (b[1] - a[1]) * e
          // the lit part of the link (sender -> pulse head), glowing in the sender's colour
          ctx.save()
          ctx.globalAlpha = Math.max(0, glow) * 0.9
          ctx.strokeStyle = t.colour
          ctx.shadowColor = t.colour
          ctx.shadowBlur = det ? 16 : 10
          ctx.lineWidth = det ? 3.5 : 2.5
          ctx.beginPath()
          ctx.moveTo(a[0], a[1])
          ctx.lineTo(hx, hy)
          ctx.stroke()
          ctx.restore()
          // the pulse head, while it is still travelling
          if (run < 1) {
            ctx.save()
            ctx.fillStyle = '#ffffff'
            ctx.shadowColor = t.colour
            ctx.shadowBlur = 14
            ctx.beginPath()
            ctx.arc(hx, hy, det ? 4.5 : 3.2, 0, Math.PI * 2)
            ctx.fill()
            ctx.restore()
          }
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
        const hearing = txs.some((t) => t.to.includes(id) && now - t.t0 >= TRAVEL_MS * 0.8 && now - t.t0 < GLOW_MS)
        if (ok) {
          ctx.fillStyle = talking ? `${col}66` : hearing ? 'rgba(238, 241, 244, 0.16)' : 'rgba(238, 241, 244, 0.07)'
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

      const text = `${sent} message${sent === 1 ? '' : 's'} shared`
      if (text !== lastInfo) {
        lastInfo = text
        setInfo(text)
      }
    }
    raf = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [])

  return (
    <div className={`comms ${detailed ? 'detailed' : ''}`} ref={wrapRef}>
      <canvas ref={canvasRef} />
      <div className="netgraph-head">
        <h2>Comms</h2>
        <span className="netgraph-info">{info}</span>
        {actions && <div className="netgraph-actions">{actions}</div>}
      </div>
    </div>
  )
}
