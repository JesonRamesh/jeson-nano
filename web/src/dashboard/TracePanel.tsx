import { useEffect, useRef, useState } from 'react'
import type { TraceRow } from '../contracts'
import { sim, useSim } from '../state/store'

function fmtValue(v: unknown): string {
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(2)
  if (typeof v === 'string') return v
  return JSON.stringify(v)
}

const CAM_W = 320
const CAM_H = 240

/** LiDAR scan evidence: 36 rays drawn heading-up, 6 m = edge. */
function ScanPolar({ scan }: { scan: number[] }) {
  const s = 160
  const r0 = s / 2 - 4
  const pts = scan
    .map((r, k) => {
      const a = Math.PI / 2 + (k * Math.PI) / 18
      const d = (Math.min(r, 6) / 6) * r0
      return `${s / 2 + d * Math.cos(a)},${s / 2 - d * Math.sin(a)}`
    })
    .join(' ')
  return (
    <svg width={s} height={s} className="evidence-scan">
      <circle cx={s / 2} cy={s / 2} r={r0} fill="none" stroke="#2a2f3a" />
      <circle cx={s / 2} cy={s / 2} r={r0 / 2} fill="none" stroke="#2a2f3a" />
      <polygon points={pts} fill="rgba(255,77,79,0.25)" stroke="#ff4d4f" />
      <line x1={s / 2} y1={s / 2} x2={s / 2} y2={6} stroke="#8a93a3" strokeDasharray="3 3" />
    </svg>
  )
}

/** The camera frame a decision was based on, with that frame's detection boxes. */
function EvidenceFrame({ frameId }: { frameId: string }) {
  const [gone, setGone] = useState(false)
  const det = sim.detectionsByFrame.get(frameId)
  return (
    <div className="evidence-frame">
      {!gone && <img src={`/api/frames/${encodeURIComponent(frameId)}`} alt={`frame ${frameId}`} onError={() => setGone(true)} />}
      {gone && <div className="gone">frame {frameId} no longer in the buffer</div>}
      {!gone &&
        det?.boxes.map((b, n) => (
          <div
            key={n}
            className={`bbox bbox-${b.label}`}
            style={{
              left: `${(b.x / CAM_W) * 100}%`,
              top: `${(b.y / CAM_H) * 100}%`,
              width: `${(b.w / CAM_W) * 100}%`,
              height: `${(b.h / CAM_H) * 100}%`,
            }}
          >
            <span>
              {b.label} {b.score.toFixed(2)}
            </span>
          </div>
        ))}
    </div>
  )
}

function Row({ row, colour, open, onToggle }: { row: TraceRow; colour: string; open: boolean; onToggle: () => void }) {
  const e = row.evidence
  const det = e.frame_id ? sim.detectionsByFrame.get(e.frame_id) : undefined
  return (
    <li className={`trace-row ${open ? 'open' : ''}`} onClick={onToggle}>
      <div className="trace-line">
        <span className="mono muted trace-t">{row.t.toFixed(1)}s</span>
        <span className="trace-drone" style={{ color: colour }}>
          {row.drone}
        </span>
        <span className={`chip action action-${row.action}`}>{row.action}</span>
        <span className="trace-reason">{row.reason}</span>
        {(e.frame_id || e.scan) && <span className="muted evidence-hint">{e.frame_id ? '📷' : '📡'}</span>}
      </div>
      {open && (
        <div className="trace-evidence" onClick={(ev) => ev.stopPropagation()}>
          <div className="evidence-row">
            <span>
              evidence: <b>{e.sensor}</b> = <span className="mono">{fmtValue(e.value)}</span>
            </span>
            {e.frame_id && <span className="mono muted">frame {e.frame_id}</span>}
            {det?.gemma?.caption && (
              <span>
                Gemma: “{det.gemma.caption}”{det.gemma.conf != null && <span className="muted"> · {det.gemma.conf.toFixed(2)}</span>}
              </span>
            )}
          </div>
          {(e.frame_id || e.scan) && (
            <div className="evidence-media">
              {e.frame_id && <EvidenceFrame frameId={e.frame_id} />}
              {e.scan && e.scan.length > 0 && <ScanPolar scan={e.scan} />}
            </div>
          )}
        </div>
      )}
    </li>
  )
}

const COLLAPSED_ROWS = 3

/**
 * Decision trace. Collapsed (default): only the 3 most recent decisions.
 * Expanded: the full history, scrolling, stuck to the bottom unless the user scrolls up.
 */
export function TracePanel() {
  const trace = useSim((s) => s.trace)
  const hello = useSim((s) => s.hello)
  const listRef = useRef<HTMLUListElement>(null)
  const stick = useRef(true)
  const [open, setOpen] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)
  const colours = new Map(hello?.drones.map((d) => [d.id, d.colour]))
  const rows = expanded ? trace : trace.slice(-COLLAPSED_ROWS)

  useEffect(() => {
    const el = listRef.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [trace, expanded])

  return (
    <section className={`trace-panel ${expanded ? 'expanded' : 'collapsed'}`}>
      <div className="trace-head">
        <h2>Decision trace</h2>
        {trace.length > COLLAPSED_ROWS && (
          <button className="trace-toggle" onClick={() => setExpanded(!expanded)}>
            {expanded ? 'Show latest 3' : `Show all (${trace.length})`}
          </button>
        )}
      </div>
      <ul
        ref={listRef}
        onScroll={(e) => {
          const el = e.currentTarget
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 30
        }}
      >
        {trace.length === 0 && <li className="muted trace-empty">No decisions yet.</li>}
        {rows.map((r) => (
          <Row
            key={r.id}
            row={r}
            colour={colours.get(r.drone) ?? '#d8dde6'}
            open={open === r.id}
            onToggle={() => setOpen(open === r.id ? null : r.id)}
          />
        ))}
      </ul>
    </section>
  )
}
