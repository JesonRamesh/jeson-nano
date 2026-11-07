import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { TraceRow } from '../contracts'
import { sim, useSim } from '../state/store'
import { noEmDash } from '../ui/text'

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
      <polygon points={pts} fill="rgba(238,241,244,0.12)" stroke="#eef1f4" />
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

/** Action code -> short plain words for the log ("MARK_BLOCKED" -> "marked blocked"). */
const ACTION_WORDS: Record<string, string> = {
  ASSIGN_AREA: 'assigned area',
  EXPLORE: 'exploring',
  INSPECT: 'inspecting',
  CONFIRM_HAZARD: 'hazard confirmed',
  REJECT_DETECTION: 'rejected detection',
  UNCONFIRMED: 'unconfirmed',
  MARK_BLOCKED: 'marked blocked',
  REPLAN: 'replanned',
  RETREAT: 'retreating',
  DRONE_LOST: 'drone lost',
  REASSIGN: 'reassigned',
  SURVIVOR_CONFIRMED: 'survivor confirmed',
  CHAIN_POSITION: 'chain position',
  NO_ROUTE: 'no route',
  ESCORT: 'escorting',
  FF_CORRECTION: 'firefighter correction',
  MISSION_COMPLETE: 'mission complete',
}
const ALERT = new Set(['DRONE_LOST', 'MARK_BLOCKED', 'NO_ROUTE', 'RETREAT'])

function Row({ row, open, onToggle, detailed }: { row: TraceRow; open: boolean; onToggle: () => void; detailed: boolean }) {
  const e = row.evidence
  const det = e.frame_id ? sim.detectionsByFrame.get(e.frame_id) : undefined
  return (
    <li
      className={`trace-row ${open ? 'open' : ''} ${ALERT.has(row.action) ? 'alert' : ''}`}
      onClick={detailed ? onToggle : undefined}
    >
      <div className="trace-line">
        <span className="trace-t">{row.t.toFixed(1)}</span>
        <span className="trace-drone">{row.drone}</span>
        {detailed && <span className="trace-action">{ACTION_WORDS[row.action] ?? row.action.toLowerCase()}</span>}
        <span className="trace-reason">{noEmDash(row.reason)}</span>
        {detailed && (e.frame_id || e.scan) && <span className="evidence-hint">{e.frame_id ? 'img' : 'scan'}</span>}
      </div>
      {open && detailed && (
        <div className="trace-evidence" onClick={(ev) => ev.stopPropagation()}>
          <div className="evidence-row">
            <span>
              {e.sensor} <span className="mono">{noEmDash(fmtValue(e.value))}</span>
            </span>
            {e.frame_id && <span className="mono">frame {e.frame_id}</span>}
            {det?.gemma?.caption && (
              <span>
                Gemma “{noEmDash(det.gemma.caption)}”{det.gemma.conf != null && <span className="mono"> {det.gemma.conf.toFixed(2)}</span>}
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

const COMPACT_ROWS = 3

/**
 * Decision log. Compact (tile): the 3 most recent decisions, one line each.
 * Detailed (tile focused): the whole history with action, evidence tags; click a row to
 * open its evidence (camera frame + boxes + Gemma caption, or the LiDAR scan).
 */
export function TracePanel({ detailed = false, actions }: { detailed?: boolean; actions?: ReactNode }) {
  const trace = useSim((s) => s.trace)
  const listRef = useRef<HTMLUListElement>(null)
  const stick = useRef(true)
  const [open, setOpen] = useState<string | null>(null)
  const rows = detailed ? trace : trace.slice(-COMPACT_ROWS)

  useEffect(() => {
    const el = listRef.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [trace, detailed])

  return (
    <div className={`trace-panel ${detailed ? 'detailed' : 'compact'}`}>
      <div className="trace-head">
        <h2>Decision log</h2>
        <span className="trace-count">{trace.length}</span>
        {actions && <div className="trace-actions">{actions}</div>}
      </div>
      <ul
        ref={listRef}
        onScroll={(e) => {
          const el = e.currentTarget
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 30
        }}
      >
        {trace.length === 0 && <li className="trace-empty">No decisions yet</li>}
        {rows.map((r) => (
          <Row
            key={r.id}
            row={r}
            detailed={detailed}
            open={open === r.id}
            onToggle={() => setOpen(open === r.id ? null : r.id)}
          />
        ))}
      </ul>
    </div>
  )
}
