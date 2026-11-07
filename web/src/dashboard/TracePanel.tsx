import { useEffect, useRef, useState } from 'react'
import type { TraceRow } from '../contracts'
import { useSim } from '../state/store'

function fmtValue(v: unknown): string {
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(2)
  if (typeof v === 'string') return v
  return JSON.stringify(v)
}

function Row({ row, colour, open, onToggle }: { row: TraceRow; colour: string; open: boolean; onToggle: () => void }) {
  const e = row.evidence
  return (
    <li className={`trace-row ${open ? 'open' : ''}`} onClick={onToggle}>
      <div className="trace-line">
        <span className="mono muted trace-t">{row.t.toFixed(1)}s</span>
        <span className="trace-drone" style={{ color: colour }}>
          {row.drone}
        </span>
        <span className={`chip action action-${row.action}`}>{row.action}</span>
        <span className="trace-reason">{row.reason}</span>
      </div>
      {open && (
        <div className="trace-evidence">
          <span>
            evidence: <b>{e.sensor}</b> = <span className="mono">{fmtValue(e.value)}</span>
          </span>
          {e.frame_id && <span className="mono muted">frame {e.frame_id}</span>}
          {e.scan && <span className="mono muted">scan: {e.scan.length} rays</span>}
        </div>
      )}
    </li>
  )
}

/** Scrolling decision trace. Sticks to the bottom unless the user scrolls up. */
export function TracePanel() {
  const trace = useSim((s) => s.trace)
  const hello = useSim((s) => s.hello)
  const listRef = useRef<HTMLUListElement>(null)
  const stick = useRef(true)
  const [open, setOpen] = useState<string | null>(null)
  const colours = new Map(hello?.drones.map((d) => [d.id, d.colour]))

  useEffect(() => {
    const el = listRef.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [trace])

  return (
    <section className="trace-panel">
      <h2>Decision trace</h2>
      <ul
        ref={listRef}
        onScroll={(e) => {
          const el = e.currentTarget
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 30
        }}
      >
        {trace.length === 0 && <li className="muted trace-empty">No decisions yet.</li>}
        {trace.map((r) => (
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
