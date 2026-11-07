import { useRef } from 'react'
import { useSocket } from '../net/useSocket'
import { sim, useSim } from '../state/store'
import { BeliefMap } from './BeliefMap'
import { BudgetPanel } from './BudgetPanel'
import { DronePanel, type DronePanelHandle } from './DronePanel'
import { TracePanel } from './TracePanel'
import './dashboard.css'

const DEFAULT_DRONES = ['D1', 'D2', 'D3', 'D4'].map((id) => ({ id, colour: '#8a93a3' }))
const TEMP_LIMIT_C = 70 // §7.4 retreat threshold

/** Act 1 layout (§10.1), 1920×1080. */
export default function Dashboard() {
  const panels = useRef<(DronePanelHandle | null)[]>([])
  const { status, versionMismatch } = useSocket({
    onMessage: (msg) => sim.handle(msg),
    onFrame: (f) => panels.current[f.channel - 1]?.showFrame(f.jpeg, f.seq),
  })
  const hello = useSim((s) => s.hello)
  const state = useSim((s) => s.state)
  const detections = useSim((s) => s.detections)
  const drones = hello?.drones ?? DEFAULT_DRONES

  return (
    <div className="dash">
      <header className="topbar">
        <strong>Swarm-Guided Rescue</strong>
        <span className={`conn conn-${status}`}>● {status}</span>
        {versionMismatch && <span className="bad">contract mismatch: server v{hello?.contract_version}</span>}
        <span className="muted mono">
          {hello ? `map ${hello.map.name}` : 'no hello yet'}
          {state && ` · t=${state.t.toFixed(1)} s · ${state.speed}× · act ${state.act}`}
          {state?.phase && ` · ${state.phase}`}
        </span>
      </header>
      <main className="act1">
        <section className="map-area">
          <BeliefMap />
        </section>
        <section className="drones">
          {drones.map((d, i) => (
            <DronePanel
              key={d.id}
              ref={(h) => {
                panels.current[i] = h
              }}
              id={d.id}
              colour={d.colour}
              state={state?.drones.find((s) => s.id === d.id)}
              detection={detections[d.id]}
              tempLimit={TEMP_LIMIT_C}
            />
          ))}
        </section>
        <TracePanel />
        <BudgetPanel />
      </main>
    </div>
  )
}
