import { useRef, useState } from 'react'
import type { StateMsg } from '../contracts'
import { useSocket } from '../net/useSocket'
import { DronePanel, type DronePanelHandle } from './DronePanel'
import './dashboard.css'

const DEFAULT_DRONES = [
  { id: 'D1', color: '#4fc3f7' },
  { id: 'D2', color: '#ba68c8' },
  { id: 'D3', color: '#ffd54f' },
  { id: 'D4', color: '#81c784' },
]

export default function Dashboard() {
  const [state, setState] = useState<StateMsg | null>(null)
  const panels = useRef<(DronePanelHandle | null)[]>([])

  const { status, hello, versionMismatch } = useSocket({
    onMessage: (msg) => {
      if (msg.type === 'state') setState(msg)
    },
    onFrame: (f) => panels.current[f.channel - 1]?.showFrame(f.jpeg, f.seq),
  })

  const drones = hello?.drones ?? DEFAULT_DRONES

  return (
    <div className="dash">
      <header className="topbar">
        <strong>Swarm-Guided Rescue</strong>
        <span className={`conn conn-${status}`}>● {status}</span>
        {versionMismatch && (
          <span className="bad">
            contract mismatch: server v{hello?.contract_version}
          </span>
        )}
        <span className="muted mono">
          {hello ? `map ${hello.map.name}` : 'no hello yet'}
          {state && ` · t=${state.t.toFixed(1)} s · ${state.speed}× · act ${state.act}`}
        </span>
      </header>
      <main className="grid4">
        {drones.map((d, i) => (
          <DronePanel
            key={d.id}
            ref={(h) => {
              panels.current[i] = h
            }}
            id={d.id}
            color={d.color}
            state={state?.drones.find((s) => s.id === d.id)}
          />
        ))}
      </main>
    </div>
  )
}
