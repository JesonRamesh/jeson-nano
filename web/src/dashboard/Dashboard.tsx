import { useEffect, useRef, useState } from 'react'
import { Scene3D } from '../scene3d/Scene3D'
import { Toasts } from '../ui/Toasts'
import { useSocket } from '../net/useSocket'
import { sim, useSim } from '../state/store'
import { Banners } from './Banners'
import { DronePanel, type DronePanelHandle } from './DronePanel'
import { TracePanel } from './TracePanel'
import './dashboard.css'

const DEFAULT_DRONES = ['D1', 'D2', 'D3', 'D4'].map((id) => ({ id, colour: '#8a93a3' }))
const TEMP_LIMIT_C = 70 // §7.4 retreat threshold

/** Browser full screen for the whole dashboard; also toggled with the F key. */
function useFullscreen() {
  const [on, setOn] = useState(() => !!document.fullscreenElement)
  useEffect(() => {
    const sync = () => setOn(!!document.fullscreenElement)
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement
      if ((e.key === 'f' || e.key === 'F') && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) toggle()
    }
    document.addEventListener('fullscreenchange', sync)
    window.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('fullscreenchange', sync)
      window.removeEventListener('keydown', onKey)
    }
  }, [])
  return { on, toggle }
}

function toggle() {
  if (document.fullscreenElement) void document.exitFullscreen()
  else void document.documentElement.requestFullscreen()
}

/** Act 1 layout (§10.1, adapted), 1920×1080: 3D view | 2×2 drone feeds, short trace below. */
export default function Dashboard() {
  const panels = useRef<(DronePanelHandle | null)[]>([])
  const fullscreen = useFullscreen()
  const { status, versionMismatch, send } = useSocket({
    onMessage: (msg) => sim.handle(msg),
    onFrame: (f) => panels.current[f.channel - 1]?.showFrame(f.jpeg, f.seq),
  })
  const hello = useSim((s) => s.hello)
  const state = useSim((s) => s.state)
  const detections = useSim((s) => s.detections)
  const drones = hello?.drones ?? DEFAULT_DRONES

  return (
    <div className="dash">
      <Toasts />
      <Banners />
      <header className="topbar">
        <strong>Swarm-Guided Rescue</strong>
        <span className={`conn conn-${status}`}>● {status}</span>
        {versionMismatch && <span className="bad">contract mismatch: server v{hello?.contract_version}</span>}
        <span className="muted mono">
          {hello ? `map ${hello.map.name}` : 'no hello yet'}
          {state && ` · t=${state.t.toFixed(1)} s · ${state.speed}× · act ${state.act}`}
          {state?.phase && ` · ${state.phase}`}
        </span>
        <button className="fullscreen-btn" onClick={fullscreen.toggle} title="Full screen (F)">
          {fullscreen.on ? '⤡ Exit full screen' : '⤢ Full screen'}
        </button>
      </header>
      <main className="act1">
        <section className="map-area">
          <Scene3D send={send} />
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
      </main>
    </div>
  )
}
