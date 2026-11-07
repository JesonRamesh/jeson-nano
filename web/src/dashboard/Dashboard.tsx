import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { useSocket } from '../net/useSocket'
import { Scene3D } from '../scene3d/Scene3D'
import { sim, useSim } from '../state/store'
import { Toasts } from '../ui/Toasts'
import { Banners } from './Banners'
import { DronePanel, type DronePanelHandle } from './DronePanel'
import { TracePanel } from './TracePanel'
import './dashboard.css'

const DEFAULT_DRONES = ['D1', 'D2', 'D3', 'D4'].map((id) => ({ id, colour: '#8590a0' }))
const TEMP_LIMIT_C = 70 // §7.4 retreat threshold
const MAP_TILE = 'map'
const FLIP_MS = 420

/** Browser full screen for the whole dashboard; also toggled with the F key. */
function useFullscreen() {
  const [on, setOn] = useState(() => !!document.fullscreenElement)
  useEffect(() => {
    const sync = () => setOn(!!document.fullscreenElement)
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return
      if (e.key === 'f' || e.key === 'F') toggleFullscreen()
    }
    document.addEventListener('fullscreenchange', sync)
    window.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('fullscreenchange', sync)
      window.removeEventListener('keydown', onKey)
    }
  }, [])
  return { on, toggle: toggleFullscreen }
}

function toggleFullscreen() {
  if (document.fullscreenElement) void document.exitFullscreen()
  else void document.documentElement.requestFullscreen()
}

function isTyping(e: KeyboardEvent) {
  return e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement
}

function fmtClock(t: number) {
  const m = Math.floor(t / 60)
  const s = t - m * 60
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`
}

/**
 * Smooth re-layout (FLIP): remember every tile's box before the layout changes, then after
 * React re-renders, start each tile from its old box and let it glide to the new one.
 * Tiles are never re-mounted (the 3D canvas and video feeds keep running).
 */
function useFlip() {
  const tiles = useRef(new Map<string, HTMLElement>())
  const before = useRef(new Map<string, DOMRect>())
  const register = useCallback(
    (key: string) => (el: HTMLElement | null) => {
      if (el) tiles.current.set(key, el)
      else tiles.current.delete(key)
    },
    [],
  )
  const capture = useCallback(() => {
    before.current = new Map([...tiles.current].map(([k, el]) => [k, el.getBoundingClientRect()]))
  }, [])
  const play = useCallback(() => {
    const anims: Animation[] = []
    for (const [k, el] of tiles.current) {
      const a = before.current.get(k)
      if (!a) continue
      const b = el.getBoundingClientRect()
      if (!b.width || !b.height) continue
      const dx = a.left - b.left
      const dy = a.top - b.top
      const sx = a.width / b.width
      const sy = a.height / b.height
      if (Math.abs(dx) < 1 && Math.abs(dy) < 1 && Math.abs(sx - 1) < 0.01 && Math.abs(sy - 1) < 0.01) continue
      const anim = el.animate(
        [
          { transformOrigin: 'top left', transform: `translate(${dx}px, ${dy}px) scale(${sx}, ${sy})` },
          { transformOrigin: 'top left', transform: 'none' },
        ],
        { duration: FLIP_MS, easing: 'cubic-bezier(0.2, 0.8, 0.2, 1)' },
      )
      anims.push(anim)
    }
    before.current.clear()
    // The 3D canvas measures itself with getBoundingClientRect (which includes the FLIP
    // transform), so ask it to re-measure once the tiles have settled.
    void Promise.all(anims.map((a) => a.finished)).then(() => window.dispatchEvent(new Event('resize')))
  }, [])
  return { register, capture, play }
}

/**
 * A tile in the stage: hover lifts it slightly; a click (not a drag, not on a control)
 * focuses it. The 3D tile only focuses on a plain click, so orbiting and the click tool
 * keep working.
 */
function Tile({
  id,
  slot,
  focus,
  label,
  register,
  onFocus,
  onBack,
  children,
}: {
  id: string
  slot: number // position in the default grid: 0 = map, 1-4 = drone feeds
  focus: string | null
  label: string
  register: (el: HTMLElement | null) => void
  onFocus: (id: string) => void
  onBack: () => void
  children: ReactNode
}) {
  const down = useRef<{ x: number; y: number } | null>(null)
  const isFocused = focus === id
  const role = focus === null ? 'grid' : isFocused ? 'focused' : 'rail'
  return (
    <section
      ref={register}
      className={`tile tile-${role} ${id === MAP_TILE ? 'tile-map' : 'tile-feed'}`}
      style={
        focus === null ? { gridArea: `t${slot}` } : isFocused ? { gridArea: '1 / 1 / 5 / 2' } : { gridColumn: '2' }
      }
      onPointerDown={(e) => (down.current = { x: e.clientX, y: e.clientY })}
      onClick={(e) => {
        const d = down.current
        const target = e.target as HTMLElement
        const moved = d ? Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5 : false
        if (moved || isFocused || target.closest('button, input, .click-mode')) return
        onFocus(id)
      }}
    >
      <div className="tile-inner">
        {children}
        <span className="corner tl" />
        <span className="corner tr" />
        <span className="corner bl" />
        <span className="corner br" />
        {isFocused ? (
          <button className="tile-back" onClick={onBack} title="Back to all views (Esc)">
            ▦ All views
          </button>
        ) : (
          <span className="tile-hint">{label} · click to focus</span>
        )}
      </div>
    </section>
  )
}

/** Act 1 layout (§10.1, adapted): 3D map + 4 drone feeds in one stage, short trace below. */
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

  // ---- focus + smooth re-arrangement ----
  const [focus, setFocusState] = useState<string | null>(null)
  const flip = useFlip()
  const setFocus = useCallback(
    (next: string | null) => {
      flip.capture()
      setFocusState(next)
    },
    [flip],
  )
  useLayoutEffect(() => flip.play(), [focus, flip])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e) || e.key !== 'Escape' || focus === null) return
      if (document.querySelector('.scene3d.click-mode')) return // Esc cancels the click tool first
      setFocus(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [focus, setFocus])

  const tileProps = (id: string, slot: number, label: string) => ({
    id,
    slot,
    label,
    focus,
    register: flip.register(id),
    onFocus: (t: string) => setFocus(t),
    onBack: () => setFocus(null),
  })

  return (
    <div className="dash">
      <Toasts />
      <Banners />
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            <i />
            <i />
            <i />
            <i />
          </span>
          <span className="brand-name">Swarm-Guided Rescue</span>
        </div>
        <div className="telemetry">
          <span className={`tm tm-link tm-${status}`}>
            <b>Link</b>
            <span className="led" />
            {status}
          </span>
          <span className="tm">
            <b>Map</b>
            {hello ? hello.map.name : '...'}
          </span>
          <span className="tm">
            <b>T+</b>
            {state ? fmtClock(state.t) : '--:--.-'}
          </span>
          <span className="tm">
            <b>Speed</b>
            {state ? `${state.speed}×` : '-'}
          </span>
          <span className="tm">
            <b>Act</b>
            {state?.act ?? '-'}
          </span>
          <span className="tm">
            <b>Phase</b>
            {state?.phase || '-'}
          </span>
          {versionMismatch && <span className="tm tm-bad">contract mismatch: server v{hello?.contract_version}</span>}
        </div>
        <div className="top-actions">
          {focus !== null && (
            <button className="btn" onClick={() => setFocus(null)} title="Back to all views (Esc)">
              ▦ All views
            </button>
          )}
          <button className="btn" onClick={fullscreen.toggle} title="Full screen (F)">
            {fullscreen.on ? 'Exit full screen' : 'Full screen'}
          </button>
        </div>
      </header>

      <main className={`stage ${focus === null ? 'stage-grid' : 'stage-focus'}`}>
        <Tile {...tileProps(MAP_TILE, 0, '3D map')}>
          <Scene3D send={send} />
        </Tile>
        {drones.map((d, i) => (
          <Tile key={d.id} {...tileProps(d.id, i + 1, `${d.id} camera`)}>
            <DronePanel
              ref={(h) => {
                panels.current[i] = h
              }}
              id={d.id}
              colour={d.colour}
              state={state?.drones.find((s) => s.id === d.id)}
              detection={detections[d.id]}
              tempLimit={TEMP_LIMIT_C}
            />
          </Tile>
        ))}
      </main>
      <TracePanel />
    </div>
  )
}
