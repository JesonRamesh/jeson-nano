import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { useSocket } from '../net/useSocket'
import { FirefighterView } from '../scene3d/FirefighterView'
import { Scene3D } from '../scene3d/Scene3D'
import type { SummaryMsg } from '../contracts'
import { sim, useSim } from '../state/store'
import { Toasts } from '../ui/Toasts'
import { Banners } from './Banners'
import { DronePanel, type DronePanelHandle } from './DronePanel'
import { NetworkGraph } from './NetworkGraph'
import { TracePanel } from './TracePanel'
import './dashboard.css'

const DEFAULT_DRONES = ['D1', 'D2', 'D3', 'D4'].map((id) => ({ id, colour: '#8590a0' }))
const TEMP_LIMIT_C = 70 // §7.4 retreat threshold
const MAP_TILE = 'map'
const LOG_TILE = 'log'
const NET_TILE = 'net'
const FF_TILE = 'ff'
const FLIP_MS = 420

/** Browser full screen for the whole dashboard; also toggled with the F key. */
function useFullscreen() {
  const [on, setOn] = useState(() => !!document.fullscreenElement)
  useEffect(() => {
    const sync = () => setOn(!!document.fullscreenElement)
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey || document.pointerLockElement) return
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
  rows,
  focus,
  label,
  register,
  onFocus,
  onBack,
  children,
}: {
  id: string
  slot: number // position in the default grid: 0 = map, 1-4 = drone feeds, 5 = log, 6 = links
  rows: number // rail length while something is focused
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
      className={`tile tile-${role} tile-${id === MAP_TILE ? 'map' : id === LOG_TILE ? 'log' : id === NET_TILE ? 'net' : id === FF_TILE ? 'ff' : 'feed'}`}
      style={
        focus === null ? { gridArea: `t${slot}` } : isFocused ? { gridArea: `1 / 1 / ${rows + 1} / 2` } : { gridColumn: '2' }
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

/** Four corner brackets; pointing out = enter full screen, pointing in = exit. */
function FullscreenIcon({ exit }: { exit: boolean }) {
  const d = exit
    ? 'M7 2v5H2M13 2v5h5M7 18v-5H2M13 18v-5h5'
    : 'M2 7V2h5M18 7V2h-5M2 13v5h5M18 13v5h-5'
  return (
    <svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="square">
      <path d={d} />
    </svg>
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
  // Only flag the link after it has been down for a moment (no flash on a quick reconnect).
  const [linkDown, setLinkDown] = useState(false)
  useEffect(() => {
    if (status === 'connected') {
      const t = setTimeout(() => setLinkDown(false), 0)
      return () => clearTimeout(t)
    }
    const t = setTimeout(() => setLinkDown(true), 2000)
    return () => clearTimeout(t)
  }, [status])

  // ---- acts ----
  // Act 2: the firefighter view takes the stage; the 3D map, log and links sit in the rail;
  // drone feeds are hidden (spec §10.1). "All views" / Esc in Act 2 comes back to the FF view.
  const act = state?.act ?? 1
  const phase = state?.phase ?? ''
  const sendAct2 = useCallback(() => send({ type: 'act2_start' }), [send])
  // Restart Act 1 from the beginning (the server loads the act1_start snapshot and re-sends hello,
  // which clears the summary card)
  const restartAct1 = useCallback(() => send({ type: 'control', action: 'reset' }), [send])
  useEffect(() => {
    // G = send in the firefighter (loads the act2_start snapshot: chain at its posts)
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey || document.pointerLockElement) return
      if ((e.key === 'g' || e.key === 'G') && sim.snap.state?.act !== 2) sendAct2()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [sendAct2])

  // ---- focus + smooth re-arrangement ----
  const [focusState, setFocusState] = useState<string | null>(null)
  const focus = focusState ?? (act === 2 ? FF_TILE : null)
  const flip = useFlip()
  const setFocus = useCallback(
    (next: string | null) => {
      flip.capture()
      setFocusState(next)
    },
    [flip],
  )
  useLayoutEffect(() => flip.play(), [focus, flip])
  // a new act starts from its default arrangement
  const prevAct = useRef(act)
  useEffect(() => {
    if (prevAct.current === act) return
    prevAct.current = act
    const t = setTimeout(() => setFocus(null), 0)
    return () => clearTimeout(t)
  }, [act, setFocus])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e) || e.key !== 'Escape' || focusState === null) return
      if (document.querySelector('.scene3d.click-mode')) return // Esc cancels the click tool first
      setFocus(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [focusState, setFocus])

  const visible = act === 2 ? 4 : 7 // tiles on the stage
  const tileProps = (id: string, slot: number, label: string) => ({
    id,
    slot,
    rows: visible - 1,
    label,
    focus,
    register: flip.register(id),
    onFocus: (t: string) => setFocus(t),
    onBack: () => setFocus(null),
  })
  const summary = useSim((s) => s.summary)
  const [summaryHidden, setSummaryHidden] = useState<SummaryMsg | null>(null)
  const summaryOpen = !!summary && summaryHidden !== summary
  useEffect(() => {
    // while the summary card is open: G = run Act 2 again, R = restart Act 1
    if (!summaryOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey || document.pointerLockElement) return
      if (e.key === 'g' || e.key === 'G') sendAct2()
      else if (e.key === 'r' || e.key === 'R') restartAct1()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [summaryOpen, sendAct2, restartAct1])

  return (
    <div className="dash">
      <Toasts />
      <Banners />
      {linkDown && <div className="link-pill">{status === 'connecting' ? 'Connecting to sim' : 'Sim link lost, reconnecting'}</div>}
      {versionMismatch && <div className="link-pill bad">Contract mismatch: server v{hello?.contract_version}</div>}

      {act === 1 && phase === 'chain_ready' && (
        <button className="act2-btn" onClick={sendAct2} title="Start Act 2 (G)">
          Send in firefighter <kbd>G</kbd>
        </button>
      )}
      {summary && summaryOpen && (
        <div className="summary-card">
          <h2>Mission complete</h2>
          <dl>
            <dt>Time</dt>
            <dd>{summary.time_s.toFixed(0)} s</dd>
            <dt>Distance walked</dt>
            <dd>{summary.distance_m.toFixed(0)} m</dd>
            <dt>Hazards avoided</dt>
            <dd>{summary.hazards_avoided}</dd>
            <dt>Corrections</dt>
            <dd>{summary.corrections}</dd>
            <dt>Re-plans</dt>
            <dd>{summary.replans}</dd>
          </dl>
          <div className="summary-actions">
            <button className="btn summary-primary" onClick={sendAct2} title="Run Act 2 again (G)">
              Run Act 2 again <kbd>G</kbd>
            </button>
            <button className="btn" onClick={restartAct1} title="Restart Act 1 (R)">
              Restart Act 1 <kbd>R</kbd>
            </button>
            <button className="btn" onClick={() => setSummaryHidden(summary)}>
              Close
            </button>
          </div>
        </div>
      )}

      <main
        className={`stage ${focus === null ? 'stage-grid' : 'stage-focus'}`}
        style={focus === null ? undefined : { gridTemplateRows: `repeat(${visible - 1}, minmax(0, 1fr))` }}
      >
        {act === 2 && (
          <Tile {...tileProps(FF_TILE, 7, 'Firefighter view')}>
            <FirefighterView send={send} />
          </Tile>
        )}
        <Tile {...tileProps(MAP_TILE, 0, '3D map')}>
          <Scene3D send={send} />
        </Tile>
        {act === 1 && drones.map((d, i) => (
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
        <Tile {...tileProps(LOG_TILE, 5, 'Decision log')}>
          <TracePanel detailed={focus === LOG_TILE} />
        </Tile>
        <Tile {...tileProps(NET_TILE, 6, 'Swarm links')}>
          <NetworkGraph
            detailed={focus === NET_TILE}
            actions={
              <button
                className="icon-btn"
                onClick={fullscreen.toggle}
                title={fullscreen.on ? 'Exit full screen (F)' : 'Full screen (F)'}
                aria-label={fullscreen.on ? 'Exit full screen' : 'Full screen'}
              >
                <FullscreenIcon exit={fullscreen.on} />
              </button>
            }
          />
        </Tile>
      </main>
    </div>
  )
}
