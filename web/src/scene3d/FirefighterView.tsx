import { Canvas, useFrame } from '@react-three/fiber'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { CalloutMsg, ClientMsg, GuidanceMsg } from '../contracts'
import { sim, useSim } from '../state/store'
import { noEmDash } from '../ui/text'
import { Dressing } from './dressing'
import { ClickObjects, Drones, Environment, Routes, SimFrame, SurvivorMarker } from './Scene3D'
import { headset } from './voice'

const CALLOUT_SHOW_MS = 5000 // how long a drone's line stays on screen

// Act 2 firefighter view (spec §9): first person through the scan, smoke limiting sight to
// about 4 m, drones glowing through it. MuJoCo owns the position (collisions); the browser
// renders at the server pose (smoothed) and owns the look direction, which it sends as an
// absolute heading with every ff_input (30 Hz).
//
// Controls: click to look around (pointer lock), WASD walk, arrow keys turn (no mouse
// needed), Esc releases the pointer.

const EYE_Z = 1.6 // m
const SEND_HZ = 30
const MOUSE_RAD_PER_PX = 0.0025
const TURN_RAD_PER_S = 1.8 // arrow keys
const FOG = new THREE.FogExp2('#202326', 0.42) // ~4 m visibility

/** Shared, mutable look + input state for the view (refs, not React state: 30-60 Hz). */
interface Look {
  yaw: number // sim radians, CCW from +x (absolute heading sent to the server)
  pitch: number
  keys: Set<string>
  init: boolean
}

function FirstPersonCamera({ getLook }: { getLook: () => Look }) {
  const pos = useMemo(() => new THREE.Vector3(), [])
  const dir = useMemo(() => new THREE.Vector3(), [])
  useFrame(({ camera, scene }, dt) => {
    scene.fog = FOG // the dressing sets its own (lighter) fog; the firefighter sees ~4 m
    const ff = sim.snap.state?.ff
    const l = getLook()
    if (!ff) return
    if (!l.init) {
      l.yaw = ff.yaw
      l.init = true
      camera.position.set(ff.x, EYE_Z, -ff.y)
    }
    // arrow keys turn
    if (l.keys.has('arrowleft')) l.yaw += TURN_RAD_PER_S * dt
    if (l.keys.has('arrowright')) l.yaw -= TURN_RAD_PER_S * dt
    // sim (x, y, z) -> three (x, z, -y)
    pos.set(ff.x, EYE_Z, -ff.y)
    camera.position.lerp(pos, 1 - Math.exp(-dt * 12))
    const cp = Math.cos(l.pitch)
    dir.set(Math.cos(l.yaw) * cp, Math.sin(l.pitch), -Math.sin(l.yaw) * cp)
    camera.lookAt(camera.position.x + dir.x, camera.position.y + dir.y, camera.position.z + dir.z)
  })
  return null
}

/** Guidance arrow: arrow_deg is relative to the firefighter's heading, CCW positive (+90 = left). */
function Arrow({ deg }: { deg: number }) {
  return (
    <svg viewBox="0 0 40 40" width="46" height="46" style={{ transform: `rotate(${-deg}deg)` }}>
      <path d="M20 4 L31 22 L23 20 L23 35 L17 35 L17 20 L9 22 Z" fill="currentColor" />
    </svg>
  )
}

/** The drones' voices: each callout is spoken once (Gemini audio, or a local voice per drone). */
function useDroneVoices(c: CalloutMsg | null) {
  const hello = useSim((s) => s.hello)
  const pushed = useRef<CalloutMsg | null>(null)
  useEffect(() => {
    headset.reset() // a new run (fresh hello after a snapshot load / reconnect): the headset opens again
  }, [hello])
  useEffect(() => {
    if (!c || pushed.current === c) return // once per line (StrictMode runs effects twice)
    pushed.current = c
    headset.push({
      text: c.text,
      speaker: c.speaker,
      severity: c.severity,
      audio: c.audio_b64 && c.audio_mime ? { b64: c.audio_b64, mime: c.audio_mime } : null,
    })
  }, [c])
  useEffect(() => () => headset.stop(), []) // leaving the view: silence
}

/** The latest drone line, shown for a few seconds under the HUD arrow. */
function CalloutBubble({ c }: { c: CalloutMsg | null }) {
  const [shown, setShown] = useState<CalloutMsg | null>(null)
  const colour = useSim((s) => s.hello?.drones.find((d) => d.id === c?.speaker)?.colour)
  useEffect(() => {
    if (!c) return
    const show = setTimeout(() => setShown(c), 0)
    const hide = setTimeout(() => setShown((cur) => (cur === c ? null : cur)), CALLOUT_SHOW_MS)
    return () => {
      clearTimeout(show)
      clearTimeout(hide)
    }
  }, [c])
  if (!shown) return null
  return (
    <div className={`ff-callout ff-callout-${shown.severity}`}>
      <span className="ff-callout-who" style={colour ? { color: colour, borderColor: colour } : undefined}>
        {shown.speaker}
      </span>
      {noEmDash(shown.text)}
    </div>
  )
}

/** Speaks guidance with a local (offline) voice through the shared headset queue; danger lines interrupt.
 *  One beep per alarm. */
function useVoiceAndAlarm(g: GuidanceMsg | null) {
  const audio = useRef<AudioContext | null>(null)
  const lastAlarm = useRef(false)
  useEffect(() => {
    if (!g) return
    if (g.say) {
      const line = { text: g.say, speaker: null, severity: g.severity }
      if (g.say.startsWith('Survivor reached')) headset.finish(line) // last word: clear the queue, then silence
      else headset.push(line)
    }
    if (g.alarm && !lastAlarm.current) {
      try {
        audio.current ??= new AudioContext()
        const ctx = audio.current
        const osc = ctx.createOscillator()
        const gain = ctx.createGain()
        osc.type = 'square'
        osc.frequency.value = 880
        gain.gain.setValueAtTime(0.0001, ctx.currentTime)
        gain.gain.exponentialRampToValueAtTime(0.15, ctx.currentTime + 0.02)
        gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.35)
        osc.connect(gain).connect(ctx.destination)
        osc.start()
        osc.stop(ctx.currentTime + 0.4)
      } catch {
        /* audio blocked until a user gesture: subtitles still show */
      }
    }
    lastAlarm.current = g.alarm
  }, [g])
}

export function FirefighterView({ send }: { send: (m: ClientMsg) => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const look = useRef<Look>({ yaw: 0, pitch: 0, keys: new Set(), init: false })
  const getLook = useCallback(() => look.current, [])
  const [locked, setLocked] = useState(false)
  const g = useSim((s) => s.guidance)
  const ffKnown = useSim((s) => !!s.state?.ff)
  const callout = useSim((s) => s.callout)
  useVoiceAndAlarm(g)
  useDroneVoices(callout)

  // input: pointer lock + mouse look, WASD + arrows (only while this view is on screen)
  useEffect(() => {
    const l = look.current
    const wrap = wrapRef.current
    const typing = (e: KeyboardEvent) => e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement
    const down = (e: KeyboardEvent) => {
      if (typing(e) || e.ctrlKey || e.metaKey || e.altKey) return
      const k = e.key.toLowerCase()
      if (['w', 'a', 's', 'd', 'arrowleft', 'arrowright', 'arrowup', 'arrowdown'].includes(k)) {
        l.keys.add(k)
        if (k.startsWith('arrow')) e.preventDefault()
      }
    }
    const up = (e: KeyboardEvent) => l.keys.delete(e.key.toLowerCase())
    const blur = () => l.keys.clear()
    const move = (e: MouseEvent) => {
      if (!wrap || document.pointerLockElement !== wrap) return
      l.yaw -= e.movementX * MOUSE_RAD_PER_PX
      l.pitch = Math.max(-1.2, Math.min(1.2, l.pitch - e.movementY * MOUSE_RAD_PER_PX))
    }
    const lockChange = () => setLocked(!!wrap && document.pointerLockElement === wrap)
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', blur)
    window.addEventListener('mousemove', move)
    document.addEventListener('pointerlockchange', lockChange)
    // 30 Hz: fwd/strafe from the keys, yaw = absolute heading
    const timer = setInterval(() => {
      if (!l.init) return
      const k = l.keys
      const fwd = (k.has('w') || k.has('arrowup') ? 1 : 0) - (k.has('s') || k.has('arrowdown') ? 1 : 0)
      const strafe = (k.has('d') ? 1 : 0) - (k.has('a') ? 1 : 0)
      send({ type: 'ff_input', fwd, strafe, yaw: Math.atan2(Math.sin(l.yaw), Math.cos(l.yaw)) })
    }, 1000 / SEND_HZ)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', blur)
      window.removeEventListener('mousemove', move)
      document.removeEventListener('pointerlockchange', lockChange)
      clearInterval(timer)
      if (wrap && document.pointerLockElement === wrap) document.exitPointerLock()
      l.keys.clear()
    }
  }, [send])

  const sev = g?.severity ?? 'info'
  return (
    <div
      ref={wrapRef}
      className={`ffview sev-${sev} ${locked ? 'locked' : ''}`}
      onClick={() => {
        if (document.pointerLockElement !== wrapRef.current) void wrapRef.current?.requestPointerLock()
      }}
    >
      <Canvas camera={{ fov: 75, near: 0.05, far: 200 }} dpr={1} onCreated={({ gl }) => {
          gl.localClippingEnabled = true
        }}>
        <color attach="background" args={['#202326']} />
        <hemisphereLight args={['#c8d2e0', '#202020', 0.9]} />
        <SimFrame>
          <Environment />
          <Routes />
          <ClickObjects />
          <SurvivorMarker />
          <Drones />
          <Dressing />
        </SimFrame>
        <FirstPersonCamera getLook={getLook} />
      </Canvas>

      {/* HUD: arrow + distance, subtitles, danger edges */}
      <div className="ff-vignette" />
      {g && (
        <div className="ff-guide">
          <Arrow deg={g.arrow_deg} />
          <span className="ff-dist">{g.distance.toFixed(0)} m</span>
        </div>
      )}
      <CalloutBubble c={callout} />
      <div className="ff-subtitle">{g?.subtitle ? noEmDash(g.subtitle) : ffKnown ? 'Waiting for guidance' : 'Waiting for the firefighter'}</div>
      {!locked && <div className="ff-hint">Click to look around · WASD to walk · arrows to turn · Esc to release</div>}
    </div>
  )
}
