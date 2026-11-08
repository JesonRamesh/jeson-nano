import { useProgress } from '@react-three/drei'
import { useEffect, useState } from 'react'
import { useSim } from '../state/store'

// Start-up cover: while the sim connects and the 3D scan / drone models stream in (the dashboard
// stutters meanwhile), blur everything and fly a drone around the four-point Gemini spark.
// It lifts once the sim is talking (hello + state) and the 3D loads have finished, plus a short
// settle so the first heavy frames (shader compile) happen behind it. Never longer than MAX_MS.

const MIN_MS = 2200 // show long enough to register, even on a warm reload
const SETTLE_MS = 1200 // after everything has loaded
const NO_ASSETS_MS = 12000 // nothing started loading this long after hello: don't wait for it
const MAX_MS = 90000
const HOLD = new URLSearchParams(location.search).has('loading') // ?loading keeps it up (to style / demo it)

// Spark outline in a 240 x 240 box (the GeminiSpark path x10), tips slightly rounded.
const SPARK =
  'M120 12 C126 68 172 114 228 120 C172 126 126 172 120 228 C114 172 68 126 12 120 C68 114 114 68 120 12 Z'
const TRAIL = 9 // liquid blobs following the drone
const LAP_S = 5.6 // one lap of the spark
const LAG_S = 0.1 // each blob trails the one before by this much

export function LoadingScreen() {
  const hello = useSim((s) => s.hello)
  const hasState = useSim((s) => s.state !== null)
  const { active, progress, total, item } = useProgress()
  const [start] = useState(() => performance.now())
  const [helloAt, setHelloAt] = useState<number | null>(null)
  const [phase, setPhase] = useState<'on' | 'leaving' | 'gone'>('on')

  useEffect(() => {
    if (hello && helloAt === null) {
      const t = setTimeout(() => setHelloAt(performance.now()), 0)
      return () => clearTimeout(t)
    }
  }, [hello, helloAt])

  const assetsDone = total > 0 && !active && progress >= 100
  const ready = !!hello && hasState && assetsDone

  useEffect(() => {
    if (phase !== 'on' || HOLD) return
    const now = performance.now()
    let wait: number
    if (ready) wait = Math.max(SETTLE_MS, MIN_MS - (now - start))
    else if (hello && hasState && total === 0 && helloAt !== null) wait = Math.max(0, NO_ASSETS_MS - (now - helloAt))
    else wait = Math.max(0, MAX_MS - (now - start))
    const t = setTimeout(() => setPhase('leaving'), wait)
    return () => clearTimeout(t)
  }, [ready, phase, start, hello, hasState, total, helloAt])

  useEffect(() => {
    if (phase !== 'leaving') return
    const t = setTimeout(() => setPhase('gone'), 900)
    return () => clearTimeout(t)
  }, [phase])

  if (phase === 'gone') return null
  const what = !hello ? 'Connecting to the swarm' : !hasState ? 'Starting the sim' : 'Loading the 3D scan'
  const pct = total > 0 ? Math.round(progress) : null
  return (
    <div className={`loading ${phase === 'leaving' ? 'leaving' : ''}`} role="status" aria-live="polite">
      <svg width="0" height="0" style={{ position: 'absolute' }} aria-hidden>
        <defs>
          {/* gooey: blur the blobs together, then sharpen the alpha into one liquid shape */}
          <filter id="loading-goo">
            <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="b" />
            <feColorMatrix in="b" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -7" />
          </filter>
          <linearGradient id="loading-spark" x1="0" y1="1" x2="1" y2="0">
            <stop offset="0" stopColor="#3a8bff" />
            <stop offset="0.45" stopColor="#3a8bff" />
            <stop offset="0.68" stopColor="#ee5257" />
            <stop offset="0.84" stopColor="#d0c42e" />
            <stop offset="1" stopColor="#4cc88b" />
          </linearGradient>
        </defs>
      </svg>
      <div className="loading-stage">
        <svg className="loading-path" viewBox="0 0 240 240" aria-hidden>
          <path d={SPARK} fill="url(#loading-spark)" opacity="0.10" />
          <path d={SPARK} fill="none" stroke="url(#loading-spark)" strokeWidth="1.5" opacity="0.55" />
        </svg>
        <div className="loading-goo" aria-hidden>
          {Array.from({ length: TRAIL }, (_, i) => (
            <span key={i} className={`loading-blob b${i % 4}`} style={{ animationDuration: `${LAP_S}s`, animationDelay: `${-(LAP_S - LAG_S * (i + 1))}s`, ['--k' as string]: 1 - i / (TRAIL + 2) }} />
          ))}
        </div>
        <div className="loading-drone" style={{ animationDuration: `${LAP_S}s` }} aria-hidden>
          <svg viewBox="-20 -20 40 40" width="40" height="40">
            <g stroke="#eef1f4" strokeWidth="2" strokeLinecap="round">
              <line x1="-11" y1="-11" x2="11" y2="11" />
              <line x1="11" y1="-11" x2="-11" y2="11" />
            </g>
            {[
              [-11, -11],
              [11, -11],
              [11, 11],
              [-11, 11],
            ].map(([x, y], i) => (
              <circle key={i} className="rotor" cx={x} cy={y} r="6" fill="rgba(168,199,250,0.18)" stroke="#a8c7fa" strokeWidth="1.2" />
            ))}
            <rect x="-5" y="-5" width="10" height="10" rx="3" fill="#0b57d0" stroke="#eef1f4" strokeWidth="1.5" />
            <circle cx="0" cy="-2" r="1.6" fill="#ee5257" />
          </svg>
        </div>
      </div>
      <div className="loading-text">
        <div className="loading-title">{what}</div>
        <div className="loading-sub">
          {pct !== null && hello ? `${pct}%` : ' '}
          {item && hello && pct !== null && pct < 100 ? ` · ${item.split('/').pop()?.split('?')[0]}` : ''}
        </div>
      </div>
    </div>
  )
}
