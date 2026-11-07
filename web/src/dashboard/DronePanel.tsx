import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import type { DetectionMsg, DroneState } from '../contracts'

export interface DronePanelHandle {
  showFrame: (jpeg: Blob, seq: number) => void
}

interface Props {
  id: string
  colour: string
  state?: DroneState
  detection?: DetectionMsg
  tempLimit: number
}

const CAM_W = 320
const CAM_H = 240
const TEMP_MAX = 120 // °C at the top of the bar

const STATUS_CLASS: Record<string, string> = {
  confirmed: 'chip-bad',
  low_conf: 'chip-warn',
  pending: 'chip-warn',
  rejected: 'chip-ok',
  unconfirmed: 'chip-warn',
}

/** LiDAR mini-polar: 36 rays, ray k at yaw + k·10°, drawn heading-up. */
function LidarPolar({ lidar, colour }: { lidar: number[]; colour: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const ctx = c.getContext('2d')!
    const s = c.width
    const r0 = s / 2 - 2
    ctx.clearRect(0, 0, s, s)
    ctx.strokeStyle = '#2a2f3a'
    ctx.lineWidth = 1
    for (const f of [1, 0.5]) {
      ctx.beginPath()
      ctx.arc(s / 2, s / 2, r0 * f, 0, Math.PI * 2)
      ctx.stroke()
    }
    if (!lidar.length) return
    ctx.fillStyle = colour + '55'
    ctx.strokeStyle = colour
    ctx.beginPath()
    lidar.forEach((r, k) => {
      // ray 0 = heading = up; CCW positive -> to the left on screen
      const a = Math.PI / 2 + (k * Math.PI) / 18
      const d = (Math.min(r, 6) / 6) * r0
      const x = s / 2 + d * Math.cos(a)
      const y = s / 2 - d * Math.sin(a)
      if (k === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.closePath()
    ctx.fill()
    ctx.stroke()
  }, [lidar, colour])
  return <canvas ref={ref} width={84} height={84} className="lidar-polar" />
}

/**
 * One drone panel: camera JPEG + bbox labels, LiDAR mini-polar, temperature bar,
 * Gemma caption + confidence, status chip. Lost -> static + "SIGNAL LOST".
 * Frames bypass React (≈10 fps × 4): the parent calls showFrame() directly.
 */
export const DronePanel = forwardRef<DronePanelHandle, Props>(function DronePanel(
  { id, colour, state, detection, tempLimit },
  ref,
) {
  const imgRef = useRef<HTMLImageElement>(null)
  const seqRef = useRef<HTMLSpanElement>(null)
  const urlRef = useRef<string | null>(null)

  useImperativeHandle(ref, () => ({
    showFrame(jpeg, seq) {
      const url = URL.createObjectURL(jpeg)
      if (imgRef.current) imgRef.current.src = url
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
      urlRef.current = url
      if (seqRef.current) seqRef.current.textContent = `#${seq}`
    },
  }))
  useEffect(() => () => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current)
  }, [])

  const alive = state?.alive ?? true
  const temp = state?.temp ?? 0
  const hot = temp > tempLimit
  const g = detection?.gemma

  return (
    <div className={`drone-panel ${alive ? '' : 'lost'}`} style={{ borderColor: colour }}>
      <div className="drone-head">
        <span className="drone-id" style={{ color: colour }}>
          {id}
        </span>
        <span className={`chip status-${(state?.status ?? '').toLowerCase()}`}>{state?.status ?? '—'}</span>
        <span className="muted drone-task">{state?.task && (alive ? state.task : `last: ${state.task}`)}</span>
        <span ref={seqRef} className="muted mono small" />
      </div>
      <div className="drone-body">
        <div className="drone-cam">
          <img ref={imgRef} alt={`${id} camera`} />
          {alive &&
            detection?.boxes.map((b, n) => (
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
          {!alive && <div className="signal-lost">SIGNAL LOST</div>}
        </div>
        <div className="drone-side">
          <LidarPolar lidar={alive ? (state?.lidar ?? []) : []} colour={colour} />
          <div className="temp" title={`${temp.toFixed(0)} °C`}>
            <div className="temp-bar">
              <div
                className={`temp-fill ${hot ? 'hot' : ''}`}
                style={{ height: `${Math.min(100, (temp / TEMP_MAX) * 100)}%` }}
              />
              <div className="temp-limit" style={{ bottom: `${(tempLimit / TEMP_MAX) * 100}%` }} />
            </div>
            <span className={`mono ${hot ? 'bad' : ''}`}>{temp.toFixed(0)}°</span>
          </div>
        </div>
      </div>
      <div className="drone-foot">
        {detection ? (
          <>
            <span className={`chip ${STATUS_CLASS[detection.status] ?? ''}`}>{detection.status.toUpperCase()}</span>
            <span className="gemma-caption">
              {g?.pending ? 'Gemma checking…' : (g?.caption ?? 'no Gemma result')}
              {g?.conf != null && !g.pending && <span className="muted"> · {g.conf.toFixed(2)}</span>}
            </span>
          </>
        ) : (
          <span className="muted">no detections</span>
        )}
      </div>
    </div>
  )
})
