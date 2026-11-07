import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import type { DetectionMsg, DroneState } from '../contracts'
import { noEmDash } from '../ui/text'

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

/**
 * LiDAR top view: the drone at the centre, heading up, 36 rays (ray k at yaw + k·10°),
 * 6 m to the edge. Drawn at device resolution so it stays crisp on the projector.
 */
function LidarTopView({ lidar, colour }: { lidar: number[]; colour: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const dpr = window.devicePixelRatio || 1
    const css = c.clientWidth || 104
    if (c.width !== Math.round(css * dpr)) {
      c.width = Math.round(css * dpr)
      c.height = Math.round(css * dpr)
    }
    const ctx = c.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const s = css
    const mid = s / 2
    const r0 = mid - 3
    ctx.clearRect(0, 0, s, s)
    // range rings at 2 / 4 / 6 m + heading tick
    ctx.strokeStyle = 'rgba(213, 218, 224, 0.14)'
    ctx.lineWidth = 1
    for (const f of [1 / 3, 2 / 3, 1]) {
      ctx.beginPath()
      ctx.arc(mid, mid, r0 * f, 0, Math.PI * 2)
      ctx.stroke()
    }
    ctx.beginPath()
    ctx.moveTo(mid, mid - r0)
    ctx.lineTo(mid, mid - r0 + 6)
    ctx.stroke()
    if (lidar.length) {
      ctx.beginPath()
      lidar.forEach((r, k) => {
        const a = Math.PI / 2 + (k * Math.PI) / 18 // ray 0 = heading = up; CCW = left
        const d = (Math.min(r, 6) / 6) * r0
        const x = mid + d * Math.cos(a)
        const y = mid - d * Math.sin(a)
        if (k === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.closePath()
      ctx.fillStyle = 'rgba(238, 241, 244, 0.10)'
      ctx.fill()
      ctx.strokeStyle = 'rgba(238, 241, 244, 0.85)'
      ctx.lineWidth = 1.1
      ctx.stroke()
    }
    // the drone itself (its colour is the one link to the 3D view)
    ctx.fillStyle = colour
    ctx.beginPath()
    ctx.moveTo(mid, mid - 4)
    ctx.lineTo(mid + 3, mid + 3)
    ctx.lineTo(mid - 3, mid + 3)
    ctx.closePath()
    ctx.fill()
  }, [lidar, colour])
  return <canvas ref={ref} className="lidar-top" />
}

/**
 * One drone feed: the camera image fills the tile; telemetry sits on it as an overlay
 * (id, status, task, temperature, frame #, LiDAR top view, detection + Gemma caption).
 * Frames bypass React (≈10 fps × 4): the parent calls showFrame() directly.
 */
export const DronePanel = forwardRef<DronePanelHandle, Props>(function DronePanel(
  { id, colour, state, detection, tempLimit },
  ref,
) {
  const imgRef = useRef<HTMLImageElement>(null)
  const seqRef = useRef<HTMLSpanElement>(null)
  const urlRef = useRef<string | null>(null)
  // Detection boxes belong to ONE frame. They show solid while that frame is on screen,
  // and otherwise fade out within ~1 s (CSS), so they never sit on newer frames.
  const boxesRef = useRef<HTMLDivElement>(null)
  const frameIdRef = useRef('') // frame on screen: "<drone>-<seq>", same format as frame_id
  const detFrameId = detection?.frame_id ?? ''
  const syncHold = () => {
    boxesRef.current?.classList.toggle('hold', frameIdRef.current === detFrameId)
  }

  useImperativeHandle(ref, () => ({
    showFrame(jpeg, seq) {
      const url = URL.createObjectURL(jpeg)
      if (imgRef.current) imgRef.current.src = url
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
      urlRef.current = url
      if (seqRef.current) seqRef.current.textContent = `F${String(seq).padStart(5, '0')}`
      frameIdRef.current = `${id}-${seq}`
      syncHold()
    },
  }))
  useEffect(syncHold) // a new detection may be for the frame already on screen
  useEffect(
    () => () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    },
    [],
  )

  const alive = state?.alive ?? true
  const temp = state?.temp ?? 0
  const hot = temp > tempLimit
  const g = detection?.gemma
  const status = alive ? (state?.status ?? 'no data').toLowerCase() : 'offline'

  return (
    <div className={`feed ${alive ? '' : 'lost'}`} style={{ ['--drone' as string]: colour }}>
      <div className="feed-cam">
        {/* 4:3 box fitted inside the tile: image, reticle and boxes share its coordinates */}
        <div className="cam-box">
        <img ref={imgRef} alt={`${id} camera`} draggable={false} />
        <div className="reticle" />
        {alive && detection && detection.boxes.length > 0 && (
          // key: a new detection (or status change) remounts, restarting the fade
          <div ref={boxesRef} key={`${detection.frame_id}|${detection.status}`} className="boxes">
            {detection.boxes.map((b, n) => (
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
        )}
        </div>
        {!alive && (
          <div className="signal-lost">
            <span>No signal</span>
          </div>
        )}

        {/* --- telemetry overlay --- */}
        <div className="feed-top">
          <span className="feed-id">{id}</span>
          <span className="feed-status">{status}</span>
          <span className="feed-task">{state?.task && noEmDash(alive ? state.task : `last: ${state.task}`)}</span>
          <span className={`feed-temp ${hot ? 'hot' : ''}`} title="Drone temperature">
            {alive ? `${temp.toFixed(0)}°C` : '--°C'}
          </span>
        </div>
        <div className="feed-bottom">
          <div className="feed-det">
            {detection ? (
              <>
                <span className="det-tag">{detection.status.replace('_', ' ')}</span>
                <span className="det-caption">
                  {g?.pending ? 'Gemma checking' : g?.caption ? noEmDash(g.caption) : 'no Gemma result'}
                  {g?.conf != null && !g.pending && <span className="det-conf">{g.conf.toFixed(2)}</span>}
                </span>
              </>
            ) : (
              <span className="det-none">no detections</span>
            )}
            <span ref={seqRef} className="feed-seq" />
          </div>
          <LidarTopView lidar={alive ? (state?.lidar ?? []) : []} colour={colour} />
        </div>
      </div>
    </div>
  )
})
