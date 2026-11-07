import { forwardRef, useImperativeHandle, useRef } from 'react'
import type { DroneState } from '../contracts'

export interface DronePanelHandle {
  showFrame: (jpeg: Blob, seq: number) => void
}

interface Props {
  id: string
  color: string
  state?: DroneState
}

/**
 * One drone camera panel. Frames bypass React state (≈10 fps × 4 drones):
 * the parent calls showFrame() and we swap the <img> src directly.
 */
export const DronePanel = forwardRef<DronePanelHandle, Props>(function DronePanel({ id, color, state }, ref) {
  const imgRef = useRef<HTMLImageElement>(null)
  const seqRef = useRef<HTMLSpanElement>(null)
  const urlRef = useRef<string | null>(null)

  useImperativeHandle(ref, () => ({
    showFrame(jpeg, seq) {
      const url = URL.createObjectURL(jpeg)
      const img = imgRef.current
      if (img) img.src = url
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
      urlRef.current = url
      if (seqRef.current) seqRef.current.textContent = `#${seq}`
    },
  }))

  const alive = state?.alive ?? true
  return (
    <div className="drone-panel" style={{ borderColor: color }}>
      <div className="drone-head">
        <span style={{ color, fontWeight: 700 }}>{id}</span>
        <span className="muted">{state ? `${state.status} · ${state.temp.toFixed(0)} °C` : '—'}</span>
        <span ref={seqRef} className="muted mono" />
      </div>
      <div className="drone-cam">
        <img ref={imgRef} alt={`${id} camera`} />
        {!alive && <div className="signal-lost">SIGNAL LOST</div>}
      </div>
    </div>
  )
})
