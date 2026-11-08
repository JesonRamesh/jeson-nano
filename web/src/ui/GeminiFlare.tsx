import { useEffect, useState } from 'react'
import { useSim } from '../state/store'

const BUDGET_FRESH_MS = 3000

/**
 * Soft, always-moving glow of the Gemini colours along the bottom edge of the 3D view:
 * Gemma is running in the background. Faint at rest, a little brighter while Gemma thinks
 * (a detection is waiting for its answer, or the perception queue is busy).
 * Pass `active` to drive it yourself (the drone tiles: only while that drone's Gemma check runs).
 */
export function GeminiFlare({ active: forced, className = '' }: { active?: boolean; className?: string } = {}) {
  const pending = useSim((s) => Object.values(s.detections).some((d) => d.gemma?.pending))
  const queue = useSim((s) => s.budget?.gemma_queue ?? 0)
  const budgetAt = useSim((s) => s.budgetAt)
  // budget arrives at 1 Hz; if it stops coming, stop claiming Gemma is busy
  const [staleAt, setStaleAt] = useState(-1)
  useEffect(() => {
    const t = setTimeout(() => setStaleAt(budgetAt), BUDGET_FRESH_MS)
    return () => clearTimeout(t)
  }, [budgetAt])
  const active = forced ?? (pending || (queue > 0 && staleAt !== budgetAt))
  return (
    <div className={`gemini-flare ${className} ${active ? 'active' : ''}`} aria-hidden>
      <div className="gf-band gf-a" />
      <div className="gf-band gf-b" />
      <div className="gf-line" />
    </div>
  )
}
