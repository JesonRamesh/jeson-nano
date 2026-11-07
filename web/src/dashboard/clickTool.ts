// Judge "change tool" (§10.2): drop debris or fire on the map, Undo the last one.
//
// The World does the authoritative safety check (free cell, ≥ 1 m from drones,
// firefighter and survivor, pool not empty) and silently refuses otherwise.
// There is no "refused" message in the contract, so the browser:
//   1. pre-checks what it can see (map bounds, walls, drones, firefighter) for an
//      instant toast, and
//   2. watches state.click_handles: if no new handle shows up soon, it toasts
//      that the sim refused (survivor too close, or pool empty).

import type { ClientMsg } from '../contracts'
import { sim } from '../state/store'
import { isFree, type FreeGrid } from '../state/grid'
import { toast } from '../ui/toast'

export type ClickMode = 'debris' | 'fire'

const CLEARANCE_M = 1.0
const CONFIRM_TIMEOUT_MS = 1500

export function placeClick(send: (m: ClientMsg) => void, grid: FreeGrid | null, mode: ClickMode, x: number, y: number) {
  const st = sim.snap.state
  if (grid && !isFree(grid, x, y)) {
    toast('Can’t place there: not a free floor cell', 'warn')
    return
  }
  const near = (px: number, py: number) => Math.hypot(px - x, py - y) < CLEARANCE_M
  const drone = st?.drones.find((d) => d.alive && near(d.x, d.y))
  if (drone) {
    toast(`Too close to ${drone.id} (needs 1 m)`, 'warn')
    return
  }
  if (st?.ff && near(st.ff.x, st.ff.y)) {
    toast('Too close to the firefighter (needs 1 m)', 'warn')
    return
  }

  const before = new Set((st?.click_handles ?? []).map((h) => h.id))
  send({ type: 'click', mode, x, y })

  const started = performance.now()
  const unsubscribe = sim.subscribe(() => {
    const handles = sim.snap.state?.click_handles ?? []
    if (handles.some((h) => !before.has(h.id))) {
      unsubscribe()
      clearTimeout(timer)
    }
  })
  const timer = setTimeout(() => {
    unsubscribe()
    if (performance.now() - started >= CONFIRM_TIMEOUT_MS)
      toast(`${mode === 'debris' ? 'Debris' : 'Fire'} refused by the sim (too close to the survivor, or none left)`, 'warn')
  }, CONFIRM_TIMEOUT_MS)
}

export function undoClick(send: (m: ClientMsg) => void) {
  if (!sim.snap.state?.click_handles.length) {
    toast('Nothing to undo', 'info')
    return
  }
  send({ type: 'undo' })
}
