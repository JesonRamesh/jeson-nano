import type { BeliefGrid } from './store'

// Colours for the swarm's belief layers (shared by the 3D overlay and its legend).
export const BELIEF_LEGEND: { label: string; colour: string }[] = [
  { label: 'searched', colour: '#4fc3f7' },
  { label: 'blocked', colour: '#ff4d4f' },
  { label: 'fire', colour: '#ff8a1f' },
  { label: 'danger > 50 °C', colour: '#ffb020' },
  { label: 'unsafe', colour: '#ce93d8' },
]

/** Belief layers -> canvas at belief resolution (1 px per 0.2 m cell, row 0 at the top). */
export function paintBelief(b: BeliefGrid, target: HTMLCanvasElement) {
  if (target.width !== b.w || target.height !== b.h) {
    target.width = b.w
    target.height = b.h
  }
  const ctx = target.getContext('2d')!
  const img = ctx.createImageData(b.w, b.h)
  const rgba = (p: number, r: number, g: number, bl: number, a: number) => {
    img.data[p] = r
    img.data[p + 1] = g
    img.data[p + 2] = bl
    img.data[p + 3] = a
  }
  for (let j = 0; j < b.h; j++) {
    const row = b.h - 1 - j // flip: belief j = 0 is the bottom of the map
    for (let i = 0; i < b.w; i++) {
      const k = j * b.w + i
      const p = (row * b.w + i) * 4
      switch (b.cells[k]) {
        case 3: // blocked_new: red hatch (diagonal stripes, 3 cells = 0.6 m period)
          if ((i + j) % 3 === 0) rgba(p, 255, 77, 79, 255)
          else rgba(p, 120, 30, 32, 230)
          break
        case 4: // fire
          rgba(p, 255, 138, 31, 255)
          break
        case 5: // danger
          rgba(p, 255, 176, 32, 140)
          break
        case 6: // unsafe
          rgba(p, 171, 71, 188, 140)
          break
        default:
          if (b.searched[k]) rgba(p, 79, 195, 247, 42)
      }
    }
  }
  ctx.putImageData(img, 0, 0)
}
