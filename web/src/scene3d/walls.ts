import type { FreeGrid } from '../state/grid'

/** Axis-aligned wall rectangle in world metres (sim frame). */
export interface WallRect {
  x: number // centre
  y: number
  w: number // size along x
  h: number // size along y
}

/**
 * Greedy merge of wall cells into rectangles (same idea as P's MuJoCo walls):
 * take each unvisited wall cell, grow right as far as possible, then grow up
 * while the whole row span is wall. Everything outside the building is "wall"
 * in grid.png, so with `trimOuter` only a 0.3 m shell next to free space is kept
 * (thin walls, like the real building) instead of one solid slab.
 */
export function mergeWalls(g: FreeGrid, trimOuter = true): WallRect[] {
  const { w, h, res, free } = g
  const used = new Uint8Array(w * h)
  const isWall = (ix: number, iy: number) => free[iy * w + ix] === 0
  const out: WallRect[] = []

  const keep = new Uint8Array(w * h)
  for (let iy = 0; iy < h; iy++)
    for (let ix = 0; ix < w; ix++) {
      if (!isWall(ix, iy)) continue
      // Keep a 3-cell (0.3 m) thick shell around free space, drop the rest of the exterior.
      if (!trimOuter) keep[iy * w + ix] = 1
      else
        outer: for (let dy = -3; dy <= 3; dy++)
          for (let dx = -3; dx <= 3; dx++) {
            const x = ix + dx
            const y = iy + dy
            if (x >= 0 && y >= 0 && x < w && y < h && free[y * w + x] === 1) {
              keep[iy * w + ix] = 1
              break outer
            }
          }
    }

  for (let iy = 0; iy < h; iy++) {
    for (let ix = 0; ix < w; ix++) {
      const k = iy * w + ix
      if (!keep[k] || used[k]) continue
      let x1 = ix
      while (x1 + 1 < w && keep[iy * w + x1 + 1] && !used[iy * w + x1 + 1]) x1++
      let y1 = iy
      grow: while (y1 + 1 < h) {
        for (let x = ix; x <= x1; x++) {
          const kk = (y1 + 1) * w + x
          if (!keep[kk] || used[kk]) break grow
        }
        y1++
      }
      for (let y = iy; y <= y1; y++) for (let x = ix; x <= x1; x++) used[y * w + x] = 1
      out.push({
        x: ((ix + x1 + 1) / 2) * res,
        y: ((iy + y1 + 1) / 2) * res,
        w: (x1 - ix + 1) * res,
        h: (y1 - iy + 1) * res,
      })
    }
  }
  return out
}
