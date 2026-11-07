import { useEffect, useState } from 'react'

/** grid.png decoded once: free[iy * w + ix] = 1, row iy = 0 at world y = 0 (bottom). */
export interface FreeGrid {
  w: number
  h: number
  res: number // metres per pixel
  free: Uint8Array
}

const cache = new Map<string, Promise<FreeGrid>>()

function load(url: string, res: number): Promise<FreeGrid> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      const c = document.createElement('canvas')
      c.width = img.width
      c.height = img.height
      const ctx = c.getContext('2d')!
      ctx.drawImage(img, 0, 0)
      const px = ctx.getImageData(0, 0, c.width, c.height).data
      const free = new Uint8Array(c.width * c.height)
      for (let row = 0; row < c.height; row++) {
        const iy = c.height - 1 - row // image rows are flipped
        for (let ix = 0; ix < c.width; ix++) free[iy * c.width + ix] = px[(row * c.width + ix) * 4] > 127 ? 1 : 0
      }
      resolve({ w: c.width, h: c.height, res, free })
    }
    img.onerror = () => reject(new Error(`could not load ${url}`))
    img.src = url
  })
}

export function useFreeGrid(url: string | undefined, res: number | undefined): FreeGrid | null {
  const [grid, setGrid] = useState<FreeGrid | null>(null)
  useEffect(() => {
    if (!url || !res) return
    let alive = true
    const key = `${url}@${res}`
    if (!cache.has(key)) cache.set(key, load(url, res))
    cache.get(key)!.then((g) => alive && setGrid(g)).catch(() => cache.delete(key))
    return () => {
      alive = false
    }
  }, [url, res])
  return grid
}

export function isFree(g: FreeGrid, x: number, y: number): boolean {
  const ix = Math.floor(x / g.res)
  const iy = Math.floor(y / g.res)
  return ix >= 0 && iy >= 0 && ix < g.w && iy < g.h && g.free[iy * g.w + ix] === 1
}
