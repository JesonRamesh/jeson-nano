// World <-> canvas mapping for the 2D belief map. World: metres, x east, y north.

export interface MapView {
  scale: number // canvas px per metre
  ox: number // canvas x of world x = 0
  oy: number // canvas y of world y = 0 (canvas y grows downward)
}

/** Fit the map into the canvas, leaving `padTop` / `padBottom` px for toolbars. */
export function fitView(
  worldW: number,
  worldH: number,
  canvasW: number,
  canvasH: number,
  pad = 12,
  padTop = pad,
  padBottom = pad,
): MapView {
  const availH = canvasH - padTop - padBottom
  const scale = Math.max(0.01, Math.min((canvasW - 2 * pad) / worldW, availH / worldH))
  const ox = (canvasW - worldW * scale) / 2
  const oy = padTop + (availH + worldH * scale) / 2
  return { scale, ox, oy }
}

export function toCanvas(v: MapView, x: number, y: number): [number, number] {
  return [v.ox + x * v.scale, v.oy - y * v.scale]
}

/** Canvas px -> world metres (for the click tool). */
export function toWorld(v: MapView, cx: number, cy: number): [number, number] {
  return [(cx - v.ox) / v.scale, (v.oy - cy) / v.scale]
}
