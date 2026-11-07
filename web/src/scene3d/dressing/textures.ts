// Small generated textures (no files, so the demo works offline).
import * as THREE from 'three'
import { rng } from './hazards'

const cache = new Map<string, THREE.Texture>()

function canvasTex(key: string, size: number, draw: (g: CanvasRenderingContext2D, s: number) => void) {
  const hit = cache.get(key)
  if (hit) return hit
  const c = document.createElement('canvas')
  c.width = c.height = size
  draw(c.getContext('2d')!, size)
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  cache.set(key, t)
  return t
}

/** Soft white dot: flames, embers. */
export const softDot = () =>
  canvasTex('dot', 64, (g, s) => {
    const r = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2)
    r.addColorStop(0, 'rgba(255,255,255,1)')
    r.addColorStop(0.35, 'rgba(255,255,255,0.6)')
    r.addColorStop(1, 'rgba(255,255,255,0)')
    g.fillStyle = r
    g.fillRect(0, 0, s, s)
  })

/** Lumpy smoke puff: a few overlapping soft blobs. */
export const smokePuff = () =>
  canvasTex('smoke', 128, (g, s) => {
    const rand = rng(7)
    for (let i = 0; i < 14; i++) {
      const x = s / 2 + (rand() - 0.5) * s * 0.45
      const y = s / 2 + (rand() - 0.5) * s * 0.45
      const rad = s * (0.18 + rand() * 0.2)
      const r = g.createRadialGradient(x, y, 0, x, y, rad)
      r.addColorStop(0, 'rgba(255,255,255,0.35)')
      r.addColorStop(1, 'rgba(255,255,255,0)')
      g.fillStyle = r
      g.fillRect(0, 0, s, s)
    }
  })

/** Scorch mark: dark, irregular, soft-edged. */
export const scorch = () =>
  canvasTex('scorch', 256, (g, s) => {
    const rand = rng(11)
    for (let i = 0; i < 40; i++) {
      const a = rand() * Math.PI * 2
      const d = rand() * s * 0.22
      const x = s / 2 + Math.cos(a) * d
      const y = s / 2 + Math.sin(a) * d
      const rad = s * (0.12 + rand() * 0.18)
      const r = g.createRadialGradient(x, y, 0, x, y, rad)
      r.addColorStop(0, 'rgba(8,6,5,0.3)')
      r.addColorStop(0.6, 'rgba(20,14,10,0.12)')
      r.addColorStop(1, 'rgba(20,14,10,0)')
      g.fillStyle = r
      g.fillRect(0, 0, s, s)
    }
  })
