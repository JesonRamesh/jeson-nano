// Smoke diffusion: a 2-D concentration field over the floor, solved every frame.
//
//   dC/dt = D ∇²C  +  S(fires)  -  k·C  +  r·(ambient - C)
//
// on a 0.25 m grid of grid.png's free cells. Walls are no-flux boundaries (smoke only
// exchanges with free neighbours), so it fills the hall around the fires, creeps down
// corridors and pools in dead ends, and never leaks through a wall. `ambient` keeps a thin
// haze over the whole floor. Rendered as a few stacked, wispy translucent layers (thinner
// higher up), dark brown-black like real smoke. Explicit Euler with sub-steps (stable for
// D·dt/h² < 0.25).
/* oxlint-disable react/immutability -- a physics field: its buffers and uniforms are mutated every frame by design */
import { useFrame } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import type { Fire } from '../../contracts'
import type { FreeGrid } from '../../state/grid'

const H = 0.25 // field cell (m)
const D = 0.35 // diffusion coefficient (m²/s)
const DECAY = 0.012 // loss (vents, settling) per s
const RELAX = 0.02 // pull toward the ambient haze per s
const SOURCE = 1.6 // per s, inside a fire's radius
const LAYERS = [0.12, 0.45, 0.85, 1.35, 1.9] // m above the floor

const VERT = /* glsl */ `
  varying vec2 vUv;
  varying vec2 vXY;
  void main() {
    vUv = uv;
    vec4 w = modelMatrix * vec4(position, 1.0);
    vXY = position.xy;
    gl_Position = projectionMatrix * viewMatrix * w;
  }
`
const FRAG = /* glsl */ `
  uniform sampler2D uC;
  uniform float uTime, uOpacity, uLayer;
  varying vec2 vUv;
  varying vec2 vXY;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
  }
  void main() {
    float c = texture2D(uC, vUv).r;
    if (c < 0.003) discard;
    // wisps: two octaves of drifting noise, different per layer
    vec2 q = vXY * 0.55 + vec2(uTime * 0.04, uTime * 0.025) + uLayer * 7.3;
    float n = 0.6 * noise(q) + 0.4 * noise(q * 2.3 - uTime * 0.06);
    float dens = c * mix(0.45, 1.35, n) * (1.0 - 0.6 * uLayer);
    float a = clamp(dens, 0.0, 1.0) * uOpacity;
    vec3 col = mix(vec3(0.20, 0.15, 0.11), vec3(0.08, 0.06, 0.05), clamp(c * 1.5, 0.0, 1.0)); // denser = darker
    gl_FragColor = vec4(col, a);
  }
`

export function SmokeField({
  grid,
  fires,
  ambient = 0.36,
  opacity = 0.44,
}: {
  grid: FreeGrid
  fires: Fire[]
  ambient?: number
  opacity?: number
}) {
  // field geometry: free cells + their free 4-neighbours (CSR lists)
  const field = useMemo(() => {
    const W = Math.ceil((grid.w * grid.res) / H)
    const Hh = Math.ceil((grid.h * grid.res) / H)
    const free = new Uint8Array(W * Hh)
    for (let j = 0; j < Hh; j++)
      for (let i = 0; i < W; i++) {
        const gx = Math.min(grid.w - 1, Math.floor(((i + 0.5) * H) / grid.res))
        const gy = Math.min(grid.h - 1, Math.floor(((j + 0.5) * H) / grid.res))
        free[j * W + i] = grid.free[gy * grid.w + gx]
      }
    const cells: number[] = []
    for (let k = 0; k < free.length; k++) if (free[k]) cells.push(k)
    const start = new Int32Array(cells.length + 1)
    const nbr: number[] = []
    cells.forEach((k, ci) => {
      start[ci] = nbr.length
      const i = k % W
      const j = (k - i) / W
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const ii = i + di
        const jj = j + dj
        if (ii >= 0 && jj >= 0 && ii < W && jj < Hh && free[jj * W + ii]) nbr.push(jj * W + ii)
      }
    })
    start[cells.length] = nbr.length
    const C = new Float32Array(W * Hh)
    for (const k of cells) C[k] = ambient
    const tex = new THREE.DataTexture(new Uint8Array(W * Hh), W, Hh, THREE.RedFormat, THREE.UnsignedByteType)
    tex.magFilter = THREE.LinearFilter
    tex.minFilter = THREE.LinearFilter
    tex.needsUpdate = true
    return { W, Hh, cells: Int32Array.from(cells), start, nbr: Int32Array.from(nbr), C, N: new Float32Array(W * Hh), tex }
  }, [grid, ambient])

  // fire sources: cells within the fire's radius (+0.4 m)
  const src = useMemo(() => {
    const S = new Float32Array(field.W * field.Hh)
    for (const f of fires) {
      const r = (f.size ?? 1.5) / 2 + 0.4
      const i0 = Math.floor((f.x - r) / H)
      const i1 = Math.ceil((f.x + r) / H)
      const j0 = Math.floor((f.y - r) / H)
      const j1 = Math.ceil((f.y + r) / H)
      for (let j = Math.max(0, j0); j <= Math.min(field.Hh - 1, j1); j++)
        for (let i = Math.max(0, i0); i <= Math.min(field.W - 1, i1); i++) {
          const d = Math.hypot((i + 0.5) * H - f.x, (j + 0.5) * H - f.y)
          if (d <= r) S[j * field.W + i] += SOURCE * (1 - d / (r + 0.01))
        }
    }
    return S
  }, [fires, field])

  const mats = useMemo(
    () =>
      LAYERS.map(
        (_, li) =>
          new THREE.ShaderMaterial({
            vertexShader: VERT,
            fragmentShader: FRAG,
            uniforms: { uC: { value: field.tex }, uTime: { value: 0 }, uOpacity: { value: opacity }, uLayer: { value: li / (LAYERS.length - 1) } },
            transparent: true,
            depthWrite: false,
            side: THREE.DoubleSide,
          }),
      ),
    [field, opacity],
  )
  useEffect(() => () => mats.forEach((m) => m.dispose()), [mats])

  useFrame(({ clock, camera }, dtFrame) => {
    const { cells, start, nbr } = field
    let { C, N } = field
    const dtMax = (0.22 * H * H) / D // stability
    let left = Math.min(0.1, dtFrame) * 1.0
    while (left > 1e-4) {
      const dt = Math.min(dtMax, left)
      left -= dt
      for (let ci = 0; ci < cells.length; ci++) {
        const k = cells[ci]
        const c = C[k]
        let lap = 0
        for (let q = start[ci]; q < start[ci + 1]; q++) lap += C[nbr[q]] - c
        const v = c + dt * ((D * lap) / (H * H) + src[k] - DECAY * c + RELAX * (ambient - c))
        N[k] = v < 0 ? 0 : v > 1 ? 1 : v
      }
      const t = C
      C = N
      N = t
    }
    field.C = C
    field.N = N
    const px = field.tex.image.data as Uint8Array
    for (let ci = 0; ci < cells.length; ci++) px[cells[ci]] = Math.min(255, C[cells[ci]] * 255)
    field.tex.needsUpdate = true
    const fade = Math.max(0.35, Math.min(1, 6 / Math.max(6, camera.position.y))) // thinner from the overview
    for (const m of mats) {
      m.uniforms.uTime.value = clock.elapsedTime
      m.uniforms.uOpacity.value = opacity * fade
    }
  })

  const w = field.W * H
  const h = field.Hh * H
  return (
    <group>
      {LAYERS.map((z, li) => (
        <mesh key={li} position={[w / 2, h / 2, z]} material={mats[li]} renderOrder={1}>
          <planeGeometry args={[w, h]} />
        </mesh>
      ))}
    </group>
  )
}
