// Rubble: a procedural pile per 1×1 m debris block (concrete chunks, slabs, a leaning
// beam, rebar) merged into one flat-shaded, vertex-coloured mesh, plus a dust patch.
import { Sparkles } from '@react-three/drei'
import { useMemo } from 'react'
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { rng } from './hazards'
import { glow, smokePuff } from './textures'

const CONCRETE = new THREE.Color('#857060') // brown shades: dusty earth / mud brick
const CONCRETE_LIGHT = new THREE.Color('#9d8670')
const BRICK = new THREE.Color('#8f4f35')
const WOOD = new THREE.Color('#6e4a2c')
const STEEL = new THREE.Color('#5c4a3c')
const RUST = new THREE.Color('#7a3f22')

const LIGHT = new THREE.Vector3(0.35, 0.25, 0.9).normalize()

function hash3(x: number, y: number, z: number) {
  const h = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453
  return h - Math.floor(h)
}
/** Smooth 3-D value noise in [0, 1]. */
function vnoise(x: number, y: number, z: number) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z)
  const xf = x - xi, yf = y - yi, zf = z - zi
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf), w = zf * zf * (3 - 2 * zf)
  const l = (a: number, b: number, t: number) => a + (b - a) * t
  const c = (dx: number, dy: number, dz: number) => hash3(xi + dx, yi + dy, zi + dz)
  return l(l(l(c(0, 0, 0), c(1, 0, 0), u), l(c(0, 1, 0), c(1, 1, 0), u), v),
    l(l(c(0, 0, 1), c(1, 0, 1), u), l(c(0, 1, 1), c(1, 1, 1), u), v), w)
}

function prep(g: THREE.BufferGeometry, color: THREE.Color, rand: () => number, m: THREE.Matrix4) {
  if (g.index) g.computeVertexNormals() // smooth rocks; boxes keep their split normals
  const geo = g.index ? g.toNonIndexed() : g
  geo.deleteAttribute('uv')
  geo.applyMatrix4(m)
  if (!g.index) geo.computeVertexNormals()
  const pos = geo.attributes.position as THREE.BufferAttribute
  const nrm = geo.attributes.normal as THREE.BufferAttribute
  const n = pos.count
  const col = new Float32Array(n * 3)
  const c = color.clone().offsetHSL((rand() - 0.5) * 0.02, (rand() - 0.5) * 0.06, (rand() - 0.5) * 0.1)
  for (let i = 0; i < n; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i)
    // baked light: soft key from above + sky fill; the dashboard scan is unlit, so is this
    const lit = Math.max(0, nrm.getX(i) * LIGHT.x + nrm.getY(i) * LIGHT.y + nrm.getZ(i) * LIGHT.z)
    const sky = 0.5 + 0.5 * nrm.getZ(i)
    const ao = 0.5 + 0.5 * Math.min(1, Math.max(0, z / 0.45)) // crevices near the floor are darker
    const mott = 0.78 + 0.32 * vnoise(x * 9, y * 9, z * 9) + 0.12 * (vnoise(x * 31, y * 31, z * 31) - 0.5) // dirt
    const k = (0.32 + 0.48 * lit + 0.2 * sky) * ao * mott
    col[i * 3] = c.r * k
    col[i * 3 + 1] = c.g * k
    col[i * 3 + 2] = c.b * k
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3))
  return geo
}

/** A broken-concrete chunk: low-detail, noise-displaced and squashed, FLAT shaded so it
 *  keeps sharp fracture facets (smooth rocks read as toys). */
function chunk(rand: () => number, r: number) {
  const g = new THREE.IcosahedronGeometry(r, 1)
  const p = g.attributes.position as THREE.BufferAttribute
  const o = rand() * 100
  const sx = 0.75 + rand() * 0.55, sy = 0.7 + rand() * 0.5, sz = 0.4 + rand() * 0.35
  // displace per unique position so shared corners stay welded (no cracks)
  const cache = new Map<string, number>()
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i)
    const key = `${x.toFixed(5)},${y.toFixed(5)},${z.toFixed(5)}`
    let d = cache.get(key)
    if (d === undefined) {
      d = 0.62 + 0.6 * vnoise(x / r * 1.4 + o, y / r * 1.4, z / r * 1.4) + 0.25 * (hash3(x * 97 + o, y * 89, z * 83) - 0.5)
      cache.set(key, d)
    }
    p.setXYZ(i, x * d * sx, y * d * sy, z * d * sz)
  }
  g.deleteAttribute('normal')
  return g
}

/** A jagged slab fragment: a thin box with its corners knocked about. */
function slab(rand: () => number, w: number, h: number, t: number) {
  const g = new THREE.BoxGeometry(w, h, t, 2, 2, 1)
  const p = g.attributes.position as THREE.BufferAttribute
  const cache = new Map<string, [number, number]>()
  for (let i = 0; i < p.count; i++) {
    const key = `${p.getX(i).toFixed(4)},${p.getY(i).toFixed(4)}`
    let j = cache.get(key)
    if (!j) {
      j = [(rand() - 0.5) * w * 0.35, (rand() - 0.5) * h * 0.35]
      cache.set(key, j)
    }
    p.setXYZ(i, p.getX(i) + j[0], p.getY(i) + j[1], p.getZ(i) + (rand() - 0.5) * t * 0.4)
  }
  return g.toNonIndexed()
}

const mtx = (x: number, y: number, z: number, rx: number, ry: number, rz: number) =>
  new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz)), new THREE.Vector3(1, 1, 1))

const geoCache = new Map<number, THREE.BufferGeometry>()

/** Height of the pile's core mound at distance d from its centre. */
const MOUND_R = 0.62
const MOUND_H = 0.62
const moundZ = (d: number) => MOUND_H * Math.sqrt(Math.max(0, 1 - (d / MOUND_R) ** 2))

function mound(rand: () => number) {
  const g = new THREE.SphereGeometry(MOUND_R, 11, 5, 0, Math.PI * 2, 0, Math.PI / 2)
  g.rotateX(Math.PI / 2) // dome up +z
  g.scale(1, 1, MOUND_H / MOUND_R)
  const p = g.attributes.position as THREE.BufferAttribute
  for (let i = 0; i < p.count; i++) {
    const z = p.getZ(i)
    const k = 0.85 + rand() * 0.3
    p.setXYZ(i, p.getX(i) * k, p.getY(i) * k, z > 0.01 ? z * (0.8 + rand() * 0.35) : 0)
  }
  g.deleteAttribute('normal')
  return g.toNonIndexed() // faceted, like packed debris (not a smooth dome)
}

function between(p0: THREE.Vector3, p1: THREE.Vector3) {
  // geometry built along +x and centred: place it from p0 to p1
  const dir = p1.clone().sub(p0)
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(1, 0, 0), dir.clone().normalize())
  return new THREE.Matrix4().compose(p0.clone().add(p1).multiplyScalar(0.5), q, new THREE.Vector3(1, 1, 1))
}

/** One pile: a dusty mound ~1.2 m across and ~0.75 m high with chunks and slabs resting on
 *  it, a beam leaning from the floor onto the top, and rebar poking out. */
function rubbleGeometry(seed: number): THREE.BufferGeometry {
  const hit = geoCache.get(seed)
  if (hit) return hit
  const rand = rng(seed)
  const parts: THREE.BufferGeometry[] = []
  parts.push(prep(mound(rand), CONCRETE, rand, new THREE.Matrix4()))
  // chunks sitting on the mound surface (half-buried)
  for (let i = 0; i < 26; i++) {
    const r = 0.06 + rand() * rand() * 0.2
    const a = rand() * Math.PI * 2
    const d = Math.sqrt(rand()) * (MOUND_R + 0.15)
    const z = moundZ(d) + r * 0.25
    const col = rand() < 0.08 ? BRICK : rand() < 0.55 ? CONCRETE_LIGHT : CONCRETE
    parts.push(prep(chunk(rand, r), col, rand, mtx(Math.cos(a) * d, Math.sin(a) * d, z, rand() * 3, rand() * 3, rand() * 3)))
  }
  // broken slabs lying on the flank, tilted to follow the slope
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2 + rand()
    const d = 0.3 + rand() * 0.15
    const g = slab(rand, 0.42 + rand() * 0.3, 0.3 + rand() * 0.2, 0.06)
    const col = i === 0 && rand() < 0.3 ? BRICK : CONCRETE_LIGHT
    const slope = Math.atan2(moundZ(d - 0.05) - moundZ(d + 0.05), 0.1)
    const m = new THREE.Matrix4().makeRotationZ(a).multiply(new THREE.Matrix4().makeRotationY(-slope))
    m.setPosition(Math.cos(a) * d, Math.sin(a) * d, moundZ(d) + 0.04)
    parts.push(prep(g, col, rand, m))
  }
  // a beam from the floor up onto the top of the pile
  {
    const a = rand() * Math.PI * 2
    const p0 = new THREE.Vector3(Math.cos(a) * 0.95, Math.sin(a) * 0.95, 0.06)
    const p1 = new THREE.Vector3(-Math.cos(a) * 0.2, -Math.sin(a) * 0.2, MOUND_H + 0.35 + rand() * 0.3)
    const g = new THREE.BoxGeometry(p0.distanceTo(p1) + 0.2, 0.11, 0.11)
    parts.push(prep(g, rand() < 0.5 ? WOOD : STEEL, rand, between(p0, p1)))
  }
  // rebar poking out of the mound
  for (let i = 0; i < 5; i++) {
    const a = rand() * Math.PI * 2
    const d = rand() * 0.45
    const len = 0.35 + rand() * 0.45
    const base = new THREE.Vector3(Math.cos(a) * d, Math.sin(a) * d, moundZ(d) - 0.05)
    const tip = base.clone().add(new THREE.Vector3(Math.cos(a) * len * 0.6, Math.sin(a) * len * 0.6, len * (0.5 + rand() * 0.5)))
    const g = new THREE.CylinderGeometry(0.011, 0.011, 1, 5).rotateZ(Math.PI / 2)
    g.scale(base.distanceTo(tip), 1, 1)
    parts.push(prep(g, RUST, rand, between(base, tip)))
  }
  // gravel spilling out around the pile, thinning with distance
  for (let i = 0; i < 70; i++) {
    const r = 0.018 + rand() * rand() * 0.05
    const a = rand() * Math.PI * 2
    const d = MOUND_R * (0.75 + Math.pow(rand(), 1.6) * 1.0)
    const g = new THREE.IcosahedronGeometry(r, 0)
    g.deleteAttribute('normal')
    const col = rand() < 0.1 ? BRICK : rand() < 0.5 ? CONCRETE_LIGHT : CONCRETE
    parts.push(prep(g, col, rand, mtx(Math.cos(a) * d, Math.sin(a) * d, r * 0.35, rand() * 3, rand() * 3, rand() * 3)))
  }
  const merged = mergeGeometries(parts, false)!
  merged.computeBoundingSphere()
  geoCache.set(seed, merged)
  return merged
}

const rubbleMat = new THREE.MeshBasicMaterial({ vertexColors: true }) // unlit, shading baked into the colours

export function RubblePile({ x, y, yaw = 0, seed = 1, dust = true }: { x: number; y: number; yaw?: number; seed?: number; dust?: boolean }) {
  const geo = useMemo(() => rubbleGeometry(seed), [seed])
  return (
    <group position={[x, y, 0]} rotation={[0, 0, yaw + seed * 0.7]}>
      <mesh geometry={geo} material={rubbleMat} castShadow={false} receiveShadow={false} />
      <mesh position={[0, 0, 0.008]} renderOrder={1}>
        <circleGeometry args={[1.0, 32]} />
        <meshBasicMaterial map={glow()} color="#000000" transparent opacity={0.5} depthWrite={false} polygonOffset polygonOffsetFactor={-1} />
      </mesh>
      <mesh position={[0, 0, 0.01]} renderOrder={1}>
        <circleGeometry args={[1.25, 24]} />
        <meshBasicMaterial map={smokePuff()} color="#7a6048" transparent opacity={0.9} depthWrite={false} polygonOffset polygonOffsetFactor={-1} />
      </mesh>
      {dust && <Sparkles count={10} scale={[1.2, 1.2, 1.4]} position={[0, 0, 0.8]} size={2} speed={0.25} color="#b39a80" opacity={0.5} />}
    </group>
  )
}
