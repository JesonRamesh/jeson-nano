// Rubble: a procedural pile per 1×1 m debris block (concrete chunks, slabs, a leaning
// beam, rebar) merged into one flat-shaded, vertex-coloured mesh, plus a dust patch.
import { Sparkles } from '@react-three/drei'
import { useMemo } from 'react'
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { rng } from './hazards'
import { smokePuff } from './textures'

const CONCRETE = new THREE.Color('#a29e97')
const CONCRETE_LIGHT = new THREE.Color('#bdb8b0')
const BRICK = new THREE.Color('#9a5f4c')
const WOOD = new THREE.Color('#8a6442')
const STEEL = new THREE.Color('#6d7178')
const RUST = new THREE.Color('#8a4c2c')

const LIGHT = new THREE.Vector3(0.35, 0.25, 0.9).normalize()

function prep(g: THREE.BufferGeometry, color: THREE.Color, rand: () => number, m: THREE.Matrix4) {
  let geo = g.index ? g.toNonIndexed() : g
  geo.deleteAttribute('uv')
  geo.applyMatrix4(m)
  geo.computeVertexNormals()
  const n = geo.attributes.position.count
  const col = new Float32Array(n * 3)
  const c = color.clone().offsetHSL(0, 0, (rand() - 0.5) * 0.12)
  const nrm = geo.attributes.normal as THREE.BufferAttribute
  for (let i = 0; i < n; i += 3) {
    // baked shading (the dashboard scan is unlit, so the rubble is too): faces toward a
    // soft overhead light are brighter, plus per-face variation
    const lit = Math.max(0, nrm.getX(i) * LIGHT.x + nrm.getY(i) * LIGHT.y + nrm.getZ(i) * LIGHT.z)
    const k = (0.42 + 0.58 * lit) * (0.9 + rand() * 0.2)
    for (let v = 0; v < 3; v++) {
      col[(i + v) * 3] = c.r * k
      col[(i + v) * 3 + 1] = c.g * k
      col[(i + v) * 3 + 2] = c.b * k
    }
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3))
  return geo
}

function chunk(rand: () => number, r: number) {
  const g = new THREE.IcosahedronGeometry(r, 1)
  const p = g.attributes.position as THREE.BufferAttribute
  for (let i = 0; i < p.count; i++) p.setXYZ(i, p.getX(i) * (0.85 + rand() * 0.3), p.getY(i) * (0.85 + rand() * 0.3), p.getZ(i) * (0.5 + rand() * 0.25))
  return g
}

const mtx = (x: number, y: number, z: number, rx: number, ry: number, rz: number) =>
  new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz)), new THREE.Vector3(1, 1, 1))

const geoCache = new Map<number, THREE.BufferGeometry>()

/** Height of the pile's core mound at distance d from its centre. */
const MOUND_R = 0.62
const MOUND_H = 0.62
const moundZ = (d: number) => MOUND_H * Math.sqrt(Math.max(0, 1 - (d / MOUND_R) ** 2))

function mound(rand: () => number) {
  const g = new THREE.SphereGeometry(MOUND_R, 14, 7, 0, Math.PI * 2, 0, Math.PI / 2)
  g.rotateX(Math.PI / 2) // dome up +z
  g.scale(1, 1, MOUND_H / MOUND_R)
  const p = g.attributes.position as THREE.BufferAttribute
  for (let i = 0; i < p.count; i++) {
    const z = p.getZ(i)
    const k = 0.85 + rand() * 0.3
    p.setXYZ(i, p.getX(i) * k, p.getY(i) * k, z > 0.01 ? z * (0.8 + rand() * 0.35) : 0)
  }
  return g
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
  for (let i = 0; i < 16; i++) {
    const r = 0.08 + rand() * 0.16
    const a = rand() * Math.PI * 2
    const d = Math.sqrt(rand()) * (MOUND_R + 0.15)
    const z = moundZ(d) + r * 0.25
    const col = rand() < 0.08 ? BRICK : rand() < 0.55 ? CONCRETE_LIGHT : CONCRETE
    parts.push(prep(chunk(rand, r), col, rand, mtx(Math.cos(a) * d, Math.sin(a) * d, z, rand() * 3, rand() * 3, rand() * 3)))
  }
  // broken slabs lying on the flank, tilted to follow the slope
  for (let i = 0; i < 3; i++) {
    const a = (i / 3) * Math.PI * 2 + rand()
    const d = 0.3 + rand() * 0.15
    const g = new THREE.BoxGeometry(0.42 + rand() * 0.3, 0.3 + rand() * 0.2, 0.06)
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
      <mesh position={[0, 0, 0.01]} renderOrder={1}>
        <circleGeometry args={[0.95, 24]} />
        <meshBasicMaterial map={smokePuff()} color="#7b746a" transparent opacity={0.9} depthWrite={false} polygonOffset polygonOffsetFactor={-1} />
      </mesh>
      {dust && <Sparkles count={10} scale={[1.2, 1.2, 1.4]} position={[0, 0, 0.8]} size={2} speed={0.25} color="#b8b0a2" opacity={0.5} />}
    </group>
  )
}
