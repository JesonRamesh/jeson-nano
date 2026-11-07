// The simulation's objects, drawn exactly as the MuJoCo drone cameras see them.
// simAssets.json is generated from sgr/world/shapes.py (`uv run python -m sgr.world.asset_spec`);
// tests/test_asset_spec.py fails if it drifts. Sim frame (z up), MuJoCo size conventions:
// ellipsoid = radii, sphere = [r], cylinder = [r, half-height] along z, box = half-extents,
// capsule = fromto + [r].
import { useFrame } from '@react-three/fiber'
import { useMemo } from 'react'
import * as THREE from 'three'
import spec from './simAssets.json'

export interface SimPart {
  type: string
  size: number[]
  rgba: number[]
  emission: number
  pos?: number[]
  fromto?: number[]
}

export const SIM_ASSETS = spec as unknown as {
  fire: { parts: SimPart[] }
  debris: { parts: SimPart[] }
  survivor: { parts: SimPart[] }
  drone: { model: string; scale: number }
}

const UP = new THREE.Vector3(0, 1, 0)

function partTransform(p: SimPart): { position: THREE.Vector3; quaternion: THREE.Quaternion; scale: THREE.Vector3 } {
  const q = new THREE.Quaternion()
  const s = new THREE.Vector3(1, 1, 1)
  if (p.type === 'capsule' && p.fromto) {
    const a = new THREE.Vector3(p.fromto[0], p.fromto[1], p.fromto[2])
    const b = new THREE.Vector3(p.fromto[3], p.fromto[4], p.fromto[5])
    q.setFromUnitVectors(UP, b.clone().sub(a).normalize())
    return { position: a.add(b).multiplyScalar(0.5), quaternion: q, scale: s }
  }
  const pos = p.pos ?? [0, 0, 0]
  if (p.type === 'cylinder') q.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2) // three: y axis -> sim: z axis
  if (p.type === 'ellipsoid') s.set(p.size[0], p.size[1], p.size[2])
  return { position: new THREE.Vector3(pos[0], pos[1], pos[2]), quaternion: q, scale: s }
}

function partGeometry(p: SimPart): THREE.BufferGeometry {
  switch (p.type) {
    case 'ellipsoid':
      return new THREE.SphereGeometry(1, 20, 14)
    case 'sphere':
      return new THREE.SphereGeometry(p.size[0], 18, 12)
    case 'cylinder':
      return new THREE.CylinderGeometry(p.size[0], p.size[0], 2 * p.size[1], 24)
    case 'box':
      return new THREE.BoxGeometry(2 * p.size[0], 2 * p.size[1], 2 * p.size[2])
    case 'capsule': {
      const [x1, y1, z1, x2, y2, z2] = p.fromto ?? [0, 0, 0, 0, 0, 0]
      return new THREE.CapsuleGeometry(p.size[0], Math.hypot(x2 - x1, y2 - y1, z2 - z1), 6, 14)
    }
    default:
      return new THREE.SphereGeometry(0.05)
  }
}

function partMaterial(p: SimPart, opacity = 1): THREE.MeshStandardMaterial {
  const [r, g, b, a] = p.rgba
  const colour = new THREE.Color(r, g, b)
  const alpha = a * opacity
  return new THREE.MeshStandardMaterial({
    color: colour,
    emissive: p.emission > 0 ? colour : new THREE.Color(0, 0, 0),
    emissiveIntensity: p.emission * 1.6, // MuJoCo's emission makes these near-self-lit
    roughness: 0.9,
    transparent: alpha < 1,
    opacity: alpha,
    depthWrite: alpha >= 1,
  })
}

/** All parts of one object. `flicker` modulates the emissive parts (fire). */
export function SimParts({
  parts,
  flicker = false,
  opacity = 1,
  phase = 0,
}: {
  parts: SimPart[]
  flicker?: boolean
  opacity?: number
  phase?: number // flicker phase, so neighbouring fires don't pulse in sync
}) {
  const items = useMemo(
    () => parts.map((p) => ({ p, geo: partGeometry(p), mat: partMaterial(p, opacity), ...partTransform(p) })),
    [parts, opacity],
  )
  useFrame(({ clock }) => {
    if (!flicker) return
    const t = clock.elapsedTime + phase
    items.forEach(({ p, mat }, i) => {
      if (p.emission > 0) mat.emissiveIntensity = p.emission * 1.6 * (0.8 + 0.25 * Math.sin(t * 9 + i * 1.7))
    })
  })
  return (
    <>
      {items.map(({ geo, mat, position, quaternion, scale }, i) => (
        <mesh key={i} geometry={geo} material={mat} position={position} quaternion={quaternion} scale={scale} />
      ))}
    </>
  )
}

/** The sim's fire (flame tongues, embers, smoke), standing on the floor at (x, y). */
export function SimFire({ x, y }: { x: number; y: number }) {
  return (
    <group position={[x, y, 0]}>
      <SimParts parts={SIM_ASSETS.fire.parts} flicker phase={x * 12.9898 + y * 78.233} />
    </group>
  )
}

/** The sim's debris block (the one LiDAR and the drones hit), on the floor at (x, y). */
export function SimDebrisBlock({ x, y, yaw = 0 }: { x: number; y: number; yaw?: number }) {
  return (
    <group position={[x, y, 0]} rotation={[0, 0, yaw]}>
      <SimParts parts={SIM_ASSETS.debris.parts} />
    </group>
  )
}

/** The sim's survivor figure (lying down, hi-vis), at (x, y) facing yaw. */
export function SimSurvivor({ x, y, yaw = 0, opacity = 1 }: { x: number; y: number; yaw?: number; opacity?: number }) {
  return (
    <group position={[x, y, 0]} rotation={[0, 0, yaw]}>
      <SimParts parts={SIM_ASSETS.survivor.parts} opacity={opacity} />
    </group>
  )
}
