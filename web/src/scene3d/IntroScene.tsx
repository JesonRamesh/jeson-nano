import { Canvas, useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { isFree, useFreeGrid, type FreeGrid } from '../state/grid'
import { useSim } from '../state/store'
import { Dressing } from './dressing'
import { DroneBody, Environment, SimFrame } from './Scene3D'

// The start screen behind the Start button: the firefighter's view from the entrance, the way
// Act 2 opens, looking slightly up at the four drones hovering just above and ahead of them, facing
// back at the camera, waiting to be sent in (Start = the firefighter deploying the swarm). Purely visual: nothing here talks to the sim. Sim frame (x east, y north,
// z up) inside <SimFrame>; the camera is set in three's frame: sim (x, y, z) -> three (x, z, -y).

const EYE_Z = 1.6 // m, same as the firefighter view
const FOG = new THREE.FogExp2('#1c1f23', 0.16) // smoky, but the drones read clearly
// Overhead arc: [ahead m, lateral (left +) as a fraction of the spread, height m]. Kept under the
// scan's 2.3 m ceiling cut (plus a little) so they hang just above the firefighter's head.
const SLOTS: [number, number, number][] = [
  [1.7, 1, 2.2],
  [2.2, 0.34, 2.4],
  [2.2, -0.34, 2.4],
  [1.7, -1, 2.2],
]
const PITCH = 0.2 // rad, looking up at them
const MAX_SPREAD = 0.95 // m from the centre line to the outer drones

/** Half the corridor width ahead of the entrance (so the outer drones never sit in a wall). */
function clearance(grid: FreeGrid | null, x: number, y: number, yaw: number): number {
  if (!grid) return MAX_SPREAD
  const lx = -Math.sin(yaw)
  const ly = Math.cos(yaw)
  let best = Infinity
  for (const ahead of [1.7, 2.2]) {
    const cx = x + Math.cos(yaw) * ahead
    const cy = y + Math.sin(yaw) * ahead
    for (const side of [1, -1]) {
      let d = 0
      while (d < 3 && isFree(grid, cx + lx * side * (d + 0.1), cy + ly * side * (d + 0.1))) d += 0.1
      best = Math.min(best, d)
    }
  }
  return Math.max(0.35, Math.min(MAX_SPREAD, best - 0.4))
}

function Camera({ x, y, yaw }: { x: number; y: number; yaw: number }) {
  useFrame(({ camera, scene, clock }) => {
    scene.fog = FOG // the dressing sets its own fog; this view wants a little more smoke
    const t = clock.elapsedTime
    // standing still, breathing: a slow sway and the head turning very slightly
    const sway = Math.sin(t * 0.6) * 0.025
    const look = yaw + Math.sin(t * 0.23) * 0.035
    const pitch = PITCH + Math.sin(t * 0.4) * 0.012
    camera.position.set(x - Math.sin(yaw) * sway, EYE_Z + Math.sin(t * 1.1) * 0.008, -(y + Math.cos(yaw) * sway))
    const cp = Math.cos(pitch)
    camera.lookAt(
      camera.position.x + Math.cos(look) * cp,
      camera.position.y + Math.sin(pitch),
      camera.position.z - Math.sin(look) * cp,
    )
  })
  return null
}

/** One waiting drone: hovers on its slot with a gentle, out-of-phase bob and drift. */
function WaitingDrone({ id, colour, at, yaw, phase }: { id: string; colour: string; at: [number, number, number]; yaw: number; phase: number }) {
  const g = useRef<THREE.Group>(null)
  useFrame(({ clock }) => {
    const t = clock.elapsedTime + phase
    if (!g.current) return
    g.current.position.set(at[0] + Math.sin(t * 0.7) * 0.04, at[1] + Math.cos(t * 0.5) * 0.04, at[2] + Math.sin(t * 1.6) * 0.06)
    g.current.rotation.z = yaw + Math.sin(t * 0.45) * 0.08
    g.current.rotation.x = Math.sin(t * 1.3) * 0.03
  })
  return (
    <group ref={g} position={at}>
      <DroneBody id={id} colour={colour} />
      <pointLight color={colour} intensity={2.2} distance={3.5} />
    </group>
  )
}

export function IntroScene() {
  const hello = useSim((s) => s.hello)
  const grid = useFreeGrid(hello?.map.grid_url, hello?.map.resolution)
  const e = hello?.entrance
  const drones = hello?.drones
  const slots = useMemo(() => {
    if (!e || !drones) return []
    const spread = clearance(grid, e.x, e.y, e.yaw)
    const c = Math.cos(e.yaw)
    const s = Math.sin(e.yaw)
    return drones.slice(0, SLOTS.length).map((d, i) => {
      const [ahead, side, z] = SLOTS[i]
      const lat = side * spread
      return { d, at: [e.x + c * ahead - s * lat, e.y + s * ahead + c * lat, z] as [number, number, number] }
    })
  }, [e, drones, grid])
  if (!e) return null
  return (
    <Canvas
      className="intro-canvas"
      camera={{ fov: 70, near: 0.05, far: 200 }}
      dpr={[1, 1.5]}
      onCreated={({ gl }) => {
        gl.localClippingEnabled = true // the scan's ceiling cut
      }}
    >
      <color attach="background" args={['#1c1f23']} />
      <hemisphereLight args={['#c8d2e0', '#202020', 0.85]} />
      <SimFrame>
        <Environment />
        <Dressing />
        {slots.map(({ d, at }, i) => (
          <WaitingDrone key={d.id} id={d.id} colour={d.colour} at={at} yaw={e.yaw + Math.PI} phase={i * 1.7} />
        ))}
      </SimFrame>
      <Camera x={e.x} y={e.y} yaw={e.yaw} />
    </Canvas>
  )
}
