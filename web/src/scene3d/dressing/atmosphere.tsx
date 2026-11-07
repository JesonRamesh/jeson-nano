// Atmosphere: smoky fog, red emergency beacons, floating dust/ash.
import { Sparkles } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { GraphNode } from '../../contracts'

const BEACON_Z = 2.35 // just under the 2.6 m ceiling

/** Up to `k` nodes spread over the map (farthest-point sampling, starting at the entrance). */
function spread(nodes: GraphNode[], k: number): GraphNode[] {
  if (!nodes.length) return []
  const out = [nodes.find((n) => n.kind === 'entrance') ?? nodes[0]]
  while (out.length < Math.min(k, nodes.length)) {
    let best: GraphNode | null = null
    let bd = -1
    for (const n of nodes) {
      const d = Math.min(...out.map((o) => Math.hypot(o.x - n.x, o.y - n.y)))
      if (d > bd) {
        bd = d
        best = n
      }
    }
    out.push(best!)
  }
  return out
}

function Beacon({ x, y, phase }: { x: number; y: number; phase: number }) {
  const light = useRef<THREE.PointLight>(null)
  const lamp = useRef<THREE.MeshBasicMaterial>(null)
  useFrame(({ clock }) => {
    // slow emergency pulse: on ~60 % of a 1.6 s cycle
    const c = (clock.elapsedTime / 1.6 + phase) % 1
    const on = Math.pow(Math.max(0, Math.sin(c * Math.PI)), 0.6)
    if (light.current) light.current.intensity = 6 * on
    if (lamp.current) lamp.current.color.setRGB(0.4 + 0.6 * on, 0.04, 0.03)
  })
  return (
    <group position={[x, y, BEACON_Z]}>
      <mesh>
        <sphereGeometry args={[0.07, 12, 8]} />
        <meshBasicMaterial ref={lamp} color="#ff1a10" toneMapped={false} />
      </mesh>
      <pointLight ref={light} color="#ff2416" intensity={6} distance={11} decay={1.5} />
    </group>
  )
}

export function Atmosphere({
  width,
  height,
  nodes,
  fog = 0.016,
  beacons = 4,
}: {
  width: number
  height: number
  nodes: GraphNode[]
  fog?: number
  beacons?: number
}) {
  const { scene } = useThree()
  const fogRef = useRef<THREE.FogExp2 | null>(null)
  // Scene-level fog: we live inside <SimFrame>, so attach="fog" would hit the group, not
  // the scene. Restore the previous fog on unmount.
  useEffect(() => {
    const prev = scene.fog
    fogRef.current = fog > 0 ? new THREE.FogExp2('#17120f', fog) : null
    // oxlint-disable-next-line react/immutability -- three.js scene objects are mutable by design
    scene.fog = fogRef.current
    return () => {
      // oxlint-disable-next-line react/immutability
      scene.fog = prev
    }
  }, [scene, fog])
  // Thick smoke at drone/chase height, thin from the overview camera (~40 m up), so the
  // whole building stays readable from above. three.js y = sim z (height).
  useFrame(({ camera }) => {
    if (fogRef.current) fogRef.current.density = fog * Math.min(1, 6 / Math.max(6, camera.position.y))
  })
  const spots = useMemo(() => spread(nodes, beacons), [nodes, beacons])
  return (
    <>
      {/* a faint warm-red fill so the whole building reads "emergency" */}
      <hemisphereLight args={['#ff6a4a', '#1a0d0a', 0.25]} />
      {spots.map((n, i) => (
        <Beacon key={n.id} x={n.x} y={n.y} phase={i * 0.37} />
      ))}
      <Sparkles
        count={Math.round(Math.min(900, width * height * 0.35))}
        scale={[width, height, 2.6]}
        position={[width / 2, height / 2, 1.3]}
        size={1.6}
        speed={0.15}
        noise={0.6}
        color="#c9b8a3"
        opacity={0.35}
      />
    </>
  )
}
