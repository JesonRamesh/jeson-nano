// Person figures for the 3D view (W): realistic but cheap. Procedural low-poly GLBs from
// tools/scan_make_people.py (CC0): ~4.6k / 7.5k triangles, 5 / 9 draw calls, no textures,
// no skeleton. Sim frame (z up, x east): the GLBs are glTF y-up with +X forward, so they get
// one +90 deg X rotation, exactly like drone.glb.
import { useGLTF } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import ffUrl from './firefighter.glb?url'
import survivorUrl from './survivor.glb?url'
import { glow } from './textures'

/** Soft dark contact shadow on the floor: grounds the figure for one textured quad. */
function ContactShadow({ sx, sy, opacity = 0.55 }: { sx: number; sy: number; opacity?: number }) {
  return (
    <mesh position={[0, 0, 0.006]} renderOrder={1}>
      <planeGeometry args={[sx, sy]} />
      <meshBasicMaterial map={glow()} color="#000000" transparent opacity={opacity} depthWrite={false} polygonOffset polygonOffsetFactor={-1} />
    </mesh>
  )
}

function useFigure(url: string, opacity: number) {
  const { scene } = useGLTF(url)
  return useMemo(() => {
    const copy = scene.clone(true) // shares geometry
    if (opacity < 1)
      copy.traverse((o) => {
        const mesh = o as THREE.Mesh
        if (!mesh.isMesh) return
        const m = (mesh.material as THREE.Material).clone()
        m.transparent = true
        m.opacity = opacity
        m.depthWrite = false
        mesh.material = m
      })
    return copy
  }, [scene, opacity])
}

/** The survivor, lying on their back along `yaw` (head toward +yaw), hi-vis jacket.
 *  Drop-in for SimSurvivor: same (x, y, yaw, opacity). Breathes slightly. */
export function SurvivorFigure({ x, y, yaw = 0, opacity = 1 }: { x: number; y: number; yaw?: number; opacity?: number }) {
  const model = useFigure(survivorUrl, opacity)
  const g = useRef<THREE.Group>(null)
  useFrame(({ clock }) => {
    if (g.current) g.current.scale.z = 1 + 0.012 * Math.sin(clock.elapsedTime * 1.6) // slow breathing
  })
  return (
    <group position={[x, y, 0]} rotation={[0, 0, yaw]}>
      <ContactShadow sx={2.3} sy={1.1} opacity={0.5 * opacity} />
      <group ref={g}>
        <group rotation={[Math.PI / 2, 0, 0]}>
          <primitive object={model} />
        </group>
      </group>
    </group>
  )
}

/** The firefighter, standing at (x, y) facing `yaw`. `moving` adds a cheap walking bob/sway. */
export function FirefighterFigure({ x, y, yaw = 0, moving = false }: { x: number; y: number; yaw?: number; moving?: boolean }) {
  const model = useFigure(ffUrl, 1)
  const g = useRef<THREE.Group>(null)
  useFrame(({ clock }) => {
    if (!g.current) return
    const t = clock.elapsedTime * 7.5 // ~1.2 steps/s
    g.current.position.z = moving ? 0.025 * Math.abs(Math.sin(t)) : 0
    g.current.rotation.x = moving ? 0.03 * Math.sin(t) : 0 // side-to-side sway
  })
  return (
    <group position={[x, y, 0]} rotation={[0, 0, yaw]}>
      <ContactShadow sx={0.95} sy={0.8} />
      <group ref={g}>
        <group rotation={[Math.PI / 2, 0, 0]}>
          <primitive object={model} />
        </group>
      </group>
    </group>
  )
}

useGLTF.preload(survivorUrl)
useGLTF.preload(ffUrl)
