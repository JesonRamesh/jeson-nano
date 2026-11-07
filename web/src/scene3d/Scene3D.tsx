import { Line, OrbitControls, useGLTF } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Component, Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import type { Handle, Transform } from '../contracts'
import { useFreeGrid, type FreeGrid } from '../state/grid'
import { CELL_CODE, sim, useSim } from '../state/store'
import { DEBRIS_SIZE } from './constants'
import { mergeWalls } from './walls'

// Everything inside <SimFrame> uses sim coordinates (metres, z up, x east, y north).
// The group rotates z-up into three.js's y-up: (x, y, z)_sim -> (x, z, -y)_three.
// That rotation is the "C = −90° about X" from spec §6.3, so the scan inside it
// only needs transform.json's M.

const WALL_H = 2.6
const BELIEF_CAP = 4000 // max instanced belief cells per layer
const NO_DRONES: { id: string; colour: string }[] = []

type CamMode = 'overview' | 'free' | string // string = drone id (chase)

function SimFrame({ children }: { children: ReactNode }) {
  return <group rotation={[-Math.PI / 2, 0, 0]}>{children}</group>
}

// ---------------------------------------------------------------- environment

function ExtrudedWalls({ grid }: { grid: FreeGrid }) {
  const rects = useMemo(() => mergeWalls(grid), [grid])
  const ref = useRef<THREE.InstancedMesh>(null)
  useLayoutEffect(() => {
    const m = new THREE.Matrix4()
    rects.forEach((r, i) => {
      m.compose(
        new THREE.Vector3(r.x, r.y, WALL_H / 2),
        new THREE.Quaternion(),
        new THREE.Vector3(r.w, r.h, WALL_H),
      )
      ref.current!.setMatrixAt(i, m)
    })
    ref.current!.instanceMatrix.needsUpdate = true
  }, [rects])
  return (
    <instancedMesh ref={ref} args={[undefined, undefined, rects.length]} castShadow={false}>
      <boxGeometry />
      <meshStandardMaterial color="#b8ad99" roughness={0.9} />
    </instancedMesh>
  )
}

function ScanModel({ url, transform }: { url: string; transform: Transform }) {
  const { scene } = useGLTF(url)
  const matrix = useMemo(() => new THREE.Matrix4().set(...(transform.M.flat() as Parameters<THREE.Matrix4['set']>)), [transform])
  return (
    <group matrix={matrix} matrixAutoUpdate={false}>
      <primitive object={scene} />
    </group>
  )
}

class Fallback extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  componentDidCatch(e: unknown) {
    console.warn('scan.glb failed to load, using extruded walls', e)
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

function Environment() {
  const hello = useSim((s) => s.hello)
  const grid = useFreeGrid(hello?.map.grid_url, hello?.map.resolution)
  if (!hello) return null
  const m = hello.map
  const w = m.size[0] * m.resolution
  const h = m.size[1] * m.resolution
  const walls = grid ? <ExtrudedWalls grid={grid} /> : null
  return (
    <>
      <mesh position={[w / 2, h / 2, -0.01]}>
        <planeGeometry args={[w, h]} />
        <meshStandardMaterial color="#2a2c31" roughness={1} />
      </mesh>
      {hello.transform ? (
        <Fallback fallback={walls}>
          <Suspense fallback={walls}>
            <ScanModel url={`/maps/${m.name}/scan.glb`} transform={hello.transform} />
          </Suspense>
        </Fallback>
      ) : (
        walls
      )}
      <mesh position={[hello.entrance.x, hello.entrance.y, 0.02]}>
        <ringGeometry args={[0.35, 0.5, 4]} />
        <meshBasicMaterial color="#ffffff" />
      </mesh>
    </>
  )
}

// ---------------------------------------------------------------- live objects

/** Drones: glowing spheres, positions smoothed between 10 Hz state updates. */
function Drones() {
  const hello = useSim((s) => s.hello)
  const ids = hello?.drones ?? NO_DRONES
  const refs = useRef<Record<string, THREE.Group | null>>({})
  const mats = useRef<Record<string, THREE.MeshStandardMaterial | null>>({})
  useFrame((_, dt) => {
    const st = sim.snap.state
    if (!st) return
    const k = 1 - Math.exp(-dt * 10)
    for (const d of st.drones) {
      const g = refs.current[d.id]
      if (!g) continue
      const z = d.alive ? d.z : 0.25 // a lost drone has fallen
      g.position.lerp(new THREE.Vector3(d.x, d.y, z), k)
      g.rotation.z = d.yaw
      const mat = mats.current[d.id]
      if (mat) mat.emissiveIntensity = d.alive ? 1.6 : 0
    }
  })
  return (
    <>
      {ids.map((d) => (
        <group key={d.id} ref={(g) => void (refs.current[d.id] = g)}>
          <mesh>
            <sphereGeometry args={[0.25, 24, 16]} />
            <meshStandardMaterial
              ref={(m) => void (mats.current[d.id] = m)}
              color={d.colour}
              emissive={d.colour}
              emissiveIntensity={1.6}
            />
          </mesh>
          {/* heading nub along +x */}
          <mesh position={[0.32, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
            <coneGeometry args={[0.08, 0.18, 12]} />
            <meshBasicMaterial color={d.colour} />
          </mesh>
          <pointLight color={d.colour} intensity={1.5} distance={3} />
        </group>
      ))}
    </>
  )
}

function FireCluster({ x, y }: { x: number; y: number }) {
  const light = useRef<THREE.PointLight>(null)
  const mats = useRef<THREE.MeshStandardMaterial[]>([])
  const seed = x * 12.9898 + y * 78.233 // stable per-fire flicker phase
  useFrame(({ clock }) => {
    const t = clock.elapsedTime + seed
    const f = 0.75 + 0.25 * Math.sin(t * 13) * Math.sin(t * 7.3)
    if (light.current) light.current.intensity = 6 * f
    mats.current.forEach((m, i) => m && (m.emissiveIntensity = 2.2 * (0.8 + 0.2 * Math.sin(t * 9 + i))))
  })
  const cones: [number, number, number][] = [
    [0, 0, 0.9],
    [0.3, 0.15, 0.6],
    [-0.25, 0.2, 0.7],
    [0.05, -0.3, 0.55],
  ]
  return (
    <group position={[x, y, 0]}>
      {cones.map(([dx, dy, hgt], i) => (
        <mesh key={i} position={[dx, dy, hgt / 2]} rotation={[Math.PI / 2, 0, 0]}>
          <coneGeometry args={[0.22, hgt, 12]} />
          <meshStandardMaterial
            ref={(m) => void (m && (mats.current[i] = m))}
            color="#ffb300"
            emissive="#ff6d00"
            emissiveIntensity={2.2}
          />
        </mesh>
      ))}
      <pointLight ref={light} position={[0, 0, 1]} color="#ff8a1f" intensity={6} distance={6} />
    </group>
  )
}

function Debris({ h }: { h: Handle }) {
  // The handle has no yaw; derive a stable small rotation from its id ("slightly rotated").
  const yaw = useMemo(() => ((h.id.split('').reduce((a, c) => a + c.charCodeAt(0), 0) % 21) - 10) * (Math.PI / 180), [h.id])
  return (
    <mesh position={[h.x, h.y, DEBRIS_SIZE.z / 2]} rotation={[0, 0, yaw]}>
      <boxGeometry args={[DEBRIS_SIZE.x, DEBRIS_SIZE.y, DEBRIS_SIZE.z]} />
      <meshStandardMaterial color="#6d5a4b" roughness={1} />
    </mesh>
  )
}

/** Judge placements (click tool). Re-renders only when the set of handles changes. */
function ClickObjects() {
  const key = useSim((s) => (s.state?.click_handles ?? []).map((h) => h.id).join(','))
  const handles = useMemo(() => sim.snap.state?.click_handles ?? [], [key]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <>
      {handles.map((h) => (h.kind === 'debris' ? <Debris key={h.id} h={h} /> : <FireCluster key={h.id} x={h.x} y={h.y} />))}
    </>
  )
}

/** What the swarm believes: confirmed fire / danger / blocked / unsafe cells on the floor. */
function BeliefCells() {
  const layers = useMemo(
    () => [
      { code: CELL_CODE.fire, color: '#ff6d00', opacity: 0.85, z: 0.03 },
      { code: CELL_CODE.danger, color: '#ffb020', opacity: 0.35, z: 0.025 },
      { code: CELL_CODE.blocked_new, color: '#ff4d4f', opacity: 0.8, z: 0.035 },
      { code: CELL_CODE.unsafe, color: '#ab47bc', opacity: 0.4, z: 0.02 },
    ],
    [],
  )
  const refs = useRef<(THREE.InstancedMesh | null)[]>([])
  const version = useRef(-1)
  useFrame(() => {
    const b = sim.belief
    if (!b || b.version === version.current) return
    version.current = b.version
    const m = new THREE.Matrix4()
    layers.forEach((layer, li) => {
      const mesh = refs.current[li]
      if (!mesh) return
      let n = 0
      for (let k = 0; k < b.cells.length && n < BELIEF_CAP; k++) {
        if (b.cells[k] !== layer.code) continue
        const i = k % b.w
        const j = Math.floor(k / b.w)
        m.makeTranslation((i + 0.5) * b.res, (j + 0.5) * b.res, layer.z)
        mesh.setMatrixAt(n++, m)
      }
      mesh.count = n
      mesh.instanceMatrix.needsUpdate = true
    })
  })
  return (
    <>
      {layers.map((l, li) => (
        <instancedMesh key={l.code} ref={(r) => void (refs.current[li] = r)} args={[undefined, undefined, BELIEF_CAP]} count={0}>
          <planeGeometry args={[0.2, 0.2]} />
          <meshBasicMaterial color={l.color} transparent opacity={l.opacity} depthWrite={false} />
        </instancedMesh>
      ))}
    </>
  )
}

function Routes() {
  const route = useSim((s) => s.state?.route)
  const old = useSim((s) => s.state?.old_routes)
  const cps = useSim((s) => s.state?.checkpoints)
  const t = useSim((s) => s.state?.t ?? 0)
  const lift = (pts: [number, number][], z: number) => pts.map(([x, y]) => [x, y, z] as [number, number, number])
  return (
    <>
      {route && route.points.length > 1 && <Line points={lift(route.points, 0.12)} color="#3ddc84" lineWidth={4} />}
      {old
        ?.filter((r) => r.until > t && r.points.length > 1)
        .map((r, i) => (
          <Line
            key={i}
            points={lift(r.points, 0.1)}
            color="#ff4d4f"
            lineWidth={3}
            dashed
            dashSize={0.5}
            gapSize={0.35}
            transparent
            opacity={Math.min(1, (r.until - t) / 6)}
          />
        ))}
      {cps?.map((c, i) => (
        <mesh key={i} position={[c.x, c.y, 0.05]}>
          <torusGeometry args={[0.5, 0.05, 8, 32]} />
          <meshBasicMaterial color="#e6f7ff" />
        </mesh>
      ))}
    </>
  )
}

function SurvivorMarker() {
  const s = useSim((st) => st.state?.survivor)
  if (!s || s.x == null || s.y == null || s.status === 'unknown') return null
  return (
    <mesh position={[s.x, s.y, 0.2]}>
      <capsuleGeometry args={[0.18, 1.2, 6, 12]} />
      <meshStandardMaterial
        color="#d4ff3d"
        emissive="#9ccc00"
        emissiveIntensity={s.status === 'confirmed' ? 0.8 : 0.3}
        transparent
        opacity={s.status === 'confirmed' ? 1 : 0.6}
      />
    </mesh>
  )
}

// ---------------------------------------------------------------- camera

const tmpTarget = new THREE.Vector3()
const tmpPos = new THREE.Vector3()

/** sim (x, y, z) -> three.js world position (matches <SimFrame>). */
const toThree = (x: number, y: number, z: number, out: THREE.Vector3) => out.set(x, z, -y)

function CameraRig({ mode }: { mode: CamMode }) {
  const controls = useRef<OrbitControlsImpl>(null)
  const { camera, size } = useThree()
  const hello = useSim((s) => s.hello)

  // Overview: look down at the map centre from the south, ~55° pitch.
  useEffect(() => {
    if (!hello || mode !== 'overview') return
    const m = hello.map
    const w = m.size[0] * m.resolution
    const h = m.size[1] * m.resolution
    // Distance that fits the whole map in the view (both axes), with some margin.
    const vHalf = THREE.MathUtils.degToRad((camera as THREE.PerspectiveCamera).fov / 2)
    const hHalf = Math.atan(Math.tan(vHalf) * (size.width / Math.max(1, size.height)))
    const dist = Math.max(w / 2 / Math.tan(hHalf), h / 2 / Math.tan(vHalf)) * 1.15
    // ~70° pitch: steep enough that the near (south) edge doesn't spill out of view.
    toThree(w / 2, h / 2, 0, tmpTarget)
    toThree(w / 2, h / 2 - dist * 0.34, dist * 0.94, tmpPos)
    camera.position.copy(tmpPos)
    controls.current?.target.copy(tmpTarget)
    controls.current?.update()
  }, [mode, hello, camera, size.width, size.height])

  // Chase: 2.5 m behind and 3 m above the drone (clear of the 2.6 m walls), looking ahead.
  useFrame((_, dt) => {
    if (mode === 'overview' || mode === 'free') return
    const d = sim.snap.state?.drones.find((x) => x.id === mode)
    if (!d) return
    const k = 1 - Math.exp(-dt * 4)
    toThree(d.x - 2.5 * Math.cos(d.yaw), d.y - 2.5 * Math.sin(d.yaw), d.z + 3, tmpPos)
    toThree(d.x + 2.5 * Math.cos(d.yaw), d.y + 2.5 * Math.sin(d.yaw), 0, tmpTarget)
    camera.position.lerp(tmpPos, k)
    if (controls.current) {
      controls.current.target.lerp(tmpTarget, k)
      controls.current.update()
    }
  })

  return <OrbitControls ref={controls} makeDefault enableDamping enabled={mode === 'overview' || mode === 'free'} />
}

// ---------------------------------------------------------------- view

export function Scene3D() {
  const [mode, setMode] = useState<CamMode>('overview')
  const drones = useSim((s) => s.hello?.drones ?? NO_DRONES)
  const hasScan = useSim((s) => !!s.hello?.transform)
  return (
    <div className="scene3d">
      <Canvas camera={{ fov: 50, near: 0.1, far: 500, position: [15, 25, 10] }} dpr={[1, 2]}>
        <color attach="background" args={['#0e1014']} />
        <hemisphereLight args={['#c8d2e0', '#202020', 0.9]} />
        <directionalLight position={[10, 30, 10]} intensity={0.8} />
        <SimFrame>
          <Environment />
          <BeliefCells />
          <Routes />
          <ClickObjects />
          <SurvivorMarker />
          <Drones />
        </SimFrame>
        <CameraRig mode={mode} />
      </Canvas>
      {!hasScan && <div className="scene3d-note">walls extruded from grid.png (no scan.glb)</div>}
      <div className="view-buttons">
        <button className={mode === 'overview' ? 'active' : ''} onClick={() => setMode('overview')}>
          Overview
        </button>
        {drones.map((d) => (
          <button
            key={d.id}
            className={mode === d.id ? 'active' : ''}
            style={{ color: d.colour }}
            onClick={() => setMode(d.id)}
          >
            {d.id}
          </button>
        ))}
        <button className={mode === 'free' ? 'active' : ''} onClick={() => setMode('free')}>
          Free
        </button>
      </div>
    </div>
  )
}
