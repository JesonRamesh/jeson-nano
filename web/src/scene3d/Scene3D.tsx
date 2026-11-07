import { Line, OrbitControls, useGLTF } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Component, Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import * as THREE from 'three'
import { SkeletonUtils, type OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import type { Handle, HelloMsg, Transform } from '../contracts'
import { useFreeGrid, type FreeGrid } from '../state/grid'
import { CELL_CODE, sim, useSim, type SimEventMsg } from '../state/store'
import { DEBRIS_SIZE } from './constants'
import { mergeWalls } from './walls'

// Everything inside <SimFrame> uses sim coordinates (metres, z up, x east, y north).
// The group rotates z-up into three.js's y-up: (x, y, z)_sim -> (x, z, -y)_three.
// That rotation is the "C = −90° about X" from spec §6.3, so the scan inside it
// only needs transform.json's M.

const WALL_H = 2.6
const BELIEF_CAP = 4000 // max instanced belief cells per layer
const MAX_FPS = 30
const NO_DRONES: { id: string; colour: string }[] = []

type CamMode = 'auto' | 'overview' | 'free' | string // string = drone id (chase)

/** Renders at most `fps` frames per second (with frameloop="demand"); stops when the tab is hidden. */
function FrameCap({ fps }: { fps: number }) {
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    const id = setInterval(() => {
      if (!document.hidden) invalidate()
    }, 1000 / fps)
    return () => clearInterval(id)
  }, [fps, invalidate])
  return null
}

function SimFrame({ children }: { children: ReactNode }) {
  return <group rotation={[-Math.PI / 2, 0, 0]}>{children}</group>
}

// ---------------------------------------------------------------- environment

/** Walls from grid.png. `ghost`: faint, see-through version drawn under the scan to fill its holes. */
function ExtrudedWalls({ grid, ghost = false }: { grid: FreeGrid; ghost?: boolean }) {
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
      {ghost ? (
        <meshStandardMaterial color="#8f877a" roughness={1} transparent opacity={0.12} depthWrite={false} />
      ) : (
        <meshStandardMaterial color="#b8ad99" roughness={0.9} />
      )}
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
      {/* Neutral floor under everything: holes / unscanned parts of the scan read as floor, not black. */}
      <mesh position={[w / 2, h / 2, -0.03]}>
        <planeGeometry args={[w, h]} />
        <meshStandardMaterial color="#4a4e57" roughness={1} />
      </mesh>
      {hello.transform ? (
        <>
          {grid && <ExtrudedWalls grid={grid} ghost />}
          <Fallback fallback={walls}>
            <Suspense fallback={walls}>
              <ScanModel url={`/maps/${m.name}/scan.glb`} transform={hello.transform} />
            </Suspense>
          </Fallback>
        </>
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

// W's CC0 drone model lives in scene3d/dressing/ (see its README). Found by glob so the build
// still works without it (the spheres are used then — cut ladder item 2).
const DRESSING_GLBS = import.meta.glob('./dressing/*.glb', { eager: true, query: '?url', import: 'default' }) as Record<
  string,
  string
>
const DRONE_GLB_URL = Object.entries(DRESSING_GLBS).find(([path]) => /drone/i.test(path))?.[1]
const DRONE_SCALE = 1.25 // model spans 0.5 m (true size); a bit larger reads better on the projector
const PROPS: [string, number][] = [
  ['prop_FL', 1],
  ['prop_RR', 1],
  ['prop_FR', -1],
  ['prop_RL', -1],
]

/** One drone body: W's GLB (accent tinted with the drone colour) or the glowing sphere fallback. */
function DroneBody({ id, colour, url }: { id: string; colour: string; url?: string }) {
  return url ? (
    <Suspense fallback={<SphereBody colour={colour} />}>
      <GlbBody id={id} colour={colour} url={url} />
    </Suspense>
  ) : (
    <SphereBody colour={colour} />
  )
}

function SphereBody({ colour }: { colour: string }) {
  return (
    <>
      <mesh>
        <sphereGeometry args={[0.25, 24, 16]} />
        <meshStandardMaterial color={colour} emissive={colour} emissiveIntensity={1.6} />
      </mesh>
      {/* heading nub along +x */}
      <mesh position={[0.32, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.08, 0.18, 12]} />
        <meshBasicMaterial color={colour} />
      </mesh>
    </>
  )
}

/** Model axes (README): +Y up, +X forward -> one +90° X rotation inside the z-up SimFrame. */
function GlbBody({ id, colour, url }: { id: string; colour: string; url: string }) {
  const { scene } = useGLTF(url)
  const model = useMemo(() => {
    const m = SkeletonUtils.clone(scene) // one copy per drone
    m.traverse((o) => {
      const mesh = o as THREE.Mesh
      if (!mesh.isMesh || !(mesh.material instanceof THREE.MeshStandardMaterial)) return
      const mat = mesh.material.clone() // own materials, so LOST styling is per drone
      if (mat.name === 'accent') {
        mat.color.set(colour)
        mat.emissive.set(colour)
      }
      mesh.material = mat
    })
    return m
  }, [scene, colour])
  useFrame((_, dt) => {
    const alive = sim.snap.state?.drones.find((d) => d.id === id)?.alive ?? true
    if (!alive) return // props stop when the drone is lost
    for (const [name, dir] of PROPS) {
      const p = model.getObjectByName(name)
      if (p) p.rotation.y += dir * 60 * dt
    }
  })
  return (
    <group rotation={[Math.PI / 2, 0, 0]} scale={DRONE_SCALE}>
      <primitive object={model} />
    </group>
  )
}

/** Drones: model (or glowing sphere) per drone, positions smoothed between 10 Hz state updates. */
function Drones() {
  const hello = useSim((s) => s.hello)
  const ids = hello?.drones ?? NO_DRONES
  const refs = useRef<Record<string, THREE.Group | null>>({})
  const lights = useRef<Record<string, THREE.PointLight | null>>({})
  const tmp = useMemo(() => new THREE.Vector3(), [])
  useFrame((_, dt) => {
    const st = sim.snap.state
    if (!st) return
    const k = 1 - Math.exp(-dt * 10)
    for (const d of st.drones) {
      const g = refs.current[d.id]
      if (!g) continue
      const z = d.alive ? d.z : 0.25 // a lost drone has fallen
      g.position.lerp(tmp.set(d.x, d.y, z), k)
      g.rotation.z = d.yaw
      // LOST: glow off, light off, tilted on the floor
      g.rotation.x = d.alive ? 0 : 0.5
      // LOST: every emissive part (sphere / model accent + nav lights) goes dark
      const glow = d.alive ? (DRONE_GLB_URL ? 1.2 : 1.6) : 0
      g.traverse((o) => {
        const m = (o as THREE.Mesh).material
        if (m instanceof THREE.MeshStandardMaterial) m.emissiveIntensity = glow
      })
      const l = lights.current[d.id]
      if (l) l.intensity = d.alive ? 1.5 : 0
    }
  })
  return (
    <>
      {ids.map((d) => (
        <group key={d.id} ref={(g) => void (refs.current[d.id] = g)}>
          <DroneBody id={d.id} colour={d.colour} url={DRONE_GLB_URL} />
          <pointLight ref={(l) => void (lights.current[d.id] = l)} color={d.colour} intensity={1.5} distance={3} />
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
const DIRECTOR_HOLD_S = 6 // how long the auto-director stays on an event
const FOCUS_BACK = 7 // m south of the focus point
const FOCUS_UP = 10 // m above it (keeps nearby walls from blocking the view)

/** sim (x, y, z) -> three.js world position (matches <SimFrame>). */
const toThree = (x: number, y: number, z: number, out: THREE.Vector3) => out.set(x, z, -y)

/** Where the camera should be for the whole-map overview, in three.js coordinates. */
function overviewPose(hello: HelloMsg, cam: THREE.PerspectiveCamera, aspect: number, pos: THREE.Vector3, target: THREE.Vector3) {
  const m = hello.map
  const w = m.size[0] * m.resolution
  const h = m.size[1] * m.resolution
  // Distance that fits the whole map in the view (both axes), with some margin.
  const vHalf = THREE.MathUtils.degToRad(cam.fov / 2)
  const hHalf = Math.atan(Math.tan(vHalf) * aspect)
  const dist = Math.max(w / 2 / Math.tan(hHalf), h / 2 / Math.tan(vHalf)) * 1.35 // margin: near edge is magnified
  // ~70° pitch: steep enough that the near (south) edge doesn't spill out of view.
  toThree(w / 2, h / 2, 0, target)
  toThree(w / 2, h / 2 - dist * 0.34, dist * 0.94, pos)
}

/** Where an event happened: its xy, else the involved drone's current position. */
function eventFocus(e: SimEventMsg): [number, number] | null {
  if (e.kind === 'false_alarm') return null // decoy beat dropped from the demo: don't cut to it
  if (e.xy) return e.xy
  const d = e.drone ? sim.snap.state?.drones.find((x) => x.id === e.drone) : undefined
  return d ? [d.x, d.y] : null
}

/**
 * Camera modes:
 *  - auto (default): the "auto-director". Sits on the overview; on each event
 *    with a place (xy or drone) it glides to that spot, holds 6 s, glides back.
 *  - overview / free: orbit controls (overview resets the pose first).
 *  - D1..D4: chase camera behind that drone.
 */
function CameraRig({ mode }: { mode: CamMode }) {
  const controls = useRef<OrbitControlsImpl>(null)
  const { camera, size } = useThree()
  const hello = useSim((s) => s.hello)
  const focus = useRef<{ seq: number; until: number; drone: string | null; xy: [number, number] | null } | null>(null)
  const lastSeq = useRef(0)

  // Overview button: jump to the overview pose, then let the user orbit.
  useEffect(() => {
    if (!hello || mode !== 'overview') return
    overviewPose(hello, camera as THREE.PerspectiveCamera, size.width / Math.max(1, size.height), tmpPos, tmpTarget)
    camera.position.copy(tmpPos)
    controls.current?.target.copy(tmpTarget)
    controls.current?.update()
  }, [mode, hello, camera, size.width, size.height])

  useFrame(({ clock }, dt) => {
    if (mode === 'overview' || mode === 'free') return
    const ctl = controls.current
    let k = 1 - Math.exp(-dt * 4)

    if (mode === 'auto') {
      if (!hello) return
      // New event? Take the newest one that has a place.
      const evs = sim.snap.events
      const newest = evs[evs.length - 1]
      if (newest && newest.seq !== lastSeq.current) {
        lastSeq.current = newest.seq
        const xy = eventFocus(newest.msg)
        if (xy) focus.current = { seq: newest.seq, until: clock.elapsedTime + DIRECTOR_HOLD_S, drone: newest.msg.xy ? null : (newest.msg.drone ?? null), xy }
      }
      const f = focus.current
      if (f && clock.elapsedTime < f.until) {
        // Follow a drone while it moves; a fixed xy stays put.
        const d = f.drone ? sim.snap.state?.drones.find((x) => x.id === f.drone) : undefined
        const [fx, fy] = d ? [d.x, d.y] : f.xy!
        toThree(fx, fy, 0, tmpTarget)
        toThree(fx, fy - FOCUS_BACK, FOCUS_UP, tmpPos)
      } else {
        focus.current = null
        overviewPose(hello, camera as THREE.PerspectiveCamera, size.width / Math.max(1, size.height), tmpPos, tmpTarget)
      }
      k = 1 - Math.exp(-dt * 2.2) // slower, cinematic glide
    } else {
      // Chase: 2.5 m behind and 3 m above the drone (clear of the 2.6 m walls), looking ahead.
      const d = sim.snap.state?.drones.find((x) => x.id === mode)
      if (!d) return
      if (d.alive) {
        toThree(d.x - 2.5 * Math.cos(d.yaw), d.y - 2.5 * Math.sin(d.yaw), d.z + 3, tmpPos)
        toThree(d.x + 2.5 * Math.cos(d.yaw), d.y + 2.5 * Math.sin(d.yaw), 0, tmpTarget)
      } else {
        // a lost drone lies on the floor: look down at it from above instead
        toThree(d.x, d.y - 1.5, 6, tmpPos)
        toThree(d.x, d.y, 0, tmpTarget)
      }
    }
    camera.position.lerp(tmpPos, k)
    if (ctl) {
      ctl.target.lerp(tmpTarget, k)
      ctl.update()
    } else camera.lookAt(tmpTarget)
  })

  return <OrbitControls ref={controls} makeDefault enableDamping enabled={mode === 'overview' || mode === 'free'} />
}

// ---------------------------------------------------------------- view

export function Scene3D() {
  const [mode, setMode] = useState<CamMode>('auto')
  const drones = useSim((s) => s.hello?.drones ?? NO_DRONES)
  const hasScan = useSim((s) => !!s.hello?.transform)
  return (
    <div className="scene3d">
      {/* frameloop="demand" + FrameCap: ~30 fps max, so the 3D view leaves GPU time for Gemma. */}
      <Canvas camera={{ fov: 50, near: 0.1, far: 500, position: [15, 25, 10] }} dpr={[1, 1.5]} frameloop="demand">
        <FrameCap fps={MAX_FPS} />
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
        <button className={mode === 'auto' ? 'active' : ''} onClick={() => setMode('auto')} title="Auto-director: cuts to each event">
          Auto
        </button>
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
