import { Line, OrbitControls, Stars, useGLTF } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Component, Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import * as THREE from 'three'
import { SkeletonUtils, type OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import type { ClientMsg, DroneState, Handle, HelloMsg, Transform } from '../contracts'
import { placeClick, undoClick, type ClickMode } from '../dashboard/clickTool'
import { useFreeGrid, type FreeGrid } from '../state/grid'
import { sim, useSim, type SimEventMsg } from '../state/store'
import { DEBRIS_SIZE } from './constants'
import { Dressing, FirefighterFigure, SurvivorFigure } from './dressing'
import { mergeWalls } from './walls'

// Everything inside <SimFrame> uses sim coordinates (metres, z up, x east, y north).
// The group rotates z-up into three.js's y-up: (x, y, z)_sim -> (x, z, -y)_three.
// That rotation is the "C = −90° about X" from spec §6.3, so the scan inside it
// only needs transform.json's M.

const WALL_H = 2.6
const MAX_FPS = 30
const NO_DRONES: { id: string; colour: string }[] = []

type CamMode = 'auto' | 'overview' | 'free' | string // string = drone id (chase)

/** Renders at most `fps` frames per second (with frameloop="demand"); stops when the tab is hidden. */
export function FrameCap({ fps }: { fps: number }) {
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    const id = setInterval(() => {
      if (!document.hidden) invalidate()
    }, 1000 / fps)
    return () => clearInterval(id)
  }, [fps, invalidate])
  return null
}

export function SimFrame({ children }: { children: ReactNode }) {
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

// "Dollhouse" cut: drop everything above 2.3 m (ceiling) so the overview sees into the rooms.
// World space is three.js y-up, so sim z = 2.3 m is y = 2.3. Needs gl.localClippingEnabled.
const CEILING_CUT = [new THREE.Plane(new THREE.Vector3(0, -1, 0), 2.3)]

/**
 * The phone scan. The Scaniverse GLB has no normals (POSITION + TEXCOORD_0 only), so lit
 * materials go flat-shaded and noisy triangles render near black. The photo texture already
 * has the lighting baked in, so draw it unlit, double-sided, with the ceiling cut away.
 */
function ScanModel({ url, transform }: { url: string; transform: Transform }) {
  const { scene } = useGLTF(url)
  const unlit = useMemo(() => {
    const copy = scene.clone(true) // shares geometry; only the materials are replaced
    copy.traverse((o) => {
      const mesh = o as THREE.Mesh
      if (!mesh.isMesh) return
      const old = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as THREE.MeshStandardMaterial
      mesh.material = new THREE.MeshBasicMaterial({
        map: old.map ?? null,
        color: old.map ? 0xffffff : (old.color ?? 0xcccccc),
        side: THREE.DoubleSide,
        clippingPlanes: CEILING_CUT,
      })
    })
    return copy
  }, [scene])
  const matrix = useMemo(() => new THREE.Matrix4().set(...(transform.M.flat() as Parameters<THREE.Matrix4['set']>)), [transform])
  return (
    <group matrix={matrix} matrixAutoUpdate={false}>
      <primitive object={unlit} />
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

export function Environment() {
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
      {/* At -0.15 m: below the lowest scanned floor (W's scan drifts from +7 cm at the
          entrance to -5 cm at the survivor), so it never covers the real floor. */}
      <mesh position={[w / 2, h / 2, -0.15]}>
        <planeGeometry args={[w, h]} />
        <meshBasicMaterial color="#77736d" /> {/* unlit like the scan: concrete grey, never black */}
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
export function DroneBody({ id, colour, url = DRONE_GLB_URL }: { id: string; colour: string; url?: string }) {
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
export function Drones() {
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

// W's disaster dressing (<Dressing />) draws flames for click fires and rubble around click
// debris. ?dressing=off turns it off (perf checks); then our simple cones stand in for fire.
const DRESSING_OFF = new URLSearchParams(window.location.search).get('dressing') === 'off'

/**
 * Judge placements (click tool). Debris is always our 2.0 m block, the same size as the
 * MuJoCo debris the drones see (W's rubble sits around its base). Fire: W's FireFX, or our
 * cones when the dressing is off. Re-renders only when the set of handles changes.
 */
export function ClickObjects() {
  const key = useSim((s) => (s.state?.click_handles ?? []).map((h) => h.id).join(','))
  const handles = useMemo(() => sim.snap.state?.click_handles ?? [], [key]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <>
      {handles.map((h) =>
        h.kind === 'debris' ? (
          <Debris key={h.id} h={h} />
        ) : DRESSING_OFF ? (
          <FireCluster key={h.id} x={h.x} y={h.y} />
        ) : null,
      )}
    </>
  )
}

/** Judge click tool in 3D: an invisible floor plane; a click on it in Debris/Fire mode places there. */
function ClickFloor({ mode, onPlace }: { mode: ClickMode | null; onPlace: (x: number, y: number) => void }) {
  const hello = useSim((s) => s.hello)
  if (!hello || !mode) return null
  const m = hello.map
  const w = m.size[0] * m.resolution
  const h = m.size[1] * m.resolution
  return (
    <mesh
      position={[w / 2, h / 2, 0.05]}
      onClick={(e) => {
        if (e.delta > 6) return // that was a camera drag, not a click
        e.stopPropagation()
        // e.point is in three.js world space: (x, z, -y)_sim
        onPlace(Math.round(e.point.x * 100) / 100, Math.round(-e.point.z * 100) / 100)
      }}
    >
      <planeGeometry args={[w, h]} />
      <meshBasicMaterial transparent opacity={0} depthWrite={false} />
    </mesh>
  )
}

export function Routes() {
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

export function SurvivorMarker() {
  // W's low-poly survivor figure (a hi-vis capsule while it loads). The belief has no
  // heading, so it lies along x. Faded until the swarm has confirmed it.
  const s = useSim((st) => st.state?.survivor)
  if (!s || s.x == null || s.y == null || s.status === 'unknown') return null
  const opacity = s.status === 'confirmed' ? 1 : 0.6
  return (
    <Suspense
      fallback={
        <mesh position={[s.x, s.y, 0.2]}>
          <capsuleGeometry args={[0.18, 1.2, 6, 12]} />
          <meshStandardMaterial
            color="#d4ff3d"
            emissive="#9ccc00"
            emissiveIntensity={s.status === 'confirmed' ? 0.8 : 0.3}
            transparent
            opacity={opacity}
          />
        </mesh>
      }
    >
      <SurvivorFigure x={s.x} y={s.y} opacity={opacity} />
    </Suspense>
  )
}

/** Act 2: the firefighter, from state.ff. Walks (bob + sway) while moving faster than 0.2 m/s. */
export function FirefighterMarker() {
  const ff = useSim((st) => st.state?.ff)
  const last = useRef<{ x: number; y: number; t: number } | null>(null)
  const [moving, setMoving] = useState(false)
  useEffect(() => {
    if (!ff) {
      last.current = null
      return
    }
    const now = performance.now() / 1000
    const prev = last.current
    last.current = { x: ff.x, y: ff.y, t: now }
    if (!prev || now - prev.t <= 0) return
    const speed = Math.hypot(ff.x - prev.x, ff.y - prev.y) / (now - prev.t)
    const m = speed > 0.2
    const id = setTimeout(() => setMoving(m), 0)
    return () => clearTimeout(id)
  }, [ff])
  if (!ff) return null
  return (
    <Suspense fallback={null}>
      <FirefighterFigure x={ff.x} y={ff.y} yaw={ff.yaw} moving={moving} />
    </Suspense>
  )
}

// ---------------------------------------------------------------- camera

const tmpTarget = new THREE.Vector3()
const tmpPos = new THREE.Vector3()
const DIRECTOR_HOLD_S = 6 // how long the auto-director stays on an event
const FOCUS_BACK = 7 // m south of the focus point
const FOCUS_UP = 10 // m above it (keeps nearby walls from blocking the view)
const AUTO_CYCLE_S = 5 // auto mode, no event showing: Overview -> D1 -> D2 -> D3 -> D4, this long each
const WASD_KEYS = 'wasdqe'

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
  // Wide view: ~70° pitch so the near (south) edge doesn't spill out of view.
  // Tall view: tilt more (~55°) so the map fills the height instead of floating in the middle.
  const back = aspect < 1 ? 0.55 : 0.34
  toThree(w / 2, h / 2, 0, target)
  toThree(w / 2, h / 2 - dist * back, dist * Math.sqrt(1 - back * back), pos)
}

/** Where an event happened: its xy, else the involved drone's current position. */
function eventFocus(e: SimEventMsg): [number, number] | null {
  if (e.kind === 'false_alarm') return null // decoy beat dropped from the demo: don't cut to it
  if (e.xy) return e.xy
  const d = e.drone ? sim.snap.state?.drones.find((x) => x.id === e.drone) : undefined
  return d ? [d.x, d.y] : null
}

/** Chase pose: 2.5 m behind and 3 m above the drone (clear of the 2.6 m walls), looking ahead. */
function chasePose(d: DroneState, pos: THREE.Vector3, target: THREE.Vector3) {
  if (d.alive) {
    toThree(d.x - 2.5 * Math.cos(d.yaw), d.y - 2.5 * Math.sin(d.yaw), d.z + 3, pos)
    toThree(d.x + 2.5 * Math.cos(d.yaw), d.y + 2.5 * Math.sin(d.yaw), 0, target)
  } else {
    // a lost drone lies on the floor: look down at it from above instead
    toThree(d.x, d.y - 1.5, 6, pos)
    toThree(d.x, d.y, 0, target)
  }
}

const isTyping = (e: KeyboardEvent) =>
  e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement

/**
 * Camera modes:
 *  - auto (default): the "auto-director". On each event with a place (xy or drone) it glides
 *    to that spot and holds 6 s. In between it cycles Overview -> D1 -> D2 -> D3 -> D4 (alive
 *    drones only), 5 s each.
 *  - overview / free: orbit controls (overview resets the pose first), plus WASD to move on the
 *    ground, Q/E down/up, Shift faster.
 *  - D1..D4: chase camera behind that drone.
 */
function CameraRig({ mode, onShot }: { mode: CamMode; onShot: (shot: string) => void }) {
  const controls = useRef<OrbitControlsImpl>(null)
  const { camera, size, invalidate } = useThree()
  const hello = useSim((s) => s.hello)
  const focus = useRef<{ seq: number; until: number; drone: string | null; xy: [number, number] | null } | null>(null)
  const lastSeq = useRef(0)
  const lastShot = useRef('')
  const keys = useRef(new Set<string>())

  // WASD / Q / E / Shift held keys. Ignored while typing and under pointer lock (the Act 2
  // firefighter view owns WASD there).
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      // Act 2: the firefighter view owns WASD
      if (isTyping(e) || document.pointerLockElement || sim.snap.state?.act === 2) return
      const k = e.key.toLowerCase()
      if (k.length === 1 && WASD_KEYS.includes(k)) {
        keys.current.add(k)
        invalidate()
      }
      if (e.key === 'Shift') keys.current.add('shift')
    }
    const up = (e: KeyboardEvent) => {
      keys.current.delete(e.key.toLowerCase())
      if (e.key === 'Shift') keys.current.delete('shift')
    }
    const clear = () => keys.current.clear()
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', clear)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', clear)
    }
  }, [invalidate])

  // WASD moves the camera AND the orbit pivot on the ground plane (three.js is y-up here).
  useFrame((_, dt) => {
    const ctl = controls.current
    if (!ctl || !(mode === 'overview' || mode === 'free') || keys.current.size === 0) return
    const k = keys.current
    const fwd = new THREE.Vector3().subVectors(ctl.target, camera.position)
    fwd.y = 0
    if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1) // looking straight down: W = north (sim +y = three -z)
    fwd.normalize()
    const right = new THREE.Vector3(-fwd.z, 0, fwd.x)
    const move = new THREE.Vector3()
    if (k.has('w')) move.add(fwd)
    if (k.has('s')) move.sub(fwd)
    if (k.has('d')) move.add(right)
    if (k.has('a')) move.sub(right)
    if (k.has('e')) move.y += 1
    if (k.has('q')) move.y -= 1
    if (move.lengthSq() === 0) return
    const speed = Math.max(3, camera.position.distanceTo(ctl.target) * 0.8) * (k.has('shift') ? 3 : 1)
    move.normalize().multiplyScalar(speed * Math.min(dt, 0.1))
    camera.position.add(move)
    ctl.target.add(move)
    ctl.update()
    invalidate() // frameloop="demand": keep rendering while a key is held
  })

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
      let shot: string
      if (f && clock.elapsedTime < f.until) {
        // Follow a drone while it moves; a fixed xy stays put.
        const d = f.drone ? sim.snap.state?.drones.find((x) => x.id === f.drone) : undefined
        const [fx, fy] = d ? [d.x, d.y] : f.xy!
        toThree(fx, fy, 0, tmpTarget)
        toThree(fx, fy - FOCUS_BACK, FOCUS_UP, tmpPos)
        shot = 'event'
      } else {
        // No event showing: cycle Overview -> D1 -> D2 -> D3 -> D4 (alive drones only).
        focus.current = null
        const alive = (sim.snap.state?.drones ?? []).filter((x) => x.alive)
        const shots = ['overview', ...alive.map((x) => x.id)]
        shot = shots[Math.floor(clock.elapsedTime / AUTO_CYCLE_S) % shots.length]
        const d = alive.find((x) => x.id === shot)
        if (d) chasePose(d, tmpPos, tmpTarget)
        else overviewPose(hello, camera as THREE.PerspectiveCamera, size.width / Math.max(1, size.height), tmpPos, tmpTarget)
      }
      if (shot !== lastShot.current) {
        lastShot.current = shot
        onShot(shot)
      }
      k = 1 - Math.exp(-dt * 2.2) // slower, cinematic glide
    } else {
      const d = sim.snap.state?.drones.find((x) => x.id === mode)
      if (!d) return
      chasePose(d, tmpPos, tmpTarget)
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

export function Scene3D({ send }: { send: (m: ClientMsg) => void }) {
  const [mode, setMode] = useState<CamMode>('auto')
  const [autoShot, setAutoShot] = useState('overview')
  const [clickMode, setClickMode] = useState<ClickMode | null>(null)
  const hello = useSim((s) => s.hello)
  const grid = useFreeGrid(hello?.map.grid_url, hello?.map.resolution)

  // Keyboard: the judge change tool has no on-screen buttons (kept clean for the audience),
  // but stays one key away: 1 = drop debris, 2 = drop fire, U = undo, Esc = cancel.
  // WASD in Auto / a chase view switches to Free so the keys move the camera.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setClickMode(null)
      if (isTyping(e) || document.pointerLockElement || e.ctrlKey || e.metaKey || e.altKey) return
      const k = e.key.toLowerCase()
      if (k === '1') setClickMode((m) => (m === 'debris' ? null : 'debris'))
      if (k === '2') setClickMode((m) => (m === 'fire' ? null : 'fire'))
      if (k === 'u') undoClick(send)
      if (k.length === 1 && WASD_KEYS.includes(k) && sim.snap.state?.act !== 2)
        setMode((m) => (m === 'overview' || m === 'free' ? m : 'free'))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [send])
  const drones = useSim((s) => s.hello?.drones ?? NO_DRONES)
  const mapCentre: [number, number, number] = hello
    ? [(hello.map.size[0] * hello.map.resolution) / 2, 0, -(hello.map.size[1] * hello.map.resolution) / 2]
    : [0, 0, 0]
  return (
    <div className={`scene3d ${clickMode ? 'click-mode' : ''}`}>
      {/* frameloop="demand" + FrameCap: ~30 fps max, so the 3D view leaves GPU time for Gemma. */}
      <Canvas
        camera={{ fov: 50, near: 0.1, far: 500, position: [15, 25, 10] }}
        dpr={[1, 1.5]}
        frameloop="demand"
        onCreated={({ gl }) => {
          gl.localClippingEnabled = true // for the scan's ceiling cut
        }}
      >
        <FrameCap fps={MAX_FPS} />
        <color attach="background" args={['#07090b']} />
        {/* Starfield behind the building (white points, gently twinkling). Centred on the map; its shader
            ignores the dressing's fog, so it stays visible through the smoke. */}
        <group position={mapCentre}>
          <Stars radius={90} depth={60} count={5000} factor={4} saturation={0} fade speed={0.5} />
        </group>
        <hemisphereLight args={['#c8d2e0', '#202020', 0.9]} />
        <directionalLight position={[10, 30, 10]} intensity={0.8} />
        <SimFrame>
          <Environment />
          <ClickFloor mode={clickMode} onPlace={(x, y) => clickMode && placeClick(send, grid, clickMode, x, y)} />
          <Routes />
          <ClickObjects />
          <SurvivorMarker />
          <FirefighterMarker />
          <Drones />
          <Dressing />
        </SimFrame>
        <CameraRig mode={mode} onShot={setAutoShot} />
      </Canvas>
      {clickMode && (
        <div className={`click-hint click-hint-${clickMode}`}>
          Click the floor to drop {clickMode} · Esc to cancel
        </div>
      )}
      <div className="view-buttons">
        <button
          className={mode === 'auto' ? 'active' : ''}
          onClick={() => setMode('auto')}
          title={`Auto-director: cuts to each event; otherwise cycles Overview, D1-D4 every ${AUTO_CYCLE_S} s`}
        >
          {mode === 'auto' ? `Auto · ${autoShot === 'overview' ? 'Overview' : autoShot === 'event' ? 'Event' : autoShot}` : 'Auto'}
        </button>
        <button className={mode === 'overview' ? 'active' : ''} onClick={() => setMode('overview')}>
          Overview
        </button>
        {drones.map((d) => (
          <button
            key={d.id}
            className={mode === d.id ? 'active' : ''}
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
