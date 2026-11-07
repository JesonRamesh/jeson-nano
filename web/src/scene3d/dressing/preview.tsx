/* oxlint-disable react/only-export-components -- dev-only entry page, not a module */
// Dev-only preview of the dressing on a real map, without the sim:
//   npm run dev  +  uv run python -m sgr.server.dev --map maps/<name>
//   http://localhost:5173/src/scene3d/dressing/preview.html?map=marshgate-route&view=fire
// Uses reveal="world" so every hazard is visible at once (the dashboard uses "belief").
import { OrbitControls, useGLTF } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { Suspense, useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import * as THREE from 'three'
import type { Scenario, Transform } from '../../contracts'
import { Dressing } from './index'

const q = new URLSearchParams(location.search)
const MAP = q.get('map') ?? 'marshgate-route'
const VIEW = q.get('view') ?? 'overview'

// [camera xyz, target xyz] in sim coordinates (z up)
const VIEWS: Record<string, [number[], number[]]> = {
  overview: [[19, -6, 46], [19, 26, 0]],
  fire: [[23.0, 34.6, 2.2], [27.8, 31.1, 0.9]],
  rubble: [[6.0, 24.6, 2.2], [9.6, 27.8, 0.5]],
  corridor: [[5.4, 6.5, 1.7], [5.4, 20, 1.3]],
  hall: [[19.0, 33.5, 1.9], [27.5, 32.0, 0.8]],
}
const toThree = (p: number[]) => new THREE.Vector3(p[0], p[2], -p[1])

// Same look as Scene3D's ScanModel: unlit (baked photo lighting), double-sided, ceiling cut at 2.3 m.
const CEILING_CUT = [new THREE.Plane(new THREE.Vector3(0, -1, 0), 2.3)]

function Scan({ transform }: { transform: Transform }) {
  const { scene } = useGLTF(`/maps/${MAP}/scan.glb`)
  const unlit = useMemo(() => {
    const copy = scene.clone(true)
    copy.traverse((o) => {
      const mesh = o as THREE.Mesh
      if (!mesh.isMesh) return
      const old = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as THREE.MeshStandardMaterial
      mesh.material = new THREE.MeshBasicMaterial({ map: old.map ?? null, side: THREE.DoubleSide, clippingPlanes: CEILING_CUT })
    })
    return copy
  }, [scene])
  const m = useMemo(() => new THREE.Matrix4().set(...(transform.M.flat() as Parameters<THREE.Matrix4['set']>)), [transform])
  return (
    <group matrix={m} matrixAutoUpdate={false}>
      <primitive object={unlit} />
    </group>
  )
}

function App() {
  const [tf, setTf] = useState<Transform | null>(null)
  const [sc, setSc] = useState<Scenario | null>(null)
  useEffect(() => {
    fetch(`/maps/${MAP}/transform.json`).then((r) => r.json()).then(setTf)
    fetch(`/api/maps/${MAP}/scenario`).then((r) => r.json()).then(setSc)
  }, [])
  const [cam, target] = VIEWS[VIEW] ?? VIEWS.overview
  if (!tf || !sc) return null
  const size: [number, number] = [tf.grid_size[0] * tf.pitch, tf.grid_size[1] * tf.pitch]
  return (
    <Canvas
      camera={{ fov: 50, near: 0.05, far: 500, position: toThree(cam) }}
      dpr={1}
      onCreated={({ gl }) => {
        gl.localClippingEnabled = true
      }}
    >
      <color attach="background" args={['#0e1014']} />
      {/* same lights as Scene3D */}
      <hemisphereLight args={['#c8d2e0', '#202020', 0.9]} />
      <directionalLight position={[10, 30, 10]} intensity={0.8} />
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <Suspense fallback={null}>
          <Scan transform={tf} />
        </Suspense>
        <Dressing mapName={MAP} size={size} nodes={sc.graph.nodes} reveal="world" />
      </group>
      <OrbitControls target={toThree(target)} makeDefault />
    </Canvas>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
