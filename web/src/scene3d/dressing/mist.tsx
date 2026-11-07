// Ground mist: a thin, low, slowly rolling haze over the whole floor.
// Many large, very faint puffs scattered over grid.png's free cells (so none float inside
// walls), each drifting and "breathing" slowly. Thicker at drone height, fades for the
// overview camera so the map stays readable from above.
import { useFrame, useThree } from '@react-three/fiber'
import { useMemo } from 'react'
import * as THREE from 'three'
import type { FreeGrid } from '../../state/grid'
import { rng } from './hazards'
import { smokePuff } from './textures'

const VERT = /* glsl */ `
  attribute vec4 aSeed;          // phase, size jitter, height, drift speed
  uniform float uTime, uPx, uSize;
  varying float vPhase;
  void main() {
    float t = uTime * aSeed.w;
    vec3 p = position + vec3(sin(t + aSeed.x * 6.28) * 0.7, cos(t * 0.8 + aSeed.x * 9.1) * 0.7,
                             aSeed.z + sin(uTime * 0.13 + aSeed.x * 4.0) * 0.06);
    vPhase = aSeed.x * 6.28 + uTime * 0.09;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uPx * uSize * (0.75 + 0.5 * aSeed.y) / max(0.1, -mv.z);
  }
`
const FRAG = /* glsl */ `
  uniform sampler2D uTex;
  uniform float uOpacity;
  varying float vPhase;
  void main() {
    float a = texture2D(uTex, gl_PointCoord).a * smoothstep(0.5, 0.2, length(gl_PointCoord - 0.5));
    float breathe = 0.55 + 0.45 * sin(vPhase);     // density slowly varies, like real haze
    gl_FragColor = vec4(0.86, 0.83, 0.79, a * uOpacity * breathe);
  }
`

export function GroundMist({ grid, density = 0.75, opacity = 0.26 }: { grid: FreeGrid; density?: number; opacity?: number }) {
  const { size: view } = useThree()
  const { geo, mat } = useMemo(() => {
    const rand = rng(4242)
    const cells: number[] = []
    for (let k = 0; k < grid.free.length; k += 3) if (grid.free[k]) cells.push(k) // subsample for speed
    const area = cells.length * 3 * grid.res * grid.res
    const n = Math.min(900, Math.round(area * density))
    const pos = new Float32Array(n * 3)
    const seed = new Float32Array(n * 4)
    for (let i = 0; i < n; i++) {
      const k = cells[Math.floor(rand() * cells.length)]
      pos[i * 3] = ((k % grid.w) + rand()) * grid.res
      pos[i * 3 + 1] = (Math.floor(k / grid.w) + rand()) * grid.res
      seed[i * 4] = rand()
      seed[i * 4 + 1] = rand()
      seed[i * 4 + 2] = 0.12 + rand() * rand() * 0.9 // mostly hugging the floor
      seed[i * 4 + 3] = 0.03 + rand() * 0.05
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 4))
    g.computeBoundingSphere()
    const m = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: { uTime: { value: 0 }, uPx: { value: 500 }, uSize: { value: 3.2 }, uTex: { value: smokePuff() }, uOpacity: { value: opacity } },
      transparent: true,
      depthWrite: false,
    })
    return { geo: g, mat: m }
  }, [grid, density, opacity])

  useFrame(({ clock, camera }) => {
    const fov = THREE.MathUtils.degToRad((camera as THREE.PerspectiveCamera).fov ?? 50)
    // oxlint-disable-next-line react/immutability -- per-frame shader uniforms are mutable by design
    mat.uniforms.uTime.value = clock.elapsedTime
    mat.uniforms.uPx.value = view.height / (2 * Math.tan(fov / 2))
    // full at drone/chase height, ~40 % from the overview (three.js y = height)
    mat.uniforms.uOpacity.value = opacity * Math.max(0.4, Math.min(1, 6 / Math.max(6, camera.position.y)))
  })
  return <points geometry={geo} material={mat} renderOrder={1} frustumCulled={false} />
}
