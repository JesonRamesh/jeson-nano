// Fire: flame particles, a smoke column, embers, a flickering light and a scorch mark.
// Sim frame (z up), placed at the fire's (x, y) on the floor. All GPU-animated.
// The dashboard draws the scan UNLIT (baked photo lighting), so the point light only
// reaches lit things (rubble, drones); the firelight on the building is faked with an
// orange-tinted floor pool + a soft halo that flicker with the light.
import { Sparkles } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { SIM_ASSETS, SimParts } from '../SimShapes'
import { rng } from './hazards'
import { glow, scorch, smokePuff, softDot } from './textures'

const VERT = /* glsl */ `
  attribute vec4 aSeed;            // phase, angle, radius, speed
  uniform float uTime, uPx, uHeight, uRadius, uPart, uSpread, uGrow;
  varying float vLife;
  void main() {
    float life = fract(uTime * aSeed.w + aSeed.x);
    float ang = aSeed.y * 6.2831 + uTime * 1.3;
    float r = aSeed.z * uRadius * mix(1.0, uSpread, life);
    float sway = sin(uTime * 3.1 + aSeed.x * 23.0) * 0.07 * life;
    vec3 p = vec3(cos(ang) * r + sway, sin(ang) * r, life * uHeight);
    vLife = life;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uPx * uPart * mix(1.0, uGrow, life) / max(0.1, -mv.z);
  }
`

// Smoke rises, then spreads out under the ceiling (uCeil) instead of making a pillar.
const SMOKE_VERT = /* glsl */ `
  attribute vec4 aSeed;
  uniform float uTime, uPx, uHeight, uRadius, uPart, uCeil;
  varying float vLife;
  void main() {
    float life = fract(uTime * aSeed.w + aSeed.x);
    float rise = life * uHeight;
    float over = max(0.0, rise - uCeil);            // distance "travelled" along the ceiling
    float ang = aSeed.y * 6.2831 + uTime * 0.25;
    float r = aSeed.z * uRadius * (0.4 + life) + over * (0.8 + aSeed.z);
    vec3 p = vec3(cos(ang) * r, sin(ang) * r, min(rise, uCeil) - aSeed.z * 0.25 * step(0.001, over));
    vLife = life;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uPx * uPart * (0.5 + 1.5 * life) / max(0.1, -mv.z);
  }
`

const FLAME_FRAG = /* glsl */ `
  uniform sampler2D uTex;
  varying float vLife;
  void main() {
    float a = texture2D(uTex, gl_PointCoord).a;
    vec3 c = mix(vec3(1.0, 0.82, 0.45), vec3(1.0, 0.42, 0.06), smoothstep(0.0, 0.35, vLife));
    c = mix(c, vec3(0.75, 0.12, 0.02), smoothstep(0.45, 0.9, vLife));
    float alpha = a * smoothstep(0.0, 0.06, vLife) * (1.0 - smoothstep(0.55, 1.0, vLife));
    gl_FragColor = vec4(c * alpha * 0.75, alpha);
  }
`

const SMOKE_FRAG = /* glsl */ `
  uniform sampler2D uTex;
  uniform float uOpacity;
  varying float vLife;
  void main() {
    float a = texture2D(uTex, gl_PointCoord).a * smoothstep(0.5, 0.25, length(gl_PointCoord - 0.5));
    vec3 c = mix(vec3(0.36, 0.27, 0.2), vec3(0.17, 0.16, 0.16), smoothstep(0.0, 0.4, vLife)); // fire-lit near the base
    float alpha = a * uOpacity * smoothstep(0.0, 0.15, vLife) * (1.0 - smoothstep(0.6, 1.0, vLife));
    gl_FragColor = vec4(c, alpha);
  }
`

function seeds(n: number, seed: number, speed: [number, number]) {
  const r = rng(seed)
  const a = new Float32Array(n * 4)
  for (let i = 0; i < n; i++) {
    a[i * 4] = r()
    a[i * 4 + 1] = r()
    a[i * 4 + 2] = Math.sqrt(r())
    a[i * 4 + 3] = speed[0] + r() * (speed[1] - speed[0])
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3)) // positions come from aSeed
  g.setAttribute('aSeed', new THREE.BufferAttribute(a, 4))
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 1.5), 4)
  return g
}

export function FireFX({ x, y, size = 1.5, smoke = true }: { x: number; y: number; size?: number; smoke?: boolean }) {
  const s = Math.max(0.6, size)
  const seed = Math.round(x * 131 + y * 71)
  const { size: view, camera } = useThree()
  const light = useRef<THREE.PointLight>(null)
  const pool = useRef<THREE.MeshBasicMaterial>(null)
  const halo = useRef<THREE.SpriteMaterial>(null)

  const flame = useMemo(() => {
    const geo = seeds(Math.round(140 * s), seed, [0.9, 1.6])
    const mat = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FLAME_FRAG,
      uniforms: {
        uTime: { value: 0 },
        uPx: { value: 500 },
        uHeight: { value: 0.75 * s + 0.4 },
        uRadius: { value: 0.32 * s },
        uPart: { value: 0.22 * s },
        uSpread: { value: 0.35 },
        uGrow: { value: 0.55 },
        uTex: { value: softDot() },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
    return { geo, mat }
  }, [s, seed])

  const smk = useMemo(() => {
    const geo = seeds(Math.round(40 * s), seed + 1, [0.06, 0.11])
    const mat = new THREE.ShaderMaterial({
      vertexShader: SMOKE_VERT,
      fragmentShader: SMOKE_FRAG,
      uniforms: {
        uTime: { value: 0 },
        uPx: { value: 500 },
        uHeight: { value: 5.5 },
        uCeil: { value: 2.15 }, // below the dashboard's 2.3 m dollhouse cut
        uRadius: { value: 0.3 * s },
        uPart: { value: 0.9 * s },
        uTex: { value: smokePuff() },
        uOpacity: { value: 0.5 },
      },
      transparent: true,
      depthWrite: false,
    })
    return { geo, mat }
  }, [s, seed])

  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    const fov = THREE.MathUtils.degToRad((camera as THREE.PerspectiveCamera).fov ?? 50)
    const px = view.height / (2 * Math.tan(fov / 2))
    for (const m of [flame.mat, smk.mat]) {
      m.uniforms.uTime.value = t
      m.uniforms.uPx.value = px
    }
    const f = 0.72 + 0.18 * Math.sin(t * 11 + seed) * Math.sin(t * 6.7) + 0.1 * Math.sin(t * 23)
    if (light.current) light.current.intensity = 9 * s * f
    if (pool.current) pool.current.opacity = 0.55 * f
    if (halo.current) halo.current.opacity = 0.28 * f
  })

  const scorchSize = 2.8 * s
  return (
    <group position={[x, y, 0]}>
      {/* the same flame tongues / embers / smoke the MuJoCo drone cameras render (shared spec) */}
      <SimParts parts={SIM_ASSETS.fire.parts} flicker phase={seed} />
      <mesh position={[0, 0, 0.012]} renderOrder={1}>
        <planeGeometry args={[scorchSize, scorchSize]} />
        <meshBasicMaterial map={scorch()} transparent depthWrite={false} polygonOffset polygonOffsetFactor={-2} />
      </mesh>
      {/* fake firelight on the unlit scan: a pool on the floor and a halo around the flames */}
      <mesh position={[0, 0, 0.02]} renderOrder={2}>
        <planeGeometry args={[6.5 * s, 6.5 * s]} />
        <meshBasicMaterial ref={pool} map={glow()} color="#ff6410" transparent opacity={0.5} depthWrite={false} fog={false} />
      </mesh>
      <sprite position={[0, 0, 0.9]} scale={[4.5 * s, 4.5 * s, 1]} renderOrder={4}>
        <spriteMaterial ref={halo} map={glow()} color="#ff7020" transparent opacity={0.25} depthWrite={false} fog={false} />
      </sprite>
      <points geometry={flame.geo} material={flame.mat} renderOrder={3} />
      {smoke && <points geometry={smk.geo} material={smk.mat} renderOrder={2} />}
      <Sparkles
        count={Math.round(24 * s)}
        scale={[1.2 * s, 1.2 * s, 2.4]}
        position={[0, 0, 1.3]}
        size={3}
        speed={1.2}
        noise={1.5}
        color="#ff9a3c"
        opacity={0.9}
      />
      <pointLight ref={light} position={[0, 0, 0.9]} color="#ff7a1a" intensity={9 * s} distance={9} decay={1.6} />
    </group>
  )
}
