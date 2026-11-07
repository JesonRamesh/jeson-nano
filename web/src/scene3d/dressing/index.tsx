// Disaster dressing for the 3D view (W, spec §11 Day 2).
//
// D mounts ONE line inside <SimFrame> in Scene3D.tsx:
//     <Dressing />
// Everything is in sim coordinates (metres, z up). It reads the map from the store's
// `hello` and, by default, shows hazards only once the swarm has found them: fire and
// blocked_new cells of the belief grid become flames/smoke and rubble. Judge clicks
// (state.click_handles) are dressed straight away. reveal="world" instead uses the
// scenario's real positions (GET /api/maps/{name}/scenario); D's server marks that
// endpoint "never the dashboard", so only use it where the team agrees (e.g. FF view).
//
// URL switches for performance checks: ?dressing=off | ?dressing=lite (no smoke/dust).
import { useFrame } from '@react-three/fiber'
import { useMemo, useRef, useState } from 'react'
import type { Fire, GraphNode } from '../../contracts'
import { useFreeGrid } from '../../state/grid'
import { sim, useSim } from '../../state/store'
import { Atmosphere } from './atmosphere'
import { FireFX } from './fire'
import { beliefDebris, beliefFires, mergeFires, useScenarioHazards, type DebrisBlock } from './hazards'
import { GroundMist } from './mist'
import { RubblePile } from './rubble'

export { FireFX } from './fire'
export { RubblePile } from './rubble'

const MODE = typeof window === 'undefined' ? '' : new URLSearchParams(window.location.search).get('dressing') ?? ''
const NO_NODES: GraphNode[] = []

function sameFires(a: Fire[], b: Fire[]) {
  return a.length === b.length && a.every((f, i) => Math.hypot(f.x - b[i].x, f.y - b[i].y) < 0.3)
}

export interface DressingProps {
  /** Overrides for use outside the live dashboard (preview page). */
  mapName?: string
  size?: [number, number] // map width, height (m)
  nodes?: GraphNode[]
  gridUrl?: string // grid.png (for the ground mist); default hello.map.grid_url
  /** 'belief' (default): only what the swarm has found so far. 'world': the scenario's
   *  real fires/debris from the start (uses the editor-only scenario endpoint). */
  reveal?: 'world' | 'belief'
}

export function Dressing({ mapName, size, nodes, gridUrl, reveal = 'belief' }: DressingProps) {
  const hello = useSim((s) => s.hello)
  const name = mapName ?? hello?.map.name
  const w = size?.[0] ?? (hello ? hello.map.size[0] * hello.map.resolution : 0)
  const h = size?.[1] ?? (hello ? hello.map.size[1] * hello.map.resolution : 0)
  const graphNodes = nodes ?? hello?.graph.nodes ?? NO_NODES
  const grid = useFreeGrid(gridUrl ?? hello?.map.grid_url, hello?.map.resolution ?? 0.1)
  const hazards = useScenarioHazards(reveal === 'world' ? name : undefined)

  // what the swarm has marked: fires (incl. the flare-up) and LiDAR-detected debris
  const [seen, setSeen] = useState<Fire[]>([])
  const [found, setFound] = useState<DebrisBlock[]>([])
  const lastV = useRef(-1)
  const lastT = useRef(0)
  useFrame(({ clock }) => {
    const b = sim.belief
    if (!b || b.version === lastV.current || clock.elapsedTime - lastT.current < 1) return
    lastV.current = b.version
    lastT.current = clock.elapsedTime
    const f = beliefFires(b)
    setSeen((old) => (sameFires(old, f) ? old : f))
    const d = beliefDebris(b)
    setFound((old) => (old.length === d.length ? old : d))
  })

  const clickKey = useSim((s) => (s.state?.click_handles ?? []).map((c) => c.id).join(','))
  const clicks = useMemo(() => sim.snap.state?.click_handles ?? [], [clickKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const fires = useMemo(
    () =>
      mergeFires([
        hazards?.fires ?? [],
        clicks.filter((c) => c.kind === 'fire').map((c) => ({ x: c.x, y: c.y, size: 1.5 })),
        seen,
      ]),
    [hazards, clicks, seen],
  )

  if (MODE === 'off' || !w || !h) return null
  const lite = MODE === 'lite'
  return (
    <group name="dressing">
      <Atmosphere width={w} height={h} nodes={graphNodes} fog={lite ? 0.025 : 0.045} />
      {grid && !lite && <GroundMist grid={grid} />}
      {fires.map((f, i) => (
        <FireFX key={`f${i}-${f.x.toFixed(1)}-${f.y.toFixed(1)}`} x={f.x} y={f.y} size={f.size} smoke={!lite} />
      ))}
      {(reveal === 'world' ? (hazards?.piles.flat() ?? []) : found).map((b) => (
        <RubblePile key={`p${b.seed}`} x={b.x} y={b.y} yaw={b.yaw} seed={b.seed} dust={!lite} />
      ))}
      {clicks
        .filter((c) => c.kind === 'debris')
        .map((c) => (
          <RubblePile key={c.id} x={c.x} y={c.y} seed={[...c.id].reduce((a, ch) => a + ch.charCodeAt(0), 0)} dust={!lite} />
        ))}
    </group>
  )
}
