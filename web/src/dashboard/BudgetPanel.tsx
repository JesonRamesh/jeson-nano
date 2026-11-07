import { useSim } from '../state/store'

const n = (v: number | null | undefined, digits = 1, unit = '') => (v == null ? 'N/A' : `${v.toFixed(digits)}${unit}`)

/** Live budget panel (§7.5). Values come from the `budget` message (1 Hz). */
export function BudgetPanel() {
  const b = useSim((s) => s.budget)
  return (
    <section className="budget-panel">
      <h2>Budget</h2>
      {!b ? (
        <p className="muted">waiting for budget…</p>
      ) : (
        <dl>
          <dt>Network</dt>
          <dd className={b.network === 'offline' ? 'ok' : b.network === 'online' ? 'bad' : 'muted'}>
            {b.network === 'offline' ? 'OFFLINE ✓' : b.network === 'online' ? 'ONLINE ⚠' : 'unknown'}
          </dd>
          <dt>Model</dt>
          <dd className="mono">{b.model_tag ?? 'N/A'}</dd>
          <dt>Gemma p50 / p95</dt>
          <dd>
            {n(b.gemma_p50_s, 1, ' s')} / {n(b.gemma_p95_s, 1, ' s')}
          </dd>
          <dt>Queue</dt>
          <dd>{b.gemma_queue}</dd>
          <dt>VRAM</dt>
          <dd>
            {b.vram_used_mb == null ? 'N/A' : `${(b.vram_used_mb / 1024).toFixed(1)} / ${n(b.vram_total_mb && b.vram_total_mb / 1024, 1)} GB`}
          </dd>
          <dt>Sim</dt>
          <dd>
            RTF {n(b.sim_rtf, 2)} · {b.speed}×
          </dd>
          <dt>Rates</dt>
          <dd>
            ctrl {n(b.control_hz, 0, ' Hz')} · cam {n(b.camera_fps, 0, ' fps')}
          </dd>
        </dl>
      )}
    </section>
  )
}
