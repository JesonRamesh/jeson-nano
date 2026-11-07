import { useEffect, useState } from 'react'
import { useSim, type SimEvent, type SimEventMsg } from '../state/store'

const SHOW_MS = 4500
const MAX_SHOWN = 3

type Tone = 'danger' | 'warn' | 'good'

/** Short notice for each event kind (spec §10.1 "banners for key events", kept small). */
function bannerFor(e: SimEventMsg): { text: string; tone: Tone } {
  const d = e.drone ?? ''
  switch (e.kind) {
    case 'fire_confirmed':
      return { text: `Fire confirmed${d && ` by ${d}`}`, tone: 'danger' }
    case 'false_alarm':
      return { text: `False alarm corrected${d && ` (${d})`}`, tone: 'good' }
    case 'drone_lost':
      return { text: `${d || 'Drone'} lost`, tone: 'danger' }
    case 'blocked':
      return { text: 'Route blocked, replanning', tone: 'warn' }
    case 'replan':
      return { text: 'New route', tone: 'good' }
    case 'survivor_found':
      return { text: 'Survivor found', tone: 'good' }
    case 'chain_formed':
      return { text: 'Relay chain formed', tone: 'good' }
    case 'no_route':
      return { text: 'No safe route, holding', tone: 'danger' }
    case 'mission_complete':
      return { text: 'Mission complete', tone: 'good' }
    case 'survivor_check':
      return { text: `Possible survivor, ${d || 'drone'} taking a closer look`, tone: 'warn' }
  }
}

/** Small notices for the latest events (top right, newest first); each stays 4.5 s. */
export function Banners() {
  const events = useSim((s) => s.events)
  const [now, setNow] = useState(() => performance.now())
  const newest = events.length ? events[events.length - 1].at : 0

  // Tick while something is on screen so banners expire. (A stale `now` is fine for
  // a brand-new event: now - at is negative, so it shows immediately.)
  useEffect(() => {
    if (!newest) return
    const id = setInterval(() => setNow(performance.now()), 250)
    const stop = setTimeout(() => clearInterval(id), SHOW_MS + 300)
    return () => {
      clearInterval(id)
      clearTimeout(stop)
    }
  }, [newest])

  const shown: SimEvent[] = events.filter((e) => now - e.at < SHOW_MS).slice(-MAX_SHOWN).reverse()
  return (
    <div className="banners">
      {shown.map((e) => {
        const b = bannerFor(e.msg)
        return (
          <div key={e.seq} className={`banner banner-${b.tone}`}>
            {b.text}
          </div>
        )
      })}
    </div>
  )
}
