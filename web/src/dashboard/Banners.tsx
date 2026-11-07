import { useEffect, useState } from 'react'
import { useSim, type SimEvent, type SimEventMsg } from '../state/store'

const SHOW_MS = 4500
const MAX_SHOWN = 2

type Tone = 'danger' | 'warn' | 'good'

/** Banner text for each event kind (spec §10.1 "Banners for key events"). */
function bannerFor(e: SimEventMsg): { text: string; tone: Tone } {
  const who = e.drone ? ` — ${e.drone}` : ''
  switch (e.kind) {
    case 'fire_confirmed':
      return { text: `FIRE CONFIRMED${who}`, tone: 'danger' }
    case 'false_alarm':
      return { text: `FALSE ALARM CORRECTED${who}`, tone: 'good' }
    case 'drone_lost':
      return { text: `${e.drone ?? 'DRONE'} SIGNAL LOST`, tone: 'danger' }
    case 'blocked':
      return { text: 'ROUTE BLOCKED — RE-PLANNING', tone: 'warn' }
    case 'replan':
      return { text: 'NEW ROUTE', tone: 'good' }
    case 'survivor_found':
      return { text: 'SURVIVOR FOUND', tone: 'good' }
    case 'chain_formed':
      return { text: 'CHECKPOINT CHAIN FORMED', tone: 'good' }
    case 'no_route':
      return { text: 'NO SAFE ROUTE — HOLDING', tone: 'danger' }
    case 'mission_complete':
      return { text: 'MISSION COMPLETE', tone: 'good' }
    case 'survivor_check':
      return { text: `POSSIBLE SURVIVOR — ${e.drone ?? 'DRONE'} TAKING A CLOSER LOOK`, tone: 'warn' }
  }
}

/** Big banners for the latest events; each stays for 4.5 s. */
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
