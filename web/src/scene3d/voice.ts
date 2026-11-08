// One speech queue for the firefighter's headset (Act 2): the HUD guidance voice and the drones' voices.
// Rules, so the headset never lags behind what is happening:
// - Lines play one after another, never on top of each other.
// - Danger lines stop everything and play at once; warnings jump the queue.
// - Every line has a shelf life: a line that waited too long is dropped, not played late.
// - Only the newest HUD line waits (the HUD describes the current state); at most MAX_QUEUE lines wait.
// - The same HUD line is not repeated within REPEAT_MS ("Follow me." flip-flopping).
// - finish(): "Mission complete" clears the queue, plays, and closes the headset until reset() (next run).
// A drone line with Gemini TTS audio plays that audio; otherwise it is spoken with a local voice, a
// different pitch per drone so they still sound like different people.
import { noEmDash } from '../ui/text'

export interface Utterance {
  text: string
  speaker: string | null // 'D1'..'D4', or null for the HUD voice
  severity: 'info' | 'warn' | 'danger'
  audio?: { b64: string; mime: string } | null
}

interface Queued extends Utterance {
  at: number // performance.now() when queued
}

// local-voice fallback per drone: [pitch, rate, voice index offset]
const DRONE_STYLE: Record<string, [number, number, number]> = {
  D1: [0.8, 1.05, 0],
  D2: [1.15, 1.1, 1],
  D3: [1.35, 1.15, 2],
  D4: [1.0, 1.05, 3],
}
const MAX_QUEUE = 3
const REPEAT_MS = 8000
const LINE_TIMEOUT_MS = 10_000 // a stuck utterance (browser bug) never blocks the queue

/** How long a line may wait before it is no longer worth saying. */
function shelfLife(u: Utterance): number {
  if (u.severity === 'danger') return 2000
  if (u.speaker === null) return u.severity === 'warn' ? 3000 : 2500 // HUD: the next state replaces it
  return u.severity === 'warn' ? 3500 : 5000 // drones
}

function firstSentence(s: string): string {
  return s.split('.')[0].trim().toLowerCase() // "Follow the drone. 63 metres to go." -> "follow the drone"
}

function localVoices(): SpeechSynthesisVoice[] {
  if (!('speechSynthesis' in window)) return []
  const all = window.speechSynthesis.getVoices().filter((v) => v.localService) // never a network voice
  const en = all.filter((v) => /^en[-_]/i.test(v.lang))
  return en.length ? en : all
}

function hudVoice(): SpeechSynthesisVoice | undefined {
  const voices = localVoices()
  return voices.find((v) => /en[-_]GB/i.test(v.lang)) ?? voices.find((v) => /en[-_]US/i.test(v.lang)) ?? voices[0]
}

class VoiceQueue {
  private q: Queued[] = []
  private busy = false
  private audio: HTMLAudioElement | null = null
  private token = 0
  private closed = false
  private lastHud = new Map<string, number>() // first sentence -> when last spoken

  push(u: Utterance) {
    if (this.closed) return
    const now = performance.now()
    if (u.speaker === null && u.severity !== 'danger') {
      const last = this.lastHud.get(firstSentence(u.text))
      if (last !== undefined && now - last < REPEAT_MS) return // just said this
      this.q = this.q.filter((x) => x.speaker !== null) // only the newest HUD line waits
    }
    const item: Queued = { ...u, at: now }
    if (u.severity === 'danger') {
      this.stop()
      this.q = [item]
    } else if (u.severity === 'warn') {
      this.q.unshift(item)
    } else {
      this.q.push(item)
    }
    while (this.q.length > MAX_QUEUE) {
      const i = this.q.findIndex((x) => x.severity === 'info') // drop the oldest routine line
      this.q.splice(i >= 0 ? i : this.q.length - 1, 1)
    }
    this.next()
  }

  /** Mission complete: say this line now, then nothing more until reset(). */
  finish(u: Utterance) {
    if (this.closed) return
    this.stop()
    this.q = [{ ...u, at: performance.now() }]
    this.next()
    this.closed = true
  }

  /** A new run (snapshot load / new session): open the headset again, forget what was said. */
  reset() {
    this.stop()
    this.q = []
    this.closed = false
    this.lastHud.clear()
  }

  stop() {
    this.token++
    if ('speechSynthesis' in window) window.speechSynthesis.cancel()
    if (this.audio) {
      this.audio.onended = null
      this.audio.onerror = null
      this.audio.pause()
      this.audio = null
    }
    this.busy = false
  }

  private next() {
    if (this.busy) return
    const now = performance.now()
    let u: Queued | undefined
    while ((u = this.q.shift()) && now - u.at > shelfLife(u)) {
      /* too old: skip */
    }
    if (!u) return
    if (u.speaker === null) this.lastHud.set(firstSentence(u.text), now)
    this.busy = true
    const token = ++this.token
    let finished = false
    const done = () => {
      if (finished || token !== this.token) return
      finished = true
      this.audio = null
      this.busy = false
      this.next()
    }
    window.setTimeout(done, LINE_TIMEOUT_MS)
    if (u.audio) {
      const line = u
      const a = new Audio(`data:${u.audio.mime};base64,${u.audio.b64}`)
      let fellBack = false
      const fallback = () => {
        if (fellBack || token !== this.token) return
        fellBack = true
        this.speakLocal(line, done)
      }
      this.audio = a
      a.onended = done
      a.onerror = fallback
      a.play().catch(fallback) // autoplay blocked: local voice
    } else {
      this.speakLocal(u, done)
    }
  }

  private speakLocal(u: Utterance, done: () => void) {
    this.audio = null
    if (!('speechSynthesis' in window)) {
      done()
      return
    }
    const ut = new SpeechSynthesisUtterance(noEmDash(u.text))
    const style = u.speaker ? DRONE_STYLE[u.speaker] : undefined
    if (style) {
      const voices = localVoices()
      if (voices.length) ut.voice = voices[style[2] % voices.length]
      ut.pitch = style[0]
      ut.rate = style[1]
    } else {
      const v = hudVoice()
      if (v) ut.voice = v
      ut.rate = 1.1
    }
    ut.onend = done
    ut.onerror = done
    window.speechSynthesis.speak(ut)
  }
}

export const headset = new VoiceQueue()
