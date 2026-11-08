// One speech queue for the firefighter's headset (Act 2): the HUD guidance voice and the drones' voices.
// - Lines play one after another, never on top of each other.
// - Danger lines (severity 'danger') stop everything and play at once; warnings jump the queue.
// - A drone line with Gemini TTS audio plays that audio; otherwise (offline / no key / TTS late) it is spoken
//   with a local voice, a different pitch per drone so they still sound like different people.
import { noEmDash } from '../ui/text'

export interface Utterance {
  text: string
  speaker: string | null // 'D1'..'D4', or null for the HUD voice
  severity: 'info' | 'warn' | 'danger'
  audio?: { b64: string; mime: string } | null
}

// local-voice fallback per drone: [pitch, rate, voice index offset]
const DRONE_STYLE: Record<string, [number, number, number]> = {
  D1: [0.8, 0.95, 0],
  D2: [1.15, 1.0, 1],
  D3: [1.35, 1.1, 2],
  D4: [1.0, 0.95, 3],
}
const MAX_QUEUE = 6
const LINE_TIMEOUT_MS = 12_000 // a stuck utterance (browser bug) never blocks the queue

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
  private q: Utterance[] = []
  private busy = false
  private audio: HTMLAudioElement | null = null
  private token = 0

  push(u: Utterance) {
    if (u.severity === 'danger') {
      this.stop()
      this.q = [u]
    } else if (u.severity === 'warn') {
      this.q.unshift(u)
    } else {
      this.q.push(u)
    }
    if (this.q.length > MAX_QUEUE) this.q = this.q.slice(0, MAX_QUEUE)
    this.next()
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
    const u = this.q.shift()
    if (!u) return
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
      const a = new Audio(`data:${u.audio.mime};base64,${u.audio.b64}`)
      let fellBack = false
      const fallback = () => {
        if (fellBack || token !== this.token) return
        fellBack = true
        this.speakLocal(u, done)
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
    }
    ut.onend = done
    ut.onerror = done
    window.speechSynthesis.speak(ut)
  }
}

export const headset = new VoiceQueue()
