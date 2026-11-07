import { useSyncExternalStore } from 'react'
import { noEmDash } from './text'

export interface Toast {
  id: number
  text: string
  kind: 'info' | 'warn' | 'error'
}

let toasts: Toast[] = []
let nextId = 1
const listeners = new Set<() => void>()
const emit = () => listeners.forEach((fn) => fn())

/** Show a short message in the corner for `ms` milliseconds. */
export function toast(text: string, kind: Toast['kind'] = 'info', ms = 3500) {
  const t = { id: nextId++, text: noEmDash(text), kind }
  toasts = [...toasts.slice(-3), t]
  emit()
  setTimeout(() => {
    toasts = toasts.filter((x) => x.id !== t.id)
    emit()
  }, ms)
}

export function useToasts(): Toast[] {
  return useSyncExternalStore(
    (fn) => {
      listeners.add(fn)
      return () => {
        listeners.delete(fn)
      }
    },
    () => toasts,
  )
}
