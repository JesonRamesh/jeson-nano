import { useCallback, useEffect, useRef, useState } from 'react'
import { CONTRACT_VERSION, type ClientMsg, type DroneFrame, type HelloMsg, type ServerMsg } from '../contracts'

export type ConnStatus = 'connecting' | 'connected' | 'disconnected'

export interface SocketHandlers {
  onMessage?: (msg: ServerMsg) => void
  onFrame?: (frame: DroneFrame) => void
}

export function parseFrame(buf: ArrayBuffer): DroneFrame | null {
  if (buf.byteLength < 6) return null
  const view = new DataView(buf)
  return {
    channel: view.getUint8(0),
    seq: view.getUint32(1), // big-endian (DataView default)
    jpeg: new Blob([buf.slice(5)], { type: 'image/jpeg' }),
  }
}

function wsUrl(): string {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${location.host}/ws`
}

/**
 * One WebSocket to the sim server with auto-reconnect.
 * Handlers are read through a ref, so callers can pass inline functions
 * without causing reconnects.
 */
export function useSocket(handlers: SocketHandlers) {
  const [status, setStatus] = useState<ConnStatus>('connecting')
  const [hello, setHello] = useState<HelloMsg | null>(null)
  const [versionMismatch, setVersionMismatch] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const handlersRef = useRef(handlers)
  useEffect(() => {
    handlersRef.current = handlers
  })

  useEffect(() => {
    let stopped = false
    let retry: ReturnType<typeof setTimeout> | undefined

    const connect = () => {
      setStatus('connecting')
      const ws = new WebSocket(wsUrl())
      ws.binaryType = 'arraybuffer'
      wsRef.current = ws

      ws.onopen = () => setStatus('connected')
      ws.onclose = () => {
        setStatus('disconnected')
        if (!stopped) retry = setTimeout(connect, 1000)
      }
      ws.onmessage = (ev) => {
        if (ev.data instanceof ArrayBuffer) {
          const frame = parseFrame(ev.data)
          if (frame) handlersRef.current.onFrame?.(frame)
          return
        }
        let msg: ServerMsg
        try {
          msg = JSON.parse(ev.data as string)
        } catch {
          return
        }
        if (msg.type === 'hello') {
          setHello(msg)
          setVersionMismatch(msg.contract_version !== CONTRACT_VERSION)
        }
        handlersRef.current.onMessage?.(msg)
      }
    }

    connect()
    return () => {
      stopped = true
      clearTimeout(retry)
      wsRef.current?.close()
    }
  }, [])

  const send = useCallback((msg: ClientMsg) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg))
  }, [])

  return { status, hello, versionMismatch, send }
}
