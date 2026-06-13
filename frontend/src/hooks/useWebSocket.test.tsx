/**
 * Regression test for the WebSocket connect/disconnect thrash.
 *
 * Bug: WebSocketProvider passes brand-new inline onOpen/onClose closures on
 * every render. When connect()/disconnect() depended on those callbacks they
 * changed identity each render, so the mount effect tore down and recreated
 * the socket continuously — the UI flapped Connecting <-> Disconnected and the
 * backend logged connection open/closed with no error.
 *
 * This test lives under src/ on purpose: the vitest `include` glob only runs
 * src/**, so tests under frontend/tests/** never execute.
 */
import { renderHook, act } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useWebSocket } from './useWebSocket'

class MockWebSocket {
  static instances: MockWebSocket[] = []
  static opened = 0
  static closed = 0
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3

  readyState = 0
  onopen: ((ev: Event) => void) | null = null
  onclose: ((ev: Event) => void) | null = null
  onerror: ((ev: Event) => void) | null = null
  onmessage: ((ev: MessageEvent) => void) | null = null

  constructor(public url: string) {
    MockWebSocket.instances.push(this)
    // Open asynchronously, like a real socket.
    queueMicrotask(() => {
      this.readyState = MockWebSocket.OPEN
      MockWebSocket.opened += 1
      this.onopen?.(new Event('open'))
    })
  }

  send() {}

  close() {
    if (this.readyState === MockWebSocket.CLOSED) return
    this.readyState = MockWebSocket.CLOSED
    MockWebSocket.closed += 1
    this.onclose?.(new Event('close'))
  }
}

describe('useWebSocket connection stability', () => {
  beforeEach(() => {
    vi.stubGlobal('WebSocket', MockWebSocket)
    // buildWsUrl() reads the stored API token; give it a working store.
    vi.stubGlobal('localStorage', {
      getItem: () => null,
      setItem: () => {},
      removeItem: () => {},
      clear: () => {},
    })
    MockWebSocket.instances = []
    MockWebSocket.opened = 0
    MockWebSocket.closed = 0
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('does not recreate the socket when the caller passes new inline callbacks each render', async () => {
    // Mirrors WebSocketProvider: a fresh onOpen/onClose closure every render.
    const { rerender } = renderHook(() =>
      useWebSocket({
        onOpen: () => {},
        onClose: () => {},
        onPriceUpdate: () => {},
      })
    )

    // Let the initial socket open.
    await act(async () => {
      await Promise.resolve()
    })
    expect(MockWebSocket.instances.length).toBe(1)
    expect(MockWebSocket.instances[0].readyState).toBe(MockWebSocket.OPEN)

    // Force several re-renders, each with brand-new inline callbacks.
    await act(async () => {
      rerender()
      rerender()
      rerender()
      await Promise.resolve()
    })

    // Exactly one socket, still open, never closed: no churn.
    expect(MockWebSocket.instances.length).toBe(1)
    expect(MockWebSocket.closed).toBe(0)
    expect(MockWebSocket.instances[0].readyState).toBe(MockWebSocket.OPEN)
  })
})
