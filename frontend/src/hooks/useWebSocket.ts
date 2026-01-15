/**
 * WebSocket hook for real-time updates.
 *
 * Provides:
 * - Connection management with auto-reconnect
 * - Price updates
 * - Bot status updates
 * - Market indicators
 * - Global stats updates
 */
import { useEffect, useRef, useCallback, useState } from 'react'

export type MessageType =
  | 'price_update'
  | 'indicator_update'
  | 'bot_update'
  | 'stats_update'
  | 'pong'

export interface PriceUpdate {
  type: 'price_update'
  symbol: string
  price: number
  bid: number
  ask: number
  change_24h: number
  timestamp: string
}

export interface IndicatorUpdate {
  type: 'indicator_update'
  symbol: string
  sentiment: number
  risk: number
  signal: 'bullish' | 'bearish' | 'neutral' | 'avoid'
  orderbook_imbalance: number
  volume_delta: number
  spread_percent: number
  volatility_regime: 'low' | 'normal' | 'high' | 'extreme'
  timestamp: string
}

export interface BotUpdate {
  type: 'bot_update'
  bot_id: number
  status: string
  pnl: number
  current_balance: number
  positions: Array<{
    trading_pair: string
    side: string
    entry_price: number
    current_price: number
    amount: number
    unrealized_pnl: number
  }>
  timestamp: string
}

export interface StatsUpdate {
  type: 'stats_update'
  total_bots: number
  running_bots: number
  total_pnl: number
  active_trades: number
  timestamp: string
}

export type WebSocketMessage = PriceUpdate | IndicatorUpdate | BotUpdate | StatsUpdate

interface UseWebSocketOptions {
  url?: string
  reconnectAttempts?: number
  reconnectInterval?: number
  onOpen?: () => void
  onClose?: () => void
  onError?: (error: Event) => void
  onMessage?: (message: WebSocketMessage) => void
  onPriceUpdate?: (update: PriceUpdate) => void
  onIndicatorUpdate?: (update: IndicatorUpdate) => void
  onBotUpdate?: (update: BotUpdate) => void
  onStatsUpdate?: (update: StatsUpdate) => void
}

interface UseWebSocketReturn {
  isConnected: boolean
  isConnecting: boolean
  subscribe: (symbols: string[]) => void
  unsubscribe: (symbols: string[]) => void
  sendMessage: (message: object) => void
  reconnect: () => void
}

const DEFAULT_WS_URL = `ws://${window.location.hostname}:8000/api/ws`

export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketReturn {
  const {
    url = DEFAULT_WS_URL,
    reconnectAttempts = 10,
    reconnectInterval = 3000,
    onOpen,
    onClose,
    onError,
    onMessage,
    onPriceUpdate,
    onIndicatorUpdate,
    onBotUpdate,
    onStatsUpdate,
  } = options

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectCountRef = useRef(0)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)

  const clearTimers = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = null
    }
  }, [])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    setIsConnecting(true)

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        setIsConnected(true)
        setIsConnecting(false)
        reconnectCountRef.current = 0

        // Start ping interval
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: 'ping' }))
          }
        }, 30000)

        onOpen?.()
      }

      ws.onclose = () => {
        setIsConnected(false)
        setIsConnecting(false)
        clearTimers()

        onClose?.()

        // Attempt reconnect
        if (reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current += 1
          const delay = reconnectInterval * Math.pow(1.5, reconnectCountRef.current - 1)
          reconnectTimeoutRef.current = setTimeout(connect, Math.min(delay, 30000))
        }
      }

      ws.onerror = (error) => {
        onError?.(error)
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage

          onMessage?.(message)

          // Route to specific handlers
          switch (message.type) {
            case 'price_update':
              onPriceUpdate?.(message as PriceUpdate)
              break
            case 'indicator_update':
              onIndicatorUpdate?.(message as IndicatorUpdate)
              break
            case 'bot_update':
              onBotUpdate?.(message as BotUpdate)
              break
            case 'stats_update':
              onStatsUpdate?.(message as StatsUpdate)
              break
          }
        } catch (e) {
          // #148: Only log parse errors in development
          if (import.meta.env.DEV) {
            console.debug('[WS] Message parse error:', e)
          }
        }
      }
    } catch (e) {
      setIsConnecting(false)
      // #148: Only log connection errors in development
      if (import.meta.env.DEV) {
        console.debug('[WS] Connection error:', e)
      }
    }
  }, [
    url,
    reconnectAttempts,
    reconnectInterval,
    clearTimers,
    onOpen,
    onClose,
    onError,
    onMessage,
    onPriceUpdate,
    onIndicatorUpdate,
    onBotUpdate,
    onStatsUpdate,
  ])

  const disconnect = useCallback(() => {
    clearTimers()
    reconnectCountRef.current = reconnectAttempts // Prevent auto-reconnect

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [clearTimers, reconnectAttempts])

  const sendMessage = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    }
  }, [])

  const subscribe = useCallback(
    (symbols: string[]) => {
      sendMessage({ action: 'subscribe', symbols })
    },
    [sendMessage]
  )

  const unsubscribe = useCallback(
    (symbols: string[]) => {
      sendMessage({ action: 'unsubscribe', symbols })
    },
    [sendMessage]
  )

  const reconnect = useCallback(() => {
    reconnectCountRef.current = 0
    disconnect()
    connect()
  }, [connect, disconnect])

  // Connect on mount
  useEffect(() => {
    connect()
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    isConnected,
    isConnecting,
    subscribe,
    unsubscribe,
    sendMessage,
    reconnect,
  }
}

export default useWebSocket
