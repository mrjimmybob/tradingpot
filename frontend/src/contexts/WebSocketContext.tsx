/**
 * WebSocket context provider for real-time updates.
 *
 * Provides global WebSocket connection and real-time data to all components.
 */
import { createContext, useContext, useState, useCallback, ReactNode, useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import {
  useWebSocket,
  PriceUpdate,
  IndicatorUpdate,
  BotUpdate,
  StatsUpdate,
} from '../hooks/useWebSocket'

interface PriceData {
  price: number
  bid: number
  ask: number
  change_24h: number
  timestamp: string
}

interface IndicatorData {
  sentiment: number
  risk: number
  signal: 'bullish' | 'bearish' | 'neutral' | 'avoid'
  orderbook_imbalance: number
  volume_delta: number
  spread_percent: number
  volatility_regime: 'low' | 'normal' | 'high' | 'extreme'
  timestamp: string
}

interface BotData {
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

interface StatsData {
  total_bots: number
  running_bots: number
  total_pnl: number
  active_trades: number
  timestamp: string
}

interface WebSocketContextType {
  // Connection state
  isConnected: boolean
  isConnecting: boolean
  reconnect: () => void

  // Subscription management
  subscribe: (symbols: string[]) => void
  unsubscribe: (symbols: string[]) => void

  // Real-time data
  prices: Map<string, PriceData>
  indicators: Map<string, IndicatorData>
  bots: Map<number, BotData>
  stats: StatsData | null

  // Helpers
  getPrice: (symbol: string) => PriceData | undefined
  getIndicator: (symbol: string) => IndicatorData | undefined
  getBotData: (botId: number) => BotData | undefined
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined)

export function useWebSocketContext() {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocketContext must be used within a WebSocketProvider')
  }
  return context
}

interface WebSocketProviderProps {
  children: ReactNode
}

export function WebSocketProvider({ children }: WebSocketProviderProps) {
  const queryClient = useQueryClient()

  // Real-time data stores
  const [prices, setPrices] = useState<Map<string, PriceData>>(new Map())
  const [indicators, setIndicators] = useState<Map<string, IndicatorData>>(new Map())
  const [bots, setBots] = useState<Map<number, BotData>>(new Map())
  const [stats, setStats] = useState<StatsData | null>(null)

  // Handle price updates
  const handlePriceUpdate = useCallback((update: PriceUpdate) => {
    setPrices((prev) => {
      const next = new Map(prev)
      next.set(update.symbol, {
        price: update.price,
        bid: update.bid,
        ask: update.ask,
        change_24h: update.change_24h,
        timestamp: update.timestamp,
      })
      return next
    })
  }, [])

  // Handle indicator updates
  const handleIndicatorUpdate = useCallback((update: IndicatorUpdate) => {
    setIndicators((prev) => {
      const next = new Map(prev)
      next.set(update.symbol, {
        sentiment: update.sentiment,
        risk: update.risk,
        signal: update.signal,
        orderbook_imbalance: update.orderbook_imbalance,
        volume_delta: update.volume_delta,
        spread_percent: update.spread_percent,
        volatility_regime: update.volatility_regime,
        timestamp: update.timestamp,
      })
      return next
    })
  }, [])

  // Handle bot updates
  const handleBotUpdate = useCallback((update: BotUpdate) => {
    setBots((prev) => {
      const next = new Map(prev)
      next.set(update.bot_id, {
        status: update.status,
        pnl: update.pnl,
        current_balance: update.current_balance,
        positions: update.positions,
        timestamp: update.timestamp,
      })
      return next
    })

    // Invalidate bot query to trigger UI update
    queryClient.invalidateQueries({ queryKey: ['bot', update.bot_id] })
  }, [queryClient])

  // Handle stats updates
  const handleStatsUpdate = useCallback((update: StatsUpdate) => {
    setStats({
      total_bots: update.total_bots,
      running_bots: update.running_bots,
      total_pnl: update.total_pnl,
      active_trades: update.active_trades,
      timestamp: update.timestamp,
    })

    // Invalidate stats query
    queryClient.invalidateQueries({ queryKey: ['stats'] })
  }, [queryClient])

  // WebSocket connection
  const { isConnected, isConnecting, subscribe, unsubscribe, reconnect } = useWebSocket({
    onPriceUpdate: handlePriceUpdate,
    onIndicatorUpdate: handleIndicatorUpdate,
    onBotUpdate: handleBotUpdate,
    onStatsUpdate: handleStatsUpdate,
    onOpen: () => {
      // #148: Use debug logging only in development
      if (import.meta.env.DEV) {
        console.debug('[WS] Connected')
      }
      // Subscribe to all data by default
      subscribe(['*'])
    },
    onClose: () => {
      // #148: Use debug logging only in development
      if (import.meta.env.DEV) {
        console.debug('[WS] Disconnected')
      }
    },
  })

  // Helpers
  const getPrice = useCallback(
    (symbol: string) => prices.get(symbol),
    [prices]
  )

  const getIndicator = useCallback(
    (symbol: string) => indicators.get(symbol),
    [indicators]
  )

  const getBotData = useCallback(
    (botId: number) => bots.get(botId),
    [bots]
  )

  const value: WebSocketContextType = {
    isConnected,
    isConnecting,
    reconnect,
    subscribe,
    unsubscribe,
    prices,
    indicators,
    bots,
    stats,
    getPrice,
    getIndicator,
    getBotData,
  }

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}

/**
 * Hook for real-time price updates for a specific symbol.
 */
export function useRealtimePrice(symbol: string) {
  const { prices, subscribe, unsubscribe } = useWebSocketContext()

  useEffect(() => {
    subscribe([symbol])
    return () => {
      unsubscribe([symbol])
    }
  }, [symbol, subscribe, unsubscribe])

  return prices.get(symbol)
}

/**
 * Hook for real-time bot status updates.
 */
export function useRealtimeBot(botId: number) {
  const { bots } = useWebSocketContext()
  return bots.get(botId)
}

/**
 * Hook for real-time global stats.
 */
export function useRealtimeStats() {
  const { stats } = useWebSocketContext()
  return stats
}

/**
 * Hook for market indicators for a symbol.
 */
export function useRealtimeIndicators(symbol: string) {
  const { indicators, subscribe, unsubscribe } = useWebSocketContext()

  useEffect(() => {
    subscribe([symbol])
    return () => {
      unsubscribe([symbol])
    }
  }, [symbol, subscribe, unsubscribe])

  return indicators.get(symbol)
}

export default WebSocketProvider
