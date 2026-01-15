/**
 * Real-time price display component.
 *
 * Shows live price updates via WebSocket with optional change indicator.
 */
import { useEffect, useState, useRef } from 'react'
import { TrendingUp, TrendingDown } from 'lucide-react'
import { useRealtimePrice } from '../contexts/WebSocketContext'

interface RealtimePriceProps {
  symbol: string
  fallbackPrice?: number
  showChange?: boolean
  className?: string
  size?: 'sm' | 'md' | 'lg'
}

export function RealtimePrice({
  symbol,
  fallbackPrice = 0,
  showChange = false,
  className = '',
  size = 'md',
}: RealtimePriceProps) {
  const priceData = useRealtimePrice(symbol)
  const [flash, setFlash] = useState<'up' | 'down' | null>(null)
  const prevPriceRef = useRef<number>(fallbackPrice)

  const price = priceData?.price ?? fallbackPrice

  // Flash animation on price change
  useEffect(() => {
    if (price !== prevPriceRef.current) {
      if (price > prevPriceRef.current) {
        setFlash('up')
      } else if (price < prevPriceRef.current) {
        setFlash('down')
      }
      prevPriceRef.current = price

      const timer = setTimeout(() => setFlash(null), 500)
      return () => clearTimeout(timer)
    }
  }, [price])

  const sizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg font-bold',
  }

  const flashClasses = {
    up: 'animate-pulse text-profit',
    down: 'animate-pulse text-loss',
  }

  const formatPrice = (p: number) => {
    if (p >= 1000) {
      return p.toLocaleString('en-US', { maximumFractionDigits: 2 })
    } else if (p >= 1) {
      return p.toFixed(4)
    } else {
      return p.toFixed(6)
    }
  }

  return (
    <span
      className={`font-mono-numbers transition-colors ${sizeClasses[size]} ${
        flash ? flashClasses[flash] : 'text-white'
      } ${className}`}
    >
      ${formatPrice(price)}
      {showChange && priceData?.change_24h !== undefined && (
        <span
          className={`ml-2 text-xs ${
            priceData.change_24h >= 0 ? 'text-profit' : 'text-loss'
          }`}
        >
          {priceData.change_24h >= 0 ? (
            <TrendingUp size={12} className="inline" />
          ) : (
            <TrendingDown size={12} className="inline" />
          )}
          {priceData.change_24h >= 0 ? '+' : ''}
          {priceData.change_24h.toFixed(2)}%
        </span>
      )}
    </span>
  )
}

/**
 * Real-time bid/ask spread display.
 */
interface RealtimeSpreadProps {
  symbol: string
  className?: string
}

export function RealtimeSpread({ symbol, className = '' }: RealtimeSpreadProps) {
  const priceData = useRealtimePrice(symbol)

  if (!priceData?.bid || !priceData?.ask) {
    return null
  }

  const spread = priceData.ask - priceData.bid
  const spreadPercent = ((spread / priceData.bid) * 100).toFixed(3)

  return (
    <div className={`text-xs text-gray-400 ${className}`}>
      <span className="text-profit">${priceData.bid.toFixed(2)}</span>
      <span className="mx-1">/</span>
      <span className="text-loss">${priceData.ask.toFixed(2)}</span>
      <span className="ml-2 text-gray-500">({spreadPercent}%)</span>
    </div>
  )
}

export default RealtimePrice
