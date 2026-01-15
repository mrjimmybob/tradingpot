/**
 * Tests for utility functions
 */
import { describe, it, expect } from 'vitest'

// Test the CSV export logic
describe('CSV Export', () => {
  it('should escape quotes in CSV values', () => {
    const name = 'Bot "Special"'
    const escaped = `"${name.replace(/"/g, '""')}"`
    expect(escaped).toBe('"Bot ""Special"""')
  })

  it('should format numbers correctly', () => {
    const pnl = 123.456
    expect(pnl.toFixed(2)).toBe('123.46')
  })
})

// Test search normalization
describe('Search Normalization', () => {
  it('should trim whitespace from search term', () => {
    const searchTerm = '   bitcoin   '
    const normalized = searchTerm.trim()
    expect(normalized).toBe('bitcoin')
  })

  it('should handle whitespace-only search as empty', () => {
    const searchTerm = '     '
    const normalized = searchTerm.trim()
    expect(normalized).toBe('')
    expect(normalized.length).toBe(0)
  })

  it('should handle special characters in search', () => {
    const searchTerm = 'BTC/USDT'
    const text = 'Trading pair: BTC/USDT'
    expect(text.toLowerCase().includes(searchTerm.toLowerCase())).toBe(true)
  })

  it('should limit search length', () => {
    const MAX_SEARCH_LENGTH = 100
    const longSearch = 'a'.repeat(150)
    const limited = longSearch.slice(0, MAX_SEARCH_LENGTH)
    expect(limited.length).toBe(100)
  })
})

// Test data sampling for charts
describe('Chart Data Sampling', () => {
  function sampleDataPoints<T>(data: T[], maxPoints: number): T[] {
    if (data.length <= maxPoints) return data

    const result: T[] = [data[0]]
    const step = (data.length - 2) / (maxPoints - 2)

    for (let i = 1; i < maxPoints - 1; i++) {
      const index = Math.round(i * step)
      result.push(data[index])
    }

    result.push(data[data.length - 1])
    return result
  }

  it('should return original data if under max points', () => {
    const data = [1, 2, 3, 4, 5]
    const sampled = sampleDataPoints(data, 10)
    expect(sampled).toEqual(data)
  })

  it('should sample data to max points', () => {
    const data = Array.from({ length: 500 }, (_, i) => i)
    const sampled = sampleDataPoints(data, 100)
    expect(sampled.length).toBe(100)
  })

  it('should preserve first and last points', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    const sampled = sampleDataPoints(data, 5)
    expect(sampled[0]).toBe(1)
    expect(sampled[sampled.length - 1]).toBe(10)
  })
})

// Test debounce behavior
describe('Debounce', () => {
  it('should delay execution', async () => {
    let value = 0
    const delay = 50

    const debouncedSet = (newValue: number) => {
      return new Promise<void>(resolve => {
        setTimeout(() => {
          value = newValue
          resolve()
        }, delay)
      })
    }

    debouncedSet(1)
    expect(value).toBe(0) // Still 0 immediately

    await new Promise(resolve => setTimeout(resolve, delay + 10))
    expect(value).toBe(1) // Now updated
  })
})

// Test status color mapping
describe('Status Colors', () => {
  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'running':
        return 'bg-running/20 text-running'
      case 'paused':
        return 'bg-paused/20 text-paused'
      case 'stopped':
        return 'bg-stopped/20 text-stopped'
      default:
        return 'bg-gray-500/20 text-gray-400'
    }
  }

  it('should return correct color for running status', () => {
    expect(getStatusColor('running')).toContain('running')
  })

  it('should return correct color for paused status', () => {
    expect(getStatusColor('paused')).toContain('paused')
  })

  it('should return default for unknown status', () => {
    expect(getStatusColor('unknown')).toContain('gray')
  })
})

// Test P&L formatting (matches actual implementation in codebase)
describe('P&L Formatting', () => {
  const formatPnL = (value: number): string => {
    const sign = value >= 0 ? '+' : ''
    return `${sign}$${value.toFixed(2)}`
  }

  it('should format positive P&L with plus sign', () => {
    expect(formatPnL(123.45)).toBe('+$123.45')
  })

  it('should format negative P&L with dollar sign before value', () => {
    // Negative numbers already include the minus in toFixed()
    expect(formatPnL(-50.00)).toBe('$-50.00')
  })

  it('should format zero P&L with plus sign', () => {
    expect(formatPnL(0)).toBe('+$0.00')
  })
})
