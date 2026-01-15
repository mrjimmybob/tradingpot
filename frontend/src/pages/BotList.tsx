import { Link, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import {
  Plus,
  Search,
  Filter,
  Play,
  Pause,
  Square,
  Trash2,
  AlertCircle,
  Bot,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  ChevronDown,
  WifiOff,
  RefreshCw,
  Download,
} from 'lucide-react'

interface BotListItem {
  id: number
  name: string
  trading_pair: string
  strategy: string
  status: string
  total_pnl: number
  is_dry_run: boolean
  created_at: string
}

type SortField = 'name' | 'trading_pair' | 'strategy' | 'status' | 'total_pnl' | 'created_at'
type SortDirection = 'asc' | 'desc'

const ITEMS_PER_PAGE = 10
const STORAGE_KEY = 'botlist_preferences'
const MAX_SEARCH_LENGTH = 100 // #166: Limit search string length
const SEARCH_DEBOUNCE_MS = 150 // #177: Debounce search for performance

// #174, #175: Export bots to CSV format
function exportToCSV(data: BotListItem[], filename: string): void {
  if (data.length === 0) return

  const headers = ['ID', 'Name', 'Trading Pair', 'Strategy', 'Status', 'Total P&L', 'Dry Run', 'Created At']
  const csvContent = [
    headers.join(','),
    ...data.map(bot => [
      bot.id,
      `"${bot.name.replace(/"/g, '""')}"`, // Escape quotes in name
      bot.trading_pair,
      bot.strategy,
      bot.status,
      bot.total_pnl.toFixed(2),
      bot.is_dry_run ? 'Yes' : 'No',
      bot.created_at,
    ].join(','))
  ].join('\n')

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = filename
  link.click()
  URL.revokeObjectURL(link.href)
}

// #177: Debounce hook for search performance
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value)

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])

  return debouncedValue
}

// #138: Persist user preferences to localStorage
interface UserPreferences {
  sortField: SortField
  sortDirection: SortDirection
  statusFilter: string
}

function loadPreferences(): Partial<UserPreferences> {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      return JSON.parse(stored)
    }
  } catch {
    // Invalid data, ignore
  }
  return {}
}

function savePreferences(prefs: Partial<UserPreferences>): void {
  try {
    const current = loadPreferences()
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...current, ...prefs }))
  } catch {
    // Storage unavailable, ignore
  }
}

// #141: Validate URL parameters
const VALID_SORT_FIELDS: SortField[] = ['name', 'trading_pair', 'strategy', 'status', 'total_pnl', 'created_at']
const VALID_SORT_DIRS: SortDirection[] = ['asc', 'desc']
const VALID_STATUSES = ['', 'running', 'paused', 'stopped', 'created']

function isValidSortField(value: string | null): value is SortField {
  return value !== null && VALID_SORT_FIELDS.includes(value as SortField)
}

function isValidSortDir(value: string | null): value is SortDirection {
  return value !== null && VALID_SORT_DIRS.includes(value as SortDirection)
}

function isValidPage(value: string | null): boolean {
  if (!value) return false
  const num = parseInt(value, 10)
  return !isNaN(num) && num > 0 && String(num) === value
}

function isValidStatus(value: string | null): boolean {
  return value === null || VALID_STATUSES.includes(value)
}

async function fetchBots(status?: string): Promise<BotListItem[]> {
  const url = status ? `/api/bots?status_filter=${status}` : '/api/bots'
  const res = await fetch(url)
  if (!res.ok) {
    if (!navigator.onLine) {
      throw new Error('No internet connection. Please check your network and try again.')
    }
    if (res.status >= 500) {
      throw new Error('Server error. The service is temporarily unavailable. Please try again later.')
    }
    if (res.status === 404) {
      throw new Error('The requested resource was not found.')
    }
    throw new Error('Failed to load bots. Please try again.')
  }
  return res.json()
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'running':
      return 'bg-running/20 text-running'
    case 'paused':
      return 'bg-paused/20 text-paused'
    case 'stopped':
      return 'bg-stopped/20 text-stopped'
    default:
      return 'bg-gray-700 text-gray-400'
  }
}

function SortIcon({ field, sortField, sortDirection }: {
  field: SortField
  sortField: SortField
  sortDirection: SortDirection
}) {
  if (field !== sortField) {
    return <ChevronUp size={14} className="text-gray-600" />
  }
  return sortDirection === 'asc'
    ? <ChevronUp size={14} className="text-accent" />
    : <ChevronDown size={14} className="text-accent" />
}

export default function BotList() {
  const [searchParams, setSearchParams] = useSearchParams()

  // #138: Load saved preferences, URL params take priority
  const savedPrefs = loadPreferences()

  // #141: Validate URL parameters, fall back to saved preferences (#138) or defaults
  const urlStatus = searchParams.get('status')
  const urlSortBy = searchParams.get('sortBy')
  const urlSortDir = searchParams.get('sortDir')
  const urlPage = searchParams.get('page')

  // Get initial values with validation
  const [statusFilter, setStatusFilter] = useState<string>(
    isValidStatus(urlStatus) ? (urlStatus || '') : (savedPrefs.statusFilter || '')
  )
  const [searchTerm, setSearchTerm] = useState(searchParams.get('search') || '')
  // #165: Normalized search term (trim whitespace for filtering)
  const normalizedSearchTerm = searchTerm.trim()
  // #177: Debounce search for better performance with rapid typing
  const debouncedSearchTerm = useDebounce(normalizedSearchTerm, SEARCH_DEBOUNCE_MS)
  const [currentPage, setCurrentPage] = useState(
    isValidPage(urlPage) ? parseInt(urlPage!, 10) : 1
  )
  const [sortField, setSortField] = useState<SortField>(
    isValidSortField(urlSortBy) ? urlSortBy : (savedPrefs.sortField || 'created_at')
  )
  const [sortDirection, setSortDirection] = useState<SortDirection>(
    isValidSortDir(urlSortDir) ? urlSortDir : (savedPrefs.sortDirection || 'desc')
  )

  // Update URL when filters change (#121)
  useEffect(() => {
    const params = new URLSearchParams()
    if (statusFilter) params.set('status', statusFilter)
    // #165: Only include search in URL if it has non-whitespace content
    if (normalizedSearchTerm) params.set('search', normalizedSearchTerm)
    if (currentPage > 1) params.set('page', currentPage.toString())
    if (sortField !== 'created_at') params.set('sortBy', sortField)
    if (sortDirection !== 'desc') params.set('sortDir', sortDirection)
    setSearchParams(params, { replace: true })
  }, [statusFilter, normalizedSearchTerm, currentPage, sortField, sortDirection, setSearchParams])

  const { data: bots, isLoading, error, refetch } = useQuery({
    queryKey: ['bots', statusFilter],
    queryFn: () => fetchBots(statusFilter || undefined),
    retry: 2,
    retryDelay: 1000,
  })

  // #167: Filter, sort, and paginate work together (#118, #120)
  // #177: Uses debounced search term for better performance
  const processedBots = useMemo(() => {
    if (!bots) return { items: [], totalPages: 0, totalItems: 0, allFiltered: [] }

    // #162, #165, #167: Filter by search term (uses debounced/trimmed search)
    // Special characters work naturally with includes() - no regex needed
    // Spaces-only search is handled by trimming (debouncedSearchTerm)
    let filtered = bots
    if (debouncedSearchTerm) {
      const searchLower = debouncedSearchTerm.toLowerCase()
      filtered = bots.filter((bot) =>
        bot.name.toLowerCase().includes(searchLower) ||
        bot.trading_pair.toLowerCase().includes(searchLower) ||
        bot.strategy.toLowerCase().includes(searchLower)
      )
    }

    // Sort (#120)
    filtered = [...filtered].sort((a, b) => {
      let comparison = 0
      switch (sortField) {
        case 'name':
          comparison = a.name.localeCompare(b.name)
          break
        case 'trading_pair':
          comparison = a.trading_pair.localeCompare(b.trading_pair)
          break
        case 'strategy':
          comparison = a.strategy.localeCompare(b.strategy)
          break
        case 'status':
          comparison = a.status.localeCompare(b.status)
          break
        case 'total_pnl':
          comparison = a.total_pnl - b.total_pnl
          break
        case 'created_at':
          comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
          break
      }
      return sortDirection === 'asc' ? comparison : -comparison
    })

    // Paginate (#118)
    const totalItems = filtered.length
    const totalPages = Math.ceil(totalItems / ITEMS_PER_PAGE)
    const startIndex = (currentPage - 1) * ITEMS_PER_PAGE
    const items = filtered.slice(startIndex, startIndex + ITEMS_PER_PAGE)

    // #174, #175: Return all filtered items (not just paginated) for export
    return { items, totalPages, totalItems, allFiltered: filtered }
  }, [bots, debouncedSearchTerm, sortField, sortDirection, currentPage])

  // Reset pagination when filters change (#119)
  const handleStatusFilterChange = (value: string) => {
    setStatusFilter(value)
    setCurrentPage(1)
    savePreferences({ statusFilter: value }) // #138: Persist preference
  }

  // #162, #165, #166: Handle special characters, spaces-only, and long search strings
  const handleSearchChange = (value: string) => {
    // #166: Limit search string length to prevent performance issues
    const limitedValue = value.slice(0, MAX_SEARCH_LENGTH)
    setSearchTerm(limitedValue)
    setCurrentPage(1)
  }

  const handleSort = (field: SortField) => {
    let newDirection: SortDirection
    if (field === sortField) {
      newDirection = sortDirection === 'asc' ? 'desc' : 'asc'
      setSortDirection(newDirection)
    } else {
      newDirection = 'asc'
      setSortField(field)
      setSortDirection(newDirection)
    }
    setCurrentPage(1)
    // #138: Persist sort preferences
    savePreferences({ sortField: field, sortDirection: newDirection })
  }

  // #174: Export all bots (unfiltered)
  const handleExportAll = useCallback(() => {
    if (bots && bots.length > 0) {
      const timestamp = new Date().toISOString().split('T')[0]
      exportToCSV(bots, `all-bots-${timestamp}.csv`)
    }
  }, [bots])

  // #175: Export filtered bots only
  const handleExportFiltered = useCallback(() => {
    if (processedBots.allFiltered.length > 0) {
      const timestamp = new Date().toISOString().split('T')[0]
      const filterSuffix = debouncedSearchTerm || statusFilter ? '-filtered' : ''
      exportToCSV(processedBots.allFiltered, `bots${filterSuffix}-${timestamp}.csv`)
    }
  }, [processedBots.allFiltered, debouncedSearchTerm, statusFilter])

  const handleDelete = async (id: number, name: string) => {
    if (confirm(`Are you sure you want to delete "${name}"?`)) {
      try {
        const res = await fetch(`/api/bots/${id}`, { method: 'DELETE' })
        if (!res.ok) {
          const data = await res.json()
          throw new Error(data.detail || 'Failed to delete bot')
        }
        refetch()
      } catch (err) {
        alert(err instanceof Error ? err.message : 'Failed to delete bot')
      }
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent"></div>
      </div>
    )
  }

  // User-friendly error message (#110)
  if (error) {
    const isNetworkError = !navigator.onLine || (error instanceof Error && error.message.includes('network'))

    return (
      <div className="flex flex-col items-center justify-center h-64 text-center">
        <div className="bg-gray-800 rounded-lg p-8 max-w-md">
          {isNetworkError ? (
            <WifiOff size={48} className="mx-auto mb-4 text-loss" />
          ) : (
            <AlertCircle size={48} className="mx-auto mb-4 text-loss" />
          )}
          <h3 className="text-xl font-semibold text-white mb-2">
            {isNetworkError ? 'Connection Error' : 'Unable to Load Bots'}
          </h3>
          <p className="text-gray-400 mb-6">
            {error instanceof Error ? error.message : 'An unexpected error occurred. Please try again.'}
          </p>
          <button
            onClick={() => refetch()}
            className="flex items-center gap-2 mx-auto px-6 py-2 bg-accent hover:bg-accent/80 text-white rounded-lg transition-colors"
          >
            <RefreshCw size={18} />
            Try Again
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Bots</h2>
        <div className="flex items-center gap-2">
          {/* #174, #175: Export buttons */}
          {bots && bots.length > 0 && (
            <div className="flex items-center gap-2">
              {(debouncedSearchTerm || statusFilter) && processedBots.allFiltered.length !== bots.length ? (
                <>
                  <button
                    onClick={handleExportFiltered}
                    className="flex items-center gap-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors text-sm"
                    title={`Export ${processedBots.allFiltered.length} filtered bots`}
                  >
                    <Download size={16} />
                    Export Filtered ({processedBots.allFiltered.length})
                  </button>
                  <button
                    onClick={handleExportAll}
                    className="flex items-center gap-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors text-sm"
                    title={`Export all ${bots.length} bots`}
                  >
                    <Download size={16} />
                    Export All ({bots.length})
                  </button>
                </>
              ) : (
                <button
                  onClick={handleExportAll}
                  className="flex items-center gap-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors text-sm"
                  title={`Export all ${bots.length} bots`}
                >
                  <Download size={16} />
                  Export ({bots.length})
                </button>
              )}
            </div>
          )}
          <Link
            to="/bots/new"
            className="flex items-center gap-2 px-4 py-2 bg-accent hover:bg-accent/80 text-white rounded-lg transition-colors"
          >
            <Plus size={18} />
            Create Bot
          </Link>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search
            size={18}
            className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
          />
          {/* #162, #166: Search input with maxLength */}
          <input
            type="text"
            placeholder="Search bots..."
            value={searchTerm}
            onChange={(e) => handleSearchChange(e.target.value)}
            maxLength={MAX_SEARCH_LENGTH}
            className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-accent"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter size={18} className="text-gray-400" />
          <select
            value={statusFilter}
            onChange={(e) => handleStatusFilterChange(e.target.value)}
            className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-accent"
          >
            <option value="">All Status</option>
            <option value="running">Running</option>
            <option value="paused">Paused</option>
            <option value="stopped">Stopped</option>
            <option value="created">Created</option>
          </select>
          {/* #165: Show clear button when there's a real filter (not just whitespace) */}
          {(statusFilter || normalizedSearchTerm) && (
            <button
              onClick={() => {
                setStatusFilter('')
                setSearchTerm('')
                setCurrentPage(1)
              }}
              className="px-3 py-2 text-gray-400 hover:text-white"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Bot Table */}
      {processedBots.items.length > 0 ? (
        <>
          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                {/* #156: Keyboard accessible table headers */}
                <thead>
                  <tr className="text-left text-gray-400 text-sm bg-gray-700/50">
                    <th
                      className="px-6 py-4 cursor-pointer hover:text-white"
                      onClick={() => handleSort('name')}
                      onKeyDown={(e) => e.key === 'Enter' && handleSort('name')}
                      tabIndex={0}
                      role="button"
                      aria-sort={sortField === 'name' ? (sortDirection === 'asc' ? 'ascending' : 'descending') : undefined}
                    >
                      <div className="flex items-center gap-1">
                        Name
                        <SortIcon field="name" sortField={sortField} sortDirection={sortDirection} />
                      </div>
                    </th>
                    <th
                      className="px-6 py-4 cursor-pointer hover:text-white"
                      onClick={() => handleSort('trading_pair')}
                      onKeyDown={(e) => e.key === 'Enter' && handleSort('trading_pair')}
                      tabIndex={0}
                      role="button"
                      aria-sort={sortField === 'trading_pair' ? (sortDirection === 'asc' ? 'ascending' : 'descending') : undefined}
                    >
                      <div className="flex items-center gap-1">
                        Trading Pair
                        <SortIcon field="trading_pair" sortField={sortField} sortDirection={sortDirection} />
                      </div>
                    </th>
                    <th
                      className="px-6 py-4 cursor-pointer hover:text-white"
                      onClick={() => handleSort('strategy')}
                      onKeyDown={(e) => e.key === 'Enter' && handleSort('strategy')}
                      tabIndex={0}
                      role="button"
                      aria-sort={sortField === 'strategy' ? (sortDirection === 'asc' ? 'ascending' : 'descending') : undefined}
                    >
                      <div className="flex items-center gap-1">
                        Strategy
                        <SortIcon field="strategy" sortField={sortField} sortDirection={sortDirection} />
                      </div>
                    </th>
                    <th
                      className="px-6 py-4 cursor-pointer hover:text-white"
                      onClick={() => handleSort('status')}
                      onKeyDown={(e) => e.key === 'Enter' && handleSort('status')}
                      tabIndex={0}
                      role="button"
                      aria-sort={sortField === 'status' ? (sortDirection === 'asc' ? 'ascending' : 'descending') : undefined}
                    >
                      <div className="flex items-center gap-1">
                        Status
                        <SortIcon field="status" sortField={sortField} sortDirection={sortDirection} />
                      </div>
                    </th>
                    <th
                      className="px-6 py-4 text-right cursor-pointer hover:text-white"
                      onClick={() => handleSort('total_pnl')}
                      onKeyDown={(e) => e.key === 'Enter' && handleSort('total_pnl')}
                      tabIndex={0}
                      role="button"
                      aria-sort={sortField === 'total_pnl' ? (sortDirection === 'asc' ? 'ascending' : 'descending') : undefined}
                    >
                      <div className="flex items-center justify-end gap-1">
                        P&L
                        <SortIcon field="total_pnl" sortField={sortField} sortDirection={sortDirection} />
                      </div>
                    </th>
                    <th className="px-6 py-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {processedBots.items.map((bot) => (
                    <tr
                      key={bot.id}
                      className="border-t border-gray-700 hover:bg-gray-700/30"
                    >
                      <td className="px-6 py-4">
                        <Link
                          to={`/bots/${bot.id}`}
                          className="text-white hover:text-accent font-medium"
                        >
                          {bot.name}
                        </Link>
                        {bot.is_dry_run && (
                          <span className="ml-2 px-2 py-0.5 text-xs bg-yellow-500/20 text-yellow-500 rounded">
                            Dry Run
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-4 text-gray-300">{bot.trading_pair}</td>
                      <td className="px-6 py-4 text-gray-300 capitalize">
                        {bot.strategy.replace(/_/g, ' ')}
                      </td>
                      <td className="px-6 py-4">
                        <span
                          className={`px-3 py-1 rounded-full text-sm capitalize ${getStatusColor(
                            bot.status
                          )}`}
                        >
                          {bot.status}
                        </span>
                      </td>
                      {/* #161: P&L with screen reader context (not just color) */}
                      <td
                        className={`px-6 py-4 text-right font-mono-numbers ${
                          bot.total_pnl >= 0 ? 'text-profit' : 'text-loss'
                        }`}
                      >
                        <span className="sr-only">{bot.total_pnl >= 0 ? 'Profit' : 'Loss'}: </span>
                        {bot.total_pnl >= 0 ? '+' : ''}${bot.total_pnl.toFixed(2)}
                      </td>
                      {/* #159: ARIA labels on icon-only buttons */}
                      <td className="px-6 py-4 text-right">
                        <div className="flex items-center justify-end gap-2" role="group" aria-label={`Actions for ${bot.name}`}>
                          {bot.status === 'running' && (
                            <button
                              className="p-2 text-paused hover:bg-gray-700 rounded"
                              aria-label={`Pause ${bot.name}`}
                              title="Pause"
                            >
                              <Pause size={16} aria-hidden="true" />
                            </button>
                          )}
                          {(bot.status === 'paused' || bot.status === 'created') && (
                            <button
                              className="p-2 text-running hover:bg-gray-700 rounded"
                              aria-label={`Start ${bot.name}`}
                              title="Start"
                            >
                              <Play size={16} aria-hidden="true" />
                            </button>
                          )}
                          {bot.status !== 'stopped' && (
                            <button
                              className="p-2 text-stopped hover:bg-gray-700 rounded"
                              aria-label={`Stop ${bot.name}`}
                              title="Stop"
                            >
                              <Square size={16} aria-hidden="true" />
                            </button>
                          )}
                          {bot.status === 'stopped' && (
                            <button
                              onClick={() => handleDelete(bot.id, bot.name)}
                              className="p-2 text-loss hover:bg-gray-700 rounded"
                              aria-label={`Delete ${bot.name}`}
                              title="Delete"
                            >
                              <Trash2 size={16} aria-hidden="true" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pagination (#118) - #159: ARIA labels on pagination buttons */}
          {processedBots.totalPages > 1 && (
            <nav className="flex items-center justify-between bg-gray-800 rounded-lg px-6 py-4" aria-label="Pagination">
              <div className="text-sm text-gray-400">
                Showing {((currentPage - 1) * ITEMS_PER_PAGE) + 1} to {Math.min(currentPage * ITEMS_PER_PAGE, processedBots.totalItems)} of {processedBots.totalItems} bots
              </div>
              <div className="flex items-center gap-2" role="group" aria-label="Page navigation">
                <button
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="p-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors"
                  aria-label="Previous page"
                >
                  <ChevronLeft size={18} aria-hidden="true" />
                </button>
                <div className="flex items-center gap-1">
                  {Array.from({ length: processedBots.totalPages }, (_, i) => i + 1)
                    .filter(page => {
                      // Show first, last, current, and adjacent pages
                      return page === 1 ||
                             page === processedBots.totalPages ||
                             Math.abs(page - currentPage) <= 1
                    })
                    .map((page, index, array) => (
                      <span key={page} className="flex items-center">
                        {index > 0 && array[index - 1] !== page - 1 && (
                          <span className="px-2 text-gray-500" aria-hidden="true">...</span>
                        )}
                        <button
                          onClick={() => setCurrentPage(page)}
                          className={`px-3 py-1 rounded-lg transition-colors ${
                            page === currentPage
                              ? 'bg-accent text-white'
                              : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                          }`}
                          aria-label={`Page ${page}`}
                          aria-current={page === currentPage ? 'page' : undefined}
                        >
                          {page}
                        </button>
                      </span>
                    ))}
                </div>
                <button
                  onClick={() => setCurrentPage(p => Math.min(processedBots.totalPages, p + 1))}
                  disabled={currentPage === processedBots.totalPages}
                  className="p-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors"
                  aria-label="Next page"
                >
                  <ChevronRight size={18} aria-hidden="true" />
                </button>
              </div>
            </nav>
          )}
        </>
      ) : (
        <div className="bg-gray-800 rounded-lg py-16 text-center">
          <Bot size={64} className="mx-auto mb-4 text-gray-600" />
          <h3 className="text-xl font-semibold text-gray-300 mb-2">
            {/* #165: Use normalizedSearchTerm (trimmed) for empty state check */}
            {normalizedSearchTerm || statusFilter ? 'No bots found' : 'No bots created yet'}
          </h3>
          <p className="text-gray-400 mb-6">
            {normalizedSearchTerm || statusFilter
              ? 'Try adjusting your filters'
              : 'Create your first trading bot to get started'}
          </p>
          {!normalizedSearchTerm && !statusFilter && (
            <Link
              to="/bots/new"
              className="inline-flex items-center gap-2 px-6 py-3 bg-accent hover:bg-accent/80 text-white rounded-lg"
            >
              <Plus size={20} />
              Create Bot
            </Link>
          )}
        </div>
      )}
    </div>
  )
}
