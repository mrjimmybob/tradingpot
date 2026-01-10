import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
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

async function fetchBots(status?: string): Promise<BotListItem[]> {
  const url = status ? `/api/bots?status_filter=${status}` : '/api/bots'
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to fetch bots')
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

export default function BotList() {
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [searchTerm, setSearchTerm] = useState('')

  const { data: bots, isLoading, error, refetch } = useQuery({
    queryKey: ['bots', statusFilter],
    queryFn: () => fetchBots(statusFilter || undefined),
  })

  const filteredBots = bots?.filter((bot) =>
    bot.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    bot.trading_pair.toLowerCase().includes(searchTerm.toLowerCase()) ||
    bot.strategy.toLowerCase().includes(searchTerm.toLowerCase())
  )

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

  if (error) {
    return (
      <div className="flex items-center justify-center h-64 text-loss">
        <AlertCircle className="mr-2" />
        Failed to load bots
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Bots</h2>
        <Link
          to="/bots/new"
          className="flex items-center gap-2 px-4 py-2 bg-accent hover:bg-accent/80 text-white rounded-lg transition-colors"
        >
          <Plus size={18} />
          Create Bot
        </Link>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search
            size={18}
            className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
          />
          <input
            type="text"
            placeholder="Search bots..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-accent"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter size={18} className="text-gray-400" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-accent"
          >
            <option value="">All Status</option>
            <option value="running">Running</option>
            <option value="paused">Paused</option>
            <option value="stopped">Stopped</option>
            <option value="created">Created</option>
          </select>
          {(statusFilter || searchTerm) && (
            <button
              onClick={() => {
                setStatusFilter('')
                setSearchTerm('')
              }}
              className="px-3 py-2 text-gray-400 hover:text-white"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Bot Table */}
      {filteredBots && filteredBots.length > 0 ? (
        <div className="bg-gray-800 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-sm bg-gray-700/50">
                  <th className="px-6 py-4">Name</th>
                  <th className="px-6 py-4">Trading Pair</th>
                  <th className="px-6 py-4">Strategy</th>
                  <th className="px-6 py-4">Status</th>
                  <th className="px-6 py-4 text-right">P&L</th>
                  <th className="px-6 py-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredBots.map((bot) => (
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
                    <td
                      className={`px-6 py-4 text-right font-mono-numbers ${
                        bot.total_pnl >= 0 ? 'text-profit' : 'text-loss'
                      }`}
                    >
                      {bot.total_pnl >= 0 ? '+' : ''}${bot.total_pnl.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        {bot.status === 'running' && (
                          <button
                            className="p-2 text-paused hover:bg-gray-700 rounded"
                            title="Pause"
                          >
                            <Pause size={16} />
                          </button>
                        )}
                        {(bot.status === 'paused' || bot.status === 'created') && (
                          <button
                            className="p-2 text-running hover:bg-gray-700 rounded"
                            title="Start"
                          >
                            <Play size={16} />
                          </button>
                        )}
                        {bot.status !== 'stopped' && (
                          <button
                            className="p-2 text-stopped hover:bg-gray-700 rounded"
                            title="Stop"
                          >
                            <Square size={16} />
                          </button>
                        )}
                        {bot.status === 'stopped' && (
                          <button
                            onClick={() => handleDelete(bot.id, bot.name)}
                            className="p-2 text-loss hover:bg-gray-700 rounded"
                            title="Delete"
                          >
                            <Trash2 size={16} />
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
      ) : (
        <div className="bg-gray-800 rounded-lg py-16 text-center">
          <Bot size={64} className="mx-auto mb-4 text-gray-600" />
          <h3 className="text-xl font-semibold text-gray-300 mb-2">
            {searchTerm || statusFilter ? 'No bots found' : 'No bots created yet'}
          </h3>
          <p className="text-gray-400 mb-6">
            {searchTerm || statusFilter
              ? 'Try adjusting your filters'
              : 'Create your first trading bot to get started'}
          </p>
          {!searchTerm && !statusFilter && (
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
