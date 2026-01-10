import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  TrendingUp,
  TrendingDown,
  Bot,
  Activity,
  DollarSign,
  AlertCircle,
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

interface Stats {
  total_bots: number
  running_bots: number
  paused_bots: number
  stopped_bots: number
  total_pnl: number
  active_trades: number
  dry_run_bots: number
}

interface PnLDataPoint {
  timestamp: string
  pnl: number
}

interface BotSummary {
  id: number
  name: string
  status: string
  trading_pair: string
  total_pnl: number
}

async function fetchStats(): Promise<Stats> {
  const res = await fetch('/api/stats')
  if (!res.ok) throw new Error('Failed to fetch stats')
  return res.json()
}

async function fetchPnLHistory(): Promise<PnLDataPoint[]> {
  const res = await fetch('/api/pnl')
  if (!res.ok) throw new Error('Failed to fetch P&L history')
  return res.json()
}

async function fetchBots(): Promise<BotSummary[]> {
  const res = await fetch('/api/bots?limit=5')
  if (!res.ok) throw new Error('Failed to fetch bots')
  return res.json()
}

function StatCard({
  title,
  value,
  icon: Icon,
  trend,
  className = '',
}: {
  title: string
  value: string | number
  icon: React.ElementType
  trend?: 'up' | 'down' | null
  className?: string
}) {
  return (
    <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-400 text-sm">{title}</p>
          <p className="text-2xl font-bold font-mono-numbers mt-1">{value}</p>
        </div>
        <div className="p-3 bg-gray-700 rounded-lg">
          <Icon size={24} className="text-accent" />
        </div>
      </div>
      {trend && (
        <div className="mt-2">
          {trend === 'up' ? (
            <TrendingUp size={16} className="text-profit inline mr-1" />
          ) : (
            <TrendingDown size={16} className="text-loss inline mr-1" />
          )}
        </div>
      )}
    </div>
  )
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'running':
      return 'text-running'
    case 'paused':
      return 'text-paused'
    case 'stopped':
      return 'text-stopped'
    default:
      return 'text-gray-400'
  }
}

export default function Dashboard() {
  const { data: stats, isLoading: statsLoading, error: statsError } = useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
  })

  const { data: pnlHistory, isLoading: pnlLoading } = useQuery({
    queryKey: ['pnl-history'],
    queryFn: fetchPnLHistory,
  })

  const { data: bots, isLoading: botsLoading } = useQuery({
    queryKey: ['bots-summary'],
    queryFn: fetchBots,
  })

  if (statsLoading || pnlLoading || botsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent"></div>
      </div>
    )
  }

  if (statsError) {
    return (
      <div className="flex items-center justify-center h-64 text-loss">
        <AlertCircle className="mr-2" />
        Failed to load dashboard data
      </div>
    )
  }

  const totalPnL = stats?.total_pnl ?? 0

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Dashboard</h2>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total P&L"
          value={`${totalPnL >= 0 ? '+' : ''}$${totalPnL.toFixed(2)}`}
          icon={DollarSign}
          className={totalPnL >= 0 ? 'border-l-4 border-profit' : 'border-l-4 border-loss'}
        />
        <StatCard
          title="Running Bots"
          value={stats?.running_bots ?? 0}
          icon={Activity}
        />
        <StatCard
          title="Total Bots"
          value={stats?.total_bots ?? 0}
          icon={Bot}
        />
        <StatCard
          title="Active Trades"
          value={stats?.active_trades ?? 0}
          icon={TrendingUp}
        />
      </div>

      {/* P&L Chart */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">P&L Over Time</h3>
        {pnlHistory && pnlHistory.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={pnlHistory}>
              <XAxis
                dataKey="timestamp"
                stroke="#6b7280"
                tick={{ fill: '#9ca3af' }}
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis
                stroke="#6b7280"
                tick={{ fill: '#9ca3af' }}
                tickFormatter={(value) => `$${value}`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '0.5rem',
                }}
                labelFormatter={(value) => new Date(value).toLocaleString()}
                formatter={(value: number) => [`$${value.toFixed(2)}`, 'P&L']}
              />
              <Line
                type="monotone"
                dataKey="pnl"
                stroke="#6366f1"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-64 flex items-center justify-center text-gray-400">
            No P&L data available yet
          </div>
        )}
      </div>

      {/* Recent Bots */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Recent Bots</h3>
          <Link
            to="/bots"
            className="text-accent hover:text-accent/80 text-sm"
          >
            View All
          </Link>
        </div>

        {bots && bots.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                  <th className="pb-3">Name</th>
                  <th className="pb-3">Pair</th>
                  <th className="pb-3">Status</th>
                  <th className="pb-3 text-right">P&L</th>
                </tr>
              </thead>
              <tbody>
                {bots.map((bot) => (
                  <tr key={bot.id} className="border-b border-gray-700/50">
                    <td className="py-3">
                      <Link
                        to={`/bots/${bot.id}`}
                        className="text-white hover:text-accent"
                      >
                        {bot.name}
                      </Link>
                    </td>
                    <td className="py-3 text-gray-300">{bot.trading_pair}</td>
                    <td className="py-3">
                      <span className={`capitalize ${getStatusColor(bot.status)}`}>
                        {bot.status}
                      </span>
                    </td>
                    <td className={`py-3 text-right font-mono-numbers ${
                      bot.total_pnl >= 0 ? 'text-profit' : 'text-loss'
                    }`}>
                      {bot.total_pnl >= 0 ? '+' : ''}${bot.total_pnl.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-400">
            <Bot size={48} className="mx-auto mb-4 opacity-50" />
            <p>No bots created yet</p>
            <Link
              to="/bots/new"
              className="inline-block mt-4 px-4 py-2 bg-accent hover:bg-accent/80 text-white rounded-lg"
            >
              Create Your First Bot
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}
