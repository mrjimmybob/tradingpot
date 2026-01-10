import { useParams, Link, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft,
  Play,
  Pause,
  Square,
  Power,
  Edit,
  Copy,
  TrendingUp,
  AlertCircle,
  Save,
  X,
  Check,
  Rocket,
} from 'lucide-react'

interface Bot {
  id: number
  name: string
  trading_pair: string
  strategy: string
  strategy_params: Record<string, unknown>
  budget: number
  compound_enabled: boolean
  current_balance: number
  running_time_hours: number | null
  stop_loss_percent: number | null
  stop_loss_absolute: number | null
  drawdown_limit_percent: number | null
  drawdown_limit_absolute: number | null
  daily_loss_limit: number | null
  weekly_loss_limit: number | null
  max_strategy_rotations: number
  is_dry_run: boolean
  status: string
  total_pnl: number
  created_at: string
  updated_at: string
  started_at: string | null
  paused_at: string | null
}

interface Position {
  id: number
  trading_pair: string
  side: string
  entry_price: number
  current_price: number
  amount: number
  unrealized_pnl: number
}

interface Order {
  id: number
  order_type: string
  trading_pair: string
  amount: number
  price: number
  fees: number
  status: string
  created_at: string
}

async function fetchBot(id: string): Promise<Bot> {
  const res = await fetch(`/api/bots/${id}`)
  if (!res.ok) {
    if (res.status === 404) throw new Error('Bot not found')
    throw new Error('Failed to fetch bot')
  }
  return res.json()
}

async function fetchPositions(id: string): Promise<Position[]> {
  const res = await fetch(`/api/bots/${id}/positions`)
  if (!res.ok) throw new Error('Failed to fetch positions')
  return res.json()
}

async function fetchOrders(id: string): Promise<Order[]> {
  const res = await fetch(`/api/bots/${id}/orders?limit=10`)
  if (!res.ok) throw new Error('Failed to fetch orders')
  return res.json()
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'running':
      return 'bg-running/20 text-running border-running'
    case 'paused':
      return 'bg-paused/20 text-paused border-paused'
    case 'stopped':
      return 'bg-stopped/20 text-stopped border-stopped'
    default:
      return 'bg-gray-700 text-gray-400 border-gray-600'
  }
}

export default function BotDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: bot, isLoading, error } = useQuery({
    queryKey: ['bot', id],
    queryFn: () => fetchBot(id!),
    enabled: !!id,
  })

  const { data: positions } = useQuery({
    queryKey: ['positions', id],
    queryFn: () => fetchPositions(id!),
    enabled: !!id,
  })

  const { data: orders } = useQuery({
    queryKey: ['orders', id],
    queryFn: () => fetchOrders(id!),
    enabled: !!id,
  })

  const startMutation = useMutation({
    mutationFn: async () => {
      const res = await fetch(`/api/bots/${id}/start`, { method: 'POST' })
      if (!res.ok) throw new Error('Failed to start bot')
      return res.json()
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['bot', id] }),
  })

  const pauseMutation = useMutation({
    mutationFn: async () => {
      const res = await fetch(`/api/bots/${id}/pause`, { method: 'POST' })
      if (!res.ok) throw new Error('Failed to pause bot')
      return res.json()
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['bot', id] }),
  })

  const stopMutation = useMutation({
    mutationFn: async () => {
      const res = await fetch(`/api/bots/${id}/stop`, { method: 'POST' })
      if (!res.ok) throw new Error('Failed to stop bot')
      return res.json()
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['bot', id] }),
  })

  const killMutation = useMutation({
    mutationFn: async () => {
      const res = await fetch(`/api/bots/${id}/kill`, { method: 'POST' })
      if (!res.ok) throw new Error('Failed to kill bot')
      return res.json()
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['bot', id] }),
  })

  const goLiveMutation = useMutation({
    mutationFn: async () => {
      const res = await fetch(`/api/bots/${id}/go-live`, { method: 'POST' })
      if (!res.ok) throw new Error('Failed to promote bot to live')
      return res.json()
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['bot', id] }),
  })

  const copyMutation = useMutation({
    mutationFn: async () => {
      const res = await fetch(`/api/bots/${id}/copy`, { method: 'POST' })
      if (!res.ok) throw new Error('Failed to copy bot')
      return res.json()
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['bots'] })
      navigate(`/bots/${data.id}`)
    },
  })

  // Edit mode state
  const [isEditing, setIsEditing] = useState(false)
  const [editForm, setEditForm] = useState({
    name: '',
    budget: 0,
    stop_loss_percent: 0,
    drawdown_limit_percent: 0,
    compound_enabled: false,
    strategy_params: {} as Record<string, unknown>,
  })
  const [editSuccess, setEditSuccess] = useState(false)

  const updateMutation = useMutation({
    mutationFn: async (data: typeof editForm) => {
      const res = await fetch(`/api/bots/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })
      if (!res.ok) throw new Error('Failed to update bot')
      return res.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['bot', id] })
      setIsEditing(false)
      setEditSuccess(true)
      setTimeout(() => setEditSuccess(false), 3000)
    },
  })

  const startEditing = () => {
    if (bot) {
      setEditForm({
        name: bot.name,
        budget: bot.budget,
        stop_loss_percent: bot.stop_loss_percent || 0,
        drawdown_limit_percent: bot.drawdown_limit_percent || 0,
        compound_enabled: bot.compound_enabled,
        strategy_params: bot.strategy_params || {},
      })
      setIsEditing(true)
    }
  }

  const cancelEditing = () => {
    setIsEditing(false)
  }

  const saveChanges = () => {
    updateMutation.mutate(editForm)
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent"></div>
      </div>
    )
  }

  if (error || !bot) {
    return (
      <div className="text-center py-16">
        <AlertCircle size={64} className="mx-auto mb-4 text-loss" />
        <h2 className="text-xl font-semibold mb-2">Bot Not Found</h2>
        <p className="text-gray-400 mb-6">
          {error instanceof Error ? error.message : 'The bot you are looking for does not exist.'}
        </p>
        <Link
          to="/bots"
          className="inline-flex items-center gap-2 text-accent hover:text-accent/80"
        >
          <ArrowLeft size={18} />
          Back to Bots
        </Link>
      </div>
    )
  }

  const canEdit = bot.status === 'paused' || bot.status === 'stopped'

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <Link
            to="/bots"
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded"
          >
            <ArrowLeft size={20} />
          </Link>
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-bold">{bot.name}</h2>
              <span
                className={`px-3 py-1 rounded-full text-sm capitalize border ${getStatusColor(
                  bot.status
                )}`}
              >
                {bot.status}
              </span>
              {bot.is_dry_run && (
                <span className="px-3 py-1 bg-yellow-500/20 text-yellow-500 rounded-full text-sm">
                  Dry Run
                </span>
              )}
            </div>
            <p className="text-gray-400 mt-1">
              {bot.trading_pair} &middot; {bot.strategy.replace(/_/g, ' ')}
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {bot.status === 'running' && (
            <button
              onClick={() => pauseMutation.mutate()}
              disabled={pauseMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-paused/20 text-paused hover:bg-paused/30 rounded-lg"
            >
              <Pause size={18} />
              Pause
            </button>
          )}
          {(bot.status === 'paused' || bot.status === 'created') && (
            <button
              onClick={() => startMutation.mutate()}
              disabled={startMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-running/20 text-running hover:bg-running/30 rounded-lg"
            >
              <Play size={18} />
              Start
            </button>
          )}
          {bot.status !== 'stopped' && (
            <button
              onClick={() => {
                if (confirm('Are you sure you want to stop this bot?')) {
                  stopMutation.mutate()
                }
              }}
              disabled={stopMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-stopped/20 text-stopped hover:bg-stopped/30 rounded-lg"
            >
              <Square size={18} />
              Stop
            </button>
          )}
          {bot.status === 'running' && (
            <button
              onClick={() => {
                if (confirm('Are you sure you want to trigger the kill switch? This will cancel all pending orders.')) {
                  killMutation.mutate()
                }
              }}
              disabled={killMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white hover:bg-red-700 rounded-lg"
            >
              <Power size={18} />
              Kill
            </button>
          )}
          {canEdit && !isEditing && (
            <button
              onClick={startEditing}
              className="flex items-center gap-2 px-4 py-2 bg-gray-700 text-white hover:bg-gray-600 rounded-lg"
            >
              <Edit size={18} />
              Edit
            </button>
          )}
          {isEditing && (
            <>
              <button
                onClick={saveChanges}
                disabled={updateMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 bg-profit text-white hover:bg-profit/80 rounded-lg"
              >
                <Save size={18} />
                {updateMutation.isPending ? 'Saving...' : 'Save'}
              </button>
              <button
                onClick={cancelEditing}
                disabled={updateMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 bg-gray-700 text-white hover:bg-gray-600 rounded-lg"
              >
                <X size={18} />
                Cancel
              </button>
            </>
          )}
          {bot.is_dry_run && bot.status !== 'running' && (
            <button
              onClick={() => {
                if (confirm('Are you sure you want to promote this bot to live trading? This will use real funds and place real orders on the exchange.')) {
                  goLiveMutation.mutate()
                }
              }}
              disabled={goLiveMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-profit text-white hover:bg-profit/80 rounded-lg"
            >
              <Rocket size={18} />
              {goLiveMutation.isPending ? 'Promoting...' : 'Go Live'}
            </button>
          )}
          <button
            onClick={() => {
              if (confirm('Create a copy of this bot?')) {
                copyMutation.mutate()
              }
            }}
            disabled={copyMutation.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-gray-700 text-white hover:bg-gray-600 rounded-lg"
          >
            <Copy size={18} />
            {copyMutation.isPending ? 'Copying...' : 'Copy'}
          </button>
        </div>
      </div>

      {/* Success Message */}
      {editSuccess && (
        <div className="bg-profit/20 border border-profit text-profit px-4 py-3 rounded-lg flex items-center gap-2">
          <Check size={18} />
          Bot configuration updated successfully!
        </div>
      )}

      {/* Edit Form */}
      {isEditing && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Edit Configuration</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Bot Name</label>
              <input
                type="text"
                value={editForm.name}
                onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-accent focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Budget (USD)</label>
              <input
                type="number"
                value={editForm.budget}
                onChange={(e) => setEditForm({ ...editForm, budget: parseFloat(e.target.value) || 0 })}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-accent focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Stop Loss (%)</label>
              <input
                type="number"
                value={editForm.stop_loss_percent}
                onChange={(e) => setEditForm({ ...editForm, stop_loss_percent: parseFloat(e.target.value) || 0 })}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-accent focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Drawdown Limit (%)</label>
              <input
                type="number"
                value={editForm.drawdown_limit_percent}
                onChange={(e) => setEditForm({ ...editForm, drawdown_limit_percent: parseFloat(e.target.value) || 0 })}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-accent focus:outline-none"
              />
            </div>
            <div className="md:col-span-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={editForm.compound_enabled}
                  onChange={(e) => setEditForm({ ...editForm, compound_enabled: e.target.checked })}
                  className="w-4 h-4 text-accent bg-gray-700 border-gray-600 rounded focus:ring-accent"
                />
                <span className="text-sm text-gray-300">Enable Compounding (add profits to budget)</span>
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-6">
          <p className="text-gray-400 text-sm">Total P&L</p>
          <p
            className={`text-2xl font-bold font-mono-numbers mt-1 ${
              bot.total_pnl >= 0 ? 'text-profit' : 'text-loss'
            }`}
          >
            {bot.total_pnl >= 0 ? '+' : ''}${bot.total_pnl.toFixed(2)}
          </p>
        </div>
        <div className="bg-gray-800 rounded-lg p-6">
          <p className="text-gray-400 text-sm">Current Balance</p>
          <p className="text-2xl font-bold font-mono-numbers mt-1">
            ${bot.current_balance.toFixed(2)}
          </p>
        </div>
        <div className="bg-gray-800 rounded-lg p-6">
          <p className="text-gray-400 text-sm">Budget</p>
          <p className="text-2xl font-bold font-mono-numbers mt-1">
            ${bot.budget.toFixed(2)}
          </p>
        </div>
        <div className="bg-gray-800 rounded-lg p-6">
          <p className="text-gray-400 text-sm">Open Positions</p>
          <p className="text-2xl font-bold font-mono-numbers mt-1">
            {positions?.length ?? 0}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Open Positions */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Open Positions</h3>
          {positions && positions.length > 0 ? (
            <div className="space-y-3">
              {positions.map((pos) => (
                <div
                  key={pos.id}
                  className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg"
                >
                  <div>
                    <p className="font-medium">{pos.trading_pair}</p>
                    <p className="text-sm text-gray-400 capitalize">
                      {pos.side} &middot; {pos.amount} @ ${pos.entry_price.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-right">
                    <p
                      className={`font-mono-numbers ${
                        pos.unrealized_pnl >= 0 ? 'text-profit' : 'text-loss'
                      }`}
                    >
                      {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                    </p>
                    <p className="text-sm text-gray-400">
                      Current: ${pos.current_price.toFixed(2)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              <TrendingUp size={32} className="mx-auto mb-2 opacity-50" />
              <p>No open positions</p>
            </div>
          )}
        </div>

        {/* Strategy Parameters */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Strategy Parameters</h3>
          <dl className="space-y-3">
            <div className="flex justify-between">
              <dt className="text-gray-400">Strategy</dt>
              <dd className="capitalize">{bot.strategy.replace(/_/g, ' ')}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-400">Compound Mode</dt>
              <dd>{bot.compound_enabled ? 'Enabled' : 'Disabled'}</dd>
            </div>
            {bot.stop_loss_percent && (
              <div className="flex justify-between">
                <dt className="text-gray-400">Stop Loss</dt>
                <dd>{bot.stop_loss_percent}%</dd>
              </div>
            )}
            {bot.drawdown_limit_percent && (
              <div className="flex justify-between">
                <dt className="text-gray-400">Drawdown Limit</dt>
                <dd>{bot.drawdown_limit_percent}%</dd>
              </div>
            )}
            {bot.running_time_hours && (
              <div className="flex justify-between">
                <dt className="text-gray-400">Running Time</dt>
                <dd>{bot.running_time_hours}h</dd>
              </div>
            )}
            {Object.entries(bot.strategy_params || {}).map(([key, value]) => (
              <div key={key} className="flex justify-between">
                <dt className="text-gray-400 capitalize">{key.replace(/_/g, ' ')}</dt>
                <dd>{String(value)}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>

      {/* Recent Orders */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Recent Orders</h3>
        {orders && orders.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                  <th className="pb-3">Type</th>
                  <th className="pb-3">Pair</th>
                  <th className="pb-3">Amount</th>
                  <th className="pb-3">Price</th>
                  <th className="pb-3">Status</th>
                  <th className="pb-3">Date</th>
                </tr>
              </thead>
              <tbody>
                {orders.map((order) => (
                  <tr key={order.id} className="border-b border-gray-700/50">
                    <td className="py-3 capitalize">
                      {order.order_type.replace(/_/g, ' ')}
                    </td>
                    <td className="py-3">{order.trading_pair}</td>
                    <td className="py-3 font-mono-numbers">{order.amount}</td>
                    <td className="py-3 font-mono-numbers">${order.price.toFixed(2)}</td>
                    <td className="py-3 capitalize">{order.status}</td>
                    <td className="py-3 text-gray-400">
                      {new Date(order.created_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-400">
            <p>No orders yet</p>
          </div>
        )}
      </div>
    </div>
  )
}
