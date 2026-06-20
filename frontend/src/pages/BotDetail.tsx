import { apiFetch } from '../lib/api'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import type { ReactNode } from 'react'
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
  Loader2,
  Activity,
} from 'lucide-react'
import { useToast } from '../components/Toast'
import { useBotActions } from '../hooks/useBotActions'
import { useRealtimeBot, useRealtimePrice } from '../contexts/WebSocketContext'
import { RealtimePrice } from '../components/RealtimePrice'

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

interface DecisionStatus {
  bot_id: number
  state: string | null
  reason: string
  symbol: string | null
  score: number | null
  threshold: number | null
  updated_at: string | null
}

interface Diagnostics {
  bot_id: number
  status: string
  current_activity: string
  paused_reason: string | null
  evaluations: {
    total: number
    runtime: number
    last_24h: number
    last_evaluation_at: string | null
    runtime_started_at: string | null
  }
  signals: {
    buy: number
    sell: number
    hold: number
    last_action: string | null
    last_reason: string | null
    last_at: string | null
  }
  top_reasons: { reason: string; count: number }[]
  blocked: {
    risk_manager: number
    min_order_size: number
    insufficient_balance: number
    cooldown: number
    position_limits: number
    exchange_validation: number
    other: number
    last_category: string | null
    last_reason: string | null
    last_at: string | null
  }
  execution: {
    successful_buys: number
    successful_sells: number
    failed_buys: number
    failed_sells: number
    last_failure_reason: string | null
    last_failure_at: string | null
  }
  market_data: {
    ticker_failures: number
    websocket_failures: number
    data_unavailable: number
    last_failure_reason: string | null
    last_failure_at: string | null
  }
}

async function fetchBot(id: string): Promise<Bot> {
  const res = await apiFetch(`/api/bots/${id}`)
  if (!res.ok) {
    if (res.status === 404) throw new Error('Bot not found')
    throw new Error('Failed to fetch bot')
  }
  return res.json()
}

async function fetchPositions(id: string): Promise<Position[]> {
  const res = await apiFetch(`/api/bots/${id}/positions`)
  if (!res.ok) throw new Error('Failed to fetch positions')
  return res.json()
}

async function fetchOrders(id: string): Promise<Order[]> {
  const res = await apiFetch(`/api/bots/${id}/orders?limit=10`)
  if (!res.ok) throw new Error('Failed to fetch orders')
  return res.json()
}

async function fetchDecisionStatus(id: string): Promise<DecisionStatus> {
  const res = await apiFetch(`/api/bots/${id}/decision-status`)
  if (!res.ok) throw new Error('Failed to fetch decision status')
  return res.json()
}

async function fetchDiagnostics(id: string): Promise<Diagnostics> {
  const res = await apiFetch(`/api/bots/${id}/diagnostics`)
  if (!res.ok) throw new Error('Failed to fetch diagnostics')
  return res.json()
}

// Map a decision state to a Tailwind color class for the status pill. Falls
// back to a neutral gray for unknown/idle states.
// Decision-state palette. DELIBERATELY uses a different colour family from the
// lifecycle badge (getStatusColor: running/paused/stopped) so a trading decision
// can never be mistaken for an operational lifecycle state. Rendered as an
// OUTLINED chip (see the Current Decision card), not a filled lifecycle pill.
function getDecisionStateColor(state: string | null): string {
  switch (state) {
    case 'Buy signal detected':
    case 'Entering position':
      return 'bg-emerald-500/10 text-emerald-300 border-emerald-500/40'
    case 'Sell signal detected':
    case 'Exiting position':
      return 'bg-rose-500/10 text-rose-300 border-rose-500/40'
    case 'Risk limit reached':
      return 'bg-orange-500/10 text-orange-300 border-orange-500/40'
    case 'Cooldown active':
    case 'Waiting for market regime':
    case 'Waiting for data':
    case 'Warming up indicators':
      return 'bg-sky-500/10 text-sky-300 border-sky-500/40'
    default:
      // Hold / Evaluating / anything else: neutral, clearly secondary.
      return 'bg-slate-500/10 text-slate-300 border-slate-500/40'
  }
}

// Small presentational helpers for the diagnostics panel.
function fmtDiagTime(iso: string | null): string {
  return iso ? new Date(iso).toLocaleTimeString() : '—'
}

function DiagRow({ label, value, danger }: { label: string; value: ReactNode; danger?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-gray-400">{label}</span>
      <span className={`font-mono-numbers ${danger ? 'text-loss' : 'text-gray-100'}`}>{value}</span>
    </div>
  )
}

const BLOCKED_LABELS: Record<string, string> = {
  risk_manager: 'Risk manager',
  min_order_size: 'Min order size',
  insufficient_balance: 'Insufficient balance',
  cooldown: 'Cooldown',
  position_limits: 'Position limits',
  exchange_validation: 'Exchange validation',
  other: 'Other',
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

// #140, #141: Validate bot ID from URL parameter
function isValidBotId(id: string | undefined): boolean {
  if (!id) return false
  const num = parseInt(id, 10)
  return !isNaN(num) && num > 0 && String(num) === id
}

export default function BotDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const toast = useToast()

  // #140, #141: Validate URL parameter before making API call
  const isValidId = isValidBotId(id)

  const { data: bot, isLoading, error } = useQuery({
    queryKey: ['bot', id],
    queryFn: () => fetchBot(id!),
    enabled: isValidId, // Only fetch if ID is valid
  })

  const { data: positions } = useQuery({
    queryKey: ['positions', id],
    queryFn: () => fetchPositions(id!),
    enabled: isValidId,
  })

  const { data: orders } = useQuery({
    queryKey: ['orders', id],
    queryFn: () => fetchOrders(id!),
    enabled: isValidId,
  })

  // Poll the in-memory decision status so the panel reflects what the engine is
  // doing in near-real-time without a websocket dependency.
  const { data: decisionStatus } = useQuery({
    queryKey: ['decision-status', id],
    queryFn: () => fetchDecisionStatus(id!),
    enabled: isValidId,
    refetchInterval: 3000,
  })

  // Poll observe-only strategy diagnostics (evaluations, signals, reasons,
  // blocked trades, execution/data health, pause reason) for the diagnostics
  // panel. Auto-refreshes so the panel reflects live activity without a refresh.
  const { data: diagnostics } = useQuery({
    queryKey: ['diagnostics', id],
    queryFn: () => fetchDiagnostics(id!),
    enabled: isValidId,
    refetchInterval: 3000,
  })

  // Real-time bot data from WebSocket
  const realtimeBotData = useRealtimeBot(parseInt(id || '0'))

  // Get real-time price for the trading pair
  useRealtimePrice(bot?.trading_pair || '')  // subscribe for side effects

  // Merge real-time data with query data
  const displayPnl = realtimeBotData?.pnl ?? bot?.total_pnl ?? 0
  const displayBalance = realtimeBotData?.current_balance ?? bot?.current_balance ?? 0
  const displayPositions = realtimeBotData?.positions ?? positions

  // Start/Pause/Stop share one implementation with the bots list via
  // useBotActions (single code path, identical endpoints + behaviour).
  const { start: startMutation, pause: pauseMutation, stop: stopMutation } = useBotActions()

  const killMutation = useMutation({
    mutationFn: async () => {
      const res = await apiFetch(`/api/bots/${id}/kill`, { method: 'POST' })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || 'Failed to activate kill switch')
      }
      return res.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['bot', id] })
      toast.warning('Kill Switch Activated', `"${bot?.name}" has been emergency stopped. All pending orders cancelled.`)
    },
    onError: (err: Error) => toast.error('Kill Switch Failed', err.message || 'Could not activate kill switch. Please try again.'),
  })

  const goLiveMutation = useMutation({
    mutationFn: async () => {
      const res = await apiFetch(`/api/bots/${id}/go-live`, { method: 'POST' })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || 'Failed to promote bot to live')
      }
      return res.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['bot', id] })
      toast.success('Bot Promoted to Live', `"${bot?.name}" will now execute real trades with real funds.`)
    },
    onError: (err: Error) => toast.error('Promotion Failed', err.message || 'Could not promote bot to live. Please try again.'),
  })

  const copyMutation = useMutation({
    mutationFn: async () => {
      const res = await apiFetch(`/api/bots/${id}/copy`, { method: 'POST' })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || 'Failed to copy bot')
      }
      return res.json()
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['bots'] })
      toast.success('Bot Copied', `Created "${data.name}" with the same configuration as "${bot?.name}"`)
      navigate(`/bots/${data.id}`)
    },
    onError: (err: Error) => toast.error('Copy Failed', err.message || 'Could not copy the bot. Please try again.'),
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
      const res = await apiFetch(`/api/bots/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || 'Failed to update bot')
      }
      return res.json()
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['bot', id] })
      setIsEditing(false)
      setEditSuccess(true)
      setTimeout(() => setEditSuccess(false), 3000)
      toast.success('Configuration Saved', `"${data.name}" settings have been updated successfully.`)
    },
    onError: (err: Error) => toast.error('Update Failed', err.message || 'Could not save changes. Please try again.'),
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

  // #140, #141: Handle invalid/malformed bot ID in URL
  if (!isValidId) {
    return (
      <div className="text-center py-16">
        <AlertCircle size={64} className="mx-auto mb-4 text-loss" />
        <h2 className="text-xl font-semibold mb-2">Invalid Bot ID</h2>
        <p className="text-gray-400 mb-6">
          The URL contains an invalid bot ID "{id}". Bot IDs must be positive numbers.
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
              <span className="text-xs uppercase tracking-wide text-gray-500">Status</span>
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
              onClick={() => pauseMutation.mutate({ id: id!, name: bot.name })}
              disabled={pauseMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-paused/20 text-paused hover:bg-paused/30 rounded-lg disabled:opacity-50"
            >
              {pauseMutation.isPending ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Pause size={18} />
              )}
              {pauseMutation.isPending ? 'Pausing...' : 'Pause'}
            </button>
          )}
          {(bot.status === 'paused' || bot.status === 'created') && (
            <button
              onClick={() => startMutation.mutate({ id: id!, name: bot.name })}
              disabled={startMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-running/20 text-running hover:bg-running/30 rounded-lg disabled:opacity-50"
            >
              {startMutation.isPending ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Play size={18} />
              )}
              {startMutation.isPending ? 'Starting...' : 'Start'}
            </button>
          )}
          {bot.status !== 'stopped' && (
            <button
              onClick={() => {
                if (confirm('Are you sure you want to stop this bot?')) {
                  stopMutation.mutate({ id: id!, name: bot.name })
                }
              }}
              disabled={stopMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-stopped/20 text-stopped hover:bg-stopped/30 rounded-lg disabled:opacity-50"
            >
              {stopMutation.isPending ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Square size={18} />
              )}
              {stopMutation.isPending ? 'Stopping...' : 'Stop'}
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
              className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white hover:bg-red-700 rounded-lg disabled:opacity-50"
            >
              {killMutation.isPending ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Power size={18} />
              )}
              {killMutation.isPending ? 'Killing...' : 'Kill'}
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
                className="flex items-center gap-2 px-4 py-2 bg-profit text-white hover:bg-profit/80 rounded-lg disabled:opacity-50"
              >
                {updateMutation.isPending ? (
                  <Loader2 size={18} className="animate-spin" />
                ) : (
                  <Save size={18} />
                )}
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
              className="flex items-center gap-2 px-4 py-2 bg-profit text-white hover:bg-profit/80 rounded-lg disabled:opacity-50"
            >
              {goLiveMutation.isPending ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Rocket size={18} />
              )}
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
            className="flex items-center gap-2 px-4 py-2 bg-gray-700 text-white hover:bg-gray-600 rounded-lg disabled:opacity-50"
          >
            {copyMutation.isPending ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Copy size={18} />
            )}
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
              displayPnl >= 0 ? 'text-profit' : 'text-loss'
            }`}
          >
            {displayPnl >= 0 ? '+' : ''}${displayPnl.toFixed(2)}
          </p>
        </div>
        <div className="bg-gray-800 rounded-lg p-6">
          <p className="text-gray-400 text-sm">Current Balance</p>
          <p className="text-2xl font-bold font-mono-numbers mt-1">
            ${displayBalance.toFixed(2)}
          </p>
        </div>
        <div className="bg-gray-800 rounded-lg p-6">
          <p className="text-gray-400 text-sm">Current Price</p>
          <RealtimePrice
            symbol={bot.trading_pair}
            fallbackPrice={0}
            showChange={true}
            size="lg"
          />
        </div>
        <div className="bg-gray-800 rounded-lg p-6">
          <p className="text-gray-400 text-sm">Open Positions</p>
          <p className="text-2xl font-bold font-mono-numbers mt-1">
            {displayPositions?.length ?? 0}
          </p>
        </div>
      </div>

      {/* Current Decision — the strategy's trading decision (NOT the bot's
          lifecycle state). Styled as a secondary OUTLINED chip with its own
          colour family so it is never mistaken for the lifecycle badge above. */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <Activity size={18} className="text-accent" />
          <h3 className="text-lg font-semibold">Current Decision</h3>
          <span className="text-xs text-gray-500">(trading decision — lifecycle status is shown next to the bot name)</span>
        </div>
        {decisionStatus ? (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="text-xs uppercase tracking-wide text-gray-500">Decision</span>
              <span
                className={`px-2.5 py-0.5 rounded-md text-sm border font-medium ${getDecisionStateColor(
                  decisionStatus.state
                )}`}
              >
                {decisionStatus.state ?? 'Idle'}
              </span>
              {decisionStatus.reason && (
                <p className="text-gray-300 text-sm">{decisionStatus.reason}</p>
              )}
            </div>
            <div className="flex items-center gap-6 text-sm">
              {decisionStatus.score != null && (
                <div className="text-right">
                  <p className="text-gray-400">Signal Score</p>
                  <p className="font-mono-numbers">
                    {decisionStatus.score.toFixed(3)}
                    {decisionStatus.threshold != null && (
                      <span className="text-gray-400">
                        {' '}/ {decisionStatus.threshold.toFixed(3)}
                      </span>
                    )}
                  </p>
                </div>
              )}
              <div className="text-right">
                <p className="text-gray-400">Last Update</p>
                <p className="font-mono-numbers">
                  {decisionStatus.updated_at
                    ? new Date(decisionStatus.updated_at).toLocaleTimeString()
                    : '—'}
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-6 text-gray-400">
            <p>No decision data yet</p>
          </div>
        )}
      </div>

      {/* Strategy Diagnostics — observe-only operability: what the bot has been
          doing, thinking, waiting for, blocked by, and why it's paused */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <Activity size={18} className="text-accent" />
          <h3 className="text-lg font-semibold">Strategy Diagnostics</h3>
        </div>
        {diagnostics ? (
          <div className="space-y-4">
            {/* Current activity — the first thing visible */}
            <div className="bg-gray-900 rounded-md p-4">
              <p className="text-gray-400 text-xs uppercase tracking-wide mb-1">
                Current Activity
              </p>
              <p className="text-sm text-gray-100">{diagnostics.current_activity}</p>
            </div>

            {/* Pause reason (only when paused) */}
            {diagnostics.paused_reason && (
              <div className="bg-loss/10 border border-loss/40 rounded-md p-4">
                <p className="text-loss text-xs uppercase tracking-wide mb-1 font-semibold flex items-center gap-1">
                  <AlertCircle size={14} /> Paused Because
                </p>
                <p className="text-sm text-gray-100">{diagnostics.paused_reason}</p>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Evaluation statistics */}
              <div className="bg-gray-900 rounded-md p-4">
                <p className="text-gray-400 text-xs uppercase tracking-wide mb-2">
                  Evaluation Statistics
                </p>
                <div className="space-y-1 text-sm">
                  <DiagRow label="Total" value={diagnostics.evaluations.total} />
                  <DiagRow label="This runtime" value={diagnostics.evaluations.runtime} />
                  <DiagRow label="Last 24h" value={diagnostics.evaluations.last_24h} />
                  <DiagRow
                    label="Last evaluation"
                    value={fmtDiagTime(diagnostics.evaluations.last_evaluation_at)}
                  />
                </div>
              </div>

              {/* Signal statistics */}
              <div className="bg-gray-900 rounded-md p-4">
                <p className="text-gray-400 text-xs uppercase tracking-wide mb-2">
                  Signal Statistics
                </p>
                <div className="space-y-1 text-sm">
                  <DiagRow label="Buy" value={diagnostics.signals.buy} />
                  <DiagRow label="Sell" value={diagnostics.signals.sell} />
                  <DiagRow label="Hold" value={diagnostics.signals.hold} />
                  <DiagRow
                    label="Last signal"
                    value={`${diagnostics.signals.last_action ?? '—'} @ ${fmtDiagTime(
                      diagnostics.signals.last_at
                    )}`}
                  />
                </div>
                {diagnostics.signals.last_reason && (
                  <p className="text-xs text-gray-500 mt-2 break-words">
                    {diagnostics.signals.last_reason}
                  </p>
                )}
              </div>

              {/* Execution statistics */}
              <div className="bg-gray-900 rounded-md p-4">
                <p className="text-gray-400 text-xs uppercase tracking-wide mb-2">
                  Execution Statistics
                </p>
                <div className="space-y-1 text-sm">
                  <DiagRow label="Successful buys" value={diagnostics.execution.successful_buys} />
                  <DiagRow label="Successful sells" value={diagnostics.execution.successful_sells} />
                  <DiagRow
                    label="Failed buys"
                    value={diagnostics.execution.failed_buys}
                    danger={diagnostics.execution.failed_buys > 0}
                  />
                  <DiagRow
                    label="Failed sells"
                    value={diagnostics.execution.failed_sells}
                    danger={diagnostics.execution.failed_sells > 0}
                  />
                </div>
                {diagnostics.execution.last_failure_reason && (
                  <p className="text-xs text-loss mt-2 break-words">
                    Last failure: {diagnostics.execution.last_failure_reason} (
                    {fmtDiagTime(diagnostics.execution.last_failure_at)})
                  </p>
                )}
              </div>

              {/* Market data health */}
              <div className="bg-gray-900 rounded-md p-4">
                <p className="text-gray-400 text-xs uppercase tracking-wide mb-2">
                  Market Data Health
                </p>
                <div className="space-y-1 text-sm">
                  <DiagRow
                    label="Ticker failures"
                    value={diagnostics.market_data.ticker_failures}
                    danger={diagnostics.market_data.ticker_failures > 0}
                  />
                  <DiagRow
                    label="Websocket failures"
                    value={diagnostics.market_data.websocket_failures}
                    danger={diagnostics.market_data.websocket_failures > 0}
                  />
                  <DiagRow
                    label="Data unavailable"
                    value={diagnostics.market_data.data_unavailable}
                    danger={diagnostics.market_data.data_unavailable > 0}
                  />
                </div>
                {diagnostics.market_data.last_failure_reason && (
                  <p className="text-xs text-loss mt-2 break-words">
                    Last: {diagnostics.market_data.last_failure_reason} (
                    {fmtDiagTime(diagnostics.market_data.last_failure_at)})
                  </p>
                )}
              </div>
            </div>

            {/* Top decision reasons — what the strategy is actually thinking */}
            <div className="bg-gray-900 rounded-md p-4">
              <p className="text-gray-400 text-xs uppercase tracking-wide mb-2">
                Top Decision Reasons
              </p>
              {diagnostics.top_reasons.length > 0 ? (
                <ul className="space-y-1 text-sm">
                  {diagnostics.top_reasons.map((r) => (
                    <li key={r.reason} className="flex items-baseline justify-between gap-3">
                      <span className="text-gray-300 break-words">{r.reason}</span>
                      <span className="font-mono-numbers text-gray-400">{r.count}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-gray-500">No signals recorded yet.</p>
              )}
            </div>

            {/* Blocked trade statistics */}
            <div className="bg-gray-900 rounded-md p-4">
              <p className="text-gray-400 text-xs uppercase tracking-wide mb-2">
                Blocked Trade Statistics
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1 text-sm">
                {Object.entries(BLOCKED_LABELS).map(([key, label]) => (
                  <DiagRow
                    key={key}
                    label={label}
                    value={(diagnostics.blocked as unknown as Record<string, number>)[key] ?? 0}
                    danger={
                      ((diagnostics.blocked as unknown as Record<string, number>)[key] ?? 0) > 0
                    }
                  />
                ))}
              </div>
              {diagnostics.blocked.last_reason && (
                <p className="text-xs text-gray-500 mt-2 break-words">
                  Last block ({diagnostics.blocked.last_category}):{' '}
                  {diagnostics.blocked.last_reason} ({fmtDiagTime(diagnostics.blocked.last_at)})
                </p>
              )}
            </div>
          </div>
        ) : (
          <div className="text-center py-6 text-gray-400">
            <p>No diagnostics yet</p>
          </div>
        )}
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
