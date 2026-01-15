import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Settings as SettingsIcon,
  Database,
  Newspaper,
  TrendingUp,
  Activity,
  MessageCircle,
  BarChart3,
  CheckCircle,
  XCircle,
  AlertCircle,
  RefreshCw,
  Power,
  PowerOff,
  Loader2,
  ChevronDown,
  ChevronUp,
  Info,
} from 'lucide-react'
import { useToast } from '../components/Toast'

// Data source types
interface DataSourceConfig {
  enabled: boolean
  has_api_key: boolean
  refresh_interval_seconds: number
  cache_ttl_seconds: number
  settings: Record<string, unknown>
  healthy: boolean
  last_fetch: string | null
  last_error: string | null
  data_age_seconds: number | null
}

interface AggregatedSignals {
  overall_sentiment: number
  confidence: number
  signal: string
  contributing_sources: string[]
  timestamp: string
  details: Record<string, unknown>
}

// API functions
async function fetchDataSources(): Promise<Record<string, DataSourceConfig>> {
  const res = await fetch('/api/data-sources/sources')
  if (!res.ok) throw new Error('Failed to fetch data sources')
  return res.json()
}

async function fetchSignals(): Promise<AggregatedSignals> {
  const res = await fetch('/api/data-sources/signals')
  if (!res.ok) throw new Error('Failed to fetch signals')
  return res.json()
}

async function updateSource(
  sourceType: string,
  data: { enabled?: boolean; api_key?: string }
): Promise<DataSourceConfig> {
  const res = await fetch(`/api/data-sources/sources/${sourceType}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error('Failed to update source')
  return res.json()
}

async function bulkUpdateSources(enabled: boolean): Promise<Record<string, DataSourceConfig>> {
  const res = await fetch('/api/data-sources/sources/bulk', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  })
  if (!res.ok) throw new Error('Failed to update sources')
  return res.json()
}

// Source metadata
const SOURCE_INFO: Record<string, {
  name: string
  icon: typeof Database
  description: string
  category: 'market' | 'sentiment'
  apiRequired: boolean
  apiLink?: string
}> = {
  news_api: {
    name: 'News API',
    icon: Newspaper,
    description: 'Crypto news headlines with sentiment analysis',
    category: 'sentiment',
    apiRequired: true,
    apiLink: 'https://newsapi.org/',
  },
  fear_greed: {
    name: 'Fear & Greed Index',
    icon: TrendingUp,
    description: 'Market sentiment indicator (0-100)',
    category: 'sentiment',
    apiRequired: false,
  },
  onchain_metrics: {
    name: 'On-chain Metrics',
    icon: Activity,
    description: 'Blockchain activity and whale tracking',
    category: 'market',
    apiRequired: true,
  },
  social_sentiment: {
    name: 'Social Sentiment',
    icon: MessageCircle,
    description: 'Twitter/Reddit mentions and sentiment',
    category: 'sentiment',
    apiRequired: true,
  },
  market_conditions: {
    name: 'Market Conditions',
    icon: BarChart3,
    description: 'Overall market health (BTC dominance, market cap)',
    category: 'market',
    apiRequired: false,
  },
}

// Toggle switch component - #156, #159: Accessible toggle
function Toggle({
  enabled,
  onToggle,
  disabled,
  label,
}: {
  enabled: boolean
  onToggle: () => void
  disabled?: boolean
  label?: string
}) {
  return (
    <button
      onClick={onToggle}
      disabled={disabled}
      role="switch"
      aria-checked={enabled}
      aria-label={label}
      className={`relative w-12 h-6 rounded-full transition-colors ${
        enabled ? 'bg-accent' : 'bg-gray-600'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
    >
      <span
        className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
          enabled ? 'left-7' : 'left-1'
        }`}
        aria-hidden="true"
      />
    </button>
  )
}

// Status indicator component
function StatusIndicator({ healthy, lastError }: { healthy: boolean; lastError: string | null }) {
  if (healthy) {
    return <CheckCircle size={16} className="text-profit" title="Healthy" />
  }
  return (
    <XCircle
      size={16}
      className="text-loss"
      title={lastError || 'Error'}
    />
  )
}

// Data source card component
function DataSourceCard({
  sourceType,
  config,
  onToggle,
  isUpdating,
}: {
  sourceType: string
  config: DataSourceConfig
  onToggle: () => void
  isUpdating: boolean
}) {
  const [expanded, setExpanded] = useState(false)
  const info = SOURCE_INFO[sourceType]
  const Icon = info?.icon || Database

  const formatAge = (seconds: number | null) => {
    if (seconds === null) return 'Never'
    if (seconds < 60) return `${seconds}s ago`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
    return `${Math.floor(seconds / 3600)}h ago`
  }

  return (
    <div className={`bg-gray-800 rounded-lg border ${config.enabled ? 'border-accent/30' : 'border-gray-700'}`}>
      <div className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${config.enabled ? 'bg-accent/20' : 'bg-gray-700'}`}>
              <Icon size={20} className={config.enabled ? 'text-accent' : 'text-gray-400'} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h4 className="font-medium">{info?.name || sourceType}</h4>
                {config.enabled && <StatusIndicator healthy={config.healthy} lastError={config.last_error} />}
                {info?.category === 'sentiment' && (
                  <span className="px-2 py-0.5 text-xs bg-yellow-500/20 text-yellow-500 rounded">
                    Sentiment
                  </span>
                )}
              </div>
              <p className="text-sm text-gray-400">{info?.description}</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {isUpdating ? (
              <Loader2 size={20} className="animate-spin text-accent" />
            ) : (
              <Toggle
                enabled={config.enabled}
                onToggle={onToggle}
                disabled={isUpdating}
                label={`${config.enabled ? 'Disable' : 'Enable'} ${info?.name || sourceType}`}
              />
            )}
            <button
              onClick={() => setExpanded(!expanded)}
              className="p-1 text-gray-400 hover:text-white"
              aria-expanded={expanded}
              aria-label={`${expanded ? 'Collapse' : 'Expand'} details for ${info?.name || sourceType}`}
            >
              {expanded ? <ChevronUp size={18} aria-hidden="true" /> : <ChevronDown size={18} aria-hidden="true" />}
            </button>
          </div>
        </div>

        {/* Expanded details */}
        {expanded && (
          <div className="mt-4 pt-4 border-t border-gray-700 space-y-3">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-400">Status:</span>{' '}
                <span className={config.enabled ? 'text-profit' : 'text-gray-500'}>
                  {config.enabled ? 'Enabled' : 'Disabled'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">API Key:</span>{' '}
                <span className={config.has_api_key ? 'text-profit' : 'text-gray-500'}>
                  {config.has_api_key ? 'Configured' : 'Not set'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Last Fetch:</span>{' '}
                <span className="text-white">{formatAge(config.data_age_seconds)}</span>
              </div>
              <div>
                <span className="text-gray-400">Refresh:</span>{' '}
                <span className="text-white">{config.refresh_interval_seconds}s</span>
              </div>
            </div>
            {config.last_error && (
              <div className="p-3 bg-loss/10 border border-loss/30 rounded-lg">
                <p className="text-sm text-loss">{config.last_error}</p>
              </div>
            )}
            {info?.apiRequired && !config.has_api_key && (
              <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <div className="flex items-start gap-2">
                  <AlertCircle size={16} className="text-yellow-500 mt-0.5" />
                  <div className="text-sm">
                    <p className="text-yellow-500">API key required</p>
                    {info.apiLink && (
                      <a
                        href={info.apiLink}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-accent hover:underline"
                      >
                        Get API key
                      </a>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// Signals panel component
function SignalsPanel({ signals }: { signals: AggregatedSignals | undefined }) {
  if (!signals || signals.contributing_sources.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-center">
        <Info size={32} className="mx-auto mb-2 text-gray-500" />
        <p className="text-gray-400">Enable data sources to see aggregated signals</p>
      </div>
    )
  }

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'bullish': return 'text-profit'
      case 'bearish': return 'text-loss'
      default: return 'text-gray-400'
    }
  }

  const getSentimentBar = (value: number) => {
    const percent = ((value + 1) / 2) * 100
    return (
      <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all ${value >= 0 ? 'bg-profit' : 'bg-loss'}`}
          style={{ width: `${percent}%`, marginLeft: value < 0 ? `${percent}%` : '50%', transform: value < 0 ? 'translateX(-100%)' : 'none' }}
        />
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h3 className="text-lg font-semibold mb-4">Aggregated Trading Signals</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div>
          <p className="text-gray-400 text-sm mb-1">Overall Signal</p>
          <p className={`text-2xl font-bold capitalize ${getSignalColor(signals.signal)}`}>
            {signals.signal}
          </p>
        </div>
        <div>
          <p className="text-gray-400 text-sm mb-1">Sentiment</p>
          <p className={`text-xl font-mono ${signals.overall_sentiment >= 0 ? 'text-profit' : 'text-loss'}`}>
            {signals.overall_sentiment >= 0 ? '+' : ''}{(signals.overall_sentiment * 100).toFixed(1)}%
          </p>
          {getSentimentBar(signals.overall_sentiment)}
        </div>
        <div>
          <p className="text-gray-400 text-sm mb-1">Confidence</p>
          <p className="text-xl font-mono text-white">
            {(signals.confidence * 100).toFixed(0)}%
          </p>
          <p className="text-xs text-gray-500">
            {signals.contributing_sources.length} source(s) active
          </p>
        </div>
      </div>
    </div>
  )
}

export default function Settings() {
  const queryClient = useQueryClient()
  const toast = useToast()
  const [updatingSource, setUpdatingSource] = useState<string | null>(null)

  const { data: sources, isLoading } = useQuery({
    queryKey: ['data-sources'],
    queryFn: fetchDataSources,
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const { data: signals } = useQuery({
    queryKey: ['data-signals'],
    queryFn: fetchSignals,
    refetchInterval: 60000, // Refresh every minute
  })

  const toggleMutation = useMutation({
    mutationFn: ({ sourceType, enabled }: { sourceType: string; enabled: boolean }) =>
      updateSource(sourceType, { enabled }),
    onSuccess: (_, { sourceType, enabled }) => {
      queryClient.invalidateQueries({ queryKey: ['data-sources'] })
      queryClient.invalidateQueries({ queryKey: ['data-signals'] })
      const sourceName = SOURCE_INFO[sourceType]?.name || sourceType
      toast.success(
        enabled ? 'Source Enabled' : 'Source Disabled',
        `${sourceName} has been ${enabled ? 'enabled' : 'disabled'}`
      )
      setUpdatingSource(null)
    },
    onError: (error) => {
      toast.error('Update Failed', error instanceof Error ? error.message : 'Failed to update source')
      setUpdatingSource(null)
    },
  })

  const bulkMutation = useMutation({
    mutationFn: bulkUpdateSources,
    onSuccess: (_, enabled) => {
      queryClient.invalidateQueries({ queryKey: ['data-sources'] })
      queryClient.invalidateQueries({ queryKey: ['data-signals'] })
      toast.success(
        enabled ? 'All Sources Enabled' : 'All Sources Disabled',
        `All external data sources have been ${enabled ? 'enabled' : 'disabled'}`
      )
    },
    onError: (error) => {
      toast.error('Update Failed', error instanceof Error ? error.message : 'Failed to update sources')
    },
  })

  const handleToggleSource = (sourceType: string, currentEnabled: boolean) => {
    setUpdatingSource(sourceType)
    toggleMutation.mutate({ sourceType, enabled: !currentEnabled })
  }

  const enabledCount = sources ? Object.values(sources).filter(s => s.enabled).length : 0
  const totalCount = sources ? Object.keys(sources).length : 0

  // Separate sources by category
  const marketSources = sources
    ? Object.entries(sources).filter(([key]) => SOURCE_INFO[key]?.category === 'market')
    : []
  const sentimentSources = sources
    ? Object.entries(sources).filter(([key]) => SOURCE_INFO[key]?.category === 'sentiment')
    : []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <SettingsIcon size={32} className="text-accent" />
          <div>
            <h2 className="text-2xl font-bold">External Data Sources</h2>
            <p className="text-gray-400 text-sm">
              Configure external data feeds for trading signals
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">
            {enabledCount}/{totalCount} enabled
          </span>
          <button
            onClick={() => bulkMutation.mutate(true)}
            disabled={bulkMutation.isPending}
            className="flex items-center gap-2 px-3 py-2 bg-profit/20 text-profit hover:bg-profit/30 rounded-lg text-sm disabled:opacity-50"
          >
            <Power size={16} />
            Enable All
          </button>
          <button
            onClick={() => bulkMutation.mutate(false)}
            disabled={bulkMutation.isPending}
            className="flex items-center gap-2 px-3 py-2 bg-gray-700 text-gray-300 hover:bg-gray-600 rounded-lg text-sm disabled:opacity-50"
          >
            <PowerOff size={16} />
            Disable All
          </button>
        </div>
      </div>

      {/* Info banner about sentiment sources */}
      <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <AlertCircle size={20} className="text-yellow-500 mt-0.5" />
          <div>
            <h4 className="font-medium text-yellow-500">About Sentiment Sources</h4>
            <p className="text-sm text-gray-300 mt-1">
              Sentiment-based sources (News, Fear/Greed, Social) provide market mood indicators but may be noisy.
              Consider starting with Market Data sources only (Market Conditions, On-chain Metrics) and
              enable sentiment sources gradually if needed.
            </p>
          </div>
        </div>
      </div>

      {/* Aggregated signals panel */}
      <SignalsPanel signals={signals} />

      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent"></div>
        </div>
      ) : (
        <>
          {/* Market Data Sources */}
          <div>
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <BarChart3 size={20} className="text-accent" />
              Market Data Sources
            </h3>
            <div className="space-y-3">
              {marketSources.map(([sourceType, config]) => (
                <DataSourceCard
                  key={sourceType}
                  sourceType={sourceType}
                  config={config}
                  onToggle={() => handleToggleSource(sourceType, config.enabled)}
                  isUpdating={updatingSource === sourceType}
                />
              ))}
            </div>
          </div>

          {/* Sentiment Sources */}
          <div>
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <MessageCircle size={20} className="text-yellow-500" />
              Sentiment Sources
              <span className="text-xs px-2 py-1 bg-yellow-500/20 text-yellow-500 rounded">
                Use with caution
              </span>
            </h3>
            <div className="space-y-3">
              {sentimentSources.map(([sourceType, config]) => (
                <DataSourceCard
                  key={sourceType}
                  sourceType={sourceType}
                  config={config}
                  onToggle={() => handleToggleSource(sourceType, config.enabled)}
                  isUpdating={updatingSource === sourceType}
                />
              ))}
            </div>
          </div>
        </>
      )}

      {/* Server Info */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Configuration</h3>
        <p className="text-gray-400 text-sm mb-4">
          API keys and advanced settings are configured in{' '}
          <code className="text-accent">backend/data_sources.yaml</code>
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Config File:</span>{' '}
            <code className="text-accent">data_sources.yaml</code>
          </div>
          <div>
            <span className="text-gray-400">API Endpoint:</span>{' '}
            <code className="text-accent">/api/data-sources</code>
          </div>
        </div>
      </div>
    </div>
  )
}
