/**
 * Strategy Reason Panel
 *
 * Shows for each bot:
 * - Current strategy
 * - Current regime
 * - Eligible strategies
 * - Blocked strategies with reasons:
 *   - Cooldown
 *   - Capacity
 *   - Regime mismatch
 *   - Risk
 */

import React from 'react';
import { CheckCircle, XCircle, Clock, TrendingUp, AlertTriangle, Activity } from 'lucide-react';

interface BlockedStrategy {
  strategy_name: string;
  blocked_reason: string;
}

interface StrategyReason {
  current_strategy: string;
  current_regime: string | null;
  eligible_strategies: string[];
  blocked_strategies: BlockedStrategy[];
}

interface StrategyReasonPanelProps {
  botId: number;
  botName: string;
  isSimulated: boolean;
}

export const StrategyReasonPanel: React.FC<StrategyReasonPanelProps> = ({
  botId,
  botName,
  isSimulated,
}) => {
  const [data, setData] = React.useState<StrategyReason | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [expanded, setExpanded] = React.useState(false);

  React.useEffect(() => {
    const fetchStrategyReason = async () => {
      try {
        setLoading(true);
        const response = await fetch(
          `/api/reports/strategy-reason/${botId}?is_simulated=${isSimulated}`
        );

        if (!response.ok) {
          throw new Error('Failed to fetch strategy reasoning');
        }

        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchStrategyReason();
  }, [botId, isSimulated]);

  const getBlockReasonIcon = (reason: string) => {
    const lowerReason = reason.toLowerCase();
    if (lowerReason.includes('cooldown')) return Clock;
    if (lowerReason.includes('capacity')) return AlertTriangle;
    if (lowerReason.includes('regime')) return Activity;
    if (lowerReason.includes('risk')) return TrendingUp;
    return XCircle;
  };

  const getBlockReasonColor = (reason: string) => {
    const lowerReason = reason.toLowerCase();
    if (lowerReason.includes('cooldown')) return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30';
    if (lowerReason.includes('capacity')) return 'text-orange-400 bg-orange-500/10 border-orange-500/30';
    if (lowerReason.includes('regime')) return 'text-blue-400 bg-blue-500/10 border-blue-500/30';
    if (lowerReason.includes('risk')) return 'text-red-400 bg-red-500/10 border-red-500/30';
    return 'text-gray-400 bg-gray-500/10 border-gray-500/30';
  };

  const formatStrategyName = (name: string) => {
    return name
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  if (loading) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-700 rounded w-1/3 mb-2"></div>
          <div className="h-3 bg-gray-700 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
        <p className="text-red-400 text-sm">{error || 'Strategy data unavailable'}</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg">
      {/* Header - Always Visible */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full p-4 flex items-center justify-between hover:bg-gray-700/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Activity className="h-5 w-5 text-accent" />
          <div className="text-left">
            <h4 className="text-white font-semibold">{botName}</h4>
            <p className="text-sm text-gray-400">
              Current: <span className="text-accent font-semibold">{formatStrategyName(data.current_strategy)}</span>
              {data.current_regime && (
                <span className="ml-2">
                  • Regime: <span className="text-blue-400">{data.current_regime}</span>
                </span>
              )}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs px-2 py-1 rounded bg-green-500/20 text-green-400 border border-green-500/30">
            {data.eligible_strategies.length} eligible
          </span>
          {data.blocked_strategies.length > 0 && (
            <span className="text-xs px-2 py-1 rounded bg-red-500/20 text-red-400 border border-red-500/30">
              {data.blocked_strategies.length} blocked
            </span>
          )}
          <svg
            className={`h-5 w-5 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Expanded Content */}
      {expanded && (
        <div className="border-t border-gray-700 p-4">
          {/* Eligible Strategies */}
          <div className="mb-4">
            <h5 className="text-sm font-semibold text-gray-400 mb-2 flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-400" />
              Eligible Strategies
            </h5>
            <div className="flex flex-wrap gap-2">
              {data.eligible_strategies.map((strategy) => (
                <div
                  key={strategy}
                  className={`px-3 py-2 rounded border ${
                    strategy === data.current_strategy
                      ? 'bg-accent/20 text-accent border-accent/30 font-semibold'
                      : 'bg-green-500/10 text-green-400 border-green-500/30'
                  }`}
                >
                  <span className="text-sm">{formatStrategyName(strategy)}</span>
                  {strategy === data.current_strategy && (
                    <span className="ml-2 text-xs opacity-75">(Active)</span>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Blocked Strategies */}
          {data.blocked_strategies.length > 0 && (
            <div>
              <h5 className="text-sm font-semibold text-gray-400 mb-2 flex items-center gap-2">
                <XCircle className="h-4 w-4 text-red-400" />
                Blocked Strategies
              </h5>
              <div className="space-y-2">
                {data.blocked_strategies.map((blocked) => {
                  const Icon = getBlockReasonIcon(blocked.blocked_reason);
                  const colorClass = getBlockReasonColor(blocked.blocked_reason);

                  return (
                    <div
                      key={blocked.strategy_name}
                      className={`p-3 rounded border ${colorClass}`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Icon className="h-4 w-4" />
                          <span className="font-semibold text-sm">
                            {formatStrategyName(blocked.strategy_name)}
                          </span>
                        </div>
                        <span className="text-xs px-2 py-1 rounded bg-black/20">
                          {blocked.blocked_reason}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Summary */}
          <div className="mt-4 pt-4 border-t border-gray-700">
            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="bg-gray-700/50 p-2 rounded">
                <p className="text-xs text-gray-400">Total Strategies</p>
                <p className="text-lg font-mono-numbers text-white font-semibold">
                  {data.eligible_strategies.length + data.blocked_strategies.length}
                </p>
              </div>
              <div className="bg-green-500/10 p-2 rounded border border-green-500/30">
                <p className="text-xs text-green-400">Eligible</p>
                <p className="text-lg font-mono-numbers text-green-400 font-semibold">
                  {data.eligible_strategies.length}
                </p>
              </div>
              <div className="bg-red-500/10 p-2 rounded border border-red-500/30">
                <p className="text-xs text-red-400">Blocked</p>
                <p className="text-lg font-mono-numbers text-red-400 font-semibold">
                  {data.blocked_strategies.length}
                </p>
              </div>
            </div>
          </div>

          {/* Regime Info (if available) */}
          {data.current_regime && (
            <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded">
              <div className="flex items-center gap-2 mb-1">
                <Activity className="h-4 w-4 text-blue-400" />
                <span className="text-sm font-semibold text-blue-400">Market Regime</span>
              </div>
              <p className="text-white font-semibold">{data.current_regime}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
