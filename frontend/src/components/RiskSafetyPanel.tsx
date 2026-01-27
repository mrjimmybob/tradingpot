/**
 * Risk & Safety Panel
 *
 * Displays global and per-bot risk metrics:
 * - Drawdown percentage
 * - Daily loss percentage
 * - Portfolio exposure
 * - Strategy capacity
 * - Kill switch state
 * - Last risk event
 *
 * Color coding:
 * - Green: Safe
 * - Amber: Near limit
 * - Red: Blocked/Critical
 */

import React from 'react';
import { AlertTriangle, Shield, TrendingDown, Activity, AlertCircle, CheckCircle, XCircle, PauseCircle } from 'lucide-react';

interface BotRiskInfo {
  bot_id: number;
  bot_name: string;
  drawdown_pct: number;
  daily_loss_pct: number;
  strategy_capacity_pct: number;
  kill_switch_state: string;
  last_risk_event: {
    timestamp: string;
    type: string;
    message: string;
  } | null;
}

interface RiskStatus {
  bots: BotRiskInfo[];
  portfolio: {
    total_exposure_pct: number;
    total_exposure_usd: number;
    total_portfolio_value: number;
    loss_caps_remaining: {
      daily: number | null;
      weekly: number | null;
    };
  };
}

interface RiskSafetyPanelProps {
  isSimulated: boolean;
  ownerId?: string;
  refreshInterval?: number; // milliseconds
}

export const RiskSafetyPanel: React.FC<RiskSafetyPanelProps> = ({
  isSimulated,
  ownerId,
  refreshInterval = 10000, // 10 seconds default
}) => {
  const [data, setData] = React.useState<RiskStatus | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  const fetchRiskStatus = React.useCallback(async () => {
    try {
      let url = `/api/reports/risk-status?is_simulated=${isSimulated}`;
      if (ownerId) {
        url += `&owner_id=${ownerId}`;
      }

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error('Failed to fetch risk status');
      }

      const result = await response.json();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [isSimulated, ownerId]);

  React.useEffect(() => {
    fetchRiskStatus();
    const interval = setInterval(fetchRiskStatus, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchRiskStatus, refreshInterval]);

  const getRiskColor = (value: number, warningThreshold: number, criticalThreshold: number) => {
    if (value >= criticalThreshold) return 'text-red-400 bg-red-500/10 border-red-500/30';
    if (value >= warningThreshold) return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30';
    return 'text-green-400 bg-green-500/10 border-green-500/30';
  };

  const getKillSwitchIcon = (state: string) => {
    switch (state) {
      case 'active':
        return <CheckCircle className="h-5 w-5 text-green-400" />;
      case 'paused':
        return <PauseCircle className="h-5 w-5 text-yellow-400" />;
      case 'stopped':
        return <XCircle className="h-5 w-5 text-red-400" />;
      default:
        return <AlertCircle className="h-5 w-5 text-gray-400" />;
    }
  };

  const getKillSwitchColor = (state: string) => {
    switch (state) {
      case 'active':
        return 'bg-green-500/10 border-green-500/30';
      case 'paused':
        return 'bg-yellow-500/10 border-yellow-500/30';
      case 'stopped':
        return 'bg-red-500/10 border-red-500/30';
      default:
        return 'bg-gray-500/10 border-gray-500/30';
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-accent"></div>
          <p className="text-gray-400 ml-3">Loading risk status...</p>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6">
        <div className="flex items-center gap-3">
          <AlertTriangle className="h-6 w-6 text-red-400" />
          <div>
            <h3 className="text-red-400 font-semibold">Risk Status Unavailable</h3>
            <p className="text-red-300 text-sm">{error || 'Failed to load risk data'}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Portfolio-Level Risk */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <Shield className="h-5 w-5 text-accent" />
          <h3 className="text-lg font-semibold text-white">Portfolio Risk</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Total Exposure */}
          <div className={`p-4 rounded border ${getRiskColor(data.portfolio.total_exposure_pct, 70, 85)}`}>
            <div className="flex items-center gap-2 mb-2">
              <Activity className="h-4 w-4" />
              <p className="text-xs font-semibold uppercase">Exposure</p>
            </div>
            <p className="text-2xl font-mono-numbers font-bold">
              {data.portfolio.total_exposure_pct.toFixed(1)}%
            </p>
            <p className="text-xs mt-1 opacity-75">
              ${data.portfolio.total_exposure_usd.toFixed(2)} / ${data.portfolio.total_portfolio_value.toFixed(2)}
            </p>
          </div>

          {/* Daily Loss Cap */}
          <div className={`p-4 rounded border ${
            data.portfolio.loss_caps_remaining.daily !== null && data.portfolio.loss_caps_remaining.daily <= 0
              ? 'text-red-400 bg-red-500/10 border-red-500/30'
              : 'text-gray-400 bg-gray-700/50 border-gray-600'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              <TrendingDown className="h-4 w-4" />
              <p className="text-xs font-semibold uppercase">Daily Loss Cap</p>
            </div>
            <p className="text-2xl font-mono-numbers font-bold">
              {data.portfolio.loss_caps_remaining.daily !== null
                ? `$${data.portfolio.loss_caps_remaining.daily.toFixed(2)}`
                : 'Unlimited'}
            </p>
            <p className="text-xs mt-1 opacity-75">Remaining today</p>
          </div>

          {/* Weekly Loss Cap */}
          <div className={`p-4 rounded border ${
            data.portfolio.loss_caps_remaining.weekly !== null && data.portfolio.loss_caps_remaining.weekly <= 0
              ? 'text-red-400 bg-red-500/10 border-red-500/30'
              : 'text-gray-400 bg-gray-700/50 border-gray-600'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              <TrendingDown className="h-4 w-4" />
              <p className="text-xs font-semibold uppercase">Weekly Loss Cap</p>
            </div>
            <p className="text-2xl font-mono-numbers font-bold">
              {data.portfolio.loss_caps_remaining.weekly !== null
                ? `$${data.portfolio.loss_caps_remaining.weekly.toFixed(2)}`
                : 'Unlimited'}
            </p>
            <p className="text-xs mt-1 opacity-75">Remaining this week</p>
          </div>
        </div>
      </div>

      {/* Per-Bot Risk */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <AlertTriangle className="h-5 w-5 text-accent" />
          <h3 className="text-lg font-semibold text-white">Bot Risk Status</h3>
          <span className="text-xs text-gray-400 ml-auto">{data.bots.length} bots</span>
        </div>

        {data.bots.length === 0 ? (
          <p className="text-gray-400 text-center py-8">No bots found</p>
        ) : (
          <div className="space-y-3">
            {data.bots.map((bot) => (
              <div
                key={bot.bot_id}
                className="bg-gray-700/50 rounded-lg p-4 border border-gray-600 hover:border-gray-500 transition-colors"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <h4 className="text-white font-semibold">{bot.bot_name}</h4>
                    <p className="text-xs text-gray-400">Bot #{bot.bot_id}</p>
                  </div>
                  <div className={`flex items-center gap-2 px-3 py-1 rounded border ${getKillSwitchColor(bot.kill_switch_state)}`}>
                    {getKillSwitchIcon(bot.kill_switch_state)}
                    <span className="text-xs font-semibold uppercase">{bot.kill_switch_state}</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {/* Drawdown */}
                  <div className={`p-2 rounded text-center border ${getRiskColor(bot.drawdown_pct, 15, 25)}`}>
                    <p className="text-xs opacity-75">Drawdown</p>
                    <p className="text-lg font-mono-numbers font-bold">
                      {bot.drawdown_pct.toFixed(1)}%
                    </p>
                  </div>

                  {/* Daily Loss */}
                  <div className={`p-2 rounded text-center border ${getRiskColor(bot.daily_loss_pct, 5, 10)}`}>
                    <p className="text-xs opacity-75">Daily Loss</p>
                    <p className="text-lg font-mono-numbers font-bold">
                      {bot.daily_loss_pct.toFixed(1)}%
                    </p>
                  </div>

                  {/* Strategy Capacity */}
                  <div className={`p-2 rounded text-center border ${getRiskColor(bot.strategy_capacity_pct, 70, 90)}`}>
                    <p className="text-xs opacity-75">Capacity</p>
                    <p className="text-lg font-mono-numbers font-bold">
                      {bot.strategy_capacity_pct.toFixed(0)}%
                    </p>
                  </div>

                  {/* Last Risk Event */}
                  <div className={`p-2 rounded text-center ${
                    bot.last_risk_event
                      ? 'bg-yellow-500/10 border border-yellow-500/30 text-yellow-400'
                      : 'bg-gray-600 border border-gray-600 text-gray-400'
                  }`}>
                    <p className="text-xs opacity-75">Last Event</p>
                    <p className="text-xs font-semibold truncate">
                      {bot.last_risk_event ? bot.last_risk_event.type : 'None'}
                    </p>
                  </div>
                </div>

                {/* Last Risk Event Detail */}
                {bot.last_risk_event && (
                  <div className="mt-3 p-2 bg-gray-800 rounded text-xs">
                    <p className="text-gray-400">
                      <span className="font-semibold text-white">
                        {new Date(bot.last_risk_event.timestamp).toLocaleString()}:
                      </span>{' '}
                      {bot.last_risk_event.message}
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Risk Legend */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <p className="text-xs font-semibold text-gray-400 mb-2">Risk Color Legend</p>
        <div className="flex flex-wrap gap-3 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-green-500/10 border border-green-500/30"></div>
            <span className="text-gray-300">Safe</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-yellow-500/10 border border-yellow-500/30"></div>
            <span className="text-gray-300">Near Limit</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-red-500/10 border border-red-500/30"></div>
            <span className="text-gray-300">Critical/Blocked</span>
          </div>
        </div>
      </div>
    </div>
  );
};
