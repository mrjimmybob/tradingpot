/**
 * Equity Curve with Events
 *
 * Enhanced P&L chart with event markers:
 * - Strategy switches
 * - Kill switches
 * - Regime changes
 * - Grid re-centers
 * - Large losses
 *
 * Hover tooltip shows event type + reason
 */

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
  Legend,
} from 'recharts';
import { Activity, AlertTriangle, TrendingDown, Zap, RotateCcw, Shuffle } from 'lucide-react';

interface EquityPoint {
  timestamp: string;
  equity: number;
}

interface EquityEvent {
  timestamp: string;
  event_type: string;
  description: string;
  bot_id: number | null;
}

interface EquityCurveData {
  curve: EquityPoint[];
  events: EquityEvent[];
}

interface EquityCurveWithEventsProps {
  isSimulated: boolean;
  ownerId?: string;
  asset?: string;
  height?: number;
}

const eventColors = {
  strategy_switch: '#6366f1', // indigo
  kill_switch: '#ef4444', // red
  regime_change: '#f59e0b', // amber
  grid_recenter: '#8b5cf6', // purple
  large_loss: '#dc2626', // dark red
  drawdown: '#f97316', // orange
  alert: '#64748b', // slate
};

const eventIcons = {
  strategy_switch: Shuffle,
  kill_switch: AlertTriangle,
  regime_change: Activity,
  grid_recenter: RotateCcw,
  large_loss: TrendingDown,
  drawdown: TrendingDown,
  alert: Zap,
};

export const EquityCurveWithEvents: React.FC<EquityCurveWithEventsProps> = ({
  isSimulated,
  ownerId,
  asset = 'USDT',
  height = 400,
}) => {
  const [data, setData] = React.useState<EquityCurveData | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [showEvents, setShowEvents] = React.useState(true);
  const [selectedEventTypes, setSelectedEventTypes] = React.useState<Set<string>>(
    new Set(Object.keys(eventColors))
  );

  React.useEffect(() => {
    const fetchEquityCurve = async () => {
      try {
        setLoading(true);
        let url = `/api/reports/equity-curve?is_simulated=${isSimulated}&asset=${asset}`;
        if (ownerId) {
          url += `&owner_id=${ownerId}`;
        }

        const response = await fetch(url);

        if (!response.ok) {
          throw new Error('Failed to fetch equity curve');
        }

        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchEquityCurve();
  }, [isSimulated, ownerId, asset]);

  const toggleEventType = (eventType: string) => {
    setSelectedEventTypes((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(eventType)) {
        newSet.delete(eventType);
      } else {
        newSet.add(eventType);
      }
      return newSet;
    });
  };

  if (loading) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-center" style={{ height }}>
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent"></div>
          <p className="text-gray-400 ml-3">Loading equity curve...</p>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-center" style={{ height }}>
          <p className="text-red-400">{error || 'Failed to load equity curve'}</p>
        </div>
      </div>
    );
  }

  // Prepare chart data by merging equity points with events
  const chartData = data.curve.map((point) => {
    const timestamp = new Date(point.timestamp).getTime();
    const eventsAtTime = data.events.filter(
      (e) => Math.abs(new Date(e.timestamp).getTime() - timestamp) < 60000 // Within 1 minute
    );

    return {
      timestamp: new Date(point.timestamp).toLocaleDateString(),
      timestampFull: point.timestamp,
      equity: point.equity,
      events: eventsAtTime,
    };
  });

  // Sample data for performance if too many points
  const maxPoints = 200;
  const sampledData =
    chartData.length > maxPoints
      ? chartData.filter((_, i) => i % Math.ceil(chartData.length / maxPoints) === 0)
      : chartData;

  // Filter events by selected types
  const filteredEvents = data.events.filter((e) => selectedEventTypes.has(e.event_type));

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload[0]) return null;

    const dataPoint = payload[0].payload;

    return (
      <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg">
        <p className="text-gray-400 text-xs mb-1">{new Date(dataPoint.timestampFull).toLocaleString()}</p>
        <p className="text-white font-mono-numbers font-semibold text-lg">
          ${dataPoint.equity.toFixed(2)}
        </p>

        {dataPoint.events && dataPoint.events.length > 0 && (
          <div className="mt-2 pt-2 border-t border-gray-700">
            <p className="text-xs font-semibold text-gray-400 mb-1">Events:</p>
            {dataPoint.events.map((event: EquityEvent, idx: number) => {
              const EventIcon = eventIcons[event.event_type as keyof typeof eventIcons] || Zap;
              const color = eventColors[event.event_type as keyof typeof eventColors] || '#64748b';

              return (
                <div key={idx} className="flex items-start gap-2 mb-1">
                  <EventIcon className="h-3 w-3 mt-0.5 flex-shrink-0" style={{ color }} />
                  <p className="text-xs text-gray-300">{event.description}</p>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  // Get unique event types for legend
  const eventTypeCounts = data.events.reduce((acc, event) => {
    acc[event.event_type] = (acc[event.event_type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Equity Curve with Events</h3>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-sm text-gray-400">
            <input
              type="checkbox"
              checked={showEvents}
              onChange={(e) => setShowEvents(e.target.checked)}
              className="rounded bg-gray-700 border-gray-600 text-accent focus:ring-accent"
            />
            Show Events
          </label>
        </div>
      </div>

      {/* Event Type Filters */}
      {showEvents && Object.keys(eventTypeCounts).length > 0 && (
        <div className="mb-4 flex flex-wrap gap-2">
          {Object.entries(eventTypeCounts).map(([eventType, count]) => {
            const EventIcon = eventIcons[eventType as keyof typeof eventIcons] || Zap;
            const color = eventColors[eventType as keyof typeof eventColors] || '#64748b';
            const isSelected = selectedEventTypes.has(eventType);

            return (
              <button
                key={eventType}
                onClick={() => toggleEventType(eventType)}
                className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-all ${
                  isSelected
                    ? 'bg-gray-700 border-2'
                    : 'bg-gray-800 border border-gray-600 opacity-50'
                }`}
                style={{
                  borderColor: isSelected ? color : undefined,
                  color: isSelected ? color : '#9ca3af',
                }}
              >
                <EventIcon className="h-3 w-3" />
                <span className="font-semibold">{eventType.replace(/_/g, ' ')}</span>
                <span className="opacity-75">({count})</span>
              </button>
            );
          })}
        </div>
      )}

      {/* Chart */}
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={sampledData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="timestamp"
              stroke="#9ca3af"
              style={{ fontSize: '12px' }}
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis
              stroke="#9ca3af"
              style={{ fontSize: '12px' }}
              tickFormatter={(value) => `$${value.toFixed(0)}`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="equity"
              stroke="#6366f1"
              strokeWidth={2}
              dot={false}
              animationDuration={sampledData.length > 100 ? 0 : 1000}
            />

            {/* Event Markers */}
            {showEvents &&
              filteredEvents.map((event, idx) => {
                const dataPoint = sampledData.find(
                  (d) =>
                    Math.abs(new Date(d.timestampFull).getTime() - new Date(event.timestamp).getTime()) <
                    60000
                );

                if (!dataPoint) return null;

                const color = eventColors[event.event_type as keyof typeof eventColors] || '#64748b';

                return (
                  <ReferenceDot
                    key={idx}
                    x={dataPoint.timestamp}
                    y={dataPoint.equity}
                    r={6}
                    fill={color}
                    stroke="#1f2937"
                    strokeWidth={2}
                    isFront={true}
                  />
                );
              })}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Stats */}
      {data.curve.length > 0 && (
        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-gray-700/50 p-2 rounded">
            <p className="text-xs text-gray-400">Starting Equity</p>
            <p className="text-sm font-mono-numbers text-white font-semibold">
              ${data.curve[0].equity.toFixed(2)}
            </p>
          </div>
          <div className="bg-gray-700/50 p-2 rounded">
            <p className="text-xs text-gray-400">Current Equity</p>
            <p className="text-sm font-mono-numbers text-white font-semibold">
              ${data.curve[data.curve.length - 1].equity.toFixed(2)}
            </p>
          </div>
          <div className="bg-gray-700/50 p-2 rounded">
            <p className="text-xs text-gray-400">Total Events</p>
            <p className="text-sm font-mono-numbers text-white font-semibold">{data.events.length}</p>
          </div>
          <div className="bg-gray-700/50 p-2 rounded">
            <p className="text-xs text-gray-400">Data Points</p>
            <p className="text-sm font-mono-numbers text-white font-semibold">{data.curve.length}</p>
          </div>
        </div>
      )}
    </div>
  );
};
