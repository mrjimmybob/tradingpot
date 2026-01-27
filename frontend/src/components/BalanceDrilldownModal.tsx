/**
 * Balance Drilldown Modal
 *
 * Shows recent balance changes with source classification:
 * - Last 20 ledger entries
 * - Cumulative total
 * - Source tags (trade, fee, funding, correction)
 */

import React from 'react';
import { X, TrendingUp, TrendingDown, DollarSign, AlertTriangle, Link as LinkIcon } from 'lucide-react';

interface BalanceDrilldownModalProps {
  asset: string;
  isSimulated: boolean;
  ownerId?: string;
  onClose: () => void;
  onTradeClick?: (tradeId: number) => void;
}

interface LedgerEntry {
  ledger_entry_id: number;
  timestamp: string;
  delta_amount: number;
  balance_after: number;
  reason: string;
  source_classification: string;
  related_trade_id: number | null;
  related_order_id: number | null;
}

interface BalanceDrilldown {
  current_balance: number;
  ledger_entries: LedgerEntry[];
  cumulative_total: number;
}

const sourceColors = {
  trade: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  fee: 'bg-red-500/20 text-red-400 border-red-500/30',
  funding: 'bg-green-500/20 text-green-400 border-green-500/30',
  correction: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  other: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
};

const sourceIcons = {
  trade: TrendingUp,
  fee: DollarSign,
  funding: TrendingUp,
  correction: AlertTriangle,
  other: LinkIcon,
};

export const BalanceDrilldownModal: React.FC<BalanceDrilldownModalProps> = ({
  asset,
  isSimulated,
  ownerId,
  onClose,
  onTradeClick,
}) => {
  const [data, setData] = React.useState<BalanceDrilldown | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [limit, setLimit] = React.useState(20);

  React.useEffect(() => {
    const fetchBalanceDrilldown = async () => {
      try {
        setLoading(true);
        let url = `/api/reports/balance-drilldown?is_simulated=${isSimulated}&asset=${asset}&limit=${limit}`;
        if (ownerId) {
          url += `&owner_id=${ownerId}`;
        }

        const response = await fetch(url);

        if (!response.ok) {
          throw new Error('Failed to fetch balance drilldown');
        }

        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchBalanceDrilldown();
  }, [asset, isSimulated, ownerId, limit]);

  // Close on escape key
  React.useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [onClose]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'decimal',
      minimumFractionDigits: 2,
      maximumFractionDigits: 8,
    }).format(value);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const handleTradeClick = (tradeId: number | null) => {
    if (tradeId && onTradeClick) {
      onTradeClick(tradeId);
    }
  };

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-gray-800 rounded-lg p-8">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent mx-auto"></div>
          <p className="text-gray-300 mt-4">Loading balance history...</p>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-gray-800 rounded-lg p-8 max-w-md">
          <h2 className="text-xl font-bold text-red-400 mb-4">Error</h2>
          <p className="text-gray-300">{error || 'Balance data not found'}</p>
          <button
            onClick={onClose}
            className="mt-4 px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="bg-gray-800 rounded-lg shadow-xl max-w-5xl w-full my-8 animate-scaleIn"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="border-b border-gray-700 p-6 flex justify-between items-center">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              Balance Drilldown: {asset}
            </h2>
            <p className="text-gray-400 text-sm mt-1">
              Last {data.ledger_entries.length} ledger entries
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
            aria-label="Close modal"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        <div className="p-6 max-h-[calc(100vh-200px)] overflow-y-auto">
          {/* Current Balance Card */}
          <div className="bg-gradient-to-r from-accent/20 to-accent/5 border border-accent/30 rounded-lg p-6 mb-6">
            <div className="flex justify-between items-center">
              <div>
                <p className="text-gray-400 text-sm">Current Balance</p>
                <p className="text-3xl font-mono-numbers font-bold text-white mt-1">
                  {formatCurrency(data.current_balance)} {asset}
                </p>
              </div>
              <div className="text-right">
                <p className="text-gray-400 text-sm">Cumulative Total</p>
                <p className="text-xl font-mono-numbers font-semibold text-gray-300 mt-1">
                  {formatCurrency(data.cumulative_total)} {asset}
                </p>
              </div>
            </div>
          </div>

          {/* Limit Selector */}
          <div className="mb-4 flex items-center gap-2">
            <label className="text-gray-400 text-sm">Show entries:</label>
            <select
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              className="bg-gray-700 text-white rounded px-3 py-1 text-sm focus:ring-2 focus:ring-accent outline-none"
            >
              <option value={10}>Last 10</option>
              <option value={20}>Last 20</option>
              <option value={50}>Last 50</option>
              <option value={100}>Last 100</option>
            </select>
          </div>

          {/* Ledger Entries Table */}
          {data.ledger_entries.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-400">No ledger entries found</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-700/50 sticky top-0">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-300">Timestamp</th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-300">Source</th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-300">Reason</th>
                    <th className="px-4 py-3 text-right text-xs font-semibold text-gray-300">Change</th>
                    <th className="px-4 py-3 text-right text-xs font-semibold text-gray-300">Balance After</th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-gray-300">Links</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-700">
                  {data.ledger_entries.map((entry) => {
                    const SourceIcon = sourceIcons[entry.source_classification as keyof typeof sourceIcons] || LinkIcon;
                    const sourceColor = sourceColors[entry.source_classification as keyof typeof sourceColors] || sourceColors.other;

                    return (
                      <tr key={entry.ledger_entry_id} className="hover:bg-gray-700/30 transition-colors">
                        <td className="px-4 py-3 text-xs text-gray-400 whitespace-nowrap">
                          {formatDate(entry.timestamp)}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs border ${sourceColor}`}>
                            <SourceIcon className="h-3 w-3" />
                            {entry.source_classification}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-xs px-2 py-1 rounded bg-gray-600 text-gray-200">
                            {entry.reason}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right font-mono-numbers">
                          <span className={`font-semibold ${
                            entry.delta_amount >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {entry.delta_amount >= 0 ? '+' : ''}
                            {formatCurrency(entry.delta_amount)}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right font-mono-numbers text-white font-semibold">
                          {formatCurrency(entry.balance_after)}
                        </td>
                        <td className="px-4 py-3 text-center">
                          <div className="flex items-center justify-center gap-2">
                            {entry.related_trade_id && (
                              <button
                                onClick={() => handleTradeClick(entry.related_trade_id)}
                                className="text-xs px-2 py-1 rounded bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 transition-colors"
                                title="View trade detail"
                              >
                                Trade #{entry.related_trade_id}
                              </button>
                            )}
                            {entry.related_order_id && (
                              <span className="text-xs px-2 py-1 rounded bg-gray-600 text-gray-300">
                                Order #{entry.related_order_id}
                              </span>
                            )}
                            {!entry.related_trade_id && !entry.related_order_id && (
                              <span className="text-gray-500 text-xs">-</span>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Summary Statistics */}
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
            {['trade', 'fee', 'funding', 'correction'].map((source) => {
              const entries = data.ledger_entries.filter(e => e.source_classification === source);
              const total = entries.reduce((sum, e) => sum + e.delta_amount, 0);
              const SourceIcon = sourceIcons[source as keyof typeof sourceIcons];
              const sourceColor = sourceColors[source as keyof typeof sourceColors];

              return (
                <div key={source} className={`p-3 rounded border ${sourceColor}`}>
                  <div className="flex items-center gap-2 mb-1">
                    <SourceIcon className="h-4 w-4" />
                    <p className="text-xs font-semibold uppercase">{source}</p>
                  </div>
                  <p className="text-sm font-mono-numbers">
                    {entries.length} entries
                  </p>
                  <p className="text-lg font-mono-numbers font-bold">
                    {total >= 0 ? '+' : ''}{formatCurrency(total)}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-gray-700 p-4 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
