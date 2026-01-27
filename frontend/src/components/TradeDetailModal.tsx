/**
 * Trade Detail Modal
 *
 * Shows complete forensic trail for a trade:
 * - Trade record
 * - Linked order
 * - Ledger entries (debit/credit)
 * - Tax lots consumed
 * - Realized gain/loss
 * - Fees and modeled costs
 */

import React from 'react';
import { X, ExternalLink, TrendingUp, TrendingDown } from 'lucide-react';

interface TradeDetailModalProps {
  tradeId: number;
  isSimulated: boolean;
  onClose: () => void;
}

interface TradeDetail {
  trade: {
    id: number;
    trading_pair: string;
    side: string;
    base_amount: number;
    quote_amount: number;
    price: number;
    fee_amount: number;
    fee_asset: string;
    executed_at: string;
    strategy_used: string;
  };
  order: {
    order_id: number;
    order_type: string;
    status: string;
    amount: number;
    price: number;
    created_at: string;
    filled_at: string | null;
  };
  ledger_entries: Array<{
    ledger_entry_id: number;
    asset: string;
    delta_amount: number;
    balance_after: number;
    reason: string;
    timestamp: string;
  }>;
  tax_lots_consumed: Array<{
    tax_lot_id: number;
    quantity_consumed: number;
    unit_cost: number;
    purchase_date: string;
  }>;
  realized_gain_loss: number | null;
  modeled_cost: number;
  realized_cost: number;
}

export const TradeDetailModal: React.FC<TradeDetailModalProps> = ({
  tradeId,
  isSimulated,
  onClose,
}) => {
  const [detail, setDetail] = React.useState<TradeDetail | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    const fetchTradeDetail = async () => {
      try {
        setLoading(true);
        const response = await fetch(
          `/api/reports/trade-detail/${tradeId}?is_simulated=${isSimulated}`
        );

        if (!response.ok) {
          throw new Error('Failed to fetch trade detail');
        }

        const data = await response.json();
        setDetail(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchTradeDetail();
  }, [tradeId, isSimulated]);

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
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 8,
    }).format(value);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-gray-800 rounded-lg p-8">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent mx-auto"></div>
          <p className="text-gray-300 mt-4">Loading trade details...</p>
        </div>
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-gray-800 rounded-lg p-8 max-w-md">
          <h2 className="text-xl font-bold text-red-400 mb-4">Error</h2>
          <p className="text-gray-300">{error || 'Trade not found'}</p>
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
        className="bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full my-8 animate-scaleIn"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="border-b border-gray-700 p-6 flex justify-between items-center">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              Trade Detail #{detail.trade.id}
              <span className={`text-sm px-2 py-1 rounded ${
                detail.trade.side === 'BUY'
                  ? 'bg-green-500/20 text-green-400'
                  : 'bg-red-500/20 text-red-400'
              }`}>
                {detail.trade.side}
              </span>
            </h2>
            <p className="text-gray-400 text-sm mt-1">
              {detail.trade.trading_pair} • {formatDate(detail.trade.executed_at)}
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
          {/* Trade Summary */}
          <section className="mb-6">
            <h3 className="text-lg font-semibold text-white mb-3">Trade Summary</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-700/50 p-3 rounded">
                <p className="text-gray-400 text-xs">Quantity</p>
                <p className="text-white font-mono-numbers font-semibold">
                  {detail.trade.base_amount.toFixed(8)}
                </p>
              </div>
              <div className="bg-gray-700/50 p-3 rounded">
                <p className="text-gray-400 text-xs">Price</p>
                <p className="text-white font-mono-numbers font-semibold">
                  {formatCurrency(detail.trade.price)}
                </p>
              </div>
              <div className="bg-gray-700/50 p-3 rounded">
                <p className="text-gray-400 text-xs">Total</p>
                <p className="text-white font-mono-numbers font-semibold">
                  {formatCurrency(detail.trade.quote_amount)}
                </p>
              </div>
              <div className="bg-gray-700/50 p-3 rounded">
                <p className="text-gray-400 text-xs">Strategy</p>
                <p className="text-white font-semibold text-sm">
                  {detail.trade.strategy_used || 'N/A'}
                </p>
              </div>
            </div>
          </section>

          {/* Cost Analysis */}
          <section className="mb-6">
            <h3 className="text-lg font-semibold text-white mb-3">Cost Analysis</h3>
            <div className="bg-gray-700/50 p-4 rounded space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Exchange Fee</span>
                <span className="text-white font-mono-numbers">
                  {formatCurrency(detail.trade.fee_amount)} {detail.trade.fee_asset}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Modeled Cost</span>
                <span className="text-white font-mono-numbers">
                  {formatCurrency(detail.modeled_cost)}
                </span>
              </div>
              <div className="flex justify-between border-t border-gray-600 pt-2">
                <span className="text-white font-semibold">Total Realized Cost</span>
                <span className="text-white font-mono-numbers font-semibold">
                  {formatCurrency(detail.realized_cost)}
                </span>
              </div>
            </div>
          </section>

          {/* Realized Gain/Loss (for SELL trades) */}
          {detail.realized_gain_loss !== null && (
            <section className="mb-6">
              <h3 className="text-lg font-semibold text-white mb-3">Realized P&L</h3>
              <div className={`p-4 rounded flex items-center gap-3 ${
                detail.realized_gain_loss >= 0
                  ? 'bg-green-500/10 border border-green-500/30'
                  : 'bg-red-500/10 border border-red-500/30'
              }`}>
                {detail.realized_gain_loss >= 0 ? (
                  <TrendingUp className="h-6 w-6 text-green-400" />
                ) : (
                  <TrendingDown className="h-6 w-6 text-red-400" />
                )}
                <div>
                  <p className="text-gray-400 text-sm">Realized Gain/Loss</p>
                  <p className={`text-2xl font-mono-numbers font-bold ${
                    detail.realized_gain_loss >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatCurrency(detail.realized_gain_loss)}
                  </p>
                </div>
              </div>
            </section>
          )}

          {/* Tax Lots Consumed (for SELL trades) */}
          {detail.tax_lots_consumed.length > 0 && (
            <section className="mb-6">
              <h3 className="text-lg font-semibold text-white mb-3">Tax Lots Consumed (FIFO)</h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-700/50">
                    <tr>
                      <th className="px-4 py-2 text-left text-xs font-semibold text-gray-300">Lot ID</th>
                      <th className="px-4 py-2 text-left text-xs font-semibold text-gray-300">Quantity</th>
                      <th className="px-4 py-2 text-left text-xs font-semibold text-gray-300">Unit Cost</th>
                      <th className="px-4 py-2 text-left text-xs font-semibold text-gray-300">Purchase Date</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-700">
                    {detail.tax_lots_consumed.map((lot) => (
                      <tr key={lot.tax_lot_id} className="hover:bg-gray-700/30">
                        <td className="px-4 py-2 text-sm text-gray-300">#{lot.tax_lot_id}</td>
                        <td className="px-4 py-2 text-sm text-white font-mono-numbers">
                          {lot.quantity_consumed.toFixed(8)}
                        </td>
                        <td className="px-4 py-2 text-sm text-white font-mono-numbers">
                          {formatCurrency(lot.unit_cost)}
                        </td>
                        <td className="px-4 py-2 text-sm text-gray-300">
                          {formatDate(lot.purchase_date)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Linked Order */}
          <section className="mb-6">
            <h3 className="text-lg font-semibold text-white mb-3">Linked Order</h3>
            <div className="bg-gray-700/50 p-4 rounded space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Order ID</span>
                <span className="text-white font-mono-numbers">#{detail.order.order_id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Type</span>
                <span className="text-white">{detail.order.order_type}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Status</span>
                <span className={`px-2 py-1 rounded text-xs ${
                  detail.order.status === 'FILLED'
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-gray-600 text-gray-300'
                }`}>
                  {detail.order.status}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Created</span>
                <span className="text-white text-sm">{formatDate(detail.order.created_at)}</span>
              </div>
              {detail.order.filled_at && (
                <div className="flex justify-between">
                  <span className="text-gray-400">Filled</span>
                  <span className="text-white text-sm">{formatDate(detail.order.filled_at)}</span>
                </div>
              )}
            </div>
          </section>

          {/* Ledger Entries */}
          <section>
            <h3 className="text-lg font-semibold text-white mb-3">Ledger Entries (Double-Entry)</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-700/50">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-semibold text-gray-300">ID</th>
                    <th className="px-4 py-2 text-left text-xs font-semibold text-gray-300">Asset</th>
                    <th className="px-4 py-2 text-left text-xs font-semibold text-gray-300">Reason</th>
                    <th className="px-4 py-2 text-right text-xs font-semibold text-gray-300">Debit</th>
                    <th className="px-4 py-2 text-right text-xs font-semibold text-gray-300">Credit</th>
                    <th className="px-4 py-2 text-right text-xs font-semibold text-gray-300">Balance After</th>
                    <th className="px-4 py-2 text-left text-xs font-semibold text-gray-300">Timestamp</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-700">
                  {detail.ledger_entries.map((entry) => (
                    <tr key={entry.ledger_entry_id} className="hover:bg-gray-700/30">
                      <td className="px-4 py-2 text-sm text-gray-300">#{entry.ledger_entry_id}</td>
                      <td className="px-4 py-2 text-sm text-white font-semibold">{entry.asset}</td>
                      <td className="px-4 py-2">
                        <span className="text-xs px-2 py-1 rounded bg-gray-600 text-gray-200">
                          {entry.reason}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-sm text-right font-mono-numbers">
                        {entry.delta_amount < 0 ? (
                          <span className="text-red-400">
                            {Math.abs(entry.delta_amount).toFixed(8)}
                          </span>
                        ) : (
                          <span className="text-gray-500">-</span>
                        )}
                      </td>
                      <td className="px-4 py-2 text-sm text-right font-mono-numbers">
                        {entry.delta_amount > 0 ? (
                          <span className="text-green-400">
                            {entry.delta_amount.toFixed(8)}
                          </span>
                        ) : (
                          <span className="text-gray-500">-</span>
                        )}
                      </td>
                      <td className="px-4 py-2 text-sm text-right font-mono-numbers text-white">
                        {entry.balance_after.toFixed(8)}
                      </td>
                      <td className="px-4 py-2 text-xs text-gray-400">
                        {formatDate(entry.timestamp)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>

        {/* Footer */}
        <div className="border-t border-gray-700 p-4 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
          >
            Close
          </button>
          <a
            href={`/api/reports/trade-detail/${tradeId}?is_simulated=${isSimulated}`}
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 bg-accent text-white rounded hover:bg-accent/80 transition-colors flex items-center gap-2"
          >
            <ExternalLink className="h-4 w-4" />
            Export JSON
          </a>
        </div>
      </div>
    </div>
  );
};
