/**
 * Tax Reporting Page
 *
 * Separate tab for tax reporting:
 * - Fiscal year selector
 * - Total realized gains
 * - Short vs long term breakdown
 * - Lot history
 * - CSV export
 *
 * DO NOT mix with trading P&L UI
 */

import React from 'react';
import { Download, FileText, TrendingUp, TrendingDown, Calendar, DollarSign } from 'lucide-react';

interface TaxSummary {
  total_realized_gain: number;
  short_term_gain: number;
  long_term_gain: number;
  lot_count: number;
  trade_count: number;
}

export const TaxReporting: React.FC = () => {
  const currentYear = new Date().getFullYear();
  const [selectedYear, setSelectedYear] = React.useState(currentYear);
  const [isSimulated, setIsSimulated] = React.useState(true);
  const [summary, setSummary] = React.useState<TaxSummary | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [exporting, setExporting] = React.useState(false);

  // Generate year options (current year + 5 years back)
  const years = Array.from({ length: 6 }, (_, i) => currentYear - i);

  React.useEffect(() => {
    fetchTaxSummary();
  }, [selectedYear, isSimulated]);

  const fetchTaxSummary = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(
        `/api/reports/tax-summary/${selectedYear}?is_simulated=${isSimulated}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch tax summary');
      }

      const data = await response.json();
      setSummary(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setSummary(null);
    } finally {
      setLoading(false);
    }
  };

  const handleExportCSV = async () => {
    try {
      setExporting(true);

      const response = await fetch(
        `/api/reports/tax-export/${selectedYear}?is_simulated=${isSimulated}`,
        { method: 'POST' }
      );

      if (!response.ok) {
        throw new Error('Failed to export tax data');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `tax_report_${selectedYear}_${isSimulated ? 'simulated' : 'live'}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Export failed');
    } finally {
      setExporting(false);
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Tax Reporting</h1>
        <p className="text-gray-400">
          Download tax reports for your trading activity. This data is separate from trading P&L and
          follows FIFO cost basis methodology.
        </p>
      </div>

      {/* Mode Banner */}
      <div className={`mb-6 p-4 rounded-lg border-2 ${
        isSimulated
          ? 'bg-blue-500/10 border-blue-500/30'
          : 'bg-red-500/10 border-red-500/30'
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`h-3 w-3 rounded-full ${
              isSimulated ? 'bg-blue-500' : 'bg-red-500'
            } animate-pulse`}></div>
            <span className={`text-lg font-bold ${
              isSimulated ? 'text-blue-400' : 'text-red-400'
            }`}>
              {isSimulated ? 'SIMULATED MODE' : 'LIVE MODE'}
            </span>
          </div>
          <button
            onClick={() => setIsSimulated(!isSimulated)}
            className="px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
          >
            Switch to {isSimulated ? 'Live' : 'Simulated'}
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 mb-6">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div>
              <label className="block text-sm font-semibold text-gray-400 mb-2">
                <Calendar className="h-4 w-4 inline mr-1" />
                Fiscal Year
              </label>
              <select
                value={selectedYear}
                onChange={(e) => setSelectedYear(Number(e.target.value))}
                className="bg-gray-700 text-white rounded px-4 py-2 text-lg font-semibold focus:ring-2 focus:ring-accent outline-none"
              >
                {years.map((year) => (
                  <option key={year} value={year}>
                    {year}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button
            onClick={handleExportCSV}
            disabled={exporting || !summary}
            className="flex items-center gap-2 px-6 py-3 bg-accent text-white rounded-lg hover:bg-accent/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {exporting ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                Exporting...
              </>
            ) : (
              <>
                <Download className="h-5 w-5" />
                Export CSV for {selectedYear}
              </>
            )}
          </button>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-12 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent mx-auto mb-4"></div>
          <p className="text-gray-400">Loading tax summary...</p>
        </div>
      )}

      {/* Error State */}
      {error && !loading && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6">
          <div className="flex items-center gap-3">
            <FileText className="h-6 w-6 text-red-400" />
            <div>
              <h3 className="text-red-400 font-semibold">Error Loading Tax Data</h3>
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Tax Summary */}
      {summary && !loading && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
            {/* Total Realized Gain */}
            <div className={`p-6 rounded-lg border-2 ${
              summary.total_realized_gain >= 0
                ? 'bg-green-500/10 border-green-500/30'
                : 'bg-red-500/10 border-red-500/30'
            }`}>
              <div className="flex items-center gap-2 mb-2">
                {summary.total_realized_gain >= 0 ? (
                  <TrendingUp className="h-5 w-5 text-green-400" />
                ) : (
                  <TrendingDown className="h-5 w-5 text-red-400" />
                )}
                <p className="text-sm font-semibold text-gray-400 uppercase">Total Realized</p>
              </div>
              <p className={`text-3xl font-mono-numbers font-bold ${
                summary.total_realized_gain >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatCurrency(summary.total_realized_gain)}
              </p>
              <p className="text-xs text-gray-400 mt-1">
                {summary.total_realized_gain >= 0 ? 'Taxable Gain' : 'Deductible Loss'}
              </p>
            </div>

            {/* Short-Term Gain */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
              <div className="flex items-center gap-2 mb-2">
                <DollarSign className="h-5 w-5 text-orange-400" />
                <p className="text-sm font-semibold text-gray-400 uppercase">Short-Term</p>
              </div>
              <p className={`text-3xl font-mono-numbers font-bold ${
                summary.short_term_gain >= 0 ? 'text-white' : 'text-red-400'
              }`}>
                {formatCurrency(summary.short_term_gain)}
              </p>
              <p className="text-xs text-gray-400 mt-1">Held &lt; 1 year (ordinary income)</p>
            </div>

            {/* Long-Term Gain */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
              <div className="flex items-center gap-2 mb-2">
                <DollarSign className="h-5 w-5 text-blue-400" />
                <p className="text-sm font-semibold text-gray-400 uppercase">Long-Term</p>
              </div>
              <p className={`text-3xl font-mono-numbers font-bold ${
                summary.long_term_gain >= 0 ? 'text-white' : 'text-red-400'
              }`}>
                {formatCurrency(summary.long_term_gain)}
              </p>
              <p className="text-xs text-gray-400 mt-1">Held ≥ 1 year (capital gains rate)</p>
            </div>

            {/* Activity Summary */}
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
              <div className="flex items-center gap-2 mb-2">
                <FileText className="h-5 w-5 text-accent" />
                <p className="text-sm font-semibold text-gray-400 uppercase">Activity</p>
              </div>
              <div className="space-y-1">
                <p className="text-white">
                  <span className="text-2xl font-mono-numbers font-bold">{summary.lot_count}</span>
                  <span className="text-sm text-gray-400 ml-2">tax lots</span>
                </p>
                <p className="text-white">
                  <span className="text-2xl font-mono-numbers font-bold">{summary.trade_count}</span>
                  <span className="text-sm text-gray-400 ml-2">sell trades</span>
                </p>
              </div>
            </div>
          </div>

          {/* Breakdown */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 mb-6">
            <h3 className="text-lg font-semibold text-white mb-4">Gain/Loss Breakdown</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 bg-gray-700/50 rounded">
                <span className="text-gray-300">Short-Term Capital Gain/Loss</span>
                <span className={`font-mono-numbers font-semibold ${
                  summary.short_term_gain >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {formatCurrency(summary.short_term_gain)}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-gray-700/50 rounded">
                <span className="text-gray-300">Long-Term Capital Gain/Loss</span>
                <span className={`font-mono-numbers font-semibold ${
                  summary.long_term_gain >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {formatCurrency(summary.long_term_gain)}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-accent/10 border border-accent/30 rounded">
                <span className="text-white font-semibold">Total Realized Gain/Loss</span>
                <span className={`font-mono-numbers font-bold text-lg ${
                  summary.total_realized_gain >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {formatCurrency(summary.total_realized_gain)}
                </span>
              </div>
            </div>
          </div>

          {/* Tax Information */}
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-blue-400 mb-3">Tax Information</h3>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-0.5">•</span>
                <span>
                  <strong>Cost Basis Method:</strong> FIFO (First-In-First-Out) - oldest lots are sold first
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-0.5">•</span>
                <span>
                  <strong>Short-Term:</strong> Assets held less than 1 year are taxed as ordinary income
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-0.5">•</span>
                <span>
                  <strong>Long-Term:</strong> Assets held 1 year or more qualify for capital gains tax rates (0%, 15%, or 20% depending on income)
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-0.5">•</span>
                <span>
                  <strong>CSV Export:</strong> Contains detailed lot-by-lot breakdown suitable for tax preparation software (TurboTax, etc.)
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-yellow-400 mt-0.5">⚠</span>
                <span className="text-yellow-400">
                  <strong>Disclaimer:</strong> This report is for informational purposes only. Consult a tax professional for official tax preparation.
                </span>
              </li>
            </ul>
          </div>
        </>
      )}

      {/* Empty State */}
      {!summary && !loading && !error && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-12 text-center">
          <FileText className="h-16 w-16 text-gray-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-400 mb-2">No Tax Data for {selectedYear}</h3>
          <p className="text-gray-500">
            No realized gains or losses recorded for the selected year.
          </p>
        </div>
      )}
    </div>
  );
};
