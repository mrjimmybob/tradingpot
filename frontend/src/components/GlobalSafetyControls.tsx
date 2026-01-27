/**
 * Global Safety Controls
 *
 * Admin-only safety features:
 * - Global "Freeze all bots"
 * - "Rebuild state from ledger"
 * - "Export all data" (ledger + trades + tax)
 */

import React from 'react';
import { AlertTriangle, Snowflake, RefreshCw, Download, Shield } from 'lucide-react';

interface GlobalSafetyControlsProps {
  isSimulated: boolean;
}

export const GlobalSafetyControls: React.FC<GlobalSafetyControlsProps> = ({ isSimulated }) => {
  const [freezing, setFreezing] = React.useState(false);
  const [rebuilding, setRebuilding] = React.useState(false);
  const [exporting, setExporting] = React.useState(false);
  const [showConfirm, setShowConfirm] = React.useState<string | null>(null);

  const handleFreezeAll = async () => {
    if (!showConfirm) {
      setShowConfirm('freeze');
      return;
    }

    try {
      setFreezing(true);
      const response = await fetch('/api/kill-all', { method: 'POST' });

      if (!response.ok) {
        throw new Error('Failed to freeze all bots');
      }

      alert('All bots have been stopped successfully');
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to freeze bots');
    } finally {
      setFreezing(false);
      setShowConfirm(null);
    }
  };

  const handleRebuildState = async () => {
    if (!showConfirm) {
      setShowConfirm('rebuild');
      return;
    }

    try {
      setRebuilding(true);
      // This endpoint would need to be implemented in the backend
      const response = await fetch('/api/admin/rebuild-state', { method: 'POST' });

      if (!response.ok) {
        throw new Error('Failed to rebuild state');
      }

      alert('State has been rebuilt from ledger successfully');
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to rebuild state');
    } finally {
      setRebuilding(false);
      setShowConfirm(null);
    }
  };

  const handleExportAll = async () => {
    try {
      setExporting(true);

      // Export all data
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const mode = isSimulated ? 'simulated' : 'live';

      // Export ledger
      const ledgerResponse = await fetch(`/api/reports/ledger-audit?is_simulated=${isSimulated}`);
      if (ledgerResponse.ok) {
        const ledgerData = await ledgerResponse.json();
        const ledgerBlob = new Blob([JSON.stringify(ledgerData, null, 2)], { type: 'application/json' });
        const ledgerUrl = window.URL.createObjectURL(ledgerBlob);
        const ledgerLink = document.createElement('a');
        ledgerLink.href = ledgerUrl;
        ledgerLink.download = `ledger_${mode}_${timestamp}.json`;
        document.body.appendChild(ledgerLink);
        ledgerLink.click();
        window.URL.revokeObjectURL(ledgerUrl);
        document.body.removeChild(ledgerLink);
      }

      // Export trades
      const tradesResponse = await fetch(`/api/reports/trades?is_simulated=${isSimulated}`);
      if (tradesResponse.ok) {
        const tradesData = await tradesResponse.json();
        const tradesBlob = new Blob([JSON.stringify(tradesData, null, 2)], { type: 'application/json' });
        const tradesUrl = window.URL.createObjectURL(tradesBlob);
        const tradesLink = document.createElement('a');
        tradesLink.href = tradesUrl;
        tradesLink.download = `trades_${mode}_${timestamp}.json`;
        document.body.appendChild(tradesLink);
        tradesLink.click();
        window.URL.revokeObjectURL(tradesUrl);
        document.body.removeChild(tradesLink);
      }

      // Export tax data for current year
      const currentYear = new Date().getFullYear();
      const taxResponse = await fetch(`/api/reports/tax-export/${currentYear}?is_simulated=${isSimulated}`, {
        method: 'POST',
      });
      if (taxResponse.ok) {
        const taxBlob = await taxResponse.blob();
        const taxUrl = window.URL.createObjectURL(taxBlob);
        const taxLink = document.createElement('a');
        taxLink.href = taxUrl;
        taxLink.download = `tax_${currentYear}_${mode}_${timestamp}.csv`;
        document.body.appendChild(taxLink);
        taxLink.click();
        window.URL.revokeObjectURL(taxUrl);
        document.body.removeChild(taxLink);
      }

      alert('All data exported successfully! Check your downloads folder.');
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to export data');
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="bg-gray-800 border-2 border-orange-500/30 rounded-lg p-6">
      <div className="flex items-center gap-2 mb-4">
        <Shield className="h-6 w-6 text-orange-400" />
        <h3 className="text-xl font-semibold text-orange-400">Global Safety Controls</h3>
        <span className="ml-auto text-xs px-2 py-1 rounded bg-orange-500/20 text-orange-400 border border-orange-500/30">
          ADMIN ONLY
        </span>
      </div>

      <div className="space-y-3">
        {/* Freeze All Bots */}
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <Snowflake className="h-5 w-5 text-red-400" />
                <h4 className="font-semibold text-red-400">Emergency Stop All Bots</h4>
              </div>
              <p className="text-sm text-gray-300">
                Immediately stop all running bots. Use this in case of market emergency or system issues.
              </p>
            </div>
            <button
              onClick={handleFreezeAll}
              disabled={freezing}
              className={`px-4 py-2 rounded font-semibold transition-colors whitespace-nowrap ${
                showConfirm === 'freeze'
                  ? 'bg-red-600 text-white hover:bg-red-700'
                  : 'bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30'
              }`}
            >
              {freezing ? 'Stopping...' : showConfirm === 'freeze' ? 'Confirm Stop' : 'Freeze All'}
            </button>
          </div>
          {showConfirm === 'freeze' && (
            <div className="mt-3 p-2 bg-red-900/30 rounded text-sm text-red-300">
              ⚠️ This will stop ALL running bots immediately. Click "Confirm Stop" to proceed or wait 10 seconds to cancel.
            </div>
          )}
        </div>

        {/* Rebuild State */}
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <RefreshCw className="h-5 w-5 text-yellow-400" />
                <h4 className="font-semibold text-yellow-400">Rebuild State from Ledger</h4>
              </div>
              <p className="text-sm text-gray-300">
                Reconstruct all bot balances and positions from the authoritative ledger. Use if cache is inconsistent.
              </p>
            </div>
            <button
              onClick={handleRebuildState}
              disabled={rebuilding}
              className={`px-4 py-2 rounded font-semibold transition-colors whitespace-nowrap ${
                showConfirm === 'rebuild'
                  ? 'bg-yellow-600 text-white hover:bg-yellow-700'
                  : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 hover:bg-yellow-500/30'
              }`}
            >
              {rebuilding ? 'Rebuilding...' : showConfirm === 'rebuild' ? 'Confirm Rebuild' : 'Rebuild State'}
            </button>
          </div>
          {showConfirm === 'rebuild' && (
            <div className="mt-3 p-2 bg-yellow-900/30 rounded text-sm text-yellow-300">
              ⚠️ This will recalculate all balances from the ledger. Bots will be paused during rebuild.
            </div>
          )}
        </div>

        {/* Export All Data */}
        <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <Download className="h-5 w-5 text-blue-400" />
                <h4 className="font-semibold text-blue-400">Export All Data</h4>
              </div>
              <p className="text-sm text-gray-300">
                Download complete data backup: ledger entries, trades, and tax records (JSON + CSV).
              </p>
            </div>
            <button
              onClick={handleExportAll}
              disabled={exporting}
              className="px-4 py-2 rounded font-semibold bg-blue-500/20 text-blue-400 border border-blue-500/30 hover:bg-blue-500/30 transition-colors whitespace-nowrap"
            >
              {exporting ? (
                <span className="flex items-center gap-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400"></div>
                  Exporting...
                </span>
              ) : (
                'Export All'
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Warning */}
      <div className="mt-4 p-3 bg-orange-900/30 border border-orange-500/30 rounded text-sm text-orange-300">
        <AlertTriangle className="h-4 w-4 inline mr-2" />
        These are administrative functions. Use with caution in production environments.
      </div>

      {/* Auto-cancel confirmation */}
      {showConfirm && (
        <div className="mt-4">
          <button
            onClick={() => setShowConfirm(null)}
            className="text-sm text-gray-400 hover:text-white underline"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
};
