/**
 * Audit & Compliance View
 *
 * Unified audit log from:
 * - alerts_log
 * - strategy_rotations
 * - ledger_invariant failures
 *
 * Filters: bot, date, severity
 */

import React from 'react';
import { AlertTriangle, Info, XCircle, Filter, Download, Calendar } from 'lucide-react';

interface AuditLogEntry {
  id: number;
  timestamp: string;
  severity: string;
  source: string;
  bot_id: number | null;
  message: string;
  details: Record<string, any> | null;
}

export const AuditCompliance: React.FC = () => {
  const [logs, setLogs] = React.useState<AuditLogEntry[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  // Filters
  const [isSimulated, setIsSimulated] = React.useState(true);
  const [selectedBotId, setSelectedBotId] = React.useState<string>('');
  const [selectedSeverity, setSelectedSeverity] = React.useState<string>('');
  const [startDate, setStartDate] = React.useState<string>('');
  const [endDate, setEndDate] = React.useState<string>('');
  const [expandedId, setExpandedId] = React.useState<number | null>(null);

  React.useEffect(() => {
    fetchAuditLog();
  }, [isSimulated, selectedBotId, selectedSeverity, startDate, endDate]);

  const fetchAuditLog = async () => {
    try {
      setLoading(true);
      setError(null);

      let url = `/api/reports/audit-log?is_simulated=${isSimulated}`;
      if (selectedBotId) url += `&bot_id=${selectedBotId}`;
      if (selectedSeverity) url += `&severity=${selectedSeverity}`;
      if (startDate) url += `&start_date=${new Date(startDate).toISOString()}`;
      if (endDate) url += `&end_date=${new Date(endDate).toISOString()}`;

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error('Failed to fetch audit log');
      }

      const data = await response.json();
      setLogs(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handleClearFilters = () => {
    setSelectedBotId('');
    setSelectedSeverity('');
    setStartDate('');
    setEndDate('');
  };

  const handleExport = () => {
    const csv = [
      ['Timestamp', 'Severity', 'Source', 'Bot ID', 'Message'],
      ...logs.map(log => [
        new Date(log.timestamp).toISOString(),
        log.severity,
        log.source,
        log.bot_id || 'N/A',
        log.message,
      ])
    ].map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `audit_log_${isSimulated ? 'simulated' : 'live'}_${new Date().toISOString()}.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'error':
        return <XCircle className="h-5 w-5 text-red-400" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-yellow-400" />;
      case 'info':
      default:
        return <Info className="h-5 w-5 text-blue-400" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'error':
        return 'bg-red-500/10 border-red-500/30 text-red-400';
      case 'warning':
        return 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400';
      case 'info':
      default:
        return 'bg-blue-500/10 border-blue-500/30 text-blue-400';
    }
  };

  const getSourceColor = (source: string) => {
    if (source === 'alerts_log') return 'bg-red-500/20 text-red-400';
    if (source === 'strategy_rotations') return 'bg-blue-500/20 text-blue-400';
    return 'bg-gray-500/20 text-gray-400';
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Audit & Compliance</h1>
        <p className="text-gray-400">
          Complete audit trail combining alerts, strategy rotations, and system events.
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

      {/* Filters */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 mb-6">
        <div className="flex items-center gap-2 mb-4">
          <Filter className="h-5 w-5 text-accent" />
          <h3 className="text-lg font-semibold text-white">Filters</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          {/* Bot ID */}
          <div>
            <label className="block text-sm font-semibold text-gray-400 mb-2">Bot ID</label>
            <input
              type="number"
              value={selectedBotId}
              onChange={(e) => setSelectedBotId(e.target.value)}
              placeholder="All bots"
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-accent outline-none"
            />
          </div>

          {/* Severity */}
          <div>
            <label className="block text-sm font-semibold text-gray-400 mb-2">Severity</label>
            <select
              value={selectedSeverity}
              onChange={(e) => setSelectedSeverity(e.target.value)}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-accent outline-none"
            >
              <option value="">All severities</option>
              <option value="error">Error</option>
              <option value="warning">Warning</option>
              <option value="info">Info</option>
            </select>
          </div>

          {/* Start Date */}
          <div>
            <label className="block text-sm font-semibold text-gray-400 mb-2">Start Date</label>
            <input
              type="datetime-local"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-accent outline-none"
            />
          </div>

          {/* End Date */}
          <div>
            <label className="block text-sm font-semibold text-gray-400 mb-2">End Date</label>
            <input
              type="datetime-local"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:ring-2 focus:ring-accent outline-none"
            />
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={handleClearFilters}
            className="px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
          >
            Clear Filters
          </button>
          <button
            onClick={handleExport}
            disabled={logs.length === 0}
            className="flex items-center gap-2 px-4 py-2 bg-accent text-white rounded hover:bg-accent/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Download className="h-4 w-4" />
            Export CSV
          </button>
        </div>
      </div>

      {/* Stats */}
      {!loading && logs.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          {['error', 'warning', 'info', 'total'].map((type) => {
            const count = type === 'total'
              ? logs.length
              : logs.filter(log => log.severity === type).length;
            const Icon = type === 'error' ? XCircle
              : type === 'warning' ? AlertTriangle
              : type === 'info' ? Info
              : Calendar;

            return (
              <div key={type} className={`p-4 rounded-lg border ${
                type === 'error' ? 'bg-red-500/10 border-red-500/30'
                : type === 'warning' ? 'bg-yellow-500/10 border-yellow-500/30'
                : type === 'info' ? 'bg-blue-500/10 border-blue-500/30'
                : 'bg-gray-700 border-gray-600'
              }`}>
                <div className="flex items-center gap-2 mb-1">
                  <Icon className={`h-4 w-4 ${
                    type === 'error' ? 'text-red-400'
                    : type === 'warning' ? 'text-yellow-400'
                    : type === 'info' ? 'text-blue-400'
                    : 'text-gray-400'
                  }`} />
                  <p className="text-xs font-semibold uppercase text-gray-400">{type}</p>
                </div>
                <p className="text-2xl font-mono-numbers font-bold text-white">{count}</p>
              </div>
            );
          })}
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-12 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent mx-auto mb-4"></div>
          <p className="text-gray-400">Loading audit log...</p>
        </div>
      )}

      {/* Error State */}
      {error && !loading && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6">
          <div className="flex items-center gap-3">
            <XCircle className="h-6 w-6 text-red-400" />
            <div>
              <h3 className="text-red-400 font-semibold">Error Loading Audit Log</h3>
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Audit Log Table */}
      {!loading && logs.length > 0 && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-700/50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-300">Timestamp</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-300">Severity</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-300">Source</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-300">Bot</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-300">Message</th>
                  <th className="px-4 py-3 text-center text-xs font-semibold text-gray-300">Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {logs.map((log) => (
                  <React.Fragment key={log.id}>
                    <tr className="hover:bg-gray-700/30 transition-colors">
                      <td className="px-4 py-3 text-sm text-gray-400 whitespace-nowrap">
                        {new Date(log.timestamp).toLocaleString()}
                      </td>
                      <td className="px-4 py-3">
                        <div className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs border ${getSeverityColor(log.severity)}`}>
                          {getSeverityIcon(log.severity)}
                          {log.severity}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`text-xs px-2 py-1 rounded ${getSourceColor(log.source)}`}>
                          {log.source}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm text-white">
                        {log.bot_id ? `#${log.bot_id}` : <span className="text-gray-500">Global</span>}
                      </td>
                      <td className="px-4 py-3 text-sm text-white max-w-md truncate">
                        {log.message}
                      </td>
                      <td className="px-4 py-3 text-center">
                        {log.details && (
                          <button
                            onClick={() => setExpandedId(expandedId === log.id ? null : log.id)}
                            className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
                          >
                            {expandedId === log.id ? 'Hide' : 'View'}
                          </button>
                        )}
                      </td>
                    </tr>
                    {expandedId === log.id && log.details && (
                      <tr>
                        <td colSpan={6} className="px-4 py-3 bg-gray-900/50">
                          <pre className="text-xs text-gray-300 overflow-x-auto p-3 bg-gray-950 rounded">
                            {JSON.stringify(log.details, null, 2)}
                          </pre>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!loading && logs.length === 0 && !error && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-12 text-center">
          <Info className="h-16 w-16 text-gray-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-400 mb-2">No Audit Entries Found</h3>
          <p className="text-gray-500">
            No entries match the current filters. Try adjusting your filter criteria.
          </p>
        </div>
      )}
    </div>
  );
};
