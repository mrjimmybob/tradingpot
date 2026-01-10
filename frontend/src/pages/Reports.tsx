import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { FileBarChart, Download, TrendingUp, DollarSign } from 'lucide-react'

interface PnLReportEntry {
  bot_id: number
  bot_name: string
  trading_pair: string
  strategy: string
  total_pnl: number
  win_count: number
  loss_count: number
  win_rate: number
  total_fees: number
}

interface PnLReportResponse {
  entries: PnLReportEntry[]
  total_pnl: number
  overall_win_rate: number
}

interface TaxReportEntry {
  date: string
  trading_pair: string
  purchase_price: number
  sale_price: number
  gain: number
  token: string
}

interface TaxReportResponse {
  entries: TaxReportEntry[]
  total_gains: number
  year: number
}

interface FeeReportEntry {
  bot_id: number
  bot_name: string
  total_fees: number
  order_count: number
}

interface FeeReportResponse {
  entries: FeeReportEntry[]
  total_fees: number
}

async function fetchPnLReport(): Promise<PnLReportResponse> {
  const res = await fetch('/api/reports/pnl')
  if (!res.ok) throw new Error('Failed to fetch P&L report')
  return res.json()
}

async function fetchTaxReport(): Promise<TaxReportResponse> {
  const res = await fetch('/api/reports/tax')
  if (!res.ok) throw new Error('Failed to fetch tax report')
  return res.json()
}

async function fetchFeeReport(): Promise<FeeReportResponse> {
  const res = await fetch('/api/reports/fees')
  if (!res.ok) throw new Error('Failed to fetch fee report')
  return res.json()
}

type TabType = 'pnl' | 'tax' | 'fees'

export default function Reports() {
  const [activeTab, setActiveTab] = useState<TabType>('pnl')

  const { data: pnlReport, isLoading: pnlLoading } = useQuery({
    queryKey: ['pnl-report'],
    queryFn: fetchPnLReport,
    enabled: activeTab === 'pnl',
  })

  const { data: taxReport, isLoading: taxLoading } = useQuery({
    queryKey: ['tax-report'],
    queryFn: fetchTaxReport,
    enabled: activeTab === 'tax',
  })

  const { data: feeReport, isLoading: feeLoading } = useQuery({
    queryKey: ['fee-report'],
    queryFn: fetchFeeReport,
    enabled: activeTab === 'fees',
  })

  const handleExportTax = () => {
    window.open('/api/reports/tax?format=csv', '_blank')
  }

  const tabs = [
    { id: 'pnl' as const, name: 'P&L Report', icon: TrendingUp },
    { id: 'tax' as const, name: 'Tax Export', icon: FileBarChart },
    { id: 'fees' as const, name: 'Fees', icon: DollarSign },
  ]

  // Calculate totals from P&L entries
  const totalWins = pnlReport?.entries.reduce((sum, e) => sum + e.win_count, 0) ?? 0
  const totalLosses = pnlReport?.entries.reduce((sum, e) => sum + e.loss_count, 0) ?? 0
  const totalTrades = totalWins + totalLosses

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <FileBarChart size={32} className="text-accent" />
        <h2 className="text-2xl font-bold">Reports</h2>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-700">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-accent text-accent'
                : 'border-transparent text-gray-400 hover:text-white'
            }`}
          >
            <tab.icon size={18} />
            {tab.name}
          </button>
        ))}
      </div>

      {/* P&L Report Tab */}
      {activeTab === 'pnl' && (
        <div className="space-y-6">
          {pnlLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent"></div>
            </div>
          ) : pnlReport ? (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-800 p-6 rounded-lg">
                  <p className="text-gray-400 text-sm">Total P&L</p>
                  <p
                    className={`text-2xl font-bold font-mono-numbers mt-1 ${
                      pnlReport.total_pnl >= 0 ? 'text-profit' : 'text-loss'
                    }`}
                  >
                    {pnlReport.total_pnl >= 0 ? '+' : ''}${pnlReport.total_pnl.toFixed(2)}
                  </p>
                </div>

                <div className="bg-gray-800 p-6 rounded-lg">
                  <p className="text-gray-400 text-sm">Overall Win Rate</p>
                  <p className="text-2xl font-bold font-mono-numbers mt-1">
                    {pnlReport.overall_win_rate.toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    {totalWins}W / {totalLosses}L
                  </p>
                </div>

                <div className="bg-gray-800 p-6 rounded-lg">
                  <p className="text-gray-400 text-sm">Total Trades</p>
                  <p className="text-2xl font-bold font-mono-numbers mt-1">
                    {totalTrades}
                  </p>
                </div>
              </div>

              {/* Bot P&L Table */}
              {pnlReport.entries.length > 0 ? (
                <div className="bg-gray-800 rounded-lg overflow-hidden">
                  <h3 className="text-lg font-semibold px-6 py-4 border-b border-gray-700">
                    P&L by Bot
                  </h3>
                  <table className="w-full">
                    <thead>
                      <tr className="text-left text-gray-400 text-sm">
                        <th className="px-6 py-3">Bot</th>
                        <th className="px-6 py-3">Pair</th>
                        <th className="px-6 py-3">Strategy</th>
                        <th className="px-6 py-3">Win Rate</th>
                        <th className="px-6 py-3">Fees</th>
                        <th className="px-6 py-3 text-right">P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pnlReport.entries.map((entry) => (
                        <tr key={entry.bot_id} className="border-t border-gray-700">
                          <td className="px-6 py-3">{entry.bot_name}</td>
                          <td className="px-6 py-3">{entry.trading_pair}</td>
                          <td className="px-6 py-3 capitalize">{entry.strategy.replace(/_/g, ' ')}</td>
                          <td className="px-6 py-3 font-mono-numbers">
                            {entry.win_rate.toFixed(1)}%
                          </td>
                          <td className="px-6 py-3 font-mono-numbers text-loss">
                            ${entry.total_fees.toFixed(2)}
                          </td>
                          <td
                            className={`px-6 py-3 text-right font-mono-numbers ${
                              entry.total_pnl >= 0 ? 'text-profit' : 'text-loss'
                            }`}
                          >
                            {entry.total_pnl >= 0 ? '+' : ''}${entry.total_pnl.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-400">
                  No bots with P&L data yet
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12 text-gray-400">
              No P&L data available
            </div>
          )}
        </div>
      )}

      {/* Tax Report Tab */}
      {activeTab === 'tax' && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <p className="text-gray-400">
              Capital gains report for tax purposes. Only includes realized gains.
            </p>
            <button
              onClick={handleExportTax}
              className="flex items-center gap-2 px-4 py-2 bg-accent hover:bg-accent/80 text-white rounded-lg transition-colors"
            >
              <Download size={18} />
              Export CSV
            </button>
          </div>

          {taxLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent"></div>
            </div>
          ) : taxReport ? (
            <>
              {/* Summary */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-800 p-6 rounded-lg">
                  <p className="text-gray-400 text-sm">Total Capital Gains ({taxReport.year})</p>
                  <p
                    className={`text-2xl font-bold font-mono-numbers mt-1 ${
                      taxReport.total_gains >= 0 ? 'text-profit' : 'text-loss'
                    }`}
                  >
                    {taxReport.total_gains >= 0 ? '+' : ''}${taxReport.total_gains.toFixed(2)}
                  </p>
                </div>
                <div className="bg-gray-800 p-6 rounded-lg">
                  <p className="text-gray-400 text-sm">Tax Year</p>
                  <p className="text-2xl font-bold font-mono-numbers mt-1">
                    {taxReport.year}
                  </p>
                </div>
              </div>

              {/* Entries Table */}
              {taxReport.entries.length > 0 ? (
                <div className="bg-gray-800 rounded-lg overflow-hidden">
                  <table className="w-full">
                    <thead>
                      <tr className="text-left text-gray-400 text-sm bg-gray-700">
                        <th className="px-4 py-3">Date</th>
                        <th className="px-4 py-3">Token</th>
                        <th className="px-4 py-3">Pair</th>
                        <th className="px-4 py-3">Purchase Price</th>
                        <th className="px-4 py-3">Sale Price</th>
                        <th className="px-4 py-3 text-right">Gain/Loss</th>
                      </tr>
                    </thead>
                    <tbody>
                      {taxReport.entries.map((entry, idx) => (
                        <tr key={idx} className="border-t border-gray-700">
                          <td className="px-4 py-3">
                            {new Date(entry.date).toLocaleDateString()}
                          </td>
                          <td className="px-4 py-3">{entry.token}</td>
                          <td className="px-4 py-3">{entry.trading_pair}</td>
                          <td className="px-4 py-3 font-mono-numbers">
                            ${entry.purchase_price.toFixed(4)}
                          </td>
                          <td className="px-4 py-3 font-mono-numbers">
                            ${entry.sale_price.toFixed(4)}
                          </td>
                          <td
                            className={`px-4 py-3 text-right font-mono-numbers ${
                              entry.gain >= 0 ? 'text-profit' : 'text-loss'
                            }`}
                          >
                            {entry.gain >= 0 ? '+' : ''}${entry.gain.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-400">
                  No taxable events recorded for {taxReport.year}
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12 text-gray-400">
              No tax data available
            </div>
          )}
        </div>
      )}

      {/* Fees Tab */}
      {activeTab === 'fees' && (
        <div className="space-y-6">
          {feeLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent"></div>
            </div>
          ) : feeReport ? (
            <>
              <div className="bg-gray-800 p-6 rounded-lg">
                <p className="text-gray-400 text-sm">Total Fees Paid</p>
                <p className="text-3xl font-bold font-mono-numbers mt-1 text-loss">
                  ${feeReport.total_fees.toFixed(2)}
                </p>
              </div>

              {feeReport.entries.length > 0 ? (
                <div className="bg-gray-800 rounded-lg overflow-hidden">
                  <h3 className="text-lg font-semibold px-6 py-4 border-b border-gray-700">
                    Fees by Bot
                  </h3>
                  <table className="w-full">
                    <thead>
                      <tr className="text-left text-gray-400 text-sm">
                        <th className="px-6 py-3">Bot</th>
                        <th className="px-6 py-3">Order Count</th>
                        <th className="px-6 py-3 text-right">Total Fees</th>
                      </tr>
                    </thead>
                    <tbody>
                      {feeReport.entries.map((entry) => (
                        <tr key={entry.bot_id} className="border-t border-gray-700">
                          <td className="px-6 py-3">{entry.bot_name}</td>
                          <td className="px-6 py-3 font-mono-numbers">{entry.order_count}</td>
                          <td className="px-6 py-3 text-right font-mono-numbers text-loss">
                            ${entry.total_fees.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-400">
                  No fee data recorded yet
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12 text-gray-400">No fee data available</div>
          )}
        </div>
      )}
    </div>
  )
}
