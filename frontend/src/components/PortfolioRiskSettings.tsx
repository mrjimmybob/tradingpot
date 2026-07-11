/**
 * Portfolio Risk Settings
 *
 * Configure account-wide risk caps that aggregate across every bot for
 * this deployment (see PortfolioRiskService / add-trading-safety-boundaries):
 * daily/weekly realized-loss caps, max drawdown, and max total exposure.
 *
 * This deployment has no multi-user/account model yet, so every bot shares
 * a single owner_id ("default") - see Bot.owner_id. There is intentionally
 * no owner picker here; all caps configured on this page apply to every bot.
 */

import { apiFetch } from '../lib/api'
import React from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ShieldAlert, Save } from 'lucide-react'
import { useToast } from './Toast'

const OWNER_ID = 'default'

interface PortfolioRiskConfig {
  owner_id: string
  daily_loss_cap_pct: number | null
  weekly_loss_cap_pct: number | null
  max_drawdown_pct: number | null
  max_total_exposure_pct: number | null
  enabled: boolean
}

async function fetchPortfolioRisk(): Promise<PortfolioRiskConfig> {
  const res = await apiFetch(`/portfolio/risk/${OWNER_ID}`)
  if (!res.ok) throw new Error('Failed to fetch portfolio risk config')
  return res.json()
}

async function savePortfolioRisk(config: PortfolioRiskConfig): Promise<PortfolioRiskConfig> {
  const res = await apiFetch('/portfolio/risk', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!res.ok) throw new Error('Failed to save portfolio risk config')
  return res.json()
}

// Percent input that stores null when cleared (a null cap = disabled for
// that check specifically, independent of the overall enabled toggle).
function CapInput({
  label,
  hint,
  value,
  onChange,
}: {
  label: string
  hint: string
  value: number | null
  onChange: (value: number | null) => void
}) {
  return (
    <label className="block">
      <span className="text-sm text-gray-300">{label}</span>
      <div className="mt-1 flex items-center gap-2">
        <input
          type="number"
          min={0}
          step={0.1}
          value={value ?? ''}
          onChange={(e) => {
            const raw = e.target.value
            onChange(raw === '' ? null : Number(raw))
          }}
          placeholder="Off"
          aria-label={label}
          className="w-28 px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-accent"
        />
        <span className="text-sm text-gray-400">%</span>
      </div>
      <p className="text-xs text-gray-500 mt-1">{hint}</p>
    </label>
  )
}

export const PortfolioRiskSettings: React.FC = () => {
  const queryClient = useQueryClient()
  const toast = useToast()

  const { data, isLoading } = useQuery({
    queryKey: ['portfolio-risk', OWNER_ID],
    queryFn: fetchPortfolioRisk,
  })

  const [draft, setDraft] = React.useState<PortfolioRiskConfig | null>(null)

  // Seed the editable draft once the server config loads, without
  // clobbering in-progress edits on background refetches.
  React.useEffect(() => {
    if (data && !draft) {
      setDraft(data)
    }
  }, [data, draft])

  const saveMutation = useMutation({
    mutationFn: savePortfolioRisk,
    onSuccess: (saved) => {
      queryClient.setQueryData(['portfolio-risk', OWNER_ID], saved)
      setDraft(saved)
      toast.success(
        'Portfolio Risk Settings Saved',
        saved.enabled
          ? 'Caps are now enforced across all bots.'
          : 'Caps are saved but disabled - trades are not blocked by them.'
      )
    },
    onError: (error) => {
      toast.error('Save Failed', error instanceof Error ? error.message : 'Failed to save settings')
    },
  })

  if (isLoading || !draft) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 flex items-center justify-center h-40">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-accent"></div>
      </div>
    )
  }

  const handleSave = () => {
    saveMutation.mutate({ ...draft, owner_id: OWNER_ID })
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex items-center gap-2 mb-2">
        <ShieldAlert size={20} className="text-accent" />
        <h3 className="text-lg font-semibold">Portfolio Risk Caps</h3>
        <span
          className={`ml-auto text-xs px-2 py-1 rounded ${
            draft.enabled
              ? 'bg-profit/20 text-profit border border-profit/30'
              : 'bg-gray-700 text-gray-400 border border-gray-600'
          }`}
        >
          {draft.enabled ? 'Enforced' : 'Not enforced'}
        </span>
      </div>
      <p className="text-gray-400 text-sm mb-4">
        Account-wide caps aggregated across every bot in this deployment. Leave a
        field blank to leave that specific cap off. The overall toggle below must
        also be on for any cap to actually block trades.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <CapInput
          label="Daily loss cap"
          hint="Blocks new trades once today's realized loss reaches this % of portfolio budget."
          value={draft.daily_loss_cap_pct}
          onChange={(v) => setDraft({ ...draft, daily_loss_cap_pct: v })}
        />
        <CapInput
          label="Weekly loss cap"
          hint="Same as daily, measured from the start of the current week."
          value={draft.weekly_loss_cap_pct}
          onChange={(v) => setDraft({ ...draft, weekly_loss_cap_pct: v })}
        />
        <CapInput
          label="Max drawdown"
          hint="Blocks new trades once portfolio balance falls this % below its starting budget."
          value={draft.max_drawdown_pct}
          onChange={(v) => setDraft({ ...draft, max_drawdown_pct: v })}
        />
        <CapInput
          label="Max total exposure"
          hint="Resizes (or blocks) buy orders that would push open exposure above this % of balance."
          value={draft.max_total_exposure_pct}
          onChange={(v) => setDraft({ ...draft, max_total_exposure_pct: v })}
        />
      </div>

      <div className="flex items-center justify-between border-t border-gray-700 pt-4">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={draft.enabled}
            onChange={(e) => setDraft({ ...draft, enabled: e.target.checked })}
            className="h-4 w-4"
          />
          <span className="text-sm text-gray-300">Enforce these caps</span>
        </label>
        <button
          onClick={handleSave}
          disabled={saveMutation.isPending}
          className="flex items-center gap-2 px-4 py-2 bg-accent/20 text-accent hover:bg-accent/30 rounded-lg text-sm disabled:opacity-50"
        >
          <Save size={16} />
          {saveMutation.isPending ? 'Saving...' : 'Save'}
        </button>
      </div>
    </div>
  )
}
