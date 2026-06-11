# Frontend Financial Transparency - Quick Start Guide

## New Pages

### 🏦 Tax Reporting (`/tax`)
Dedicated tax compliance interface with fiscal year selector, short/long-term gain breakdown, and CSV export.

**Navigate to**: Click "Tax" in sidebar or visit http://localhost:5173/tax

### 🛡️ Audit & Compliance (`/audit`)
Unified audit log with filtering (bot, severity, date range) and CSV export.

**Navigate to**: Click "Audit" in sidebar or visit http://localhost:5173/audit

---

## New Components (Ready to Use)

### 1. Trade Detail Modal
```tsx
import { TradeDetailModal } from '../components/TradeDetailModal';

<TradeDetailModal
  tradeId={123}
  isSimulated={true}
  onClose={() => setShowModal(false)}
/>
```
**Shows**: Trade + Order + Ledger Entries + Tax Lots + Realized P&L + Costs

---

### 2. Balance Drilldown Modal
```tsx
import { BalanceDrilldownModal } from '../components/BalanceDrilldownModal';

<BalanceDrilldownModal
  asset="USDT"
  isSimulated={true}
  onClose={() => setShowModal(false)}
  onTradeClick={(id) => openTradeDetail(id)}
/>
```
**Shows**: Last 20 ledger entries with source tags (trade/fee/funding/correction)

---

### 3. Risk & Safety Panel
```tsx
import { RiskSafetyPanel } from '../components/RiskSafetyPanel';

<RiskSafetyPanel isSimulated={true} refreshInterval={10000} />
```
**Shows**: Drawdown %, Daily Loss %, Exposure %, Kill Switch State (auto-refreshes)

---

### 4. Equity Curve with Events
```tsx
import { EquityCurveWithEvents } from '../components/EquityCurveWithEvents';

<EquityCurveWithEvents isSimulated={true} asset="USDT" height={400} />
```
**Shows**: P&L chart + Event markers (strategy switches, kill switches, losses)

---

### 5. Strategy Reason Panel
```tsx
import { StrategyReasonPanel } from '../components/StrategyReasonPanel';

<StrategyReasonPanel botId={1} botName="My Bot" isSimulated={true} />
```
**Shows**: Current strategy, eligible strategies, blocked strategies with reasons

---

### 6. Global Safety Controls
```tsx
import { GlobalSafetyControls } from '../components/GlobalSafetyControls';

<GlobalSafetyControls isSimulated={true} />
```
**Shows**: Freeze All, Rebuild State, Export All Data (admin-only)

---

## Quick Integration Examples

### Add to Dashboard
```tsx
// src/pages/Dashboard.tsx
import { RiskSafetyPanel } from '../components/RiskSafetyPanel';
import { EquityCurveWithEvents } from '../components/EquityCurveWithEvents';

// Add after stat cards:
<RiskSafetyPanel isSimulated={true} />
<EquityCurveWithEvents isSimulated={true} height={400} />
```

### Add to Bot Detail
```tsx
// src/pages/BotDetail.tsx
import { useState } from 'react';
import { TradeDetailModal } from '../components/TradeDetailModal';
import { StrategyReasonPanel } from '../components/StrategyReasonPanel';

const [selectedTradeId, setSelectedTradeId] = useState<number | null>(null);

// Add to page:
<StrategyReasonPanel botId={bot.id} botName={bot.name} isSimulated={bot.is_dry_run} />

// In trade table:
<button onClick={() => setSelectedTradeId(trade.id)}>
  View Details
</button>

// Add modal:
{selectedTradeId && (
  <TradeDetailModal
    tradeId={selectedTradeId}
    isSimulated={bot.is_dry_run}
    onClose={() => setSelectedTradeId(null)}
  />
)}
```

### Add to Settings
```tsx
// src/pages/Settings.tsx
import { GlobalSafetyControls } from '../components/GlobalSafetyControls';

// Add at bottom:
<GlobalSafetyControls isSimulated={isSimulated} />
```

---

## File Locations

**Components**:
- `frontend/src/components/TradeDetailModal.tsx`
- `frontend/src/components/BalanceDrilldownModal.tsx`
- `frontend/src/components/RiskSafetyPanel.tsx`
- `frontend/src/components/EquityCurveWithEvents.tsx`
- `frontend/src/components/StrategyReasonPanel.tsx`
- `frontend/src/components/GlobalSafetyControls.tsx`

**Pages**:
- `frontend/src/pages/TaxReporting.tsx`
- `frontend/src/pages/AuditCompliance.tsx`

**Routing**:
- `frontend/src/App.tsx` (routes added)
- `frontend/src/components/Layout.tsx` (navigation updated)

---

## Color Coding (Quick Reference)

**Risk Levels**:
- 🟢 Green = Safe
- 🟡 Amber = Near Limit
- 🔴 Red = Critical/Blocked

**P&L**:
- 🟢 Green = Profit
- 🔴 Red = Loss

**Bot Status**:
- 🔵 Blue = Running
- 🟡 Yellow = Paused
- ⚫ Gray = Stopped

**Event Types**:
- 🔵 Indigo = Strategy Switch
- 🔴 Red = Kill Switch
- 🟡 Amber = Regime Change
- 🟣 Purple = Grid Recenter
- 🔴 Dark Red = Large Loss

---

## Testing the Implementation

### 1. Start Backend
```bash
cd backend
python -m uvicorn app.main:app --reload
```

### 2. Start Frontend
```bash
cd frontend
npm install  # if needed
npm run dev
```

### 3. Navigate to New Pages
- Tax: http://localhost:5173/tax
- Audit: http://localhost:5173/audit

### 4. Test Components
- Click any balance → Opens Balance Drilldown
- Click any trade → Opens Trade Detail
- View Dashboard → See Risk Panel + Equity Curve

---

## Safety Features

✅ **Data Separation**: All components require `isSimulated` parameter
✅ **Mode Banners**: Large SIMULATED/LIVE indicators
✅ **Confirmations**: Double-confirm destructive operations
✅ **Read-Only**: All endpoints are query-only (no mutations)

---

## Troubleshooting

**"Trade not found"**
→ Check `is_simulated` parameter matches trade mode

**Empty balance drilldown**
→ Verify ledger has entries for the asset

**Charts not rendering**
→ Ensure `recharts` is installed: `npm install recharts`

**Risk panel not updating**
→ Check API endpoint `/api/reports/risk-status` is responding

---

## Next Steps

1. ✅ Backend API endpoints implemented
2. ✅ Frontend components created
3. ✅ Routes added to App.tsx
4. ✅ Navigation updated in Layout
5. 🔄 **TODO**: Integrate components into existing pages (Dashboard, BotDetail)
6. 🔄 **TODO**: Add bot creation safety warnings
7. 🔄 **TODO**: Test with real data

---

## Full Documentation

See `FRONTEND_IMPLEMENTATION_SUMMARY.md` for:
- Detailed component documentation
- API endpoint reference
- Integration examples
- Performance optimizations
- Accessibility features
- Testing recommendations

---

**Status**: ✅ All components implemented and ready for integration
**Backend**: ✅ All API endpoints implemented
**Frontend**: ✅ All components + pages created
**Routes**: ✅ Updated
**Navigation**: ✅ Updated
