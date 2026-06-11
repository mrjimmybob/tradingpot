# New API Endpoints Quick Reference

## Money Forensics

### Trade Detail
```
GET /api/reports/trade-detail/{trade_id}?is_simulated=true
```
Returns: Trade + Order + Ledger Entries + Tax Lots + Realized P&L

### Balance Drilldown
```
GET /api/reports/balance-drilldown?is_simulated=true&asset=USDT&limit=20
```
Returns: Current Balance + Last 20 Ledger Entries + Source Classification

---

## Risk Monitoring

### Risk Status
```
GET /api/reports/risk-status?is_simulated=true
```
Returns: Per-Bot Risk Metrics + Portfolio Exposure + Kill Switch States

---

## Performance Visualization

### Equity Curve
```
GET /api/reports/equity-curve?is_simulated=true&asset=USDT
```
Returns: Equity Time Series + Event Overlays (strategy switches, alerts, etc.)

---

## Strategy Analysis

### Strategy Reasoning
```
GET /api/reports/strategy-reason/{bot_id}?is_simulated=true
```
Returns: Current Strategy + Eligible Strategies + Blocked Strategies (with reasons)

---

## Tax Reporting

### Tax Summary
```
GET /api/reports/tax-summary/{year}?is_simulated=true
```
Returns: Total Realized Gain + Short/Long-Term Breakdown + Lot Count

---

## Audit & Compliance

### Audit Log
```
GET /api/reports/audit-log?is_simulated=true&bot_id=1&severity=error
```
Returns: Unified Log (Alerts + Strategy Rotations + Invariant Failures)

---

## Common Parameters

- **is_simulated** (required): `true` for dry-run, `false` for live
- **owner_id** (optional): Filter by owner
- **bot_id** (optional): Filter by specific bot
- **start_date** (optional): ISO 8601 datetime
- **end_date** (optional): ISO 8601 datetime
- **severity** (optional): `info`, `warning`, `error`

---

## Response Status Codes

- **200**: Success
- **400**: Invalid parameters
- **404**: Resource not found
- **500**: Server error

---

## Interactive API Docs

Visit: **http://localhost:8000/docs**

Test all endpoints directly in browser with Swagger UI.
