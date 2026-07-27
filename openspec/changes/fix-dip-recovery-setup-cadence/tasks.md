## Premise (binding for every task)
Restore the cadence the existing defaults assume. **No task may tune a
parameter, change a threshold or multiplier, alter the trading theory, or
redesign the strategy.**

## 1. Bar-Denominated Setup Inputs
- [x] 1.1 Derive a bar-close series from the strategy's accumulated bars.
- [x] 1.2 Use it for the reference-high window, the EMA slope, the regime gate,
      the spike guard, and the warm-up gate.

## 2. Per-Bar Lifecycle
- [x] 2.1 Advance the setup path only when a bar completes; hold otherwise.
- [x] 2.2 Keep exit management on every evaluation.

## 3. Verification
- [x] 3.1 Tests: setup progress and confirmation counters are unchanged by
      evaluation frequency; the reference high spans bars; exits still fire
      mid-bar.
- [x] 3.2 Full backend test suite.
- [x] 3.3 Before/after measurement at 1h and 15m; confirm no other strategy moved.
- [x] 3.4 Update the docstring and DIP_RECOVERY_CADENCE_INVESTIGATION.md.

### Discoveries during implementation
- **A bar-based warm-up gate would have left a resumed bot'''s position
  unmanaged.** A bot restoring state saved before bar aggregation has no bars,
  so waiting for `atr_period` of them would leave a real open position without
  exit management for that long — the hazard `_PERSISTED_PRICE_HISTORY_LEN`
  already exists to prevent. The gate is now skipped whenever a position is
  open; exit levels are locked at entry and ATR is floored, so only ENTRY needs
  a warm window. Caught by existing exit-path tests.
- **The reference high reads bar HIGHS, not bar closes.** Closes would silently
  discard every intra-bar peak and understate the decline being detected; highs
  preserve what the tick series used to give.
- **Exits deliberately bypass the per-bar gate**, which is placed after exit
  management. Deferring a stop to bar close would have been a risk regression.
- **Backtests barely moved (1h: 342→345 trades) and the 4h all-strategy baseline
  is identical**, because one candle completes one bar so the setup path still
  advances every candle. The change bites live, where it turns a 60-second
  lookback back into the 60-minute one the defaults describe.
