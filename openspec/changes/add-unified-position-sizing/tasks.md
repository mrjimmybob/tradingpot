## 1. Implementation
- [ ] 1.1 Extract the existing risk_percent/ATR sizing formula (currently
      duplicated in `_strategy_trend_following`,
      `_strategy_volatility_breakout`, `_strategy_dip_recovery`) into one
      shared function.
- [ ] 1.2 Add tests proving the extraction is behavior-identical for the 3
      strategies already using this formula (before/after backtest parity).
- [ ] 1.3 Apply the shared sizing function to `_strategy_mean_reversion`
      (ceiling = today's `order_size_percent`).
- [ ] 1.4 Apply it to `_strategy_grid`, preserving the convex depth
      multiplier as a multiplier on top of the new base size.
- [ ] 1.5 Apply it to `_strategy_dca` (ceiling = today's
      `amount_percent`/`amount_usd`).
- [ ] 1.6 Confirm `MIN_ORDER_USD` floor and `_BUY_BALANCE_FRACTION` cap still
      apply identically after consolidation.
- [ ] 1.7 Re-run backtests for all 6 strategies before/after; document the
      delta.
