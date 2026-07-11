## 1. Backtest Engine Short Support
- [ ] 1.1 Extend `BacktestPortfolio` to allow negative `base_amount` (short
      position) with its own entry/cover accounting, separate from the
      existing long-only `apply_buy`/`apply_sell`.
- [ ] 1.2 Add borrow/funding cost modeling to `BacktestExecutionModel` for
      short positions (pessimistic-by-default, not zero).
- [ ] 1.3 Add liquidation modeling: a short position whose adverse move
      would breach margin must be force-closed at the liquidation price
      within the backtest, not ride out silently.
- [ ] 1.4 Tests: short P&L accounting correctness, liquidation forces a
      close, no lookahead introduced (extend the existing
      `test_decision_never_sees_a_future_candle` pattern to short signals).

## 2. First Short-Capable Strategy
- [ ] 2.1 Add a mirror-image short variant to `_strategy_trend_following`
      (or a clearly-scoped sibling), reusing its existing EMA/ATR logic.
- [ ] 2.2 Backtest across bull/bear/chop windows.
- [ ] 2.3 Walk-forward validate (via `add-strategy-validation-tooling`)
      before considering this strategy for live wiring — do not proceed to
      Section 3 without a positive out-of-sample result.

## 3. Live Execution (gated on Section 2 passing validation)
- [ ] 3.1 Confirm exchange support for short/perpetual order placement
      (`backend/app/services/exchange.py`).
- [ ] 3.2 Wire live order placement for short/cover, default-disabled
      per-bot config flag.
- [ ] 3.3 Extend `PortfolioRiskService`'s exposure calculation
      (`portfolio_risk.py`) to include short notional (depends on
      `add-trading-safety-boundaries`'s aggregation fix already being in
      place, otherwise this inherits the same broken-aggregation bug).
- [ ] 3.4 Manual dry-run validation period before any live capital.
