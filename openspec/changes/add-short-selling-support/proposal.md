# Change: Add short-selling / two-sided trading support

## Why
Every strategy and the entire backtest engine are long-only spot
(`backtesting/portfolio.py` docstring: "long-only spot portfolio ledger";
`BacktestPortfolio.apply_sell` clamps to never go negative; no strategy ever
produces a short position despite `PositionSide.SHORT` already existing in
the model enum with unrealized-P&L math for it). A long-only bot structurally
cannot profit in sustained bearish or declining range-bound markets — exactly
the regimes where `trend_following`/`mean_reversion`/`volatility_breakout`
are currently losing. This is the highest-cost, highest-risk item in the
roadmap (it introduces margin/liquidation risk that doesn't exist today) and
should be built last, after the validation tooling exists to actually prove a
short strategy has edge before it trades real capital.

## What Changes
- Extend `BacktestPortfolio` to support negative `base_amount` (a short
  position), with borrow/funding cost modeling in the backtest execution
  model.
- Wire at least one strategy (recommend starting with `trend_following`'s
  mirror-image short: EMA(short) < EMA(long) and price < EMA(long)) to emit
  short/cover signals producing `PositionSide.SHORT` positions.
- Extend live execution to place short/margin orders on exchanges that
  support it (the existing perpetual-funding-rate lookup code in
  `exchange.py` suggests this was already being scoped for Hyperliquid-style
  perpetuals).
- Extend the portfolio risk service's exposure calculation to account for
  short exposure (currently only sums long position notional).

## Impact
- Affected specs: short-selling
- Affected code: `backend/app/backtesting/portfolio.py`,
  `backend/app/backtesting/execution_model.py`,
  `backend/app/services/trading_engine.py`,
  `backend/app/services/exchange.py`, `backend/app/models/position.py`
  (already has `PositionSide.SHORT`), `backend/app/services/portfolio_risk.py`
- Depends on: `add-strategy-validation-tooling` (a short strategy must show
  positive out-of-sample expectancy via walk-forward before being trusted),
  `add-unified-position-sizing` (short position sizing should use the same
  volatility-scaled convention, extended for borrow cost)
