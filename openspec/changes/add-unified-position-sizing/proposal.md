# Change: Add unified volatility-scaled position sizing across all strategies

## Why
Only 3 of 6 strategies (`trend_following`, `volatility_breakout`,
`dip_recovery`) size positions proportionally to risk and volatility
(`risk_percent × balance / (ATR × multiplier)`, verified as three
near-identical copies of the same formula). The other 3 (`dca_accumulator`,
`adaptive_grid`, `mean_reversion`) use flat fixed-% or fixed-$ notional
regardless of volatility or the strategy's own conviction — the same order
size in a calm market and a violent one. A central expectancy gate
(`evaluate_reward_risk`, `trading_engine.py:258-289`) already blocks trades
that don't clear fees, but it's pass/fail — it never scales size up for
high-conviction setups or down for marginal ones. This change gives every
strategy the same, consistent, volatility-aware sizing convention and routes
it through the portfolio risk service so total exposure stays bounded
account-wide (once that service's aggregation is fixed).

## What Changes
- Add one shared position-sizing function (volatility/ATR-scaled,
  risk-percent-of-balance based) that all 6 strategies call, replacing each
  strategy's ad hoc sizing formula.
- Preserve each strategy's existing minimum-order-floor and
  balance-fraction-cap safety behavior (`MIN_ORDER_USD`,
  `_BUY_BALANCE_FRACTION`) — this is a sizing-formula consolidation, not a
  safety-margin change.
- Ensure sized orders are still checked against the portfolio risk service's
  exposure cap.

## Impact
- Affected specs: position-sizing
- Affected code: `backend/app/services/trading_engine.py`
  (`_strategy_dca`, `_strategy_grid`, `_strategy_mean_reversion`, and the
  existing sizing logic in `_strategy_trend_following`,
  `_strategy_volatility_breakout`, `_strategy_dip_recovery`, consolidated)
- Depends on: `add-trading-safety-boundaries` (the exposure cap must
  aggregate correctly across bots for this to matter beyond a single bot);
  recommended to sequence after `add-strategy-validation-tooling` so the
  sizing change can be validated via backtest/walk-forward before/after,
  since it changes trade sizes and therefore results.
