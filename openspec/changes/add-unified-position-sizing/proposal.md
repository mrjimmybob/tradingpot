# Change: Add unified volatility-scaled position sizing across all strategies

## Why
Only 3 of 6 strategies (`trend_following`, `volatility_breakout`,
`dip_recovery`) size positions proportionally to risk and volatility
(`risk_percent × balance / (ATR × multiplier)`, verified as three
near-identical copies of the same formula). The other 3 (`dca_accumulator`,
`adaptive_grid`, `mean_reversion`) use flat fixed-% or fixed-$ notional
regardless of volatility or how much measurable evidence supports the
trade — the same order size in a calm market and a violent one. A central expectancy gate
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
- **✓ Revision applied — no longer blocked on design.** `add-strategy-
  decision-framework` (Tier 1.5) requires position sizing to take the
  per-trade **Evidence-Based Decision Score** as an input (its Pillar 5,
  "Decision-Score-Weighted Position Sizing"). That framework's Phase 0 has
  now shipped `DecisionScoreEngine`/`DecisionScoreResult`
  (`backend/app/services/strategy_framework/decision_score.py`), and this
  change's `design.md` has been revised accordingly: the shared sizing
  function's signature now takes a `decision_score: DecisionScoreResult`
  parameter and applies a deterministic `size_multiplier(decision_score)`
  on top of the existing ATR/risk-percent formula — see `design.md`'s
  Decisions section for the full specification.
  - **Still blocked on implementation ordering, not design**: per
    `design.md`'s Migration Plan precondition, a strategy cannot be sized
    via this change until that SAME strategy has completed its own
    `add-strategy-decision-framework` Phase 1-6 migration (it needs a real
    `DecisionScoreResult` to size from — one does not exist for any
    strategy until that strategy migrates). Do not implement this change's
    Phase N sizing for a given strategy before that strategy's own
    framework migration phase lands.
