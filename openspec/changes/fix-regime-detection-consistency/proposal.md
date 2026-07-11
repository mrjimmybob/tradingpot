# Change: Fix regime detection consistency and volatility_breakout's regime gate

## Why
`_strategy_volatility_breakout` gates entries on `"volatility_expanding"` but
actually tests volatility *level* (an ATR percentile: low/medium/high), not
*direction* (rate of change) — verified at `trading_engine.py:3467-3491`:
`volatility_state` is relabeled directly ("high" -> "volatility_expanding")
with no rate-of-change check. A market that has been choppy and high-vol for
a week reads identically to one that broke out of compression one bar ago,
even though the strategy's whole thesis is "enter on the transition out of
compression." A second, more correct detector
(`_detect_market_regime_bar_based`) already computes a real
`volatility_direction` (expanding/contracting/stable, with 3-bar persistence
before flipping) — but only `_strategy_auto` uses it. `_strategy_dca`,
`_strategy_mean_reversion`, and `_strategy_volatility_breakout` each call the
weaker, price-only `_detect_market_regime` instead, so different strategies
reason about "regime" using different, inconsistent detectors with no shared
validated definition.

## What Changes
- Give `_strategy_volatility_breakout` access to a real volatility-direction
  signal (reusing its own already-maintained bar history) instead of
  relabeling a level.
- Fix the entry gate to require genuine expansion, not just "currently high."
- Document which detector each strategy uses and why, so this inconsistency
  isn't silently reintroduced by a future strategy.

## Impact
- Affected specs: regime-detection
- Affected code: `backend/app/services/trading_engine.py`
  (`_detect_market_regime`, `_detect_market_regime_bar_based`,
  `_strategy_volatility_breakout`'s regime-gating section at `:3467-3491`,
  `_get_strategy_capabilities` at `:6083-6109`)
- Does not change: `_strategy_dca`'s or `_strategy_mean_reversion`'s regime
  gates (their existing trend_state-based logic was audited and found
  conceptually sound — this change is scoped to the volatility_direction gap
  only).
