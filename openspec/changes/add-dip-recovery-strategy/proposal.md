# Change: Add Dip Recovery (Reversal Momentum) strategy

## Why
Every existing strategy either follows an established trend (trend_following,
funding_carry), trades a stable range (adaptive_grid, mean_reversion, volatility_breakout),
or accumulates on a fixed clock (dca_accumulator). None of them specifically target the
pullback-reversal setup: a significant, volatility-relative decline that then shows
confirmed reversal off a local low. dca_accumulator/mean_reversion will buy into an
ongoing decline; this gap means the bot has no strategy that waits for a confirmed bounce
before committing capital after a sharp drop.

## What Changes
- Add a new, independent strategy `dip_recovery`: monitors for a significant decline
  (adaptive to current volatility via an ATR-percent proxy), tracks the resulting low, and
  only buys after price has reversed off that low by an adaptive margin with optional
  momentum confirmation. Never buys while price is still falling.
- Explicit lifecycle state machine (IDLE → TRACKING_DROP → WAITING_REVERSAL → LONG_OPEN →
  COOLDOWN → IDLE), persisted like every other strategy's runtime state.
- ATR-based exits (take-profit, monotonic trailing stop, emergency stop), a maximum
  position duration, and a loss-aware cooldown (a losing exit pauses longer than a winning
  one).
- Integrate into Auto Mode: regime-based eligibility (`_get_strategy_capabilities`) and a
  numeric opportunity score (`_compute_opportunity_score`) shared with the strategy's own
  diagnostics via one formula (`_dip_recovery_score_from_ratios`).
- Integrate into diagnostics/explanations via the existing `ExplanationBuilder` - every
  evaluation reports the current lifecycle state, the value being monitored, the required
  threshold, and the distance to the next action.
- No changes to any existing strategy's logic, parameters, or execution behavior.

## Impact
- Affected specs: dip-recovery-strategy (new)
- Affected code:
  - `backend/app/services/trading_engine.py` — new strategy method and helpers, new
    shared `_calc_price_atr_proxy` utility, `_PERSISTED_STATE_ATTRS`,
    `_get_strategy_executor`, `_get_strategy_capabilities`, `_compute_opportunity_score`,
    `validate_dip_recovery_params`.
  - `backend/app/routers/config.py` — `StrategyInfo` entry (parameters/defaults exposed
    to the existing frontend strategy-config UI; no UI code changed).
  - `backend/app/routers/bots.py` — cross-field parameter validation registration.
  - `backend/app/services/strategy_capacity.py` — capacity-tracking entry (unlimited,
    matching the other alpha strategies).
  - `backend/tests/test_dip_recovery_strategy.py` — new regression test suite (26 tests).
