# Change: Compute dip_recovery's ATR from time-based bars, not evaluation ticks

## Why
`dip_recovery` measures volatility with `_calc_price_atr_proxy` — the absolute
tick-to-tick price change, averaged over `atr_period` **evaluation ticks**. That
quantity has no fixed unit of time: it means whatever the caller's cadence
happens to be.

The live loop evaluates every ~1 second (`trading_engine.py:1067`, with
`_execute_strategy` called every iteration at `:913`), and `dip_recovery` is the
only strategy that does not aggregate ticks into bars — it reads
`bar_interval_seconds` (`:5819`) but uses it solely for proposal validity. Its
14-tick ATR therefore spans 14 *seconds*, while every threshold derived from it
must clear an absolute fee hurdle.

Measured on real BTCUSDT data, the take-profit target (`3 × ATR`) at the live
cadence is **0.019%** in Feb 2022 and **0.016%** in 2024 Q1 against a **0.25%**
round-trip fee hurdle. The strategy's own fee-viability gate (`:6358`) therefore
refuses **every** entry: it cannot open a position live, in any market condition.

This is a known, already-solved bug in this codebase. `trend_following` carries
the fix and the reason (`:3910`):

> Replace tick-level ATR with bar-based ATR. The original code computed
> `|price[i] − price[i-1]|` at 1 Hz → ATR ≈ $1-3 on BTC, placing stops well
> inside the fee hurdle (guaranteed loss on every trade). 60-second bar H-L
> ranges (~$50-200 on BTC) reflect actual volatility.

`dip_recovery` still uses the superseded approach and is now the only caller of
`_calc_price_atr_proxy` (`:5794`). Full evidence:
`DIP_RECOVERY_CADENCE_INVESTIGATION.md`.

## What Changes
- Aggregate ticks into `bar_interval_seconds` bars inside `dip_recovery`'s own
  persisted state, and compute ATR from bar high-low ranges — the identical
  mechanism `trend_following` uses (`tf_bars` / `_calc_bar_atr`).
- Apply the same fee-coverage floor `trend_following` applies, so ATR can never
  place a stop inside the round-trip fee hurdle during bar warm-up.
- Correct the stale documentation that describes the tick-based approximation.

## What Explicitly Does NOT Change
- **No parameter value, threshold, or multiplier is tuned.** The whole point is
  that the existing defaults are already calibrated for one tick per minute
  (14-bar ATR = 14 min, 60-tick lookback = 60 min, `setup_expiry_minutes` 240 =
  4× that lookback); restoring that cadence is what makes them mean what they
  say.
- No change to the trading theory, the lifecycle state machine, entry/exit
  conditions, sizing, or the regime and decision-score gates.
- No change to any other strategy, to `BacktestEngine`, or to the validation
  tooling.

## Impact
- Affected specs: dip-recovery-strategy (new capability record for the
  volatility-source requirement).
- Affected code: `backend/app/services/trading_engine.py` —
  `_dip_recovery_default_state` (two new state keys) and
  `_strategy_dip_recovery` (bar accumulation + ATR source).
- Affected docs: the strategy docstring, and
  `DIP_RECOVERY_CADENCE_INVESTIGATION.md`'s proposed-correction section.
- **Behaviour change, live:** every ATR-derived distance (drop threshold,
  recovery confirmation, take-profit, trailing stop, emergency stop, position
  sizing) is computed from a 60-second volatility estimate instead of a
  1-second one. In backtests the change is small by construction — one candle
  produces one bar, so a bar's range is the candle-to-candle move the tick proxy
  already measured — which is why a before/after measurement is part of this
  change rather than an afterthought.
