# Change: Advance dip_recovery's setup logic per bar, not per evaluation

## Why
`fix-dip-recovery-bar-atr` moved this strategy's *volatility* measurement onto
time-based bars. Its **setup logic** was left tick-denominated and is the same
defect: `reference_high_lookback_ticks` (60), `ema_slope_period` (5) and
`min_ticks_without_new_low` (2) count **evaluations**, and the regime gate reads
the raw tick price series.

The live loop evaluates about once a second, so at runtime the "recent high" a
decline is measured against spans ~60 **seconds** rather than the 60 minutes the
defaults describe, the EMA slope is a 5-second slope, and "no new low for 2
ticks" means 2 seconds. A strategy whose stated thesis is a significant decline
followed by a confirmed reversal is instead looking for a 1.5% collapse inside
one minute.

## What Changes
- Derive a bar-close series from the bars the strategy already aggregates, and
  use it for every setup input: the reference-high window, the EMA slope, the
  regime gate, the spike guard, and the warm-up gate.
- Advance the setup lifecycle **once per completed bar** rather than on every
  evaluation, so the confirmation counters denote bars of market time.
- Leave exit management on every evaluation: a stop must react to price when it
  moves, not at the end of a bar.

## What Explicitly Does NOT Change
- **No parameter value, threshold, or multiplier is tuned.** Restoring the
  cadence is what makes the existing defaults mean what they say.
- No change to the trading theory, the lifecycle states, the entry or exit
  conditions, sizing, or the decision-score and fee-viability gates.
- No change to any other strategy or to the validation tooling.

## Impact
- Affected specs: dip-recovery-strategy.
- Affected code: `backend/app/services/trading_engine.py` —
  `_strategy_dip_recovery` and its setup helper.
- **Behaviour change, live:** setup detection now observes minute bars instead of
  seconds. In backtests the change is again near-nil by construction, because one
  candle completes one bar, so the setup path still advances on every candle —
  which is why a before/after measurement is part of this change.
