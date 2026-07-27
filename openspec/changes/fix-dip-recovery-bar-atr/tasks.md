## Premise (binding for every task)
Replace the obsolete volatility calculation with the bar-based implementation
`trend_following` already adopted, preserving the intended behaviour of the
existing defaults. **No task may tune a parameter, change a threshold or
multiplier, alter the trading theory, or redesign the strategy.**

## 1. Bar Aggregation
- [x] 1.1 Add `dr_bars` / `dr_current_bar` to `_dip_recovery_default_state()`,
      and backfill them on persisted state before any early return, so a state
      dict saved before this change restores cleanly.
- [x] 1.2 Accumulate high/low/close into the current bar on every evaluation and
      roll it over after `bar_interval_seconds`, trimming to the last 100 bars —
      the identical mechanism `trend_following` uses for `tf_bars`.

## 2. ATR Source
- [x] 2.1 Compute ATR as the mean high-low range of the last `atr_period` bars,
      replacing `_calc_price_atr_proxy`.
- [x] 2.2 Apply the same fee-coverage floor `trend_following` applies
      (`2 × fee + _VIABILITY_SAFETY_MARGIN_PCT`), so warm-up cannot produce a
      stop inside the fee hurdle.
- [x] 2.3 Confirm `_calc_price_atr_proxy` has no remaining callers, and either
      remove it or document why it stays.

## 3. Documentation
- [x] 3.1 Update the `_strategy_dip_recovery` docstring: it now measures
      volatility over bars, and the cadence-sensitivity warning is resolved
      rather than merely disclosed.
- [x] 3.2 Update `DIP_RECOVERY_CADENCE_INVESTIGATION.md` to record the
      correction and its measured impact.

## 4. Verification
- [x] 4.1 Tests: ATR is unchanged by evaluation cadence within a bar; the fee
      floor holds during warm-up; the bar-based ATR takes over once available;
      persisted state without the new keys restores cleanly.
- [x] 4.2 Full backend test suite.
- [x] 4.3 Before/after walk-forward + benchmark measurement of `dip_recovery` at
      1h (its usable range) and at a finer timeframe, using the validation
      tooling, with the cadence warning checked.
- [x] 4.4 Confirm no other strategy's measurements moved.

## 5. Same-Cadence-Mismatch Sweep
- [x] 5.1 Check whether any other defect stems from the same tick-vs-time
      mismatch in `dip_recovery` and record the finding, fixing only what falls
      within "replace the obsolete volatility calculation".
      **Found and recorded, deliberately NOT fixed:**
      `reference_high_lookback_ticks` (60), `ema_slope_period` (5) and
      `min_ticks_without_new_low` (2) still count EVALUATIONS. At the live ~1s
      cadence the "recent high" therefore spans ~60 seconds rather than the 60
      minutes the defaults imply, so live entries now require a fast drop
      (≥1.5% in about a minute). Same class of mismatch, but correcting it
      changes what the strategy means by a "setup" — entry semantics, not the
      volatility calculation — so it is outside this change's stated scope.
      Documented in the docstring and the investigation report.

### Discoveries during implementation
- **Lifecycle resets were discarding the bar history.** Five sites rebuilt state
  via `_dip_recovery_default_state()` (setup expiry, cooldown, the defensive
  LONG_OPEN reset, entry, exit). Bar history is observed market data, not
  lifecycle state: wiping it on every reset would drop the strategy back onto
  the fee-coverage floor for `atr_period` bars each time, and setups expire
  often enough that it would have sat on the floor almost permanently —
  silently undoing this fix. Caught by a test asserting measured volatility
  dominates the floor. Now routed through `_dip_recovery_reset_state()`, which
  carries `dr_bars`/`dr_current_bar` across a reset, with its own regression
  test.
- **`_calc_price_atr_proxy` is retained, not deleted.** It has no remaining
  strategy callers (asserted by a test), but it is a shared utility and the new
  tests use it to demonstrate that the two cadences disagree by >10x under the
  old semantics — i.e. to stop the cadence-independence test passing vacuously.
- **Test harness convention.** Existing dip_recovery tests drive the strategy in
  a tight loop under the real clock, so no bar would ever close and they would
  sit on the fee floor. They now pass `bar_interval_seconds=0` ("close a bar
  every call"), which makes each bar's range the change since the previous call
  — exactly the tick semantics they were written against, so every original
  assertion keeps its meaning. This is the convention `trend_following` already
  documents.
- **Backtest results barely moved (341→342 trades at 1h), which is the expected
  result**: one candle produces one bar, so a bar's range is the candle-to-candle
  move the tick proxy already measured. The all-strategy 4h baseline is
  byte-identical for all six strategies.
