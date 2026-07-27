# Investigation: is `dip_recovery` correctly configured?

**Date:** 2026-07-27
**Question:** the validation tooling proved the 4h/1d measurements of
`dip_recovery` were invalid. That does not answer whether the strategy itself is
correctly configured. This report answers that.

> **Status: corrected.** The defect described below was fixed on 2026-07-27 by
> `fix-dip-recovery-bar-atr`. See *Correction applied* at the end for the
> measured before/after impact and what deliberately remains.

**Verdict: genuine strategy design defect.** `dip_recovery` denominates its
volatility estimate in *evaluation ticks* while the thresholds that consume it
must clear an *absolute* fee hurdle. The live engine evaluates roughly once per
second, at which cadence the strategy's own fee-viability guard blocks **every**
entry — in one of the most volatile months on record, by a factor of 13. It
cannot open a position in live operation, and this is independent of the
measurement problem found earlier.

No code was changed. A proposed minimal correction is at the end.

---

## 1. What timeframes is `dip_recovery` intended to support?

Its design is explicit that it is **tick-driven**: it operates on the shared tick
`_price_histories` and approximates True Range from tick-to-tick price changes
because "no OHLC candles are available for tick-driven bots"
(`add-dip-recovery-strategy/design.md`, decisions at lines 24–41).

Its parameters are denominated in two different units, which pins the intended
cadence:

| Tick-denominated | Wall-clock-denominated |
|---|---|
| `atr_period` = 14 ticks | `setup_expiry_minutes` = 240 |
| `reference_high_lookback_ticks` = 60 | `max_position_duration_minutes` = 720 |
| `ema_slope_period` = 5 | `cooldown_seconds` = 300 |
| `min_ticks_without_new_low` = 2 | `loss_cooldown_seconds` = 1800 |

These two families are only mutually coherent at **one tick per minute**:

- 14-tick ATR = 14 minutes, 60-tick lookback = 60 minutes — the classic 14/60
  period pairing.
- `setup_expiry` 240 min = 4× the 60-minute lookback window.
- `max_position_duration` 720 min = 12 hours; cooldowns 5 and 30 minutes.

At any other cadence the two families decouple. **The defaults were calibrated
for a 60-second evaluation cadence.**

## 2. Is `setup_expiry_minutes = 240` fixed, or should it scale?

It is a fixed design decision, and it is *correct* for the intended 60s cadence
(240 evaluations, 4× the lookback window). It should **not** be made to scale.

The problem is not this parameter's value. It is that the *other* parameters are
denominated in ticks, so the ratio between them changes with cadence while this
one does not. Scaling `setup_expiry_minutes` to the timeframe would paper over
the unit mismatch rather than fix it, and would silently change what the
strategy means by "a setup" at every timeframe.

## 3. Do other time-based parameters suffer the same issue?

Yes — all four wall-clock parameters degrade together, because the issue is the
cadence, not any individual value. Measured against candle interval:

| Parameter | Value | Evaluations at 1h | at 4h | at 1d |
|---|---|---|---|---|
| `cooldown_seconds` | 5 min | 0.08 | 0.02 | 0.003 |
| `loss_cooldown_seconds` | 30 min | 0.5 | 0.125 | 0.02 |
| `setup_expiry_minutes` | 240 min | 4 | **1** | 0.17 |
| `max_position_duration_minutes` | 720 min | 12 | 3 | 0.5 |

At 4h, `setup_expiry` gets exactly one evaluation — the cause of the six years of
zeros. The cooldowns stop binding at every timeframe of 1h or coarser, so the
strategy re-enters faster than it would live.

## 4. Does it assume a high-frequency cadence incompatible with candle backtests?

It assumes **60 seconds**. The live loop actually runs at **~1 second**
(`trading_engine.py:1067`, `await asyncio.sleep(1)`), and `_execute_strategy` is
called every iteration with no throttle (`:913`).

Every other strategy bridges that gap by aggregating ticks into bars and acting
only on bar close — `adaptive_grid` (`:1894`), `mean_reversion` (`:3024`),
`trend_following` (`:3925`), `volatility_breakout` (`:4759`), `auto` (`:6845`).

**`dip_recovery` is the only strategy with no bar aggregation.** It reads
`bar_interval_seconds` (`:5819`) but uses it solely for proposal validity, never
to gate evaluation. So it acts on every ~1-second tick, and its 14-tick ATR spans
14 *seconds*.

This is not a new discovery in this codebase. `trend_following` carries this
comment at `:3910`:

> Replace tick-level ATR with bar-based ATR. The original code computed
> `|price[i] − price[i-1]|` at 1 Hz → ATR ≈ $1-3 on BTC, placing stops well
> inside the fee hurdle (guaranteed loss on every trade). 60-second bar H-L
> ranges (~$50-200 on BTC) reflect actual volatility.

`dip_recovery` still uses exactly the approach that comment describes as a bug,
via `_calc_price_atr_proxy` — which, since that fix, **only `dip_recovery` calls**
(`:5794`). Its own docstring claims the approximation is done "exactly like
`trend_following`" (`:5711`). That claim is now false.

### The consequence, measured

`dip_recovery` gates its own entries on fee viability (`:6358`): it refuses to
enter unless its take-profit target (`3 × ATR`) clears round-trip fees plus a
safety margin — 0.25% at the default 0.1% fee. Real BTCUSDT data:

| Period | 1m ATR% (median) | Take-profit at 1m | Take-profit at ~1s (live) | Hurdle |
|---|---|---|---|---|
| Feb 2022 (extreme volatility) | 0.0499% | 0.150% | **0.019%** | 0.25% |
| 2024 Q1 (normal) | 0.0406% | 0.122% | **0.016%** | 0.25% |

*(1s figures scale the measured 1m values by √60, the random-walk relation; no
1-second data is stored. The extrapolated ~$3.65 ATR on BTC corroborates the
"$1-3" figure in the `trend_following` comment.)*

**At the live cadence the take-profit target is 13× short of the fee hurdle in
the most volatile month on record.** The guard fires on every entry attempt. The
strategy does not lose money — it correctly refuses to trade — but it can never
trade at all.

A second, independent blocker compounds it: the drop threshold is
`max(min_drop_percent, 2.5 × ATR%)`, so at fine cadence the 1.5% floor binds, and
the tracked "recent high" spans 60 ticks = **60 seconds**. Live, the strategy is
waiting for a 1.5% fall within one minute.

## 5. Are the defaults internally consistent across supported timeframes?

**No — and there is no timeframe at which all of them are consistent.**

| Cadence | ATR / fee viability | `setup_expiry` | Usable? |
|---|---|---|---|
| ~1s (live) | blocked, 13× short | 14,400 evals | **No** — cannot enter |
| 1m | blocked at median, passes at p90 | 240 evals | Marginal |
| 5m–15m | passes | 48–16 evals | **Yes** |
| 1h | passes | 4 evals | Marginal |
| 4h | passes | **1 eval** | **No** — setups cannot resolve |
| 1d | passes | 0.17 evals | **No** |

The valid operating range is roughly **5m–1h tick spacing**, bounded below by fee
viability and above by setup expiry. The live system runs at ~1 second — outside
that range, at the wrong end.

The defaults are self-consistent *as a set* (they describe a coherent 60-second
strategy). They are inconsistent with the cadence the strategy is actually
executed at.

---

## Classification

Not a measurement limitation — that was the separate, already-fixed 4h/1d issue.
Not merely a default-value inconsistency — no change of values fixes it, because
the units themselves are cadence-dependent.

**A genuine strategy design issue:** a volatility estimate denominated in
evaluation ticks, consumed by thresholds that must clear an absolute fee hurdle.
Its meaning changes whenever the cadence changes, and at the cadence it actually
runs at, it is ~30× too small.

## Proposed minimal correction

Give `dip_recovery` the 60-second bar aggregation every other strategy already
has, and drive its ATR from bar high-low ranges rather than tick-to-tick deltas —
precisely the change already made to `trend_following` for this identical bug.

This is the smallest correction that addresses the cause rather than a symptom:

- It restores the cadence the defaults were calibrated for, so **no parameter
  value needs to change** — 14 bars = 14 minutes and 60 bars = 60 minutes become
  true again, and `setup_expiry` returns to 4× the lookback window.
- It reuses an established in-repo pattern and the parameter `dip_recovery`
  already declares (`bar_interval_seconds`), rather than inventing logic.
- It makes the docstring's "exactly like `trend_following`" claim true.

Two smaller alternatives were considered and rejected:

- *Add only a fee-coverage floor to the ATR* (as `trend_following` also has).
  This would unblock entries but leave the volatility estimate wrong, so stops
  and targets would be set from a floor rather than from measured volatility, and
  the 60-tick lookback would still span 60 seconds.
- *Rescale the wall-clock parameters to the cadence.* This treats the symptom,
  changes what a "setup" means at every timeframe, and is parameter tuning.

---

## Correction applied (2026-07-27)

Implemented as `fix-dip-recovery-bar-atr`. `dip_recovery` now aggregates ticks
into `bar_interval_seconds` bars in its own persisted state and computes ATR
from bar high-low ranges, with the same fee-coverage floor `trend_following`
applies. **No parameter value, threshold, or multiplier was changed.**

### Measured impact — live cadence (the defect)

Driven at ~1 evaluation/second over 35,940 ticks of a BTCUSDT-derived price path:

| | ATR | ATR% | Take-profit (3×ATR) | vs 0.25% hurdle |
|---|---|---|---|---|
| Before | $1.62 | 0.0038% | 0.0114% | **BLOCKED** |
| After | $106.62 | 0.2500% | 0.7500% | **PASSES** |

A **66× increase**. The $1.62 measured "before" independently corroborates the
"$1-3 on BTC" figure recorded in `trend_following`'s comment. On this particular
path the fee-coverage floor sets the result (raw 60-second bar range averaged
$15.64 against a $106.62 floor at that price); on a live feed with genuine
intra-minute noise the bar range contributes more — `trend_following` documents
60-second ranges of $50-200 on BTC. Either component clears the hurdle; the old
tick ATR could not.

### Measured impact — backtests (near-nil, as predicted)

| Timeframe | | Trades | Expectancy | Win rate | Profit factor |
|---|---|---|---|---|---|
| 1h | before | 341 | −27.40 | 32.6% | 0.61 |
| 1h | after | 342 | −26.94 | 33.0% | 0.62 |
| 15m | before | 416 | −20.36 | 29.1% | 0.53 |
| 15m | after | 417 | −20.52 | 27.8% | 0.54 |

This near-identity is the expected result and is itself evidence the change is
what it claims to be: in a backtest one candle produces one bar, so a bar's range
is the candle-to-candle move the tick proxy already measured. The change bites
live, where 60 one-second ticks now collapse into one bar. It also confirms the
fix is not a disguised parameter change — **`dip_recovery` still loses money at
1h with its shipped defaults**, and that finding stands unaltered.

The all-strategy 4h baseline is **byte-identical** before and after for all six
strategies: nothing else moved.

---

## Second correction: the setup logic (2026-07-27)

Implemented as `fix-dip-recovery-setup-cadence`. The first fix moved *volatility*
onto bars; the setup logic was still tick-denominated, which is the same defect.
`reference_high_lookback_ticks` (60), `ema_slope_period` (5) and
`min_ticks_without_new_low` (2) counted **evaluations**, and the regime gate read
the raw tick series.

The setup lifecycle now advances **once per completed bar** and reads bar
highs/closes; exits still run on every evaluation, ahead of the per-bar gate, so
a stop reacts when price moves. Again **no parameter value was changed.**

### Measured impact — live cadence

A 2% decline spread over 30 minutes (an ordinary dip), evaluated once a second.
`bar_interval_seconds=1` reproduces the old 60-evaluation lookback exactly:

| | Lookback spans | Result |
|---|---|---|
| Before | 60 seconds | **IDLE** — never detected |
| After | 60 minutes | **TRACKING_DROP** — armed, reference high 100.0 |

The strategy's entire thesis is "a significant decline, then a confirmed
reversal". Before this fix, at its real cadence, **any decline slower than about
a minute was invisible to it** — it could only ever have seen a flash crash.

### Measured impact — backtests (near-nil again)

| Timeframe | | Trades | Expectancy | Profit factor |
|---|---|---|---|---|
| 1h | before → after | 342 → 345 | −26.94 → −25.73 | 0.62 → 0.63 |
| 15m | before → after | 417 → 417 | −20.52 → −20.71 | 0.54 → 0.54 |

Same reason as before: one candle completes one bar, so the setup path still
advances on every candle. The all-strategy 4h baseline is again identical for all
six strategies. **`dip_recovery` still loses money at 1h with its shipped
defaults** — neither correction changed that, and neither was intended to.

### Two safety points worth recording

- **Exits are not deferred to bar close.** The per-bar gate sits behind exit
  management; deferring a stop to the end of a bar would have been a real risk
  regression.
- **The warm-up gate is skipped while a position is open.** A bar-based warm-up
  would otherwise leave a bot resuming from pre-bar-aggregation state holding an
  unmanaged position for `atr_period` bars — the hazard
  `_PERSISTED_PRICE_HISTORY_LEN` already exists to prevent. Caught by a test.

### Still bounded above

`setup_expiry_minutes` (240) is unchanged and still caps the usable evaluation
interval: at 4h a setup gets exactly one evaluation and can never be confirmed.
The validation tooling's cadence check warns about this automatically.
