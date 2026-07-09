## Context
Strategies in this codebase are implemented as `TradingEngine` methods (not a separate
strategies package), dispatched through `_get_strategy_executor`, with runtime state kept
in per-bot dicts on the engine instance and persisted wholesale into `Bot.strategy_state`
via `_collect_bot_state`/`_restore_bot_state` (see `_PERSISTED_STATE_ATTRS`). Auto Mode is a
separate meta-policy (`_strategy_auto`) that ranks all eligible strategies via
`_get_strategy_capabilities` (regime eligibility) and `_compute_opportunity_score`
(0-10 live-market opportunity), then delegates to the winning strategy's own executor.
Dip Recovery follows this exact architecture - no new state system, no new dispatch
mechanism.

## Goals / Non-Goals
- Goals:
  - A pullback-reversal strategy that never buys into an active decline - only after a
    confirmed, adaptive-threshold reversal off a tracked low.
  - Every threshold adapts to current volatility (ATR-percent proxy); no BTC-specific or
    otherwise pair-specific magic numbers.
  - Full persistence through the existing generic mechanism (survives restart/deploy),
    with a single source of truth for state shape.
  - Auto Mode eligibility + numeric opportunity scoring, sharing one score formula with
    the strategy's own diagnostics.
- Non-Goals:
  - No OHLC candle infrastructure was added - like `trend_following`, this strategy
    approximates ATR/True-Range from tick price history (no exchange candle-fetching
    exists in this codebase for these bots).
  - No new indicator infrastructure (RSI, volume, order-book) - the confirmation filters
    use only what's already observable from tick price history (EMA slope, no-new-low
    count), per explicit instruction not to fabricate unavailable data.
  - No change to `_ALPHA_STRATEGIES` (risk-management-triggered rotation) - out of scope;
    only Auto Mode selection was requested.

## Decisions
- Decision: Implement on the shared tick `_price_histories` (same source
  `trend_following` uses), with a new shared `_calc_price_atr_proxy` method (True Range
  approximated as `abs(price[i] - price[i-1])`, averaged over `atr_period`).
  - Why: `trend_following`'s identical ATR-proxy formula already exists, but as a private
    closure inside `_strategy_trend_following`, not reusable. Rather than refactor
    `trend_following` to call a shared method (touching pinned, tested behavior for a
    strategy explicitly out of scope), the formula is extracted once as a new method with
    a docstring explaining why `trend_following` itself was left untouched. Every future
    tick-based strategy can now reuse it instead of re-deriving the same math a third
    time.
- Decision: 6-state lifecycle machine (IDLE, TRACKING_DROP, WAITING_REVERSAL,
  ENTRY_ARMED, LONG_OPEN, COOLDOWN) as required, with `ENTRY_ARMED` reported only via
  `ExplanationBuilder.state()` on the single tick a BUY fires - it is never itself written
  to persisted state (the next persisted value is `LONG_OPEN`).
  - Why: entry confirmation and the BUY decision happen atomically in one evaluation (the
    tick recovery/confirmation conditions are met IS the tick that buys) - there is no
    additional tick of latency to persist a separate "armed" state across, and inventing
    one would only delay entry without adding safety.
- Decision: `reference_high` is derived from a rolling window of the shared price history
  (`reference_high_lookback_ticks`) while IDLE, then locked into state the moment
  TRACKING_DROP is entered (same "lock at transition" pattern as `entry_atr` elsewhere in
  this file).
  - Why: keeps IDLE state-free (single source of truth = price history) while giving
    TRACKING_DROP/WAITING_REVERSAL a stable reference that does not drift while a setup is
    being tracked.
- Decision: one shared scoring formula, `_dip_recovery_score_from_ratios(decline_ratio,
  recovery_ratio)`, called both by `_compute_opportunity_score`'s `dip_recovery` branch
  (fed ratios approximated from Auto Mode's own 60s bar history) and by the strategy's own
  explanation (fed its exact, precisely-tracked ratios).
  - Why: Auto Mode ranking and the strategy's own "Opportunity score" diagnostic must
    never disagree about what a given setup is worth; defining the curve once prevents
    that class of bug.
- Decision: independent, ATR-locked "emergency stop" measured from entry price (not from
  the trailing high), validated to always be wider than the trailing stop
  (`validate_dip_recovery_params`).
  - Why: in normal operation the trailing stop always triggers first (it's tighter); the
    emergency stop is a defense-in-depth safety net for the abnormal case where
    `entry_atr`/`trailing_stop` are missing or corrupted after a restart (see the
    "Restart during LONG_OPEN" regression test), matching this codebase's existing
    "belt-and-suspenders" risk-control style (e.g. `MAX_CONSECUTIVE_REJECTIONS`).
- Decision: a single-tick move larger than `spike_guard_atr_multiplier x ATR` is excluded
  from updating the tracked reference-high/lowest-price (but still recorded in price
  history, so it still affects volatility estimates).
  - Why: required "handle extreme single candle spikes" invalidation case, implemented
    from data already computed (ATR) rather than a fabricated outlier filter.
- Decision: defensive self-heal - if persisted state says `LONG_OPEN` but
  `_get_bot_positions` returns no position (closed outside the strategy's knowledge, or a
  crash between execution and the next state save), reset cleanly to `IDLE` instead of
  falling into the TRACKING_DROP/WAITING_REVERSAL branch with stale/absent low/high
  fields.
  - Why: found by testing (an unguarded `current_price < lowest_price` crashed with
    `None` when this desync occurred). A real, if narrow, production possibility -
    handling it defensively costs one `if` and prevents a crash loop.

## Risks / Trade-offs
- The tick-based ATR proxy is coarser than a real OHLC ATR (as `trend_following` already
  accepts). Acceptable: it is the established pattern for every tick-driven strategy in
  this codebase, and no candle-fetching infrastructure exists to do better without adding
  a new external dependency.
- Many configurable parameters (18). All have documented sane-crypto defaults and are
  validated (`validate_dip_recovery_params`); none are hardcoded in the strategy body.

## Migration Plan
Purely additive: a new strategy name (`dip_recovery`) and new dict entries in existing
maps. No schema migration needed (`Bot.strategy_state` is already a JSON column; no new
column). Existing bots on other strategies are unaffected - confirmed by regression tests
pinning the previous dispatch table, capability entries, and `_PERSISTED_STATE_ATTRS`
list.

## Open Questions
- None outstanding. Defaults were chosen conservatively (later entry, wider stops) per the
  brief's explicit preference for reduced downside risk over catching the exact bottom;
  they can be tuned via `strategy_params` per-bot without code changes if live/paper
  results suggest otherwise.
