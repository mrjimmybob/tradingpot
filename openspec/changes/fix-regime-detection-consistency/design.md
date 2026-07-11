## Context
Two independently-implemented market-regime detectors exist in
`trading_engine.py`: `_detect_market_regime` (plain price-list input, used by
`_strategy_dca`, `_strategy_mean_reversion`, `_strategy_volatility_breakout`)
and `_detect_market_regime_bar_based` (OHLC bars, used only by
`_strategy_auto`). Only the bar-based one computes true volatility
*direction* (expanding/contracting/stable via rate-of-change, with a 3-bar
persistence requirement before flipping, `:5207-5223` / `:5249-5291`). The
price-only one only classifies a *level* (low/medium/high, an ATR
percentile). `volatility_breakout` needs direction for its
compression-then-breakout thesis but only has access to the price-only
detector's level, so its regime gate silently tests the wrong concept
(verified directly at `trading_engine.py:3467-3491`).

## Goals / Non-Goals
- Goals:
  - `volatility_breakout`'s regime gate reflects actual volatility
    expansion (direction), not merely a high level.
  - Every strategy reasons about "regime" using a validated,
    consistently-defined detector — or, where two detectors remain, their
    purposes are clearly documented and non-overlapping.
- Non-Goals:
  - Replacing the heuristic threshold-based approach with a learned/ML
    regime model — a much larger bet, out of scope here. (These thresholds
    could eventually be empirically validated via the walk-forward/optimizer
    work in `add-strategy-validation-tooling`.)
  - Changing `mean_reversion`'s or `dca`'s regime gates — audited and found
    conceptually consistent with their theses; not touched by this change.

## Decisions
- Decision: `volatility_breakout` already builds and maintains its own bar
  history (`state["bars"]`) for its Bollinger/ATR calculations — reuse that
  directly with the bar-based detector's volatility-direction logic instead
  of calling the separate price-only `_detect_market_regime`. This avoids
  building a second bar series just to get a coarser answer.
  - Alternative considered: backport `volatility_direction` into the
    price-only detector so all three of its consumers get it "for free."
    Rejected as the primary fix — `dca`/`mean_reversion` don't need
    direction today, and it's more surface area to validate at once. Worth
    revisiting later for full consistency, once the walk-forward harness can
    validate any resulting change to those two strategies' behavior.

## Risks / Trade-offs
- Changing the entry gate changes *when* `volatility_breakout` enters —
  this is a behavior change, not a pure bug fix, and must be re-backtested
  rather than assumed to be strictly better. Use
  `add-strategy-validation-tooling` once available; at minimum, run a manual
  before/after backtest across bull, bear, and chop windows.
- **Correction found during implementation**: `regime_allows_entry`/
  `volatility_regime_name` are computed but never actually consulted by the
  entry decision. `is_breakout = compression_satisfied and last_bar_close >
  upper_band` (`trading_engine.py:3752`) is the only behavioral entry gate;
  a prior, already-in-place change deliberately removed the regime veto as a
  hard gate (see the in-code comment at `:3684-3692`: "the compression-then-
  breakout sequence already encodes the volatility thesis... the separate
  volatility veto was redundant"). So this fix changes what gets *computed
  and reported* (the diagnostic/`_explain` "regime" value, and what
  `auto_mode`'s capability table's comments accurately describe) but does
  **not** change any trade decision for the standalone strategy path today
  — the "re-backtest before/after" risk above does not apply; a before/after
  backtest is expected to show identical trades, and that's the correct
  outcome, not a null result. The class docstring's claim ("Pauses during
  unfavorable regimes... to prevent entries in wrong conditions") is
  therefore also stale/inaccurate as of this discovery and is corrected as
  part of this change's docstring task, not left implying a gate that
  doesn't fire.
- Re-enabling `regime_allows_entry` as an actual hard entry gate (which
  would make the docstring's claim true again) is explicitly **not** done by
  this change — that's a real behavior change requiring its own validation
  cycle via `add-strategy-validation-tooling`, and is out of scope for a
  change whose approved goal was detector/gate *correctness*, not
  re-architecting when the strategy trades.

## Migration Plan
1. Add volatility-direction computation reusing `volatility_breakout`'s own
   bar-aggregation state.
2. Switch the regime gate to require direction == "expanding".
3. Re-run backtests across at least 3 historical regimes and compare against
   the pre-change baseline — task-complete is not the same as
   validated-profitable.

## Open Questions
- Should `mean_reversion`/`dca` also eventually move off the price-only
  detector for consistency? Deferred — no evidence today that they need
  volatility direction specifically.
