# Strategy Audit: `volatility_breakout`

`_strategy_volatility_breakout`, `backend/app/services/trading_engine.py:3312-4413`.

**Certification status**: ☐ Draft ☑ Under review ☐ Certified (date: ____ —
pending human sign-off on the before/after backtest comparison and the
Self-Audit findings; both recorded below)
**Certified against**: `add-strategy-decision-framework` 10-pillar
standard, Architecture Freeze revision (Phase 1 migration)

This document supersedes the pre-migration current-state audit. The
original findings (coverage 5/14, Pillar 2 "computed but not enforced",
Pillar 3/7 "not present") are preserved below each pillar as **Pre-Phase-1
finding** for traceability, followed by what Phase 1 changed.

## Pillar 1 — Theory

**Pre-Phase-1 finding**: docstring described mechanics well but never
named the market inefficiency being exploited or stated explicit failure
conditions. UNDOCUMENTED.

**Phase 1 (now documented in the function's docstring, `trading_engine.py:
3319-3327`, and restated here for the audit record)**:

- **Market Inefficiency**: a liquidity-vacuum / stop-run inefficiency.
  Price coiled inside a multi-bar compression (a thin order book and
  suppressed realized volatility) tends to move fast and far once it
  breaks the range.
- **Why The Edge Should Exist**: the same tight range that compresses
  volatility also concentrates resting stop and breakout orders just
  beyond it — retail stop clusters above recent resistance, breakout
  algos waiting for a range break, and market makers who have been
  quoting a tight, low-inventory-risk range. The initial move out of the
  range is often mechanically self-reinforcing (triggered stops become
  new market orders that trigger further stops), not merely a random
  continuation — a structural, not purely statistical, effect.
- **Evidence**: this strategy has NOT yet been through
  `add-strategy-validation-tooling`'s walk-forward validation (that
  tooling does not exist yet — Tier 2, not implemented). This is a
  documented hypothesis awaiting empirical validation, not a proven edge
  — see Pillar 9 and the Backtesting section below for what the initial
  bull/bear/sideways runs actually showed, honestly, without over-claiming.
- **Conditions For Success**: a genuine, multi-bar compression (tight
  Bollinger Band width relative to recent history) followed by a
  decisive close beyond the band, while the bar-based volatility
  DIRECTION detector confirms expansion is actually underway (Pillar 2)
  — not merely that price touched a level.
- **Conditions For Failure**: choppy, directionless markets that produce
  frequent shallow band touches without genuine compression (false
  "breakouts" that immediately mean-revert — the "failed breakout" exit
  exists specifically to bound this failure mode's cost); low-liquidity
  gap markets where a "breakout" is a data artifact, not real order flow
  (this strategy has no volume data and cannot detect this directly —
  documented limitation, see Self-Audit finding 9 below).
- **Assumptions** (the objective, falsifiable conditions this strategy's
  edge depends on, and the source list for Pillar 10's
  `StrategyProposal.assumptions`): (1) the breakout level (BB upper band)
  is not invalidated by a close back inside the range within
  `failed_breakout_bars`; (2) the current volatility regime remains
  "expanding" per the bar-based direction detector; (3) the compression
  episode is not itself falsified by price falling back below the mean
  before a breakout confirms.

## Pillar 2 — Market Suitability

**Pre-Phase-1 finding**: `regime_allows_entry` was computed
(`_detect_market_regime_bar_based`) but never consulted by the entry
condition — the single most acute finding in the original audit.

**Phase 1**: closed. `MarketSuitabilityGate().evaluate(current_regime,
allowed_regimes)` is called at `trading_engine.py:3596`, and its
`is_suitable` result is now an ACTUAL hard gate — a fully armed,
tight, mature compression setup is still refused (NO_TRADE) if
`suitability.is_suitable` is `False` (`trading_engine.py:4100-4111`,
"PRECONDITION 2"). The `regime_filter_enabled` disable switch that
existed pre-migration has been REMOVED — a strategy in this framework
cannot opt out of Pillar 2.

- **Declared `allowed_regimes`**: `["volatility_expanding"]` (default;
  strategy-param overridable). Matches `_get_strategy_capabilities()`'s
  own declaration (`trading_engine.py` ~6187) for Auto Mode dispatch —
  both now route through the SAME `MarketSuitabilityGate` tag convention.
- **Enforcement citation**: `trading_engine.py:4100` (`if not
  suitability.is_suitable:`) — a hard `return`, not a logged diagnostic.
- **Test proving the gate blocks entries**:
  `tests/test_volatility_breakout_framework_migration.py::
  TestMarketSuitability::test_unsuitable_regime_blocks_entry_despite_strong_setup`
  — constructs a fully armed, tight, large-magnitude breakout setup and
  proves it is still refused when the regime reads "contracting".

## Pillar 3 — Evidence-Based Decision Score

**Pre-Phase-1 finding**: entry fired on a single boolean AND of two
conditions — no accumulated multi-factor score. Not present.

**Phase 1**: closed. `DecisionScoreEngine().score(...)` called at
`trading_engine.py:4194`, replacing the boolean `is_breakout` gate.
**Threshold**: `decision_score_threshold` = 40.0 (0-100 scale, strategy-
param overridable) — an initial, certification-pending value (see
Backtesting section for the evidence informing it; NOT recalibrated
between backtest runs per the Backtesting section's own methodology).

| Evidence Item | Measurement | Normalization | Weight | Reason it contributes to edge |
|---|---|---|---|---|
| Breakout magnitude | `(last_bar_close - upper_band) / atr` | `clamp(x / 1.0, -1, 1)` | 30 | A genuine breakout is already a meaningful fraction of the measured range beyond the compression boundary, not a marginal tick above it — larger magnitude is stronger evidence the liquidity-vacuum move has actually started, not noise around the band. |
| Compression maturity | `armed_bars / min_compression_bars` | `clamp((x - 1.0) * 2, -1, 1)` | 25 | The longer price coils before breaking out, the more resting stop/breakout orders concentrate just beyond the range, and the stronger the eventual move tends to be once it releases. |
| Compression tightness | `1 - (armed_min_width / percentile_value)` | `clamp(x / 0.5, -1, 1)` | 25 | A tighter compression (BB width well below the historical compression threshold) implies a smaller, more concentrated order-book vacuum and therefore a more convex expected move once it releases. |
| Volatility expansion strength | `(recent-7-bar-ATR-avg / older-7-bar-ATR-avg) - 1.0` | `clamp(x / 0.3, -1, 1)` | 20 | The compression-then-breakout thesis requires volatility to actually be expanding, not merely that price crossed a static band level — independent confirmation beyond the pass/fail Market Suitability Gate. |

Weights sum to 100. All four items are pure functions of bar-derived
data already collected by this strategy (`trading_engine.py:4137-4192`)
— no volume data is used (this implementation has none available; see
Self-Audit finding 9).

- **Determinism test**:
  `TestDecisionScoreCalculation::test_decision_score_deterministic_across_identical_calls`.
- **Compression episode capture**: `compression_bars`/`compression_min_width`
  reset to 0/None the instant a breakout bar is no longer "compressed" by
  definition — `armed_compression_bars`/`armed_compression_min_width` are
  captured (LOCKED) at the moment of arming (`trading_engine.py:3970-3983`)
  specifically so the Evidence Items above read the real compression
  episode's statistics, not a transient post-reset value.
- **Example Evidence Report** (from
  `TestEvidenceGeneration::test_evidence_report_names_all_four_factors`'s
  fixture — a strong, mature setup): all four items score near their
  maximum, total ≈ 100/100, well above the 40.0 threshold.

## Pillar 4 — Parameter Adaptation

**Pre-Phase-1 finding**: only the stop distance scaled with ATR; every
period/threshold determining "compressed"/"breaking out" was fixed.

**Phase 1**:

| Parameter | Classification | Formula (if ADAPTIVE) / Reason (if FIXED) |
|---|---|---|
| `atr_stop_multiplier` | **ADAPTIVE** | `AdaptiveParameterResolver.atr_percentile_scaled_multiplier(atr_percentile=atr/avg_atr_20, base=2.0, min=1.0, max=4.0)`, `trading_engine.py:3616-3623`. Resolved every cycle, LOCKED at entry. |
| `min_compression_bars` (5) | FIXED | A debounce count against noise, not a quantity that should itself scale with volatility. |
| `take_profit_rr_multiple` (3.0) | FIXED | A convexity target from this strategy's "rare, convex" thesis (a few winners must fund many small, bounded losses) — a design choice, not a volatility-derived quantity. |
| `max_holding_bars` (48) | FIXED | A timeout/debounce count: "has this breakout converted into a fast move yet," not a value that should scale with current volatility. |
| `bb_period` (20), `bb_std` (2.0) | FIXED | Standard Bollinger Band convention; changing these changes what "compression" MEANS, not how the strategy adapts to current conditions. |
| `compression_percentile` (20), `failed_breakout_bars` (3), `cooldown_hours` (24) | FIXED | Debounce/threshold constants, not volatility-derived — no evidence yet that these should scale (a certification-phase, backtest-informed decision if that evidence emerges). |

## Pillar 5 — Decision-Score-Weighted Position Sizing

**Pre-Phase-1 finding**: risk-scaled (partial credit) — `risk_amount =
balance * risk_percent`, `position_coins = risk_amount / (atr *
atr_stop_mult)`. No Decision Score/exposure/drawdown input.

**Phase 1**: `trading_engine.py:4256-4274`. `size_multiplier =
0.5 + clamp((decision_score.total - threshold) / (100 - threshold), 0, 1)`
— a marginal-score trade (barely clears 40.0) sizes toward 0.5x; a
maximal-score trade (100/100) sizes toward 1.5x. Deterministic,
reproducible (same standard Pillar 3 holds `DecisionScoreEngine` to).
`risk_amount = balance * risk_percent * size_multiplier`, then the
existing ATR-based stop-distance formula converts risk to a coin count.
**Not yet wired to the shared `add-unified-position-sizing` cross-strategy
function** — that change is not implemented yet (0/7 tasks); this is this
strategy's own certification-phase implementation of the SAME
requirement (Pillar 5), to be consolidated once that shared change lands
(its `design.md` has already been revised, Phase 0.7, to accept exactly
this `DecisionScoreResult` input).

- **Test**: `TestStrategyProposalGeneration::test_buy_proposal_is_well_formed`
  asserts `decision_score_size_multiplier` is present in
  `adaptive_parameters_used`; the underlying multiplier math itself is
  exercised implicitly by every BUY-path test since sizing is unconditional.

## Pillar 6 — Continuous Trade Management

**Pre-Phase-1 finding**: partial — failed-breakout and a monotonic-
tighten-only trailing stop. No take-profit, no time-based exit.

**Phase 1**:

| Check | Implemented? | Citation |
|---|---|---|
| (a) Thesis/suitability still passing | ☑ (informational) | `TradeManagementMonitor.evaluate(...)`, `trading_engine.py:3730-3746` — feeds `edge_status`/diagnostics; does NOT force-close (see Design Decision below and Pillar 7). |
| (b) Volatility changed materially | ☑ | `stop_tighten_check` lambda, `atr_percentile < 0.7` — informational (the locked entry stop distance itself does not retroactively change, per "risk never expands"; reflected in future ENTRIES' adaptive resolution, not this open position). |
| (c) Stop should tighten | ☑ | Same hook as (b); the monotonic trailing stop (`trading_engine.py:3862-3874`) already tightens on every new high using the LOCKED entry ATR/multiplier. |
| (d) Partial profit should be taken | ☑ (as a full take-profit, not partial) | `take_profit_price` locked at entry = `entry_price + (stop_distance * take_profit_rr_multiple)`, checked every tick (`trading_engine.py:3789-3792`). See Self-Audit finding 5 for why this is a full, not partial, exit. |

**Design decision**: a suitability failure (thesis check (a)) does NOT by
itself force-close an open position — the strategy already has four
independent, price-based exits (failed breakout, take-profit, trailing
stop, time-stop) doing the actual risk control, matching how the
strategy's OTHER exits already work. Forcing an exit purely on a regime
re-read was assessed as an unvalidated behavior change beyond this
migration's scope (see Self-Audit finding 2).

**Pillar 6 gap closure** (the audit's explicit finding): take-profit and
time-stop exits added — `trading_engine.py:3789-3823`.

## Pillar 7 — Strategy Edge Management

**Pre-Phase-1 finding**: no performance fields in the strategy's own
state dict; Not present.

**Phase 1**: closed. `StrategyEdgeManager` instance at
`self._volatility_breakout_edge_manager` (lazily initialized,
`trading_engine.py:3346-3348`), `.evaluate(...)` called every cycle
(`trading_engine.py:3643-3648`).

- **Configuration**: `StrategyEdgeManager()` defaults
  (`min_sample_size=20`, `outcome_window=50`, `expectancy_floor=0.0`,
  `win_rate_floor=0.35`) — **NOT yet recalibrated for this specific
  strategy**; these are the shared module's generic placeholders (see
  `edge_management.py`'s own docstring). Recalibrating them against this
  strategy's actual trade-frequency/expectancy profile is flagged as
  outstanding certification work, to be informed by real trade history or
  `add-strategy-validation-tooling` once available.
- **Category A condition**: `regime_outside_suitable_range = not
  suitability.is_suitable` (the SAME Pillar 2 gate result).
- **Category B condition**: `parameter_mismatch_evidence` is cited only
  when `atr_percentile` (current ATR vs. its 20-bar average) drifts
  outside `[0.6, 1.5]` — evidence the `atr_stop_multiplier` BASE (not the
  live-adaptive value) may need certification-phase recalibration
  (`trading_engine.py:3634-3641`).
- **Category C threshold**: whatever remains after A and B are ruled out
  — see `edge_management.py`'s own structural ordering (this module is
  frozen; not modified by this strategy).
- **New-entry gating**: a Category C classification blocks new entries
  (`blocking_reasons` check, `trading_engine.py:4212-4216`) but NEVER
  force-closes an existing position — verified by
  `TestStrategyEdgeManagement::test_edge_management_never_force_closes_open_position`.
- **Tests**: `TestStrategyEdgeManagement::test_category_c_blocks_new_entry_despite_qualifying_score`,
  `::test_healthy_history_does_not_block_entry`,
  `::test_edge_management_never_force_closes_open_position`.

## Pillar 8 — Self-Diagnostics

**Pre-Phase-1 finding**: warm-up/exit/entry gates explained; sizing and
fee-viability gap never surfaced via `ExplanationBuilder`.

**Phase 1**: closed — `.check()` calls added for Decision Score threshold
clearance (`trading_engine.py:4198-4202`), fee viability
(`trading_engine.py:4304-4307`), Market Suitability, and Strategy Edge
disqualification (`trading_engine.py:4048-4055`). The well-known
`edge_status_category` metric key
(`explanation_persistence.py`'s documented convention) is populated on
every cycle (`trading_engine.py:3760`, `3878`, `4045`), so Phase 0.6's
`Order.edge_management_category` persistence now populates correctly for
this strategy's executed trades — the first strategy for which that
column is non-null.

- **`Order.decision_explanation` populated**: confirmed via
  `TestEvidenceGeneration::test_explanation_contains_all_evidence_metrics`
  (checks the explanation dict directly) — full end-to-end persistence
  verification (`_execute_trade` writing the Order row) is Phase 0.6's
  own test suite (`test_order_decision_explanation.py`), reused unchanged
  here since this strategy calls no code outside the documented flow.

## Pillar 9 — Performance Expectations

**Pre-Phase-1 finding**: frequency only ("enters rarely, few trades/month").
UNDOCUMENTED beyond that.

**Phase 1** (documented expectations — see Backtesting section below for
what the initial validation runs actually showed against these):

- **Expected trade frequency**: rare — a few times a month in a real
  market with genuine multi-week compression cycles; the Decision Score
  threshold (40.0) is deliberately non-trivial, expected to further
  reduce frequency versus the pre-migration boolean gate.
- **Expected holding time**: short — capped at `max_holding_bars` (48
  bars; 48 minutes at the default 60s bar interval) unless the take-
  profit or trailing stop resolves it sooner.
- **Expected win rate**: LOW-to-MODERATE (a rare/convex strategy — most
  attempts are expected to be small, bounded losses; profitability
  depends on a minority of winners reaching the 3:1 reward:risk target,
  not on a high hit rate). No numeric target set — no walk-forward
  evidence exists yet to ground one honestly.
- **Expected profit factor / drawdown**: not yet quantified — flagged
  explicitly as UNDOCUMENTED pending `add-strategy-validation-tooling`,
  rather than inventing a plausible-sounding number.
- **Typical vs. worst-case conditions**: typical = a clear regime
  transition (compression breaking into a trending or high-volatility
  phase); worst-case = a choppy, directionless market producing repeated
  shallow "breakouts" that immediately fail (bounded by
  `failed_breakout_bars`, but still a fee/slippage drag per attempt).
- **Buy-and-hold comparison**: not evaluated — this strategy is long-only
  and rare-entry by design, not intended as a buy-and-hold substitute;
  comparing it to buy-and-hold in a single bull-market window would
  repeat the exact methodology error `dca_accumulator`'s "winning" result
  already demonstrated is misleading (see the roadmap's Key Finding #1).

## Pillar 10 — Strategy Proposal Interface

**Pre-Phase-1 finding**: returns `TradeSignal` directly. Not present.

**Phase 1**: closed — this strategy now internally constructs a
`StrategyProposal` for EVERY branch (entry, exit, hold-with-position,
every no-trade reason) and translates it via the shared
`StandaloneAdapter.to_trade_signal(...)` before returning — verified by
`_capture_proposals` spying in every test in
`test_volatility_breakout_framework_migration.py`.

- **`execution_intent` mapping**: entry (armed + suitable + score cleared
  + edge OK + not in cooldown) -> `BUY` + `OPEN_POSITION`. Every exit
  (failed breakout, take-profit, trailing stop, time-stop) -> `SELL` +
  `CLOSE_POSITION`. No exit condition in this strategy ever partially
  reduces a position, so `REDUCE_POSITION`/`ADD_TO_POSITION` are unused
  by this strategy (valid — the pairing table permits, does not require,
  every intent). Holding an open position with no exit trigger -> `HOLD` +
  `HOLD_POSITION`. Not-yet-armed or unsuitable-regime or score-below-
  threshold or Category-C or cooldown-active -> `NO_TRADE` +
  `NO_ACTION`.
- **`validity.valid_until` interval**: `generated_at +
  max(bar_interval_seconds, 1)` seconds (`trading_engine.py:3428-3431`) —
  tied to this strategy's own bar cadence; the `max(..., 1)` floor exists
  only so a test harness's `bar_interval_seconds=0` convention ("close a
  bar every call") never produces a zero-duration validity window, which
  `ProposalValidity` correctly rejects.
- **`assumptions` list**: see Pillar 1 above — the three objective,
  falsifiable conditions are attached to every entry/blocked-entry
  proposal (`trading_engine.py:4204-4210`).
- **`expected_edge_estimate`**: always `None` — no
  `add-strategy-validation-tooling` result exists yet for this strategy;
  never self-computed (mechanically impossible per `EdgeEstimate`'s own
  sourcing enforcement in the frozen `proposal.py`).
- **Behavior-preservation test**:
  `TestStandaloneAdapterCompatibility::test_buy_signal_fields_match_adapter_translation`
  and `::test_expired_proposal_would_be_discarded_by_adapter`; the fixed-
  scenario reward:risk behavior-preservation test from Phase 0.11 itself
  (`test_strategy_framework_standalone_adapter.py`) is reused unmodified,
  confirming the adapter's own contract is untouched.

## Backtesting

**Methodology**: BTCUSDT, 1-minute bars, `$10,000` starting balance, default
strategy parameters (none tuned between or within runs — this is
validation, not optimization, per this change's own Backtesting
instructions), 0.1%/side fee, no spread/slippage modeled. Three ~6-week
windows, chosen for clear regime character within the year-labels this
project already uses for bull/bear/chop (`fix-regime-detection-
consistency`'s tasks.md: bull=2023, bear=2022, chop=2024 H1), narrowed
from a full year to a representative sub-window so each run completes in
reasonable time:

| Window | Dates | Character |
|---|---|---|
| Bull | 2023-10-01 – 2023-11-15 | ETF-speculation rally |
| Bear | 2022-11-01 – 2022-12-15 | FTX-collapse aftermath |
| Chop | 2024-04-01 – 2024-05-15 | Post-halving consolidation |

**A data-quality caveat discovered while running these**: the backtest
CLI's `Trades` metric undercounts real closed round-trips. Tracing every
order fill directly (bypassing the summary metric) for the bull window
showed 5 complete buy→sell round trips in both the pre- and post-migration
code, while the CLI reported 0 and 2 respectively. Root cause: `portfolio.
apply_sell`'s full-close detection (`base_amount <= 1e-12`,
`backend/app/backtesting/portfolio.py`) and the engine's separate DB-
position tracker (`_close_or_reduce_position`) are driven off the same
converted `amount_base`, but this strategy's exit signals size the sell as
`pos.amount * current_price` (a dollar amount reconverted at the *next*
bar's fill price) rather than the engine's `amount=0`/`None` "close
everything" sentinel — a decision-to-fill price gap of even a fraction of
a cent leaves float dust in the portfolio ledger that the DB-position side
doesn't see, permanently breaking closed-trade accounting for that ledger.
This was a **pre-existing bug in the shared backtest engine**, reproduced
identically on the pre-migration code (this phase did not introduce it,
and did not fix it — out of scope per this phase's Architecture Freeze;
flagged here as a candidate follow-up for whoever owns `app/backtesting/`).

**RESOLVED (pre-Phase-2 infrastructure review)**: fixed in
`backend/app/backtesting/engine.py::_apply_signal` — the sell path now
resolves the exit's base quantity against the DECISION price (the price
the strategy saw and multiplied by), mirroring production's
`_execute_trade` STEP 2.5, instead of dividing the exit notional by the
fill price. Full-position exits now settle to zero and record their closed
round-trip exactly. Regression test:
`tests/test_backtesting_engine.py::TestFullCloseAccounting::
test_full_close_records_trade_when_fill_price_rises`. The round-trip counts
in the table below were captured under the buggy accounting and are left
as the honest historical record of this run; they are not re-derived here.
Ending balance, return %, and total fees ARE reliable regardless (cash/
equity update on every fill, independent of the closed-trade list) — the
table below uses those, plus the bull window's traced round-trip count
where available.

| Window | Return % | Buy-and-hold % | Round trips | Fees | Notes |
|---|---|---|---|---|---|
| Bull (post-migration) | -1.63% | +31.84% | 5 (traced) | $99.15 | Identical entries/exits/prices to pre-migration (see below) |
| Bull (pre-migration) | -1.64% | +31.84% | 5 (traced) | $99.14 | Same 5 setups, same outcomes — the migration was behaviorally a no-op in this specific sample |
| Bear (post-migration only) | -0.29% | -13.07% | 2 (CLI-reported; likely undercounted per above) | $140.41 | |
| Chop (post-migration only) | -2.18% | -13.59% | 4 (CLI-reported; likely undercounted per above) | $118.71 | |

Bear and chop were run post-migration only (a scoping decision made mid-
session to bound backtest runtime/resource use on this machine — each
1-minute-resolution run takes 10-20 minutes single-threaded); they
validate that the migrated strategy behaves sensibly in those regimes, not
a pre/post regression check. The bull window's pre/post comparison is the
one that actually exercises "did the migration change behavior," and it
did not, in this sample.

**Why Pillar 2 enforcement made no observable difference here**: all 5
bull-window entries occurred while `regime_allows_entry` already read
`True` under BOTH the old (diagnostic-only) and new (hard-gate) code —
none of this sample's setups happened to arm while the regime read
"contracting." This backtest sample does not exercise the one behavior
change Task 1.2 was specifically about; that Pillar 2 enforcement is
instead verified directly by `TestMarketSuitability::
test_unsuitable_regime_blocks_entry_despite_strong_setup` (a synthetic
setup constructed specifically to arm during an unsuitable regime).

**Interpretation, honestly, per the theory (not just the code)**: every
trade attempt across all three windows was a loss (0% win rate, ~11
combined attempts). The strategy's own theory (Pillar 1) predicts exactly
this shape most of the time — a rare, convex setup where most attempts are
small, bounded losses funding an occasional large winner at the 3:1
reward:risk target — but zero winners were observed in this sample. With
only ~11 attempts across 4.5 months of (non-contiguous) data, this sample
is too small to distinguish three different explanations: (a) the theory
is sound and a winning breakout simply didn't occur in these particular
windows (rare-event strategies need many more samples before an absence of
winners is meaningful), (b) `decision_score_threshold` (40.0) or
`take_profit_rr_multiple` (3.0) — both explicitly flagged as first-pass,
uncalibrated values in Pillar 4/9 — are miscalibrated, or (c) the theory
itself is weaker than hypothesized. Per this change's own Backtesting
instructions, this is not being resolved by tuning parameters against
this sample (that would be curve-fitting to 11 trades); it requires
`add-strategy-validation-tooling`'s walk-forward validation across many
more regime windows, which does not exist yet (Pillar 1/9 both already
flag this honestly rather than inventing a number).

**Buy-and-hold comparison, taken at face value**: the strategy preserved
capital decisively better than buy-and-hold in both adverse regimes (bear
-0.29% vs -13.07%; chop -2.18% vs -13.59%) simply by being mostly in cash
and losing small amounts on rare failed attempts, and lagged buy-and-hold
substantially in the bull window (-1.63% vs +31.84%) exactly as Pillar 9
already documents this strategy is not intended to compete with a strong
sustained uptrend. Directionally consistent with the theory's stated role
(a rare, convex, capital-preserving component — not a buy-and-hold
substitute), though the small sample means this is an observation, not
statistical validation.

## Certification Checklist

- [x] Deterministic — `TestDecisionScoreCalculation::
      test_decision_score_deterministic_across_identical_calls`.
- [x] Immutable — enforced at runtime by the frozen `StrategyProposal`
      dataclass (Phase 0.10, not modified by this phase).
- [x] Assumptions documented — objective, falsifiable, traced to Pillar 1
      (see table above).
- [x] Expiration defined — `validity.valid_until` set and tested
      (`TestStandaloneAdapterCompatibility::test_expired_proposal_would_be_discarded_by_adapter`).
- [x] Execution intent consistent with direction — enforced structurally
      by `StrategyProposal.__post_init__`'s pairing table; this strategy
      never constructs an off-table pairing (would raise at construction,
      and every test exercising a BUY/SELL/HOLD/NO_TRADE path passed).
- [x] Evidence measurable — Pillar 3 table complete, all four items pure
      functions of bar-derived data.
- [x] Explanation reproducible — Evidence Report renders identically for
      the same inputs (Pillar 3 determinism test); persisted via Pillar 8.
- [x] No subjective information — no factor in the Pillar 3 table could
      not be expressed as a deterministic Measurement + Normalization pair.
- [x] Expected edge sourced correctly — always `None`, mechanically
      enforced (see Pillar 10 above).
- [x] Before/after backtest comparison recorded — see Backtesting section
      above.
- [x] All existing tests pass (1152/1152 full suite); new tests cover
      every pillar above (21 new tests,
      `test_volatility_breakout_framework_migration.py`).

## Self-Audit (critical review against trading theory, not code)

See the top-level task's Self-Audit output for the full 9-question
critique. Headline findings carried into this document:

1. Behaves like a professional discretionary breakout trader in
   structure (waits for a real setup, sizes by conviction, has a hard
   stop and a defined target) but is more mechanical/less discretionary
   than a human, by design — that is the point of the framework, not a
   flaw.
2. Should refuse to trade: choppy/no-compression markets (implemented,
   Precondition 1), contracting/non-expanding volatility (implemented,
   Pillar 2), a demonstrated losing streak with no explanation
   (implemented, Pillar 7). NOT implemented: liquidity/spread-based
   refusal (no order-book or volume data available — an honest,
   documented gap, not silently ignored).
3. Every executed BUY is justified by the rendered Evidence Report
   (4 named, weighted, deterministic factors).
4. Every rejected trade is justified — one of: not armed, unsuitable
   regime, score below threshold, Category C, or cooldown, each with a
   specific citation, never a bare "no."
5. Arbitrary thresholds still present: `decision_score_threshold` (40.0),
   `take_profit_rr_multiple` (3.0), `max_holding_bars` (48),
   `min_atr_stop_multiplier`/`max_atr_stop_multiplier` (1.0/4.0) are
   first-pass, not yet empirically calibrated — explicitly flagged, not
   hidden. `atr_stop_multiplier` itself is no longer arbitrary (adaptive).
6. Minimizes unnecessary trades more than the pre-migration version: a
   real multi-factor score, a non-trivial threshold, AND an edge-health
   gate all had to independently agree, versus a single boolean AND.
7. Protects against prolonged losing periods via Pillar 7 (Category C
   blocks new entries) — but does not (and per design SHALL NOT)
   force-close a position already open when a losing period is detected.
8. Every proposal satisfies the `StrategyProposal` specification —
   verified by the full certification checklist above and 21 dedicated
   tests.
9. Remaining weaknesses for production review: no volume/order-book
   confirmation (theory names a liquidity-vacuum mechanism this
   implementation cannot directly observe); `StrategyEdgeManager`
   thresholds are generic defaults, not calibrated to this strategy's
   real trade frequency; Decision Score weights/threshold are principled
   but uncalibrated; sizing is a local, strategy-specific implementation
   of Pillar 5 pending `add-unified-position-sizing`'s shared consolidation.

**Findings from actually running the backtests** (not assumable from code
review alone — see the Backtesting section above for full detail):

10. Every trade attempt observed across all three regime windows (~11
    total) was a loss — 0% empirical win rate in this sample. This is
    NOT proof the theory or implementation is wrong: the theory itself
    predicts a low hit rate funded by rare large winners, and ~11
    attempts is far too small a sample for a rare-event strategy to
    surface one. But it is also not evidence the edge exists. Honestly
    unresolved pending `add-strategy-validation-tooling`'s walk-forward
    validation across many more windows — deliberately NOT "fixed" by
    retuning `decision_score_threshold`/`take_profit_rr_multiple` against
    this sample, which would be curve-fitting to 11 trades, exactly what
    this change's Backtesting instructions warn against.
11. The bull-window before/after comparison exercised the migration's
    *interface* change (`TradeSignal` -> `StrategyProposal`) but not its
    headline behavior change (Pillar 2 becoming a hard gate) — every
    setup in that sample happened to arm during an already-suitable
    regime, so the enforcement point never fired. The regression suite's
    `test_unsuitable_regime_blocks_entry_despite_strong_setup` is what
    actually proves the gate blocks, not this backtest window; a reviewer
    should not read "no observable behavior change in backtesting" as
    "Pillar 2 enforcement doesn't matter."
12. Running real backtests surfaced a bug this migration did NOT
    introduce and did NOT fix: the shared backtest engine's `Trades`
    metric undercounts real closed round-trips for this strategy (traced
    fills showed 5 real round trips in a window the CLI reported as 0
    pre-migration / 2 post-migration — see Backtesting section above for
    the mechanism). Filed here rather than silently worked around,
    because a strategy this framework certifies should have its
    performance measurable by the tooling that reports on it — this is a
    latent gap in that tooling, out of this phase's scope
    (`backend/app/backtesting/`, not `strategy_framework/`) but worth a
    follow-up change. **RESOLVED in the pre-Phase-2 infrastructure review**
    — see the Backtesting section's "RESOLVED" note above.
