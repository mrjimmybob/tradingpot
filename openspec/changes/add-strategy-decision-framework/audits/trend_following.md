# Strategy Audit: `trend_following`

`_strategy_trend_following`, `backend/app/services/trading_engine.py`.

**Certification status**: ☐ Draft ☑ Under review ☐ Certified (date: ____ —
pending human sign-off on the before/after backtest comparison and the
Self-Audit findings; both recorded below)
**Certified against**: `add-strategy-decision-framework` 10-pillar
standard, Architecture Freeze revision (Phase 2 migration)

This document supersedes the pre-migration current-state audit (coverage
5/14; **zero enforced market suitability** — the highest-urgency finding of
the six). The original findings are preserved below each pillar as
**Pre-Phase-2 finding** for traceability, followed by what Phase 2 changed.

## Pillar 1 — Theory

**Pre-Phase-2 finding**: docstring stated the mechanism (EMA crossover, ATR
stops, noise-resistant confirmation) but named no inefficiency rationale, no
explicit failure conditions, no assumptions. UNDOCUMENTED.

**Phase 2 (now documented in the function's docstring and restated here):**

- **Market Inefficiency**: the documented **time-series-momentum anomaly**.
  Established price trends persist longer than an efficient market would
  allow.
- **Why The Edge Should Exist**: information diffuses gradually rather than
  instantly; trend-following capital flows (CTAs, momentum funds) mechanically
  reinforce an existing move; and behavioral effects (herding, the disposition
  effect — winners sold too early, losers held too long) delay full repricing.
  The combination makes an already-established trend more likely than chance to
  continue for some time.
- **Evidence**: NOT yet through `add-strategy-validation-tooling`'s
  walk-forward validation (that tooling does not exist yet — Tier 2). A
  documented hypothesis awaiting empirical validation, not a proven edge — see
  Pillar 9 and the Backtesting section for what the initial bull/bear/chop runs
  actually showed, honestly.
- **Conditions For Success**: a genuine, established up-trend — fast EMA
  above slow EMA AND price above the slow EMA — confirmed by the bar-based
  regime detector reading `trend_up` (or early `volatility_expanding`), with a
  trend separation meaningful relative to current volatility (not just noise
  around flat EMAs).
- **Conditions For Failure**: choppy, directionless (regime `trend_flat`)
  markets that produce EMA crossovers with no follow-through (whipsaws — the
  confirmed trend-break exit exists to bound this cost); sharp reversals after
  entry (bounded by the trailing stop locked at entry ATR); a decayed edge with
  no regime or parameter explanation (Category C — Pillar 7 blocks new entries).
- **Assumptions** (objective, falsifiable — the source for Pillar 10's
  `StrategyProposal.assumptions`): (1) trend direction unchanged — the fast EMA
  remains above the slow EMA (EMA ordering not reversed); (2) price remains
  above the slow EMA (participation in the trend holds); (3) the current market
  regime remains within the strategy's allowed set.

## Pillar 2 — Market Suitability

**Pre-Phase-2 finding**: NO regime detection call anywhere in the function.
Regime awareness existed ONLY externally in `_get_strategy_capabilities()`
for Auto dispatch — a **standalone** trend_following bot never checked market
regime before entering, at all. The single most acute finding in the whole
audit.

**Phase 2**: closed, and built from scratch (unlike volatility_breakout,
which already *computed* a suitability value — this strategy had nothing).
`MarketSuitabilityGate().evaluate(current_regime, allowed_regimes)` is called
every cycle and its `is_suitable` result is an ACTUAL hard gate — a strong,
high-score up-trend setup is still refused (NO_TRADE) when the regime is
unsuitable.

- **Declared `allowed_regimes`**: `["trend_up", "volatility_expanding"]`
  (default; strategy-param overridable) — matches
  `_get_strategy_capabilities()["trend_following"]` exactly, so the standalone
  gate and Auto-mode eligibility route through the SAME regime-tag convention
  and can never disagree.
- **Regime source**: `_detect_market_regime_bar_based(state["tf_bars"], …)` —
  the same canonical bar-based detector `fix-regime-detection-consistency` made
  authoritative. `tf_bars` (the strategy's existing 60s H-L-C aggregation) are
  reused directly.
- **Enforcement citation**: the entry path collects `blocking_reasons`; `not
  suitability.is_suitable` is one, and any blocking reason yields a
  `Direction.NO_TRADE` / `ExecutionIntent.NO_ACTION` proposal (a hard `return`
  via the Standalone Adapter, not a logged diagnostic).
- **Test proving the gate blocks entries**:
  `tests/test_trend_following_framework_migration.py::TestMarketSuitability::
  test_unsuitable_regime_blocks_entry_despite_strong_setup` — a strong up-trend
  setup (high Decision Score) with `allowed_regimes=["trend_down"]` is refused.
- **Bar/regime history is now PRESERVED across exits** (previously the exit
  reset replaced the whole state dict, dropping `tf_bars`), so the regime gate
  does not have to re-warm from zero after every trade.

## Pillar 3 — Evidence-Based Decision Score

**Pre-Phase-2 finding**: entry fired on a single boolean AND
(`current_price > ema_long AND ema_short > ema_long`), gated only by a fixed
confirmation-loop counter (a noise filter, not a score). Not present.

**Phase 2**: closed. `DecisionScoreEngine().score(...)` replaces the boolean
gate. The pre-migration hard N-loop confirmation gate is folded into the score
as the "Confirmation persistence" Evidence Item (noise defense expressed as
measurable evidence rather than a boolean count). **Threshold**:
`decision_score_threshold` = 40.0 (0–100 scale, strategy-param overridable) —
an initial, certification-pending value.

| Evidence Item | Measurement | Normalization | Weight | Reason it contributes to edge |
|---|---|---|---|---|
| Trend strength | `(ema_short - ema_long) / ema_long` | `clamp(x / 0.02, -1, 1)` | 35 | The core momentum signal: a larger fast-over-slow EMA separation is a more established trend, which the momentum anomaly says is more likely to persist. A negative separation is counter-trend and correctly scores against entry. |
| Price participation | `(current_price - ema_long) / ema_long` | `clamp(x / 0.02, -1, 1)` | 25 | Price itself must participate, not just the smoothed EMAs. Price above the slow EMA confirms the up-trend is current; price below it (even with EMAs still ordered up) is early evidence the move is fading. |
| Confirmation persistence | `entry_confirmation_count / entry_confirmation_loops` | `clamp(x, -1, 1)` | 20 | A momentum signal persisting across consecutive bars is less likely noise than a single-bar flicker — the noise defense the pre-migration confirmation-loop gate provided, now measurable evidence. |
| Volatility-normalized trend | `(ema_short - ema_long) / atr` | `clamp(x / 1.0, -1, 1)` | 20 | The EMA separation must be meaningful relative to current noise, not just to price level. Measuring in ATR units confirms the trend stands out above bar-to-bar volatility, independent of the percentage measure above. |

Weights sum to 100. All four items are pure functions of EMA/ATR values the
strategy already computes — no new indicator was introduced.

- **Determinism test**: `TestDecisionScoreCalculation::
  test_decision_score_deterministic_across_identical_calls`.
- **Single-factor insufficiency test**: `TestDecisionScoreCalculation::
  test_single_factor_alone_is_insufficient` — a near-flat market where only
  Confirmation persistence is positive does NOT clear the threshold.

## Pillar 4 — Parameter Adaptation

**Pre-Phase-2 finding**: only the trailing-stop distance and a fee-floor used
live ATR; every entry threshold and lookback period was fixed. The
`long_period` docstring/code mismatch (docstring said 200, code used 100) was
flagged.

**Phase 2**:

| Parameter | Classification | Formula (if ADAPTIVE) / Reason (if FIXED) |
|---|---|---|
| `atr_multiplier` (base 2.0) | **ADAPTIVE** | `AdaptiveParameterResolver.atr_percentile_scaled_multiplier(atr_percentile = bar_atr / avg(tf_atr_history[-20:]), base=2.0, min=1.0, max=4.0)`. Resolved every cycle, LOCKED at entry (risk never expands mid-trade). |
| `min_atr_multiplier` / `max_atr_multiplier` (1.0 / 4.0) | FIXED | Bounds for the adaptive resolution above. |
| `short_period` (50), `long_period` (100) | FIXED | EMA lookbacks define what "the trend" MEANS; changing them changes the strategy's identity, not how it adapts to current conditions. **The `long_period` docstring/code mismatch is fixed — docstring and default now both read 100.** |
| `entry_confirmation_loops` (3), `exit_confirmation_loops` (2) | FIXED | Debounce counts against noise/whipsaw, not volatility-derived quantities. |
| `cooldown_seconds` (300) | FIXED | An anti-churn debounce. |
| `atr_period` (14), `bar_interval_seconds` (60) | FIXED | Standard measurement windows; changing them changes what ATR/regime MEAN, not how the strategy adapts. |

## Pillar 5 — Decision-Score-Weighted Position Sizing

**Pre-Phase-2 finding**: risk-scaled (partial credit) — `risk_amount =
balance * risk_percent`, `position_coins = risk_amount / (atr * atr_mult)`. No
Decision Score / exposure / drawdown input.

**Phase 2**: `size_multiplier = 0.5 + clamp((decision_score.total -
threshold) / (100 - threshold), 0, 1)` — a marginal-score trade (barely clears
40) sizes toward 0.5x; a maximal-score trade (100/100) toward 1.5x.
Deterministic and reproducible. `risk_amount = balance * risk_percent *
size_multiplier`, then the existing ATR-based stop-distance formula (using the
LOCKED adaptive multiplier) converts risk to a coin count.
**Not yet wired to the shared `add-unified-position-sizing` function** — that
change is not implemented yet (0/7 tasks); this is this strategy's own
certification-phase implementation of the SAME requirement (Pillar 5), to be
consolidated once that shared change lands (its `design.md` already accepts a
`DecisionScoreResult` input, Phase 0.7).

- **Test**: `TestStrategyProposalGeneration::test_buy_proposal_is_well_formed`
  asserts `decision_score_size_multiplier` is present in
  `adaptive_parameters_used`.

## Pillar 6 — Continuous Trade Management

**Pre-Phase-2 finding**: FULL — two conditions re-evaluated every tick:
trailing stop on new highs (locked entry ATR, immediate) and a
confirmed trend-break exit (`exit_confirmation_loops` consecutive ticks below
the slow EMA). No partial profit-taking. This was already the second-strongest
Pillar 6 of the six, so Phase 2 preserves both exits rather than adding a
take-profit (unlike Phase 1's volatility_breakout, whose Pillar 6 had a real
gap).

**Phase 2**: both exits preserved, now expressed as `StrategyProposal`s
(`SELL` / `CLOSE_POSITION`) routed through the Standalone Adapter. The trailing
stop distance now uses the LOCKED adaptive multiplier (`entry_stop_multiplier`)
rather than the fixed `atr_multiplier` param. A suitability failure does NOT
force-close an open position — the two price-based exits do the risk control,
matching the design's "never force-close" rule (Pillar 7) and how the strategy
already worked.

| Check | Implemented? | Citation |
|---|---|---|
| Trailing stop tightens on new highs | ☑ | `has_position` branch; monotonic, uses LOCKED entry ATR x LOCKED adaptive multiplier. |
| Confirmed trend-break exit | ☑ | price < slow EMA, confirmed over `exit_confirmation_loops`. |
| Thesis/suitability re-read every cycle | ☑ (informational) | regime + edge evaluated every cycle; feeds diagnostics, never force-closes. |

## Pillar 7 — Strategy Edge Management

**Pre-Phase-2 finding**: no streak/expectancy/self-pause logic anywhere. Not
present.

**Phase 2**: closed. `StrategyEdgeManager` instance at
`self._trend_following_edge_manager`, `.evaluate(...)` called every cycle;
`.record_decision_score(...)` on every entry evaluation; `.record_trade_
outcome(...)` on every exit (only when a local entry price is known — never a
fabricated P&L for an imported/legacy position).

- **Configuration**: `StrategyEdgeManager()` defaults (`min_sample_size=20`,
  `outcome_window=50`, `expectancy_floor=0.0`, `win_rate_floor=0.35`) — **NOT
  yet recalibrated for this specific strategy**; the shared module's generic
  placeholders. Recalibration is flagged outstanding certification work,
  informed by real trade history or `add-strategy-validation-tooling`.
- **Category A condition**: `regime_outside_suitable_range = not
  suitability.is_suitable` (the SAME Pillar 2 gate result).
- **Category B condition**: `parameter_mismatch_evidence` cited only when
  `bar_atr` percentile (vs its 20-bar average) drifts outside `[0.6, 1.5]` —
  evidence the `atr_multiplier` BASE may need certification-phase recalibration.
- **Category C**: whatever remains after A and B are ruled out (module frozen).
- **New-entry gating**: a Category C classification is a `blocking_reason`
  (NO_TRADE) but NEVER force-closes an open position — verified by
  `TestStrategyEdgeManagement::test_edge_management_never_force_closes_open_position`.
- **Tests**: `::test_category_c_blocks_new_entry_despite_qualifying_score`,
  `::test_healthy_history_does_not_block_entry`,
  `::test_edge_management_never_force_closes_open_position`.

## Pillar 8 — Self-Diagnostics

**Pre-Phase-2 finding**: PARTIAL — cooldown/price-vs-EMA/crossover/confirmation
checks present; NO `.check()` for the risk-based sizing calculation or the
min-order floor; initial stop placement reported only as a metric, never
asserted pass/fail.

**Phase 2**: closed — `.check()` calls added for Decision Score threshold
clearance, Market Suitability, Strategy Edge disqualification, Cooldown,
**Position size >= min order** (the sizing gap), **Fee viability**, and
**Initial stop below entry** (the stop-placement gap). The well-known
`edge_status_category` metric key is populated every cycle, so Phase 0.6's
`Order.edge_management_category` persistence populates for this strategy's
executed trades.

- **`Order.decision_explanation` persistence**: verified end-to-end by Phase
  0.6's own suite (`test_order_decision_explanation.py`), reused unchanged here
  since this strategy calls no code outside the documented flow.

## Pillar 9 — Performance Expectations

**Pre-Phase-2 finding**: UNDOCUMENTED beyond a bare "long" holding-time label
in the external capability table.

**Phase 2** (documented expectations — see Backtesting for what the initial
runs showed against these):

- **Expected trade frequency**: low-to-moderate — a handful of trades over a
  multi-week trending window; the Decision Score threshold (40.0) plus the hard
  regime gate is expected to reduce frequency versus the pre-migration boolean
  gate (which entered on any EMA cross regardless of regime).
- **Expected holding time**: long — rides a trend until the trailing stop or a
  confirmed trend break resolves it.
- **Expected win rate**: MODERATE, with a right-skewed P&L profile typical of
  trend following (a minority of large trend-capture winners funding many
  small, bounded whipsaw losses). No numeric target set — no walk-forward
  evidence exists yet to ground one honestly.
- **Expected profit factor / drawdown**: not yet quantified — UNDOCUMENTED
  pending `add-strategy-validation-tooling`, rather than inventing a number.
- **Typical vs. worst-case conditions**: typical = a sustained directional
  trend (the setup it is built for); worst-case = a choppy `trend_flat` market
  producing repeated crossovers with no follow-through (now largely refused by
  the Pillar 2 regime gate, which the pre-migration version lacked entirely).
- **Buy-and-hold comparison**: not evaluated as a target — trend following is a
  timing/risk-control strategy, not a buy-and-hold substitute; the Backtesting
  section reports the observed B&H figure per window for context only.

## Pillar 10 — Strategy Proposal Interface

**Pre-Phase-2 finding**: returned `TradeSignal` directly at every return
point. `TradeSignal.score`/`.threshold` never populated. Not present.

**Phase 2**: closed — the strategy now constructs a `StrategyProposal` for
EVERY branch (entry, both exits, hold-with-position, every no-trade reason) and
translates it via `StandaloneAdapter.to_trade_signal(...)` before returning —
verified by `_capture_proposals` spying in the migration test suite.

- **`execution_intent` mapping**: confirmed up-trend entry -> `BUY` +
  `OPEN_POSITION`. Every exit (trailing stop, confirmed trend break) -> `SELL`
  + `CLOSE_POSITION`. Holding an open position with no exit trigger (and the
  trend-break-confirmation-building tick) -> `HOLD` + `HOLD_POSITION`.
  Not-yet-trending / unsuitable-regime / score-below-threshold / Category-C /
  cooldown-active -> `NO_TRADE` + `NO_ACTION`. `ADD_TO_POSITION` /
  `REDUCE_POSITION` are unused by this single-position strategy (valid — the
  pairing table permits, does not require, every intent).
- **`validity.valid_until`**: `generated_at + max(bar_interval_seconds, 1)`
  seconds — tied to this strategy's bar cadence; the `max(…, 1)` floor only so a
  test harness's `bar_interval_seconds=0` never produces a zero-duration window
  (which `ProposalValidity` rejects).
- **`assumptions`**: the three objective, falsifiable conditions from Pillar 1,
  attached to every entry / blocked-entry / hold proposal.
- **`expected_edge_estimate`**: always `None` — no
  `add-strategy-validation-tooling` result exists yet; never self-computed
  (mechanically enforced by `EdgeEstimate`'s sourcing rule in frozen
  `proposal.py`).
- **Behavior-preservation**: `TestStandaloneAdapterCompatibility::
  test_buy_signal_matches_adapter_translation` (the returned `TradeSignal`'s
  action/amount/score/threshold come from the proposal via the adapter) and
  `::test_expired_proposal_would_be_discarded_by_adapter`. The state-persistence
  regression suite (`test_production_hardening.py`) confirms the restored-
  trailing-stop exit and DB state roundtrip are unchanged by the migration.

## Backtesting

**Methodology**: BTCUSDT, 1-minute bars, `$10,000` starting balance, default
strategy parameters (none tuned between or within runs — validation, not
optimization, per this change's own Backtesting instructions), 0.1%/side fee,
no spread/slippage modeled. Same three ~6-week windows Phase 1 used
(bull=2023, bear=2022, chop=2024 H1). "Before" = the pre-migration
trend_following at commit `0408436` (which already includes the fixed
backtest trade-accounting engine, so before/after differ ONLY in strategy
behavior, not in how trades are counted). All six runs use the corrected
engine.

| Window | Dates | Character |
|---|---|---|
| Bull | 2023-10-01 – 2023-11-15 | ETF-speculation rally |
| Bear | 2022-11-01 – 2022-12-15 | FTX-collapse aftermath |
| Chop | 2024-04-01 – 2024-05-15 | Post-halving consolidation |

**Before vs. after (full pre/post comparison across all three windows):**

| Window | | Return % | Trades | Win rate | Profit factor | Max DD % | Fees | Buy-and-hold % |
|---|---|---|---|---|---|---|---|---|
| Bull | before | **−71.90%** | **716** | 12.8% | 0.29 | 71.92% | $8,215.71 | +31.84% |
| Bull | after | **−2.15%** | **50** | 26.0% | 0.87 | 7.49% | $765.36 | +31.84% |
| Bear | before | **−77.96%** | **703** | 10.8% | 0.22 | 78.05% | $6,983.08 | −13.07% |
| Bear | after | **−15.10%** | **62** | 17.7% | 0.28 | 15.75% | $889.41 | −13.07% |
| Chop | before | **−73.11%** | **663** | 15.1% | 0.32 | 73.43% | $7,498.37 | −13.59% |
| Chop | after | **−13.69%** | **75** | 25.3% | 0.37 | 15.23% | $1,208.39 | −13.59% |

**What the migration changed — the headline, and exactly the audit's
diagnosis confirmed by measurement**: the pre-migration strategy was
*catastrophic in every regime* — a −72% to −78% loss with **663–716 trades**
per 6-week window and **$7,000–$8,200 in fees**, including a −71.90% loss in a
BULL market that rose +31.84%. That is the framework's founding complaint made
concrete: a strategy that "overtrades, trades in the wrong market, loses
money" — here it literally churned itself to death on fees regardless of what
the market did, because its only entry logic was a boolean EMA cross with **no
regime awareness whatsoever**.

The migration cut trade count by **~90–93%** (716→50, 703→62, 663→75), cut the
loss by **50–70 percentage points** in every window, cut max drawdown from
~72–78% to **7–16%**, and cut fees by **~85–90%**. This is the Pillar 2 hard
regime gate + the Pillar 3 Decision-Score threshold + the Pillar 7 edge gate
all doing exactly what they were built to do: refuse the vast majority of the
low-quality, wrong-regime entries the old boolean gate fired on.

**Interpretation, honestly (not over-claiming)**:

- The migrated strategy is still **net-negative in all three windows**
  (−2.15% / −15.10% / −13.69%). The migration fixed the catastrophic
  overtrading pathology; it did NOT manufacture a proven edge, and it does not
  claim to. Positive expectancy for a trend follower requires either a
  genuinely sustained trend (none of these ~6-week windows contained one the
  strategy's long-only, regime-gated logic could ride cleanly) or calibrated
  thresholds — deliberately NOT tuned here (that would be curve-fitting), and
  explicitly deferred to `add-strategy-validation-tooling` (Pillars 4/7/9 all
  flag the thresholds as uncalibrated first-pass values).
- **Bull**: capital preservation is dramatically better (−2.15% vs −71.90%),
  but it lags buy-and-hold (+31.84%) — the regime gate + threshold kept it
  mostly in cash, so it neither bled out nor captured the rally. Consistent
  with Pillar 9: this is a timing/risk-control strategy, not a buy-and-hold
  substitute.
- **Bear**: −15.10% is slightly worse than buy-and-hold (−13.07%) but a **63
  percentage-point improvement** over the pre-migration −77.96%. A long-only
  trend follower is structurally disadvantaged in a sustained downtrend (few
  valid up-trends, residual whipsaws); the framework bounded the damage rather
  than eliminating it.
- **Chop**: −13.69% ≈ buy-and-hold (−13.59%), versus the pre-migration −73.11%
  — the single clearest demonstration that the regime gate now keeps the
  strategy out of the choppy conditions that previously destroyed it.
- Trade count dropped sharply but is **not zero** in any window (50/62/75) —
  satisfying the acceptance criterion that the strategy still trades in
  reasonable conditions. The remaining 50–75 trades per window are the
  regime-gate-passing entries that still net small losses at these uncalibrated
  thresholds; calibration is expected to reduce them further.

**Backtest-engine note**: these runs use the trade-accounting fix from the
pre-Phase-2 infrastructure review (commit `0408436`) — the `Trades` counts
above are the real closed-round-trip counts, no longer undercounted (the bug
`audits/volatility_breakout.md` discovered and this review fixed).

## Certification Checklist

- [x] Deterministic — `TestDecisionScoreCalculation::
      test_decision_score_deterministic_across_identical_calls`.
- [x] Immutable — enforced at runtime by the frozen `StrategyProposal`
      dataclass (Phase 0.10, not modified by this phase).
- [x] Assumptions documented — objective, falsifiable, traced to Pillar 1.
- [x] Expiration defined — `validity.valid_until` set and tested
      (`test_expired_proposal_would_be_discarded_by_adapter`).
- [x] Execution intent consistent with direction — enforced structurally by
      `StrategyProposal.__post_init__`; `TestStrategyProposalGeneration::
      test_every_proposal_has_consistent_intent_pairing` asserts every pairing
      this strategy emits.
- [x] Evidence measurable — Pillar 3 table complete, all four items pure
      functions of EMA/ATR data.
- [x] Explanation reproducible — Evidence Report renders identically for the
      same inputs (Pillar 3 determinism test); persisted via Pillar 8.
- [x] No subjective information — no factor could not be expressed as a
      deterministic Measurement + Normalization pair.
- [x] Expected edge sourced correctly — always `None`, mechanically enforced.
- [x] Before/after backtest comparison recorded — see Backtesting section
      (full pre/post across bull/bear/chop: trades cut ~90%, losses cut
      50–70pp, drawdown cut from ~72–78% to 7–16%).
- [x] All existing tests pass (1169/1169 full suite); new tests cover every
      pillar (16 new tests,
      `test_trend_following_framework_migration.py`).

## Self-Audit (critical review against trading theory, not code)

1. Behaves like a disciplined trend follower: enters only WITH a confirmed
   up-trend and a suitable regime, sizes by conviction (Decision Score), rides
   with a trailing stop, exits on a confirmed trend break. More mechanical than
   a discretionary trader by design — the point of the framework.
2. Should refuse to trade: choppy/`trend_flat` markets (implemented — Pillar 2
   hard gate, which the pre-migration version entirely lacked), weak/ambiguous
   trends (implemented — Decision Score threshold), a demonstrated losing streak
   with no explanation (implemented — Pillar 7 Category C). NOT implemented:
   liquidity/spread-based refusal (no order-book or volume data — an honest,
   documented gap).
3. Every executed BUY is justified by four named, weighted, deterministic
   Evidence factors; every rejected entry cites a specific blocking reason
   (unsuitable regime / score below threshold / Category C / cooldown), never a
   bare "no".
4. Arbitrary thresholds still present and explicitly flagged (not hidden):
   `decision_score_threshold` (40.0), `entry/exit_confirmation_loops` (3/2),
   `cooldown_seconds` (300), the adaptive multiplier bounds (1.0/4.0). The
   `atr_multiplier` itself is no longer arbitrary (adaptive).
5. Minimizes unnecessary trades far more than the pre-migration version: a
   multi-factor score, a non-trivial threshold, a hard regime gate, AND an
   edge-health gate must all independently agree, versus a single boolean AND
   with no regime awareness at all.
6. Protects against prolonged losing periods via Pillar 7 (Category C blocks
   new entries) — but does not (and per design SHALL NOT) force-close a
   position already open when a losing period is detected.
7. Remaining weaknesses for production review: no volume/order-book data;
   `StrategyEdgeManager` thresholds are generic defaults, not calibrated to this
   strategy; Decision Score weights/threshold are principled but uncalibrated;
   sizing is a local implementation of Pillar 5 pending
   `add-unified-position-sizing`'s shared consolidation. All flagged, none
   hidden.
</content>
</invoke>
