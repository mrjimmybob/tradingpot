# Strategy Audit: `dip_recovery`

`_strategy_dip_recovery` (+ `_dip_recovery_manage_setup`,
`_dip_recovery_manage_exit`, `_dip_recovery_exit_signal`),
`backend/app/services/trading_engine.py`.

**Certification status**: ☐ Draft ☑ Under review ☐ Certified (date: ____ —
pending human sign-off on the before/after backtest comparison and the
Self-Audit findings)
**Certified against**: `add-strategy-decision-framework` 10-pillar
standard, Architecture Freeze revision (Phase 3 migration)

This document supersedes the pre-migration current-state audit (coverage
7/14 — the highest of the six, best foundation). The original findings are
preserved below each pillar as **Pre-Phase-3 finding**, followed by what
Phase 3 changed.

## Pillar 1 — Theory

**Pre-Phase-3 finding**: strongest of the six — a clear "buy the bounce AFTER
a decline, never the falling knife" thesis, but no *why* (the inefficiency)
and no explicit failure conditions. UNDOCUMENTED (partial).

**Phase 3 (now documented in the docstring and here):**

- **Market Inefficiency**: short-horizon **overreaction / panic-selling mean
  reversion**. Sharp, fear-driven declines routinely overshoot fair value as
  forced sellers (liquidations, stop cascades, capitulation) transact
  price-insensitively; once that flow exhausts, price tends to revert part of
  the overshoot.
- **Why The Edge Should Exist**: the sellers driving a panic leg are
  non-fundamental and finite. When they exhaust, the marginal buyer no longer
  faces that pressure and price snaps back — the "bounce" this strategy
  targets. Entering only AFTER a confirmed reversal (not during the decline)
  trades a later entry for materially lower odds of catching a still-falling
  knife.
- **Evidence**: NOT yet through `add-strategy-validation-tooling`'s
  walk-forward validation (Tier 2, not implemented). A documented hypothesis,
  not a proven edge — see Pillar 9 and the Backtesting section.
- **Conditions For Success**: a genuine ATR-scaled decline, then a confirmed
  reversal off a local low (recovery clears an adaptive threshold, short EMA
  rising, no new low for N ticks) while the regime is a declining/volatile one
  the strategy is suited to.
- **Conditions For Failure**: a persistent downtrend that produces repeated
  false reversals (each bounce fails and makes a new low — bounded per attempt
  by the trailing/emergency stops and the loss-aware cooldown); a low-liquidity
  gap where the "decline" is a data artifact (no volume data — documented
  limitation).
- **Assumptions** (objective, falsifiable — source for Pillar 10's
  `assumptions`): (1) no new low since entry (the tracked bottom is not
  undercut); (2) the confirmed reversal is not reversed (price does not fall
  back below the recovery threshold off the low); (3) the current regime
  remains within the strategy's allowed set.

## Pillar 2 — Market Suitability

**Pre-Phase-3 finding**: NOT ENFORCED. No regime check anywhere; regime
awareness existed only in the Auto capability table. A standalone bot never
checked regime before entering.

**Phase 3**: closed. `MarketSuitabilityGate().evaluate(current_regime,
allowed_regimes)` is called every cycle and is a hard gate on new entries (a
fully confirmed reversal is still refused — NO_TRADE — in an unsuitable
regime). Verified by `tests/test_dip_recovery_framework_migration.py::
TestMarketSuitability::test_unsuitable_regime_blocks_entry_despite_confirmed_reversal`.

- **Declared `allowed_regimes`**: `["trend_down", "volatility_high",
  "volatility_expanding"]` (default; param-overridable) — matches
  `_get_strategy_capabilities()["dip_recovery"]`.

### Why the PRICE-based regime detector (documented decision, not a silent assumption)

dip_recovery is **tick-driven**: it keeps a tick `price_history` and computes
an ATR *proxy* from tick-to-tick changes (`_calc_price_atr_proxy`); it has NO
OHLC bar aggregation (that is its Pillar 4 design — thresholds adapt to a tick
ATR proxy, no candle machinery). It therefore uses
**`_detect_market_regime(price_history, …)`** — the documented price-only
regime variant built for exactly this kind of strategy — rather than
`_detect_market_regime_bar_based(...)`, which requires OHLC bars this strategy
does not have. Feeding the bar-based detector would have required inventing a
bar-aggregation layer purely to satisfy the gate, which the price-only variant
exists specifically to avoid.

**Consequences (recorded here, not hidden):**

1. **`volatility_expanding` never fires in the standalone path.** The
   price-only detector returns `trend_state` / `volatility_state` /
   `liquidity_state` but **no `volatility_direction`** field. The shared
   `MarketSuitabilityGate.regime_tags()` therefore falls back to the default
   `volatility_stable`, so the `volatility_expanding` tag in this strategy's
   declared `allowed_regimes` **can never match** when the strategy runs
   standalone. It is retained in the declaration for consistency with the Auto
   capability table (Auto's bar-based path *does* evaluate it) and for the day
   this strategy might be fed bar data, but it is inert here today. Asserted by
   `TestMarketSuitability::test_suitable_regime_allows_entry` (which checks
   `"volatility_expanding" not in regime_tags`).
2. **Observable behavioural limitation** (the honest one): with
   `volatility_expanding` inert, standalone suitability rests on **`trend_down`
   or `volatility_high`**. In practice this is not a real narrowing of when the
   strategy trades, because a genuine panic decline — exactly this strategy's
   setup — reads as a declining and/or high-volatility regime, so `trend_down`
   / `volatility_high` cover it. The one edge case NOT covered standalone: a
   market with expanding-but-not-yet-high volatility that is also flat/rising
   in trend (an early volatility expansion with no decline). That is not a
   dip-recovery setup anyway (there is no dip to recover from), so no genuine
   opportunity is lost — but it IS a documented difference between the
   standalone gate and Auto's gate, recorded rather than glossed over.
3. **Liquidity is always `normal`** from the price-only detector (not
   measurable from close prices alone); dip_recovery declares no liquidity tag,
   so this has no effect on its gate.

- **Regime source citation**: `_detect_market_regime(price_history, None)` in
  `_strategy_dip_recovery`, immediately after positions are fetched.

## Pillar 3 — Evidence-Based Decision Score

**Pre-Phase-3 finding**: PARTIAL — `entry_ready` was a triple boolean AND, but
an `opportunity_score` was ALREADY computed (`_dip_recovery_score_from_ratios`)
and only written to diagnostics, never consulted by the entry decision. The
closest any strategy came to Pillar 3.

**Phase 3**: closed. `DecisionScoreEngine().score(...)` now gates entry
(threshold `decision_score_threshold` = 40.0, param-overridable). The formerly
diagnostic-only opportunity signals are formalized as Evidence Items. The
pre-migration "never buy on the way down" filters (recovery cleared its
adaptive threshold, short EMA rising, no new low) are **preserved as HARD
safety preconditions** — the Decision Score evaluates the QUALITY of an
already-safe setup, it does not override those filters.

| Evidence Item | Measurement | Normalization | Weight | Reason it contributes to edge |
|---|---|---|---|---|
| Decline depth | `decline_ratio` (decline % / ATR %) | `clamp(x / 2.5, -1, 1)` | 35 | Overreaction thesis: the deeper the panic decline relative to this pair's own ATR, the more likely it overshot fair value and the larger the mean-reversion bounce. |
| Recovery strength | `recovery_ratio` (recovery % off low / ATR %) | `clamp(x / 1.0, -1, 1)` | 30 | A reversal that has already retraced a meaningful fraction of an ATR is stronger evidence the bottom is in — the "confirmed reversal, not a falling knife" core. |
| Reversal momentum (EMA slope) | `1.0 if short EMA rising else 0.0` | `clamp(x, -1, 1)` | 20 | A rising short EMA confirms the bounce has momentum, not a single mean-reverting tick that resumes falling. |
| Base stability (no new low) | `min(ticks_since_new_low / min_ticks, 1.5)` | `clamp(x, -1, 1)` | 15 | The longer price holds without a new low, the more sellers are exhausted and the safer the entry. |

Weights sum to 100. **Structural single-factor insufficiency**: the largest
single weight (35) is below the default threshold (40), so no single Evidence
Item can carry a trade — verified by `TestDecisionScoreCalculation::
test_no_single_factor_reaches_the_threshold` and
`::test_unreachable_threshold_blocks_entry` (the score genuinely gates entry,
not just the preconditions).

- **Determinism test**: `::test_decision_score_deterministic`.

## Pillar 4 — Parameter Adaptation

**Pre-Phase-3 finding**: PARTIAL — the **best of the six**. The entry
thresholds themselves are already ATR-adaptive: `drop_threshold =
max(min_drop_percent, atr_percent * drop_atr_multiplier)` and the analogous
`recovery_threshold`. No Phase 3 task adds a new adaptive parameter (unlike
Phases 1–2, which made a fixed stop multiplier adaptive) — this strategy
already met the pillar's substance.

**Phase 3**: no code change; classification documented.

| Parameter | Classification | Reason |
|---|---|---|
| `drop_threshold` / `recovery_threshold` | **ADAPTIVE** | `max(floor, atr_percent × multiplier)` — the actual entry thresholds scale with live tick-ATR %. |
| `min_drop_percent` (1.5), `drop_atr_multiplier` (2.5), `min_recovery_percent` (0.5), `recovery_atr_multiplier` (0.8) | FIXED | Floors and multipliers *defining* the adaptive formula; not themselves volatility-derived. |
| `take_profit/trailing_stop/emergency_stop_atr_multiplier` (3.0/1.5/5.0) | FIXED (ATR-scaled) | The exit distances are ATR-scaled and LOCKED at entry (risk never expands); the multipliers are design choices from the strategy's reward:risk shape. |
| `atr_period` (14), `reference_high_lookback_ticks` (60), `ema_slope_period` (5), `min_ticks_without_new_low` (2), durations/cooldowns, `spike_guard_atr_multiplier` (6.0) | FIXED | Measurement windows / debounce constants, not volatility-derived. |

## Pillar 5 — Decision-Score-Weighted Position Sizing

**Pre-Phase-3 finding**: risk-scaled (partial) — `risk_amount = balance *
risk_percent`, coins from the ATR trailing distance. No Decision Score input.

**Phase 3**: `size_multiplier = 0.5 + clamp((score - threshold) / (100 -
threshold), 0, 1)` scales the risked amount (marginal → 0.5x, maximal → 1.5x),
then the existing ATR-trailing-distance formula converts risk to coins.
Deterministic; recorded in `adaptive_parameters_used`. Not yet wired to the
shared `add-unified-position-sizing` function (not implemented); this is the
certification-phase local implementation, same as Phases 1–2.

## Pillar 6 — Continuous Trade Management

**Pre-Phase-3 finding**: FULL — the most elaborate of the six: take-profit,
monotonic trailing stop, wider emergency stop, and max-duration exit, with
loss-aware cooldown routing. No gap.

**Phase 3**: unchanged behaviourally. All four exits are preserved exactly and
now emitted as `SELL` / `CLOSE_POSITION` proposals via the Standalone Adapter;
each records its outcome with the Strategy Edge Manager (Pillar 7) before state
is cleared.

## Pillar 7 — Strategy Edge Management

**Pre-Phase-3 finding**: NOT present.

**Phase 3**: closed. `StrategyEdgeManager` at
`self._dip_recovery_edge_manager`; `.evaluate(...)` every cycle,
`.record_decision_score(...)` on entry evaluation, `.record_trade_outcome(...)`
on every exit (only when a real entry price is known — a defensively-adopted
entry has no true P&L and is not recorded, to avoid corrupting statistics).

- **Configuration**: shared-module defaults (`min_sample_size=20`,
  `win_rate_floor=0.35`, `expectancy_floor=0.0`) — NOT recalibrated for this
  strategy; flagged outstanding certification work.
- **Category A**: `regime_outside_suitable_range = not
  suitability.is_suitable`.
- **Category B**: **never cited** for this strategy — its thresholds are
  already ATR-adaptive, so there is no single fixed base-multiplier the manager
  could point at as miscalibrated. Degradation classifies as A (regime) or C
  (edge gone) only. Documented so the absence of B is a deliberate,
  reasoned choice, not an oversight.
- **Category C**: blocks new entries; NEVER force-closes an open position —
  `::test_edge_management_never_force_closes_open_position`.
- **Tests**: `::test_category_c_blocks_new_entry`,
  `::test_healthy_history_allows_entry`.

## Pillar 8 — Self-Diagnostics

**Pre-Phase-3 finding**: PARTIAL — one notable gap: **no `exp.check()` for the
emergency stop** despite it triggering sells; also no check for entry sizing /
fee-viability.

**Phase 3**: closed — added `.check()` for "Emergency stop hit"
(`::test_emergency_stop_check_is_surfaced`), "Position size >= min order",
"Fee viability", plus the Decision Score / Market suitability / Strategy edge
checks. `edge_status_category` metric populated every cycle for Order
persistence.

## Pillar 9 — Performance Expectations

**Pre-Phase-3 finding**: UNDOCUMENTED.

**Phase 3** (documented — see Backtesting for measured results):

- **Expected trade frequency**: low-to-moderate — only after a real ATR-scaled
  decline confirms a reversal; the regime gate + Decision Score threshold
  reduce frequency further vs. the pre-migration triple-boolean gate.
- **Expected holding time**: medium — rides the bounce to the take-profit or a
  trailing/emergency stop / max-duration exit.
- **Expected win rate**: MODERATE — a mean-reversion bounce strategy expects a
  reasonable hit rate on confirmed reversals, with losses bounded by the
  stops; profitability depends on the bounces clearing round-trip fees. No
  numeric target — no walk-forward evidence to ground one honestly.
- **Expected profit factor / drawdown**: not yet quantified — UNDOCUMENTED
  pending `add-strategy-validation-tooling`.
- **Typical vs. worst-case**: typical = a sharp panic decline into a genuine
  reversal; worst-case = a persistent downtrend of repeated failed bounces
  (bounded per attempt by stops + loss-aware cooldown).
- **Buy-and-hold comparison**: reported per window in Backtesting for context
  only; this is a timing/risk-control strategy, not a buy-and-hold substitute.

## Pillar 10 — Strategy Proposal Interface

**Pre-Phase-3 finding**: returns `TradeSignal` directly. Already computed
`opportunity_score` and set `expected_move_pct`/`expected_risk_pct` — useful
existing computations carried forward. Not present.

**Phase 3**: closed — every return across all four methods now constructs a
`StrategyProposal` translated via `StandaloneAdapter.to_trade_signal(...)` —
verified by `_capture_proposals` spying in the migration suite.

- **`execution_intent` mapping**: recovery-confirmed entry -> `BUY` +
  `OPEN_POSITION`. Every exit (take-profit, trailing stop, emergency stop,
  max-duration) -> `SELL` + `CLOSE_POSITION`. Holding an open position ->
  `HOLD` + `HOLD_POSITION`. Every tracking / waiting / cooldown / blocked-entry
  branch -> `NO_TRADE` + `NO_ACTION`. `ADD_TO_POSITION` / `REDUCE_POSITION`
  unused (single-position strategy).
- **`validity.valid_until`**: `generated_at + max(bar_interval_seconds, 1)` s
  (`_dip_recovery_validity_seconds`).
- **`assumptions`**: the three objective conditions from Pillar 1.
- **`expected_edge_estimate`**: always `None`, mechanically enforced.
- **`expected_move_pct` / `expected_risk_pct`**: carried into the adapter call
  from the strategy's own take-profit / trailing-stop distances (the existing
  viability computation, preserved).
- **Behavior-preservation**: `TestStandaloneAdapterCompatibility::
  test_buy_signal_matches_adapter_translation` and
  `::test_expired_proposal_would_be_discarded`; the full pre-migration state
  machine remains covered by `test_dip_recovery_strategy.py` (25 unchanged +
  1 reason-assertion update).

## Backtesting

**Methodology**: BTCUSDT, 1-minute bars, `$10,000` starting balance, default
strategy parameters (none tuned between or within runs — validation, not
optimization), 0.1%/side fee, no spread/slippage modeled. Same three ~6-week
windows Phases 1–2 used. "Before" = pre-migration dip_recovery at commit
`bc51efb` (which already includes the fixed backtest trade-accounting engine,
so before/after differ ONLY in strategy behaviour). All six runs use the
corrected engine.

**Before vs. after (full pre/post across all three windows):**

| Window | | Return % | Trades | Win rate | Profit factor | Max DD % | Fees | Buy-and-hold % |
|---|---|---|---|---|---|---|---|---|
| Bull | before | −7.58% | 24 | 20.8% | 0.14 | 7.86% | $451.77 | +31.84% |
| Bull | after | −6.05% | 13 | 15.4% | 0.08 | 6.40% | $250.27 | +31.84% |
| Bear | before | −19.27% | 76 | 25.0% | 0.28 | 19.70% | $1,348.50 | −13.07% |
| Bear | after | −8.96% | 20 | 20.0% | 0.17 | 8.96% | $381.18 | −13.07% |
| Chop | before | −11.18% | 72 | 25.0% | 0.33 | 11.95% | $1,356.20 | −13.59% |
| Chop | after | −7.83% | 47 | 25.5% | 0.37 | 8.66% | $905.63 | −13.59% |

**What the migration changed**: unlike trend_following, dip_recovery was
**never a catastrophic overtrader** — the pre-migration code was already the
best-implemented of the six (7/14: real regime-adaptive thresholds, elaborate
four-way exit management, a computed opportunity score). So the "before" is a
*moderate* loser (24–76 trades, −8% to −19%), not a −72% fee-bleed. The
migration cut trades in every window (24→13, 76→20, 72→47), cut fees ~45–70%,
cut max drawdown in every window, and **improved the return in every window**
— most in the bear (−19.27%→−8.96%, now BETTER than buy-and-hold −13.07%) and
chop (−11.18%→−7.83%, now BETTER than buy-and-hold −13.59%). The Pillar 2
regime gate + Pillar 3 Decision Score + Pillar 7 edge gate all did their job:
refuse more of the low-quality entries.

**Interpretation, honestly (this is the decisive finding for this strategy)**:
the migrated strategy is **still net-negative in all three windows**, with a
**profit factor below 1.0 in every window (0.08 / 0.17 / 0.37)** and a win rate
of only 15–25%. Critically, the improvement came almost entirely from *trading
less* — fewer fee-losing marginal entries — NOT from the remaining trades
becoming profitable. The residual trades still have negative expectancy in
every regime. So unlike trend_following (where the migration's headline was
fixing a gross architectural defect), here the code was already correct and the
architecture is now complete, yet the strategy **still cannot demonstrate a
positive edge** in bull, bear, OR chop.

The honest caveat, per this change's own discipline: a *definitive* "no edge"
verdict requires `add-strategy-validation-tooling`'s walk-forward validation
across many more windows (which does not exist yet), and the Decision Score
threshold / weights are uncalibrated first-pass values (deliberately NOT tuned
against these windows — that would be curve-fitting). But across three
independent regime windows, correctly implemented and architecturally complete,
the measured result is a consistent net loss with sub-1.0 profit factor — which
is evidence *against* a measurable edge at these settings, not merely an
absence of evidence for one.

**Backtest-engine note**: these runs use the trade-accounting fix from the
pre-Phase-2 infrastructure review (commit `0408436`); the trade counts are the
real closed-round-trip counts.

## Certification Checklist

- [x] Deterministic — `test_decision_score_deterministic`.
- [x] Immutable — frozen `StrategyProposal` dataclass.
- [x] Assumptions documented — objective, falsifiable, traced to Pillar 1.
- [x] Expiration defined — `validity.valid_until` set and tested.
- [x] Execution intent consistent with direction — structurally enforced;
      `test_all_proposals_have_valid_intent_pairings`.
- [x] Evidence measurable — Pillar 3 table complete.
- [x] Explanation reproducible — determinism test; persisted via Pillar 8.
- [x] No subjective information — every factor a deterministic Measurement +
      Normalization pair.
- [x] Expected edge sourced correctly — always `None`, mechanically enforced.
- [x] Before/after backtest comparison recorded — see Backtesting section
      (full pre/post across bull/bear/chop: trades cut 35–74%, return improved
      in every window, but profit factor stays below 1.0 in all three —
      evidence against a measurable edge at these settings).
- [x] All existing tests pass (1185/1185 full suite); 16 new tests
      (`test_dip_recovery_framework_migration.py`).

## Self-Audit (critical review against trading theory, not code)

1. Behaves like a disciplined reversal trader: never buys the falling knife
   (hard preconditions preserved), sizes by conviction, has layered stops and a
   max-duration net. More mechanical than a discretionary trader by design.
2. Should refuse to trade: a still-falling market (implemented — recovery
   precondition), an unsuitable regime (implemented — Pillar 2), a weak setup
   (implemented — Decision Score), a demonstrated losing streak with no
   explanation (implemented — Pillar 7 Category C). NOT implemented:
   liquidity/volume-based refusal (no such data — documented).
3. The one honest architectural nuance is fully documented, not hidden: the
   price-only regime detector makes `volatility_expanding` inert standalone
   (Pillar 2, consequence 2). No genuine dip-recovery opportunity is lost by
   it, but it IS a standalone-vs-Auto difference and is recorded as such.
4. Category B is deliberately never cited (thresholds already adaptive) —
   documented as a reasoned choice, not an oversight.
5. Remaining weaknesses for production review: Edge Manager thresholds are
   generic defaults; Decision Score weights/threshold are principled but
   uncalibrated; sizing is a local Pillar 5 implementation pending
   `add-unified-position-sizing`. All flagged, none hidden.
</content>
