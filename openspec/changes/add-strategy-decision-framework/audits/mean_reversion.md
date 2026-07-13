# Strategy Audit: `mean_reversion`

`_strategy_mean_reversion`, `backend/app/services/trading_engine.py`.

**Certification status**: ☐ Draft ☑ Under review ☐ Certified (date: ____ —
pending human sign-off on the before/after backtest comparison and the
Self-Audit findings)
**Certified against**: `add-strategy-decision-framework` 10-pillar
standard, Architecture Freeze revision (Phase 4 migration)

This document supersedes the pre-migration current-state audit (coverage
6/14 — best-governed on suitability/exits, but zero quality scoring, flat
sizing, zero edge detection). Original findings preserved as **Pre-Phase-4
finding** below each pillar.

## Pillar 1 — Theory

**Pre-Phase-4 finding**: second-strongest — a clear anti-trend/bounded-risk/
regime-aware thesis and stated failure mode, but no *why* (the inefficiency)
and no assumptions. UNDOCUMENTED (partial).

**Phase 4 (now documented in the docstring and here):**

- **Market Inefficiency**: short-horizon **overreaction in a range-bound
  market**. When transient order-flow imbalance (not new information) pushes
  price to a statistical extreme — a lower Bollinger Band, ~1.8 std below a
  rolling mean — the move tends to revert toward the mean as liquidity
  providers and value buyers step in.
- **Why The Edge Should Exist**: in a genuinely ranging market there is no
  fundamental reason for price to sit at an extreme; the push is temporary
  microstructure noise, and mean-reversion flow (market-making inventory
  rebalancing, contrarian value buying) pulls it back. The edge exists ONLY
  while the market is ranging — a trend turns "cheap" into "cheaper", which is
  why the regime gate and the trend-flip force-exit are the core risk control.
- **Evidence**: NOT yet through `add-strategy-validation-tooling`'s
  walk-forward validation (Tier 2, not implemented). A documented hypothesis —
  see Pillar 9 and the Backtesting section.
- **Conditions For Success**: a range-bound regime (trend_flat / high
  volatility), price at/below the lower band (a real statistical extreme), a
  band wide enough to clear round-trip fees, and reversion toward the locked
  mean target within the time stop.
- **Conditions For Failure**: a trend (the extreme keeps extending — bounded by
  the hard stop, the time stop, and the regime force-exit); a collapsed/narrow
  band (no room to profit after fees — the viability gate refuses it).
- **Assumptions** (objective, falsifiable): (1) range/band structure not broken
  — price stays within a mean-reverting range, not the start of a trend; (2) the
  regime remains range-bound (trend_flat / high-volatility, not trending).

## Pillar 2 — Market Suitability

**Pre-Phase-4 finding**: FULL — the reference implementation. Real enforced
gate (`allowed_regimes=["trend_flat", "volatility_high"]`) computed from the
strategy's own bar closes, entry blocked when the regime doesn't match, and a
trend-flip force-exit.

**Phase 4**: unchanged behaviourally; now **routed through the shared
`MarketSuitabilityGate`** for consistency with Auto dispatch (the gate
reproduces the exact tag-matching the strategy already did). Uses the price-only
`_detect_market_regime` on bar closes — sufficient here because this strategy's
`allowed_regimes` reference only LEVEL/trend tags (`trend_flat`,
`volatility_high`), never a *direction* tag, so the absence of
`volatility_direction` from the price-only detector loses nothing (contrast
dip_recovery, whose declared `volatility_expanding` IS inert — mean_reversion
declares no such tag). The `regime_filter_enabled` switch is KEPT (unlike
volatility_breakout's removal) as a test-isolation affordance; Pillar 2 is
enforced by default and remains the reference hard gate. The trend-flip
force-exit (Pillar 6) is unchanged.

## Pillar 3 — Evidence-Based Decision Score

**Pre-Phase-4 finding**: NOT present. Entry was a single condition
(`last_bar_close <= lower_band`).

**Phase 4**: closed. `DecisionScoreEngine().score(...)` gates entry (threshold
`decision_score_threshold` = 40.0, param-overridable). The band-touch is
PRESERVED as a hard precondition (mean reversion only ever fades a statistical
extreme); the Decision Score grades the QUALITY of a genuine touch.

| Evidence Item | Measurement | Normalization | Weight | Reason it contributes to edge |
|---|---|---|---|---|
| Reversion target distance | `(sma - current_price)/current_price` | `clamp(x / 0.02, -1, 1)` | 35 | Overreaction reversion: the reward is the distance from the oversold price back to the mean; a larger gap to the SMA is a larger expected reversion move. |
| Oversold penetration | `(lower_band - last_bar_close)/lower_band` | `clamp(x / 0.005, -1, 1)` | 35 | The further price has pushed BELOW the ~2-std band, the more likely the move is transient overreaction, and the stronger the snap-back tends to be. |
| Band width adequacy | `(upper_band - lower_band)/sma` | `clamp((x - 0.01)/0.03, -1, 1)` | 30 | Reversion is only worth trading if the range is wide enough to profit after fees; a collapsed band means no room. |

Weights sum to 100 and are deliberately set so **no single Evidence Item's
weight (max 35) reaches the default threshold (40)** — a trade always requires
multiple factors to agree (the multi-factor requirement; verified by
`TestDecisionScoreCalculation::test_no_single_factor_reaches_the_threshold`).
Weights are otherwise uncalibrated first-pass values. Determinism verified by
`::test_decision_score_deterministic`; the score genuinely gates entry (not just
the precondition) by `::test_unreachable_threshold_blocks_entry`.

## Pillar 4 — Parameter Adaptation

**Pre-Phase-4 finding**: not present beyond the ATR-scaled hard stop; the
`bollinger_std` docstring/code mismatch (2.0 vs 1.8) flagged.

**Phase 4**: no adaptive parameter added (not a Phase 4 task). Classification
documented; the **`bollinger_std` mismatch is fixed** (docstring and default
both read 1.8).

| Parameter | Classification | Reason |
|---|---|---|
| `bollinger_std` (1.8), `bollinger_period` (20) | FIXED | Define what "a statistical extreme" MEANS; changing them changes the strategy's identity, not how it adapts. |
| `atr_stop_multiplier` (2.0) | FIXED (ATR-scaled) | The stop distance is ATR-scaled and LOCKED at entry; the multiplier is a design choice. |
| `max_hold_bars` (10), `cooldown_seconds` (300), `order_size_percent` (20), `atr_period` (14), `bar_interval_seconds` (60) | FIXED | Debounce/measurement/sizing constants, not volatility-derived. |

## Pillar 5 — Decision-Score-Weighted Position Sizing

**Pre-Phase-4 finding**: FLAT — 20% of balance, no Decision Score input.

**Phase 4**: `size_multiplier = 0.5 + clamp((score - threshold)/(100 -
threshold), 0, 1)` scales the flat `order_size_percent` (marginal → 0.5x,
maximal → 1.5x). Recorded in `adaptive_parameters_used`. Not yet wired to the
shared `add-unified-position-sizing` function (not implemented); certification-
phase local implementation, same as Phases 1–3.

## Pillar 6 — Continuous Trade Management

**Pre-Phase-4 finding**: FULL — four conditions re-evaluated every tick:
regime-flip force-exit, locked target, locked hard stop, time stop.

**Phase 4**: unchanged behaviourally. All four exits are preserved and now
emitted as `SELL` / `CLOSE_POSITION` proposals via the Standalone Adapter; each
records its outcome with the Strategy Edge Manager (Pillar 7).

## Pillar 7 — Strategy Edge Management

**Pre-Phase-4 finding**: NOT present.

**Phase 4**: closed. `StrategyEdgeManager` at
`self._mean_reversion_edge_manager`; `.evaluate(...)` every cycle,
`.record_decision_score(...)` on entry, `.record_trade_outcome(...)` on every
exit (only with a real entry price).

- **Configuration**: shared-module defaults — NOT recalibrated for this
  strategy; flagged outstanding certification work.
- **Category A**: `regime_outside_suitable_range = not suitability.is_suitable`.
- **Category B**: **never cited** — this phase makes no parameter adaptive
  (`bollinger_std`/`period` and the ATR stop multiplier are FIXED by design), so
  there is no adaptive base to point at as miscalibrated. Degradation classifies
  as A (regime) or C (edge gone) only. Documented as a reasoned choice.
- **Category C**: blocks new entries; NEVER force-closes an open position —
  `::test_edge_management_never_force_closes_open_position`.

## Pillar 8 — Self-Diagnostics

**Pre-Phase-4 finding**: PARTIAL — gaps: no `.check()` for the fee-viability
gate (silent hold), the sizing/min-order floor, or the hard-stop value at entry.

**Phase 4**: closed — added `.check()` for "Fee viability", "Position size >=
min order", "Initial hard stop below entry", plus the Decision Score / Strategy
edge checks. `edge_status_category` metric populated for Order persistence.

## Pillar 9 — Performance Expectations

**Pre-Phase-4 finding**: UNDOCUMENTED.

**Phase 4** (documented — see Backtesting for measured results):

- **Expected trade frequency**: low-to-moderate — only at a genuine lower-band
  extreme in a ranging regime; the score threshold reduces frequency further.
- **Expected holding time**: short — bounded by `max_hold_bars` (10) unless the
  mean target or a stop resolves it sooner.
- **Expected win rate**: MODERATE-to-HIGH by design (a defensive statistical
  strategy expects many small mean-reversion wins), but each win is small (to
  the mean) and losses are bounded (hard/time/regime stops) — profitability
  hinges on the wins clearing fees. No numeric target — no walk-forward
  evidence to ground one.
- **Expected profit factor / drawdown**: not yet quantified — UNDOCUMENTED
  pending `add-strategy-validation-tooling`.
- **Typical vs. worst-case**: typical = a choppy range oscillating around a
  stable mean; worst-case = a range that breaks into a trend (bounded by the
  regime force-exit and stops).
- **Buy-and-hold comparison**: reported per window in Backtesting for context
  only; a defensive range strategy is not a buy-and-hold substitute.

## Pillar 10 — Strategy Proposal Interface

**Pre-Phase-4 finding**: returns `TradeSignal` directly; already set
`expected_move_pct`/`expected_risk_pct` (carried forward). Not present.

**Phase 4**: closed — every return constructs a `StrategyProposal` translated
via `StandaloneAdapter.to_trade_signal(...)`.

- **`execution_intent` mapping**: band-touch entry -> `BUY` + `OPEN_POSITION`.
  Every exit (regime flip, mean target, hard stop, time stop) -> `SELL` +
  `CLOSE_POSITION`. Holding -> `HOLD` + `HOLD_POSITION`. Warmup / unsuitable /
  cooldown / below-threshold / no-touch -> `NO_TRADE` + `NO_ACTION`.
- **`validity.valid_until`**: `generated_at + max(bar_interval_seconds, 1)` s.
- **`assumptions`**: the two objective conditions from Pillar 1.
- **`expected_edge_estimate`**: always `None`, mechanically enforced.
- **`expected_move_pct`/`expected_risk_pct`**: carried into the adapter call
  from the reversion distance / hard-stop distance (existing computation).
- **Behavior-preservation**: `TestStandaloneAdapterCompatibility` tests; the
  pre-migration regime/exit behaviour stays covered by the other suites (the MR
  fixtures there isolated from the new score gate via
  `decision_score_threshold: 0.0`, mirroring Phase 2's approach).

## Backtesting

**Methodology**: BTCUSDT, 1-minute bars, `$10,000` starting balance, default
strategy parameters (none tuned between or within runs), 0.1%/side fee, no
spread/slippage modeled. Same three ~6-week windows as Phases 1–3. "Before" =
pre-migration mean_reversion at commit `73bc164` (already on the fixed backtest
engine, so before/after differ only in strategy behaviour). All six runs use
the corrected engine.

**Before vs. after (full pre/post across all three windows):**

| Window | | Return % | Trades | Win rate | Profit factor | Max DD % | Fees | Buy-and-hold % |
|---|---|---|---|---|---|---|---|---|
| Bull | before | −11.31% | 303 | 21.5% | 0.13 | 11.31% | $1,140.03 | +31.84% |
| Bull | after | **+0.01%** | 10 | 80.0% | 1.03 | 0.20% | $27.37 | +31.84% |
| Bear | before | −15.04% | 389 | 23.7% | 0.19 | 15.07% | $1,437.27 | −13.07% |
| Bear | after | **−1.44%** | 20 | 25.0% | 0.10 | 1.50% | $62.20 | −13.07% |
| Chop | before | −23.97% | 667 | 24.0% | 0.15 | 23.98% | $2,316.25 | −13.59% |
| Chop | after | **−1.33%** | 20 | 15.0% | 0.15 | 1.53% | $61.44 | −13.59% |

**What the migration changed**: the pre-migration strategy was a **catastrophic
overtrader** — despite already having the reference regime gate (Pillar 2) and
four-way exits (Pillar 6), it fired on EVERY lower-band touch within a suitable
regime, with no quality filter. In a choppy range that means hundreds of tiny
fee-losing round-trips: **303–667 trades per 6-week window**, losing 11–24% in
EVERY regime — including −23.97% in the chop window that is supposedly its home
turf. The migration's newly-added Pillar 3 Decision Score is the missing quality
gate: it cut trade count **~95–97%** (303→10, 389→20, 667→20), collapsed max
drawdown from 11–24% to **0.2–1.5%**, cut fees ~95%, and lifted the return in
every window to roughly breakeven (+0.01% / −1.44% / −1.33%), now beating
buy-and-hold in both the bear and chop windows.

**Interpretation, honestly**: the dominant, measured effect is the correction of
an architectural defect — the absence of any trade-quality gate let a
regime-suitable-but-low-quality band touch fire hundreds of times and bleed the
account on fees. Adding the Decision Score removed that pathology almost
entirely.

But the correction did **not** reveal a positive edge — it revealed capital
preservation through selectivity. Post-migration the strategy barely trades
(10–20 trades/window), and its per-trade **profit factor stays at or below ~1.0
(1.03 bull, 0.10 bear, 0.15 chop)**: the bull window is a breakeven +0.01%, and
the few trades it still takes in bear/chop remain net-losing (PF 0.10/0.15). So
while the return numbers improved dramatically, the surviving trades do not
demonstrate positive expectancy; the improvement is "stopped doing the losing
thing", not "found a winning thing". A definitive edge ruling awaits
`add-strategy-validation-tooling`'s walk-forward validation, and the Decision
Score threshold/weights are uncalibrated first-pass values (deliberately not
tuned to these windows).

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
- [x] Evidence measurable — Pillar 3 table complete; single-factor
      insufficiency verified.
- [x] Explanation reproducible — determinism test; persisted via Pillar 8.
- [x] No subjective information — every factor a deterministic Measurement +
      Normalization pair.
- [x] Expected edge sourced correctly — always `None`, mechanically enforced.
- [x] Before/after backtest comparison recorded — see Backtesting section
      (full pre/post across bull/bear/chop: trades cut ~95–97%, return lifted
      from −11/−15/−24% to ~breakeven, but per-trade profit factor stays ≤1.0 —
      pathology corrected, positive edge not demonstrated).
- [x] All existing tests pass (1201/1201 full suite); 16 new tests
      (`test_mean_reversion_framework_migration.py`).

## Self-Audit (critical review against trading theory, not code)

1. Behaves like a disciplined defensive range trader: fades only genuine
   statistical extremes in a ranging regime, sizes by conviction, exits fast on
   the mean / a stop / a regime flip.
2. Should refuse to trade: a trending market (implemented — Pillar 2 gate +
   force-exit), a low-conviction touch (implemented — Decision Score), a
   fee-uncoverable narrow band (implemented — viability gate), a demonstrated
   losing streak with no explanation (implemented — Pillar 7 Category C). NOT
   implemented: order-book/volume-based refusal (no such data — documented).
3. Category B is deliberately never cited (no adaptive parameter this phase) —
   documented as a reasoned choice, not an oversight.
4. Remaining weaknesses for production review: Edge Manager thresholds are
   generic defaults; Decision Score weights/threshold are principled (multi-
   factor-enforcing) but uncalibrated; sizing is a local Pillar 5 implementation
   pending `add-unified-position-sizing`. All flagged, none hidden.
</content>
