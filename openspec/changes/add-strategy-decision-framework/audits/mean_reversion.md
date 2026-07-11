# Strategy Audit: mean_reversion

`_strategy_mean_reversion`, `backend/app/services/trading_engine.py:2100-2696`.
Coverage score: **6/14**. Best-governed of the six on suitability/exits;
still zero quality scoring, zero adaptive parameters beyond the stop, flat
sizing, zero failure detection.

## Pillar 1 — Theory: UNDOCUMENTED (partial, second-strongest)

Docstring (2107-2124) states thesis ("ANTI-TREND, BOUNDED-RISK,
REGIME-AWARE... for range-bound markets"), states when it fails ("NOT
designed to hold through trends"), and states its character ("DEFENSIVE
STATISTICAL STRATEGY, not a conviction trade"). Still no explanation of
*why* the mean-reversion edge exists (microstructure/liquidity-provision
rationale) or an explicit assumptions list. **Authoring the remaining
theory is a certification task.**

## Pillar 2 — Market Suitability: Full

Real, enforced gate (2268-2307): `allowed_regimes=["trend_flat",
"volatility_high"]` (2271), computed from the strategy's own bar closes
via `_detect_market_regime` (2288) — fixing a real prior bug where the
shared tick buffer was empty for standalone bots (comment, 2278-2286).
Entry blocked at 2576-2582 if the regime doesn't match; force-exit on
trend flip at 2441-2463. **This is the reference implementation the other
five strategies' Pillar 2 remediation should look like.**

## Pillar 3 — Evidence-Based Decision Score: Not present

Single condition: `last_bar_close <= lower_band` (2597). No multi-factor
score; `TradeSignal.score`/`.threshold` fields exist on the signal type
(228-229) but mean_reversion never populates them.

## Pillar 4 — Parameter Adaptation: Not present (stop distance only)

Hardcoded: `bollinger_period=20`, `bollinger_std=1.8` (2140 — **docstring
at 2130 states default 2.0, a doc/code mismatch to fix during
certification**), `atr_stop_multiplier=2.0`, `max_hold_bars=10`,
`order_size_percent=20`, `cooldown_seconds=300`. Only the hard stop
distance scales with ATR (2664); the profit target locks to the live SMA
at entry (2663) but that's a locking mechanism, not volatility-adaptation.

## Pillar 5 — Position Sizing: Flat

`buy_amount = min(balance * order_size_percent, balance *
_BUY_BALANCE_FRACTION)` (2633-2636) — flat 20% of balance. No
Decision Score, exposure, or drawdown input.

## Pillar 6 — Trade Management: Full

Four conditions re-evaluated every tick while a position is open:
regime-flip force-exit (2441-2463), locked target reached (2482-2504),
locked hard stop (2508-2531), and time stop (`bars_since_entry >=
max_hold_bars`, 2535-2557). No partial-profit taking.

## Pillar 7 — Strategy Edge Management: Not present

No win/loss streak, expectancy, or self-pause logic anywhere in the
function (Auto Mode's external `_score_strategy` doesn't count — that's
cross-strategy orchestration, not this strategy's own awareness).

## Pillar 8 — Self-Diagnostics: Partial

Broad coverage: entry (2389-2395), exit checks (2376-2387), regime
(2387-2389), cooldown (2390). Gaps: no `.check()` for the fee-viability
gate (2619-2628, returns hold silently with no explain call), no
`.check()` for the sizing/min-order floor (2638-2650), and the hard-stop
value is reported only as a metric (2370), never asserted as a pass/fail
check at entry time.

## Pillar 9 — Performance Expectations: UNDOCUMENTED

No stated trade frequency, win rate, profit factor, drawdown, or
buy-and-hold comparison in the docstring. **To be authored at
certification.**

## Pillar 10 — Strategy Proposal Interface: Not present

Returns `TradeSignal` directly (e.g. `trading_engine.py:2686`,
`return TradeSignal(action="buy", ...)`) — no `StrategyProposal` concept
exists yet. This strategy already sets `expected_move_pct`/
`expected_risk_pct` on its `TradeSignal` (2684-2686) — the closest thing
to a real `expected_edge_estimate` found in the audit — a useful existing
computation to carry forward into the new field rather than recompute.
