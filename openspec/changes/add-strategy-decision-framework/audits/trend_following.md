# Strategy Audit: trend_following

`_strategy_trend_following`, `backend/app/services/trading_engine.py:2823-3272`.
Coverage score: **5/14**. **Zero enforced market suitability** — highest
urgency of the six per the user's first-named symptom.

## Pillar 1 — Theory: UNDOCUMENTED (partial)

Docstring (2830-2848) states mechanism (EMA crossover, ATR stops,
"noise-resistant confirmation") but no inefficiency rationale (why momentum
should persist), no explicit failure conditions, no assumptions section.
Thinner than mean_reversion's equivalent. **Authoring real theory is a
certification task.**

## Pillar 2 — Market Suitability: NOT ENFORCED (confirmed)

No regime detection call anywhere in the function (verified by full read
of lines 2823-3272). Regime awareness exists *only* externally, in
`_get_strategy_capabilities()["trend_following"]["allowed_regimes"]`
(line 6137/6134-6136 comment: "has no internal regime gate of its own —
this table is its ONLY regime awareness under Auto"). **A standalone
trend_following bot (not dispatched via Auto Mode) never checks market
regime before entering, at all.** This is the most acute example of the
user's complaint ("trade in inappropriate markets... ignore market
regime") found in the audit.

## Pillar 3 — Evidence-Based Decision Score: Not present

Entry is `current_price > ema_long AND ema_short > ema_long` (2995, 3057),
gated only by a fixed confirmation-loop counter (noise filter, not a
score) — `entry_confirmation_loops` default 3 (3061-3069).

## Pillar 4 — Parameter Adaptation: Not present (stop distance only)

Hardcoded: `short_period=50`, `long_period=100` (2851 — **docstring at
2842 claims default 200, a doc/code mismatch to fix during certification**),
`atr_multiplier=2.0`, `entry_confirmation_loops=3`,
`exit_confirmation_loops=2`, `cooldown_seconds=300`. Only the trailing-stop
distance and a fee-floor use live ATR (2935-2953); entry thresholds and
lookback periods are all fixed regardless of regime/volatility.

## Pillar 5 — Position Sizing: Risk-scaled (partial credit)

`risk_amount = balance * risk_percent`; `position_coins = risk_amount /
(atr * atr_multiplier)` (3081-3086) — genuinely volatility-scaled, unlike
dca/grid/mean_reversion's flat sizing. Still no Decision Score, portfolio
exposure, or drawdown input beyond the flat `risk_percent`.

## Pillar 6 — Trade Management: Full

Two conditions re-evaluated every tick: trailing stop on new highs using
locked entry ATR (3180-3212, immediate, no confirmation), and trend-break
exit requiring `exit_confirmation_loops` consecutive ticks below
EMA(long) (3215-3252). No partial profit-taking, no volatility-spike or
momentum-fading early exit beyond these two.

## Pillar 7 — Strategy Edge Management: Not present

No streak/expectancy/self-pause logic anywhere in the function.

## Pillar 8 — Self-Diagnostics: Partial

Covers cooldown, price-vs-EMA, crossover, confirmation counts (2997-3021)
and trailing-stop/trend-break checks (3025-3037). Gaps: no `.check()` for
the risk-based sizing calculation (3081-3095) or the min-order floor
(3102-3113); initial stop placement (3123) is reported only as a metric,
never asserted as pass/fail.

## Pillar 9 — Performance Expectations: UNDOCUMENTED

No stated trade frequency, win rate, profit factor, drawdown, or
buy-and-hold comparison in the docstring; only a bare "long" holding-time
label exists externally in the capability table (6139), not in the
strategy itself. **To be authored at certification.**

## Pillar 10 — Strategy Proposal Interface: Not present

Returns `TradeSignal` directly at every return point (e.g.
`trading_engine.py:3130`, `return TradeSignal(action="buy", ...)`) — no
`StrategyProposal` concept exists yet. `TradeSignal.score`/`.threshold`
are never populated by this strategy (confirmed under Pillar 3), so even
the existing observability-only precedent for a scored recommendation is
unused here today.
