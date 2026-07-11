# Strategy Audit: dip_recovery

`_strategy_dip_recovery`, `backend/app/services/trading_engine.py:4008-4612`
(exit management in `_dip_recovery_manage_exit`, 4505-4612).
Coverage score: **7/14** (highest of the six — best starting point once its
one glaring gap, Pillar 2, is closed).

## Pillar 1 — Theory: UNDOCUMENTED (partial, strongest of the six)

Docstring (4015-4069) states a clear thesis: "Captures the bounce AFTER a
significant decline, not the decline itself... never buys into a market
that is still falling," and names the trade-off explicitly ("accepts a
later entry... in exchange for materially reduced downside risk"). Still
no discussion of *why* the inefficiency exists (overreaction/panic-selling
mean reversion) or explicit failure-mode conditions (persistent downtrends
producing repeated false reversals). **Authoring the remaining theory is a
certification task**, but this strategy has the best head start.

## Pillar 2 — Market Suitability: NOT ENFORCED (confirmed)

No regime check anywhere in `_strategy_dip_recovery` or its helper
functions. Only market-condition awareness is the capability-table entry
Auto Mode uses (`allowed_regimes: ["trend_down", "volatility_expanding",
"volatility_high"]`, ~6187-6199), with an explicit comment: "has no
internal regime gate of its own... this table is its ONLY regime awareness
under Auto." A standalone dip_recovery bot never checks regime before
entering, same gap as trend_following.

## Pillar 3 — Evidence-Based Decision Score: Partial (infrastructure exists, unused)

`entry_ready = recovery_ok and ema_ok and no_new_low_ok` (4344) is a
triple boolean AND, not a weighted score. However, an `opportunity_score`
**is already computed** (`_dip_recovery_score_from_ratios`, 4182, 4351) —
it's just only written into `exp.update()` for diagnostics (4188, 4359),
never referenced in `entry_ready` or any gate. **This is the closest any
strategy comes to Pillar 3 today** — the scoring mechanism exists, it just
needs to be wired into the actual entry decision instead of computed and
discarded (the same pattern as Pillar 2's gap in this strategy and in
volatility_breakout).

## Pillar 4 — Parameter Adaptation: Partial (best of the six)

Hardcoded: `atr_period=14`, `min_drop_percent=1.5`,
`drop_atr_multiplier=2.5`, `min_recovery_percent=0.5`,
`ema_slope_period=5`, `max_position_duration_minutes=720`,
`cooldown_seconds=300` (4070-4088). But genuinely adaptive:
`drop_threshold = max(min_drop_pct, atr_percent * drop_atr_mult)` and
`recovery_threshold = max(min_recovery_pct, atr_percent *
recovery_atr_mult)` (4167-4168) — these scale the actual *entry*
thresholds with live ATR%, not just the exit-stop distance like the other
ATR-sizing strategies.

## Pillar 5 — Position Sizing: Risk-scaled (partial credit)

Same convention as trend_following/volatility_breakout:
`risk_amount = balance * risk_percent`, `position_coins = risk_amount /
(atr * trailing_atr_mult)` (4393-4397). No Decision Score/exposure/
drawdown input.

## Pillar 6 — Trade Management: Full (most elaborate of the six)

Four-way confluence in `_dip_recovery_manage_exit` (4505-4612):
take-profit (4582), monotonic-tighten trailing stop (4538-4541, 4587),
wider emergency stop (4545-4547, 4593), and time-based max-duration exit
(4598), with loss-aware cooldown routing that differs per exit reason.

## Pillar 7 — Strategy Edge Management: Not present

No self-tracked win-rate/streak state in `_dip_recovery_default_state()`
(3989-4006) or anywhere in the function.

## Pillar 8 — Self-Diagnostics: Partial, one notable gap

Warm-up, decline tracking, setup/reversal, and most exit management are
explained (4107, 4184-4224, 4361-4388, 4561-4579). **Gap: no `exp.check()`
for "Emergency stop hit"** despite `emergency_stop` being computed,
included in `exp.update()` (4564), and actually used to trigger sells
(4593) — an operator watching diagnostics would never see this exit
condition being evaluated, only see it fire. Also no `.check()`/`.metric()`
for the entry-side position sizing or fee-viability check (4415-4435).

## Pillar 9 — Performance Expectations: UNDOCUMENTED

No stated trade frequency, win rate, profit factor, drawdown, or
buy-and-hold comparison in the docstring or file. **To be authored at
certification.**

## Pillar 10 — Strategy Proposal Interface: Not present

Returns `TradeSignal` directly (e.g. `trading_engine.py:4460`,
`return TradeSignal(action="buy", ...)`) — no `StrategyProposal` concept
exists yet. This strategy already computes `opportunity_score`
(Pillar 3) and sets `expected_move_pct`/`expected_risk_pct` on its
`TradeSignal` — both useful existing computations to carry forward into
`decision_score`/`expected_edge_estimate` rather than building from
scratch, consistent with this strategy having the best existing
foundation of the six across most pillars.
