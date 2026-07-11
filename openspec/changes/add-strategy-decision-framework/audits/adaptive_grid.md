# Strategy Audit: adaptive_grid

`_strategy_grid`, `backend/app/services/trading_engine.py:1365-2060`
(helpers `_get_grid_state`/`_save_grid_state`/`calculate_atr_proxy`,
2062-2098). Coverage score: **6/14**. Most mechanically complex of the
six (virtual grid inventory, not a single position) — sequenced last
among the "already has a suitability gate" group because of that
implementation complexity, not because it's less urgent.

## Pillar 1 — Theory: UNDOCUMENTED (partial)

Docstring (1372-1408) gives design philosophy ("manufacturing process,"
"converts cash to crypto at favorable prices," long-biased for crypto)
and risk controls — closer to an edge rationale than dca's, but still no
explicit statement of *why* a mean-reversion/range-bound edge should
exist statistically, nor a formal failure-mode discussion beyond "pauses
in trends/high volatility." **Authoring the remaining theory is a
certification task.**

## Pillar 2 — Market Suitability: Full

Real, enforced gate (1579-1611): regime computed from bar closes, checked
against `allowed_regimes` (default `["trend_flat","volatility_medium"]`,
1426); returns `hold` with a `WAITING_REGIME` explain state if
unsuitable. Also has TWO additional capital-preservation gates beyond
regime: a drawdown kill switch (1687-1713) and an ATR-distance kill
switch (1715-1746), both of which force a cooldown period (1542-1576).

## Pillar 3 — Evidence-Based Decision Score: Not present

Entry/exit fires on a single condition: price crossing a pre-computed
grid level (1819, 1823). No multi-factor Decision Score accumulation.

## Pillar 4 — Parameter Adaptation: Partial (second-best of the six)

Genuinely adaptive: grid spacing/range from live ATR (`grid_range = atr *
atr_range_multiplier`, 1621; `grid_spacing = grid_range / grid_count`,
1622), spacing floored to a fee-aware minimum (1635-1639), soft recenter
at 50% half-range drift (1653), and order size scaled by grid depth
(`size_multiplier = depth_multiplier ** (depth-1)`, 1942) — this is
entry-relevant adaptation, not just an exit-stop distance. Hardcoded:
`grid_count=10` (fixed 70/30 buy/sell split, 1430-1431),
`atr_range_multiplier=8.0`, `base_order_size_percent=5`,
`depth_multiplier=1.5`, `max_drawdown_percent=15`,
`kill_atr_multiplier=3.0`, `cooldown_after_kill_hours=2`.

## Pillar 5 — Position Sizing: Flat base + depth multiplier (partial credit)

`order_size_usd = min(initial_capital * base_order_size_pct *
size_multiplier, balance * _BUY_BALANCE_FRACTION)` (1943-1946). Base is a
fixed % of *initial* capital, scaled only by grid depth — not by any
Decision Score, portfolio exposure, or correlation. Recent
drawdown is used only as a kill-switch trigger (1689), never to
proactively scale size down before the kill threshold.

## Pillar 6 — Trade Management: Partial

Every completed bar re-evaluates regime (1584), drawdown/ATR-distance
kill checks (1687, 1718), and soft-recenters (1648-1664) — real
continuous re-evaluation at the grid level. But it operates on virtual
grid inventory, not an open position with stop/target/partial-profit
mechanics; individual fills exit only via opposing grid-level crossings
(1958-2060), not thesis-invalidation logic.

## Pillar 7 — Strategy Edge Management: Not present (telemetry only)

Tracks `kill_switch_count`, `lifetime_return_pct`,
`lifetime_max_drawdown_pct` (1451-1457, 1672-1685) but only as telemetry
— never read back to modulate future sizing/pausing beyond the fixed
drawdown/ATR thresholds already covered under Pillar 2. The existing kill
switches are a real circuit-breaker mechanism (a good foundation to build
Pillar 7 on top of) but are fixed-threshold, not win-rate/expectancy-based
self-awareness as the standard requires.

## Pillar 8 — Self-Diagnostics: Partial, one significant gap

Covers bar-wait, cooldown, regime block, and full pre-decision math
(1496, 1554, 1600, 1830-1900). **Gap: both kill switches (drawdown,
1687-1713; ATR-distance, 1715-1746) only `logger.warning`/return — neither
calls `self._explain()` at all**, so a kill switch firing is completely
invisible to the diagnostics UI, the single largest Pillar 8 gap found in
the audit. Also no `.check()`/`.metric()` on the final sizing math or the
virtual cash/crypto sufficiency checks (1961, 2026).

## Pillar 9 — Performance Expectations: UNDOCUMENTED

Docstring lists risk controls but no expected trade frequency, holding
time, win rate, profit factor, drawdown target, or buy-and-hold
comparison. **To be authored at certification.**

## Pillar 10 — Strategy Proposal Interface: Not present

Returns `TradeSignal` directly per grid-level fill — no `StrategyProposal`
concept exists yet. This strategy's virtual grid-inventory model (not a
single position) is the least obvious fit for a per-tick `direction`
value of the six — see `../tasks.md` Phase 5.6 for the specific
`HOLD`-vs-"no level crossed" question its migration needs to resolve.
