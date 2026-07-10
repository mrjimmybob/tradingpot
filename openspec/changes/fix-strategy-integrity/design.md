## Context
Two independent defects surfaced in the same audit and touch the same code path (`TradingEngine._strategy_auto`
and position lifecycle), so they are bundled into one change with two capability deltas.

`funding_carry` (backend/app/services/trading_engine.py:3345) is registered in:
- `_ALPHA_STRATEGIES` (trading_engine.py:154-161)
- the strategy dispatch map (trading_engine.py:1133: `"funding_carry": self._strategy_funding_carry`)
- `validate_funding_carry_params` (trading_engine.py:306), imported and wired into
  `_STRATEGY_PARAM_VALIDATORS` in `backend/app/routers/bots.py:15,28`
- `_get_strategy_capabilities()` (trading_engine.py:6423-6430), the table Auto Mode uses for regime-based
  eligibility filtering
- the `StrategyInfo` catalog in `backend/app/routers/config.py:117-130`, served by `GET /strategies`
- a strategy-name branch at trading_engine.py:5797

Its only unique dependency, `backend/app/services/funding_diagnostic.py` (`compute_funding_stats`), has no
other caller anywhere in `backend/app` — confirmed by search — so it becomes dead code once the strategy is
removed and is deleted rather than left orphaned.

No DB enum or foreign key lists valid strategy names. `Bot.strategy` is `String(50)`, validated only at the
API layer against the `STRATEGIES` list; `Order.strategy_used` and `Trade.strategy_used` are plain nullable
strings — so nothing else in the schema needs to change shape to remove `funding_carry` rows. Per explicit
operator decision, existing `funding_carry` data is disposable test data (it never traded real capital) and
is deleted outright rather than preserved: the one bot that ever used it (`id=7`, `TestBot13-FC`) is a
dry-run test bot, and its positions/orders/trades are deleted along with it. No compatibility branch is kept
for rendering a `funding_carry` value that might still appear in the data.

For ownership: `auto_state["current_strategy"]` (trading_engine.py:4930 on) already survives a service
restart today — it rides along in `Bot.strategy_state` via `_PERSISTED_STATE_ATTRS` (trading_engine.py:125-148)
and `_collect_bot_state`/`_restore_bot_state` (trading_engine.py:8503-8549). The gap is not restart durability
of the bot-level pointer; it is that the pointer is the *only* record of ownership, is mutable at any tick,
and is never cross-checked against the actual open `Position` row. The switch branch at
trading_engine.py:5162-5197 (`elif best_strategy != current_strategy`) reassigns it whenever a competitor
strategy scores higher, with no check for `has_open_position` — unlike the `pinned_for_exit` branch just
above it (trading_engine.py:5102-5141), which already does the right thing for the *other* way ownership can
be threatened (current strategy losing eligibility). Both paths need to agree, and the source of truth needs
to move from a mutable in-memory/JSON pointer to a fact recorded on the position itself.

## Goals / Non-Goals
- Goals:
  - Remove funding_carry from every selection, validation, and capability surface without breaking
    historical reads.
  - Make "which strategy owns this open position" a persisted fact on the position, not an inference from
    logs or a mutable bot-level pointer.
  - Make Auto Mode's dispatch during an open position unconditionally follow that persisted fact.
- Non-Goals:
  - Multi-position-per-bot support.
  - Rewriting strategy alpha logic or scoring/eligibility rules used when no position is open.
  - Building the backtester itself (tracked separately in `add-historical-backtesting`), though its
    simulated position bookkeeping is required to reuse the same `owning_strategy` field rather than invent
    a parallel mechanism.

## Decisions
- Decision: Add `owning_strategy` (String), `entry_reason` (String), and `entry_strategy_state` (JSON,
  nullable) columns to `Position`, populated at the moment a position-opening order is recorded, via an
  additive migration (`backend/migrations/006_add_position_ownership.sql`), following this repo's existing
  additive-only, hand-written SQL migration convention (see `backend/migrations/README.md`) — not Alembic.
  - Why: `Position` is the one row that exists for exactly the lifetime ownership needs to be tracked over;
    tying the fact to it (rather than to `Bot` or to log text) makes it queryable, restart-safe by
    construction, and independently verifiable against `auto_state["current_strategy"]`.
- Decision: In `_strategy_auto`, once `has_open_position` is true, read `owning_strategy` off the open
  `Position` and dispatch to that executor unconditionally — do not consult the scoring/eligibility branches
  at all while a position is open. Reselection (scoring, hysteresis, switching) only runs when
  `has_open_position` is false.
  - Why: Removes the class of bug entirely rather than patching the one branch that was missing a check;
    a future third way to flip the pointer cannot reintroduce the same defect if the dispatch never reads
    the pointer while a position is open.
  - Consequence: `auto_state["current_strategy"]` becomes a pure entry-allocation hint ("what would Auto pick
    next, if free to choose") and is no longer authoritative for dispatch during an open position. Keep it
    for observability/UI (`GET /reports/strategy-performance`'s `current_strategy` field) but stop treating
    it as ownership.
- Decision: Self-heal on mismatch. If `auto_state["current_strategy"]` and the open position's
  `owning_strategy` ever disagree (e.g., a crash between opening the position and persisting state), the
  position's `owning_strategy` wins and `auto_state["current_strategy"]` is corrected to match on the next
  tick, with a warning logged.
  - Why: The persisted position fact is closer to the source of truth (written transactionally with the
    order that opened the position) than the bot-level JSON snapshot (written on a slower save cadence).

## Risks / Trade-offs
- Adding columns to `Position` requires every code path that opens a position to be updated to populate them;
  missing one silently produces a position with `owning_strategy=NULL`. Mitigate with a test asserting every
  strategy executor that can open a position populates it, and treat `NULL` as "unowned — do not auto-switch
  away from whatever is currently selected" rather than crashing.
- Because `funding_carry`'s data and bot are deleted outright (not left in place), there is no "stranded
  open position under a removed strategy" case to handle for this change — the position is deleted along
  with the bot. This does mean the removal order matters operationally: stop/delete the `funding_carry` bot
  and its rows *before* or *atomically with* removing the executor, not after, so nothing references a
  nonexistent strategy mid-deploy. In general (for any *future* strategy removal that isn't paired with a
  full data purge), the executor function would need to stay reachable for exit-only purposes after removal
  from the selectable catalog — that general rule is not needed here since there is no data left to exit.

## Migration Plan
1. Delete the `funding_carry` bot (`id=7`, `TestBot13-FC`) and its positions/orders/trades from the database
   (see tasks.md) — disposable dry-run test data per explicit operator decision.
2. Add the additive `Position` columns (nullable, no backfill required for remaining bots).
3. Populate `owning_strategy`/`entry_reason` at every position-open call site in `trading_engine.py`.
4. Change `_strategy_auto` dispatch to read ownership from the open position rather than
   `auto_state["current_strategy"]` when a position is open.
5. Remove `funding_carry` from catalog/validation/capabilities/dispatch/state-handling entirely — no
   compatibility branch retained.
6. Delete `funding_diagnostic.py`, `FUNDING_CARRY.md`, and the funding_carry-only tests; update the
   funding_carry-referencing tests to use a remaining valid strategy.

## Open Questions
- None blocking. Confirmed in this environment: bot id 7 (`TestBot13-FC`, dry-run, `is_dry_run=1`, budget
  $100) was `RUNNING` with strategy `funding_carry` and one open dry-run position (~0.00031 BTC on
  `BTC/USDT`) at audit time. Per explicit operator decision this bot and its data are deleted as disposable
  test data rather than flattened/preserved.
