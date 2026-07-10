## 1. Purge funding_carry data (disposable test data, per explicit operator decision)
- [x] 1.1 Delete bot id 7 (`TestBot13-FC`) and all `positions`/`orders`/`trades`/`strategy_performance_metrics`/
      `strategy_rotations` rows referencing it, plus any other row anywhere in the database with
      `strategy`/`strategy_used = 'funding_carry'`.

## 2. Position ownership schema
- [x] 2.1 Add `owning_strategy` (String, nullable), `entry_reason` (String, nullable), and
      `entry_strategy_state` (JSON, nullable) columns to `Position` (backend/app/models/position.py).
- [x] 2.2 Add `backend/migrations/006_add_position_ownership.sql` (additive, per this repo's migration
      convention) and wire it into `run_migrations.py`.

## 3. Auto Mode ownership enforcement
- [x] 3.1 Populate `owning_strategy`/`entry_reason`/`entry_strategy_state` at every position-open call site in
      `trading_engine.py` (each `_strategy_*` executor that can open a position).
- [x] 3.2 In `_strategy_auto`, when `has_open_position` is true, read `owning_strategy` off the open `Position`
      and dispatch to that executor unconditionally, bypassing the scoring/eligibility/switch branches
      entirely (trading_engine.py:5085-5211).
- [x] 3.3 Implement the self-heal rule: if `auto_state["current_strategy"]` disagrees with the open position's
      `owning_strategy`, correct the in-memory/persisted pointer to match and log a warning.
- [x] 3.4 Confirm reselection (scoring, hysteresis, switching) only executes when no position is open.

## 4. Remove funding_carry
- [x] 4.1 Remove the `StrategyInfo` entry from `STRATEGIES` in `backend/app/routers/config.py`.
- [x] 4.2 Remove `validate_funding_carry_params` wiring from `backend/app/routers/bots.py`
      (`_STRATEGY_PARAM_VALIDATORS`, import).
- [x] 4.3 Remove from `trading_engine.py`: `_ALPHA_STRATEGIES` entry, dispatch-map entry, the
      `_strategy_funding_carry` method, `validate_funding_carry_params`, the `_get_strategy_capabilities`
      table entry, and the `strategy_name == "funding_carry"` branch.
- [x] 4.4 Delete `backend/app/services/funding_diagnostic.py` (dead code once 4.3 lands — no other caller).
- [x] 4.5 Delete `FUNDING_CARRY.md`; update the `funding_carry` example in `DEPLOYMENT.md` to a remaining
      valid strategy.

## 5. Tests
- [x] 5.1 Delete `test_funding_carry_strategy.py` and `test_funding_diagnostic.py`.
- [x] 5.2 Update every other test file that uses `funding_carry` as a fixture strategy name (see design.md
      for the file list) to use a remaining valid strategy instead, preserving what each test was actually
      checking.
- [x] 5.3 Add a test: `funding_carry` cannot be selected via `POST /bots` or `validate_strategy_params`.
- [x] 5.4 Add a test: a report/UI endpoint rendering a trade with an arbitrary/unrecognized `strategy_used`
      string does not raise (generic robustness regression; `funding_carry` data itself is purged, not kept
      around to test against).
- [x] 5.5 Add a test reproducing the exact bug: Strategy A opens a position; a market move makes Strategy B
      score higher while A is still entry-eligible; assert Auto Mode still dispatches to A until the position
      closes, and B never touches it.
- [x] 5.6 Add a test: after a simulated service restart (state save/reload), ownership still resolves to the
      strategy that opened the position, even if the in-memory `auto_state["current_strategy"]` were wrong.

## 6. Validation
- [x] 6.1 Run the full backend test suite; fix regressions.
- [x] 6.2 `openspec validate fix-strategy-integrity --strict --no-interactive`.
