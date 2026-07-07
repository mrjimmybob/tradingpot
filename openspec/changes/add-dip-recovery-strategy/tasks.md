## 1. Implementation
- [x] 1.1 Implement `_strategy_dip_recovery` core engine method and lifecycle state
      machine (IDLE/TRACKING_DROP/WAITING_REVERSAL/ENTRY_ARMED/LONG_OPEN/COOLDOWN),
      including adaptive ATR-based thresholds, confirmation filters, entry sizing, exit
      management, loss-aware cooldown, setup expiry, and spike guard
      (`backend/app/services/trading_engine.py`).
- [x] 1.2 Wire persistence: add `_dip_recovery_states` to `_PERSISTED_STATE_ATTRS` and
      register `dip_recovery` in `_get_strategy_executor`'s dispatch table.
- [x] 1.3 Integrate Auto Mode: add `dip_recovery` to `_get_strategy_capabilities`, add the
      shared `_dip_recovery_score_from_ratios` helper, and add the `dip_recovery` branch
      to `_compute_opportunity_score`.
- [x] 1.4 Register the strategy in configuration/validation: `StrategyInfo` entry in
      `app/routers/config.py`, `validate_dip_recovery_params` registered in
      `app/routers/bots.py`, capacity entry in `app/services/strategy_capacity.py`.
- [x] 1.5 Add regression tests covering all 12 required scenarios plus parameter
      validation and end-to-end API wiring (`backend/tests/test_dip_recovery_strategy.py`,
      26 tests).
- [x] 1.6 Run the full backend suite to confirm no regressions (899 passed) and write this
      OpenSpec change record.
