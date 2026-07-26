"""Read-only strategy validation tooling (measurement, never optimisation).

This package answers one question and only one question: *how does a strategy
behave when it is run, unchanged, with the exact parameters it is actually
configured with?* It measures and explains; it does not improve.

The boundary is binding (see
``openspec/changes/add-strategy-validation-tooling/proposal.md``):

- It SHALL NOT search a parameter space, tune parameters, or select a "better"
  parameter set.
- It SHALL NOT write ``strategy_params`` (or any strategy parameter) to any
  bot, database, or configuration - ever.
- It SHALL call :meth:`BacktestEngine.run_candles` unmodified, inheriting that
  engine's proven no-lookahead guarantee rather than re-implementing replay.

That boundary is enforced structurally, not just documented: this package has
no code path that constructs an alternative parameter set and no import of any
persistence layer, and ``backend/tests/test_validation_measurement_boundary.py``
walks every module's AST on every test run to keep it that way.

Parameter optimisation is a separate, future change. It does not live here.
"""
from .measurement import (
    FixedConfig,
    Measurement,
    MeasurementSpan,
    measure_fixed_config,
    select_candles,
)

__all__ = [
    "FixedConfig",
    "Measurement",
    "MeasurementSpan",
    "measure_fixed_config",
    "select_candles",
]
