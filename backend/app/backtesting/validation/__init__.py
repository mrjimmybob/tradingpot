"""Read-only strategy validation tooling (measurement, never optimisation).

This package answers one question and only one question: *how does a strategy
behave when it is run, unchanged, with the exact parameters it is actually
configured with?* It measures and explains; it does not improve.

The boundary is binding (see the ``strategy-validation-measurement`` capability
in ``openspec/specs/``, and the change that introduced it,
``openspec/changes/archive/2026-07-26-add-strategy-validation-tooling/``):

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
from .edge_record import (
    MIN_TRADING_WINDOWS_FOR_A_RECORD,
    ValidatedEdgeRecord,
    build_validated_edge_record,
    edge_record_blockers,
    format_edge_record_report,
)
from .regime import (
    UNCLASSIFIED,
    RegimeBreakdown,
    RegimeBucket,
    RegimeTimeline,
    bucket_trades_by_regime,
    build_regime_timeline,
    canonical_regime_detector,
    format_regime_report,
    regime_label_full,
    regime_label_trend,
)
from .walk_forward import (
    MS_PER_DAY,
    SkippedWindow,
    WalkForwardMeasurement,
    format_walk_forward_report,
    plan_windows,
    resolve_span,
    run_walk_forward,
)

__all__ = [
    # measurement primitive
    "FixedConfig",
    "Measurement",
    "MeasurementSpan",
    "measure_fixed_config",
    "select_candles",
    # out-of-sample measurement across rolling windows
    "MS_PER_DAY",
    "SkippedWindow",
    "WalkForwardMeasurement",
    "format_walk_forward_report",
    "plan_windows",
    "resolve_span",
    "run_walk_forward",
    # regime-conditioned breakdown of measured trades
    "UNCLASSIFIED",
    "RegimeBreakdown",
    "RegimeBucket",
    "RegimeTimeline",
    "bucket_trades_by_regime",
    "build_regime_timeline",
    "canonical_regime_detector",
    "format_regime_report",
    "regime_label_full",
    "regime_label_trend",
    # validated measurement record (reported only, never wired into runtime)
    "MIN_TRADING_WINDOWS_FOR_A_RECORD",
    "ValidatedEdgeRecord",
    "build_validated_edge_record",
    "edge_record_blockers",
    "format_edge_record_report",
]
