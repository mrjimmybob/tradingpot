"""The measurement/optimisation boundary, enforced mechanically.

``add-strategy-validation-tooling`` is binding on one point above all others:
the tooling measures a strategy's given, fixed configuration and explains the
result - it never optimises, never searches a parameter space, and never
writes a strategy parameter anywhere. A comment saying so is worth nothing the
first time someone adds "just a small sweep"; these tests walk the AST of
*every* module under ``app/backtesting/validation/`` on every run, so the
boundary is checked against the code that actually exists rather than the code
that existed when the docstring was written.

Structural checks (this file's first half) are deliberately about *shape* -
what the package may import, name, and assign. Behavioural checks (second
half) prove the same boundary from the outside: the operator's dict comes back
untouched, and every window sees byte-identical parameters.
"""
from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

from app.backtesting.candle import Candle
from app.backtesting.data_provider import CsvHistoricalDataProvider
from app.backtesting.engine import BacktestEngine
from app.backtesting.execution_model import BacktestExecutionModel
from app.backtesting.validation import (
    FixedConfig,
    Measurement,
    MeasurementSpan,
    measure_fixed_config,
    select_candles,
)

_VALIDATION_PACKAGE = Path(__file__).resolve().parents[1] / "app" / "backtesting" / "validation"


def _validation_modules() -> List[Tuple[Path, ast.Module]]:
    """Every Python module in the validation package, parsed.

    Discovered by walking the directory rather than listing filenames, so a
    module added tomorrow is covered by these guards automatically - the guard
    must not be something a new file can be added *around*.
    """
    modules = []
    for path in sorted(_VALIDATION_PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        modules.append((path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))))
    return modules


def _walk(modules: List[Tuple[str, ast.Module]]) -> Iterator[Tuple[str, ast.AST]]:
    for label, tree in modules:
        for node in ast.walk(tree):
            yield label, node


def _package_modules() -> List[Tuple[str, ast.Module]]:
    return [(path.name, tree) for path, tree in _validation_modules()]


def _parse(source: str, label: str = "<synthetic>") -> List[Tuple[str, ast.Module]]:
    return [(label, ast.parse(source))]


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


# Vocabulary that only appears once code has started *choosing* rather than
# measuring. Matched against defined names and argument names, never against
# free text, so a docstring explaining "this does not optimise" cannot trip it.
_OPTIMISATION_FRAGMENTS = (
    "optimi",  # optimise / optimize / optimizer / optimal
    "tune",
    "sweep",
    "grid_search",
    "gridsearch",
    "param_search",
    "search_space",
    "param_grid",
    "param_sets",
    "candidate",
    "calibrat",
    "best_param",
)


def _is_optimisation_name(name: str) -> bool:
    lowered = name.lower()
    return any(fragment in lowered for fragment in _OPTIMISATION_FRAGMENTS)


# --- detectors -------------------------------------------------------------
# Each takes parsed modules and returns a list of offences. Written as plain
# functions rather than inlined into tests so the meta-tests at the bottom can
# run them against deliberately-offending source and prove they actually bite -
# a guard nobody has ever seen fail is not a guard.


def _param_writeback_offenders(modules) -> List[str]:
    """Writes *through* an object to a strategy-parameter name.

    A bare ``params = ...`` (or an annotated dataclass field declaration
    ``params: Mapping = ...``) binds a local/field *name*; it cannot reach a
    bot row, a table, or a config file. What can is ``bot.strategy_params = ...``
    or ``config["params"] = ...``, so those are what this looks for.
    """
    offenders = []
    for label, node in _walk(modules):
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            written = None
            if isinstance(target, ast.Attribute):
                written = target.attr
            elif isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                written = target.slice.value
            if written in ("strategy_params", "params"):
                offenders.append(f"{label}:{node.lineno} -> {written}")
    return offenders


def _setattr_writeback_offenders(modules) -> List[str]:
    """The back door around the assignment detector: ``setattr`` and
    ``object.__setattr__`` write attributes with no assignment node at all."""
    offenders = []
    for label, node in _walk(modules):
        if not isinstance(node, ast.Call) or _called_name(node) not in ("setattr", "__setattr__"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value == "strategy_params":
                offenders.append(f"{label}:{node.lineno} setattr(..., 'strategy_params')")
    return offenders


_FORBIDDEN_IMPORTS = {
    "sqlalchemy",
    "aiosqlite",
    "sqlite3",
    "app.models",
    "app.database",
    "app.core.config",
    "app.config",
    "app.services.bot_service",
}


def _persistence_import_offenders(modules) -> List[str]:
    offenders = []
    for label, node in _walk(modules):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
        else:
            continue
        for name in imported:
            if name in _FORBIDDEN_IMPORTS or name.split(".")[0] in _FORBIDDEN_IMPORTS:
                offenders.append(f"{label}:{node.lineno} imports {name}")
    return offenders


def _file_write_offenders(modules) -> List[str]:
    offenders = []
    for label, node in _walk(modules):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        if name not in ("open", "write_text", "write_bytes", "unlink", "remove"):
            continue
        if name == "open":
            mode_args = list(node.args[1:2]) + [
                kw.value for kw in node.keywords if kw.arg == "mode"
            ]
            modes = [a.value for a in mode_args if isinstance(a, ast.Constant)]
            if not modes:
                continue  # open(path) defaults to read
            if all("r" in m and "+" not in m for m in modes):
                continue  # explicitly read-only
        offenders.append(f"{label}:{node.lineno} calls {name}(...)")
    return offenders


def _optimisation_definition_offenders(modules) -> List[str]:
    offenders = []
    for label, node in _walk(modules):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if _is_optimisation_name(node.name):
                offenders.append(f"{label}:{node.lineno} defines {node.name}")
    return offenders


def _parameter_set_argument_offenders(modules) -> List[str]:
    """The signature-level tell for a search: an argument holding *many*
    configurations. A measurement function takes exactly one."""
    offenders = []
    for label, node in _walk(modules):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
        for arg in all_args:
            if _is_optimisation_name(arg.arg):
                offenders.append(f"{label}:{node.lineno} {node.name}({arg.arg})")
    return offenders


class TestPackageIsDiscoverable:
    def test_guard_actually_finds_the_package(self):
        """A guard that silently scans zero files passes forever. Anchor it."""
        modules = _validation_modules()
        assert modules, f"No modules found under {_VALIDATION_PACKAGE}"
        names = {path.name for path, _ in modules}
        assert "__init__.py" in names
        assert "measurement.py" in names


class TestNoParameterWriteback:
    """"SHALL NOT write ``strategy_params`` to any bot, database, or config."""

    def test_no_module_assigns_to_strategy_params(self):
        offenders = _param_writeback_offenders(_package_modules())
        assert not offenders, (
            f"The validation package must never assign a strategy parameter: {offenders}"
        )

    def test_no_module_sets_strategy_params_via_setattr(self):
        offenders = _setattr_writeback_offenders(_package_modules())
        assert not offenders, f"No strategy_params may be set via setattr: {offenders}"

    def test_no_module_imports_a_persistence_layer(self):
        """No ORM, no session, no settings module - there is simply nothing
        here that *could* reach a bot row or a config file. The engine owns its
        own isolated in-memory DB; the validation package never opens one."""
        offenders = _persistence_import_offenders(_package_modules())
        assert not offenders, (
            f"The validation package must import no persistence layer: {offenders}"
        )

    def test_no_module_opens_a_file_for_writing(self):
        """Reports are printed, not written over anything. Any future report
        artifact must be an explicit, reviewed change to this guard - not an
        accident that lands next to a config file."""
        offenders = _file_write_offenders(_package_modules())
        assert not offenders, f"The validation package must not write files: {offenders}"


class TestNoParameterSearch:
    """"SHALL NOT search a parameter space, tune, or select a better set."""

    def test_no_definition_uses_optimisation_vocabulary(self):
        offenders = _optimisation_definition_offenders(_package_modules())
        assert not offenders, (
            f"Optimisation belongs to a separate, future change: {offenders}"
        )

    def test_no_function_accepts_a_set_of_parameter_sets(self):
        offenders = _parameter_set_argument_offenders(_package_modules())
        assert not offenders, f"No parameter-search argument may exist: {offenders}"

    def test_public_api_exposes_no_optimisation_entry_point(self):
        import app.backtesting.validation as validation

        exported = list(getattr(validation, "__all__", []))
        assert exported, "The validation package must declare an explicit __all__"
        for name in exported:
            assert not _is_optimisation_name(name), f"__all__ exposes {name!r}"
            assert hasattr(validation, name), f"__all__ names missing symbol {name!r}"


class TestTheGuardsActuallyBite:
    """Meta-tests: each detector, run against source that violates it.

    Without these, every guard above would keep passing if a detector were
    quietly broken (a renamed AST field, an inverted condition) - the suite
    would report a boundary it was no longer checking.
    """

    def test_detects_writeback_to_a_bot_row(self):
        assert _param_writeback_offenders(_parse("bot.strategy_params = tuned\n"))

    def test_detects_writeback_through_a_dict_key(self):
        assert _param_writeback_offenders(_parse("config['strategy_params'] = tuned\n"))

    def test_allows_a_local_binding_and_a_field_declaration(self):
        assert not _param_writeback_offenders(_parse("params = {}\n"))
        assert not _param_writeback_offenders(
            _parse("class C:\n    params: dict = {}\n")
        )

    def test_detects_setattr_writeback(self):
        assert _setattr_writeback_offenders(
            _parse("setattr(bot, 'strategy_params', tuned)\n")
        )
        assert _setattr_writeback_offenders(
            _parse("object.__setattr__(bot, 'strategy_params', tuned)\n")
        )

    def test_detects_persistence_imports(self):
        assert _persistence_import_offenders(_parse("import sqlalchemy\n"))
        assert _persistence_import_offenders(_parse("from app.models import Bot\n"))
        assert _persistence_import_offenders(_parse("from sqlalchemy.orm import Session\n"))
        assert not _persistence_import_offenders(
            _parse("from app.backtesting.engine import BacktestEngine\n")
        )

    def test_detects_file_writes_but_permits_reads(self):
        assert _file_write_offenders(_parse("open(p, 'w')\n"))
        assert _file_write_offenders(_parse("open(p, mode='a')\n"))
        assert _file_write_offenders(_parse("open(p, 'r+')\n"))
        assert _file_write_offenders(_parse("path.write_text(s)\n"))
        assert not _file_write_offenders(_parse("open(p, 'r')\n"))
        assert not _file_write_offenders(_parse("open(p)\n"))

    def test_detects_optimisation_definitions(self):
        assert _optimisation_definition_offenders(_parse("def optimize_params():\n    pass\n"))
        assert _optimisation_definition_offenders(_parse("def tune_strategy():\n    pass\n"))
        assert _optimisation_definition_offenders(_parse("class ParamSweep:\n    pass\n"))
        assert _optimisation_definition_offenders(
            _parse("async def find_best_params():\n    pass\n")
        )
        assert not _optimisation_definition_offenders(
            _parse("def measure_fixed_config():\n    pass\n")
        )

    def test_detects_parameter_set_arguments(self):
        assert _parameter_set_argument_offenders(_parse("def f(param_grid):\n    pass\n"))
        assert _parameter_set_argument_offenders(_parse("def f(*, candidates):\n    pass\n"))
        assert _parameter_set_argument_offenders(_parse("def f(search_space=None):\n    pass\n"))
        assert not _parameter_set_argument_offenders(_parse("def f(config, span):\n    pass\n"))


class TestEngineIsReusedUnmodified:
    def test_measurement_calls_run_candles_and_nothing_else(self):
        """Spec: "it calls ``BacktestEngine.run_candles`` unmodified". The
        measurement module must therefore reach the engine through that one
        entry point - not by reimplementing replay or poking at internals."""
        source = (_VALIDATION_PACKAGE / "measurement.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        called_engine_attrs = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "engine"
        }
        assert called_engine_attrs == {"run_candles"}, (
            f"Measurement must use only run_candles, got {called_engine_attrs}"
        )


# ---------------------------------------------------------------------------
# Behavioural half: the same boundary, proved from the outside.
# ---------------------------------------------------------------------------


def _oscillating_candles(n=240, base=100.0, amplitude=0.05, start_ms=1704067200000, step_ms=60_000):
    """Deterministic sine wave - the same fixture shape the engine's own tests
    use, so mean_reversion gets both entries and exits with no randomness."""
    candles = []
    for i in range(n):
        price = base * (1 + amplitude * math.sin(i / 12.0))
        candles.append(Candle(
            timestamp=start_ms + i * step_ms,
            datetime="d", symbol="TESTUSD",
            open=price, high=price * 1.001, low=price * 0.999, close=price,
            base_volume=1.0, quote_volume=price, trade_count=1,
        ))
    return candles


_MEAN_REVERSION_PARAMS = {
    "bar_interval_seconds": 0, "regime_filter_enabled": False,
    "bollinger_period": 10, "atr_period": 10, "cooldown_seconds": 0,
    "decision_score_threshold": 0.0,
}


def _engine() -> BacktestEngine:
    return BacktestEngine(
        data_provider=CsvHistoricalDataProvider(),
        execution_model=BacktestExecutionModel(fee_pct=0.1),
    )


def _config(**overrides) -> FixedConfig:
    params = dict(_MEAN_REVERSION_PARAMS)
    params.update(overrides)
    return FixedConfig(strategy="mean_reversion", trading_pair="TEST/USD", params=params)


class TestFixedConfigIsImmutable:
    def test_params_cannot_be_mutated_through_the_config(self):
        config = _config()
        with pytest.raises(TypeError):
            config.params["bollinger_period"] = 999  # type: ignore[index]

    def test_config_fields_cannot_be_reassigned(self):
        config = _config()
        with pytest.raises(Exception):
            config.strategy = "trend_following"  # type: ignore[misc]

    def test_config_deep_copies_the_operators_dict(self):
        """Mutating the dict *after* constructing the config must not change
        what the config measures - the operator's dict and the measured
        configuration are separate objects from the moment of construction."""
        operator_params = dict(_MEAN_REVERSION_PARAMS)
        config = FixedConfig("mean_reversion", "TEST/USD", operator_params)
        before = config.params_fingerprint

        operator_params["bollinger_period"] = 999
        assert config.params["bollinger_period"] == 10
        assert config.params_fingerprint == before

    def test_params_for_run_hands_out_a_fresh_copy_each_time(self):
        config = _config()
        first = config.params_for_run()
        second = config.params_for_run()
        assert first == second
        assert first is not second

        first["bollinger_period"] = 999
        assert config.params["bollinger_period"] == 10
        assert config.params_for_run()["bollinger_period"] == 10

    def test_fingerprint_is_order_independent_and_value_sensitive(self):
        a = FixedConfig("s", "P", {"x": 1, "y": 2})
        b = FixedConfig("s", "P", {"y": 2, "x": 1})
        c = FixedConfig("s", "P", {"x": 1, "y": 3})
        assert a.params_fingerprint == b.params_fingerprint
        assert a.params_fingerprint != c.params_fingerprint

    def test_rejects_a_nonsensical_configuration(self):
        with pytest.raises(ValueError):
            FixedConfig("", "TEST/USD", {})
        with pytest.raises(ValueError):
            FixedConfig("mean_reversion", "", {})
        with pytest.raises(ValueError):
            FixedConfig("mean_reversion", "TEST/USD", {}, starting_balance=0.0)


class TestMeasurementSpan:
    def test_rejects_an_inverted_span(self):
        with pytest.raises(ValueError):
            MeasurementSpan(start_ms=2_000, end_ms=1_000)

    def test_unbounded_span_contains_everything(self):
        span = MeasurementSpan()
        assert span.contains(0)
        assert span.contains(10**13)

    def test_span_bounds_are_inclusive_on_both_ends(self):
        span = MeasurementSpan(start_ms=100, end_ms=200)
        assert span.contains(100) and span.contains(200)
        assert not span.contains(99) and not span.contains(201)

    def test_select_candles_slices_without_reordering(self):
        candles = _oscillating_candles(n=10)
        span = MeasurementSpan(start_ms=candles[2].timestamp, end_ms=candles[6].timestamp)
        selected = select_candles(candles, span)
        assert [c.timestamp for c in selected] == [c.timestamp for c in candles[2:7]]


class TestMeasurementIsReadOnly:
    @pytest.mark.asyncio
    async def test_measurement_does_not_mutate_the_operators_params(self):
        operator_params = dict(_MEAN_REVERSION_PARAMS)
        snapshot = copy.deepcopy(operator_params)
        config = FixedConfig("mean_reversion", "TEST/USD", operator_params)

        await measure_fixed_config(_engine(), _oscillating_candles(), config)

        assert operator_params == snapshot
        assert dict(config.params) == snapshot

    @pytest.mark.asyncio
    async def test_measurement_does_not_mutate_the_candles_it_is_given(self):
        candles = _oscillating_candles()
        snapshot = list(candles)
        await measure_fixed_config(_engine(), candles, _config())
        assert candles == snapshot

    @pytest.mark.asyncio
    async def test_repeated_measurement_of_one_config_is_identical(self):
        """Determinism is what makes a measurement a measurement. Two runs of
        the same fixed config over the same candles must agree exactly - if
        they did not, no per-window comparison downstream would mean anything."""
        candles = _oscillating_candles()
        config = _config()
        first = await measure_fixed_config(_engine(), candles, config)
        second = await measure_fixed_config(_engine(), candles, config)

        assert first.params_fingerprint == second.params_fingerprint
        assert first.num_trades == second.num_trades
        assert first.ending_balance == pytest.approx(second.ending_balance)
        assert first.expectancy_per_trade == pytest.approx(second.expectancy_per_trade)


class TestMeasurementResult:
    @pytest.mark.asyncio
    async def test_measurement_carries_provenance_and_metrics(self):
        candles = _oscillating_candles()
        config = _config()
        measurement = await measure_fixed_config(_engine(), candles, config)

        assert isinstance(measurement, Measurement)
        assert measurement.strategy == "mean_reversion"
        assert measurement.trading_pair == "TEST/USD"
        assert measurement.params_fingerprint == config.params_fingerprint
        assert measurement.num_candles == len(candles)
        assert measurement.first_candle_ms == candles[0].timestamp
        assert measurement.last_candle_ms == candles[-1].timestamp
        assert measurement.starting_balance == config.starting_balance
        assert measurement.num_trades > 0
        assert not measurement.is_empty
        assert len(measurement.trades) == measurement.num_trades
        assert isinstance(measurement.trades, tuple)
        assert isinstance(measurement.equity_curve, tuple)

    @pytest.mark.asyncio
    async def test_metrics_match_the_engines_own_result(self):
        """The measurement layer must not recompute (or "improve") any metric -
        it is a projection of exactly what BacktestEngine reported."""
        candles = _oscillating_candles()
        config = _config()

        direct = await _engine().run_candles(
            candles, "TEST/USD", "mean_reversion", dict(_MEAN_REVERSION_PARAMS), 10_000.0,
            quiet=True,
        )
        measured = await measure_fixed_config(_engine(), candles, config)

        assert measured.num_trades == direct.num_trades
        assert measured.win_rate_pct == pytest.approx(direct.win_rate)
        assert measured.profit_factor == pytest.approx(direct.profit_factor)
        assert measured.expectancy_per_trade == pytest.approx(direct.expectancy_per_trade)
        assert measured.max_drawdown_pct == pytest.approx(direct.max_drawdown_pct)
        assert measured.ending_balance == pytest.approx(direct.ending_balance)
        assert measured.total_return_pct == pytest.approx(direct.total_return_pct)
        assert measured.buy_and_hold_return_pct == pytest.approx(direct.buy_and_hold_return_pct)
        assert measured.total_fees_paid == pytest.approx(direct.total_fees_paid)

    @pytest.mark.asyncio
    async def test_measuring_a_sub_span_measures_only_those_candles(self):
        candles = _oscillating_candles()
        span = MeasurementSpan(start_ms=candles[50].timestamp, end_ms=candles[150].timestamp)
        measurement = await measure_fixed_config(_engine(), candles, _config(), span)

        assert measurement.num_candles == 101
        assert measurement.first_candle_ms == candles[50].timestamp
        assert measurement.last_candle_ms == candles[150].timestamp
        for trade in measurement.trades:
            assert span.contains(trade.entry_timestamp)
            assert span.contains(trade.exit_timestamp)

    @pytest.mark.asyncio
    async def test_an_unmeasurable_span_raises_rather_than_fabricating_zeros(self):
        candles = _oscillating_candles()
        span = MeasurementSpan(start_ms=candles[0].timestamp, end_ms=candles[0].timestamp)
        with pytest.raises(ValueError, match="at least 2"):
            await measure_fixed_config(_engine(), candles, _config(), span)

    @pytest.mark.asyncio
    async def test_a_span_with_no_trades_is_reported_not_hidden(self):
        """An empty window is a real result. It must come back as a measurement
        with zero trades, not an exception and not a silently skipped window."""
        candles = _oscillating_candles(n=240, amplitude=0.0)  # flat: nothing to revert to
        measurement = await measure_fixed_config(_engine(), candles, _config())
        assert measurement.num_trades == 0
        assert measurement.is_empty
        assert measurement.expectancy_per_trade == 0.0
