"""The validated measurement record, and the line it must not cross.

Two things are under test. First, that a record built from genuinely
out-of-sample windows is constructible, correct, and carries the framework's
validated-source marker - the contract ``add-strategy-decision-framework``
defined and left unfulfilled. Second, and more important, that producing it
changes nothing at runtime: no live ``StrategyProposal`` gains an
``expected_edge_estimate``, because wiring the record into decisions is a
behaviour change reserved for a separate, reviewable change.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import pytest

from app.backtesting.candle import Candle
from app.backtesting.results import TradeRecord, compute_result
from app.backtesting.validation import (
    FixedConfig,
    MeasurementSpan,
    ValidatedEdgeRecord,
    build_validated_edge_record,
    edge_record_blockers,
    format_edge_record_report,
    run_walk_forward,
)
from app.backtesting.validation.edge_record import MIN_TRADING_WINDOWS_FOR_A_RECORD
from app.services.strategy_framework.proposal import (
    VALIDATED_EDGE_SOURCE,
    EdgeEstimate,
    InvalidProposalError,
    StrategyProposal,
)

DAY = 86_400_000
START_MS = 1704067200000

_APP_DIR = Path(__file__).resolve().parents[1] / "app"
_VALIDATION_DIR = _APP_DIR / "backtesting" / "validation"


def _candles(count: int, step_ms: int = DAY // 4) -> List[Candle]:
    return [
        Candle(
            timestamp=START_MS + i * step_ms, datetime="d", symbol="TESTUSD",
            open=100.0, high=100.1, low=99.9, close=100.0,
            base_volume=1.0, quote_volume=100.0, trade_count=1,
        )
        for i in range(count)
    ]


class _ScriptedEngine:
    """Returns pre-agreed per-window results so the pooled arithmetic below is
    hand-checkable rather than dependent on a strategy's behaviour."""

    def __init__(self, pnls_per_window: List[List[float]]):
        self._script = list(pnls_per_window)
        self.calls = 0

    async def run_candles(self, candles, trading_pair, strategy, params, balance, quiet=True):
        self.calls += 1
        pnls = self._script[self.calls - 1] if self.calls <= len(self._script) else []
        ts = candles[0].timestamp
        trades = [
            TradeRecord(
                entry_timestamp=ts + i, exit_timestamp=ts + i + 1, strategy=strategy,
                entry_price=100.0, exit_price=100.0 + pnl, fees=0.0,
                gross_pnl=pnl, net_pnl=pnl, exit_reason="scripted",
            )
            for i, pnl in enumerate(pnls)
        ]
        equity = [(ts, balance), (ts + 1, balance + sum(pnls))]
        return compute_result(balance, balance + sum(pnls), trades, equity, 0.0, 0.0)


async def _walk_forward(script, window_days=25, step_days=None, candle_count=400):
    return await run_walk_forward(
        _ScriptedEngine(script),
        _candles(candle_count),
        FixedConfig("mean_reversion", "TEST/USD", {"a": 1}),
        window_ms=window_days * DAY,
        step_ms=step_days * DAY if step_days else None,
    )


class TestRecordIsProducedFromOutOfSampleWindows:
    @pytest.mark.asyncio
    async def test_a_record_is_constructible_and_validated_source(self):
        result = await _walk_forward([[10.0, -5.0], [8.0, -4.0], [6.0], [-3.0]])
        record = build_validated_edge_record(result)

        assert isinstance(record, ValidatedEdgeRecord)
        assert isinstance(record.estimate, EdgeEstimate)
        assert record.estimate.source == VALIDATED_EDGE_SOURCE
        assert record.is_validated_source

    @pytest.mark.asyncio
    async def test_the_estimate_pools_every_window_and_the_arithmetic_is_right(self):
        # 6 trades: +10, -5, +8, -4, +6, -3  ->  net +12 over 6 trades.
        result = await _walk_forward([[10.0, -5.0], [8.0, -4.0], [6.0], [-3.0]])
        estimate = build_validated_edge_record(result).estimate

        assert estimate.sample_size == 6
        assert estimate.expectancy == pytest.approx(12.0 / 6)
        assert estimate.win_rate == pytest.approx(3 / 6)
        assert estimate.profit_factor == pytest.approx(24.0 / 12.0)

    @pytest.mark.asyncio
    async def test_win_rate_is_a_fraction_as_the_framework_requires(self):
        """BacktestResult reports win rate as a percentage; EdgeEstimate
        validates it as a fraction and rejects anything outside [0, 1]."""
        result = await _walk_forward([[1.0, 1.0], [1.0, 1.0], [1.0], [1.0]])
        estimate = build_validated_edge_record(result).estimate
        assert estimate.win_rate == pytest.approx(1.0)

        result = await _walk_forward([[-1.0, -1.0], [-1.0, -1.0], [-1.0], [-1.0]])
        estimate = build_validated_edge_record(result).estimate
        assert estimate.win_rate == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_the_record_carries_the_configuration_it_describes(self):
        result = await _walk_forward([[1.0], [2.0], [3.0], [4.0]])
        record = build_validated_edge_record(result)

        assert record.strategy == "mean_reversion"
        assert record.trading_pair == "TEST/USD"
        assert record.params_fingerprint == result.params_fingerprint
        assert record.num_windows == result.num_windows
        assert record.num_trading_windows == len(result.windows_with_trades)
        assert record.consistency == result.consistency

    @pytest.mark.asyncio
    async def test_a_record_reflects_pooled_windows_not_the_best_one(self):
        """One strong window must not carry the estimate: the pooled expectancy
        has to sit below the winning window's own."""
        result = await _walk_forward([[100.0], [-10.0], [-10.0], [-10.0]])
        estimate = build_validated_edge_record(result).estimate

        best_window = max(w.expectancy_per_trade for w in result.windows_with_trades)
        assert estimate.expectancy < best_window
        assert estimate.expectancy == pytest.approx(70.0 / 4)


class TestRecordIsRefusedWhenItWouldMislead:
    @pytest.mark.asyncio
    async def test_a_single_window_cannot_produce_a_validated_record(self):
        """One window is a backtest wearing a different name."""
        result = await _walk_forward([[5.0, 6.0]], window_days=10_000)
        assert result.num_windows == 1
        blockers = edge_record_blockers(result)

        assert blockers
        assert any("not corroboration" in b for b in blockers)
        assert build_validated_edge_record(result) is None

    @pytest.mark.asyncio
    async def test_trades_confined_to_one_window_cannot_produce_a_record(self):
        """Four windows planned, one window traded - the estimate would rest
        entirely on that window, which is in-sample by another name."""
        result = await _walk_forward([[5.0, 6.0], [], [], []])
        assert result.num_windows == 4
        assert len(result.windows_with_trades) < MIN_TRADING_WINDOWS_FOR_A_RECORD

        assert any("in-sample by another name" in b for b in edge_record_blockers(result))
        assert build_validated_edge_record(result) is None

    @pytest.mark.asyncio
    async def test_overlapping_windows_cannot_produce_a_record(self):
        """Pooling overlapping windows would count shared trades twice."""
        result = await _walk_forward([[1.0]] * 12, window_days=30, step_days=10)
        assert result.windows_overlap

        assert any("more than once" in b for b in edge_record_blockers(result))
        assert build_validated_edge_record(result) is None

    @pytest.mark.asyncio
    async def test_no_trades_cannot_produce_a_record(self):
        result = await _walk_forward([[], [], [], []])
        blockers = edge_record_blockers(result)

        assert any("nothing to estimate" in b for b in blockers)
        # With zero trading windows the "resting on a single window" wording
        # would be plainly wrong, so it must not appear.
        assert not any("single window" in b for b in blockers)
        assert build_validated_edge_record(result) is None

    @pytest.mark.asyncio
    async def test_a_clean_multi_window_measurement_has_no_blockers(self):
        result = await _walk_forward([[1.0], [2.0], [3.0], [4.0]])
        assert edge_record_blockers(result) == ()


class TestCaveatsTravelWithTheRecord:
    @pytest.mark.asyncio
    async def test_a_small_sample_is_flagged(self):
        result = await _walk_forward([[1.0], [2.0], [3.0], [4.0]])
        record = build_validated_edge_record(result)
        assert any("confidence in it is low" in c for c in record.caveats)

    @pytest.mark.asyncio
    async def test_a_mixed_sign_across_windows_is_flagged(self):
        result = await _walk_forward([[10.0], [-8.0], [-6.0], [-9.0]])
        record = build_validated_edge_record(result)
        assert result.consistency == "mixed"
        assert any("describes none of them" in c for c in record.caveats)

    @pytest.mark.asyncio
    async def test_an_infinite_profit_factor_is_flagged(self):
        result = await _walk_forward([[5.0], [6.0], [7.0], [8.0]])
        record = build_validated_edge_record(result)
        assert record.estimate.profit_factor == float("inf")
        assert any("not an unbounded edge" in c for c in record.caveats)

    @pytest.mark.asyncio
    async def test_every_record_says_it_is_not_wired_into_runtime(self):
        result = await _walk_forward([[1.0], [2.0], [3.0], [4.0]])
        record = build_validated_edge_record(result)
        assert any("remains None at runtime" in c for c in record.caveats)


class TestReport:
    @pytest.mark.asyncio
    async def test_a_produced_record_is_reported_with_its_provenance(self):
        result = await _walk_forward([[10.0, -5.0], [8.0, -4.0], [6.0], [-3.0]])
        report = format_edge_record_report(result)

        assert "Validated measurement record" in report
        assert "NOT PRODUCED" not in report
        assert VALIDATED_EDGE_SOURCE in report
        assert result.params_fingerprint in report
        assert "Sample size" in report and "Expectancy per trade" in report
        assert "remains None at runtime" in report

    @pytest.mark.asyncio
    async def test_a_refused_record_states_why_plainly(self):
        result = await _walk_forward([[5.0, 6.0], [], [], []])
        report = format_edge_record_report(result)

        assert "NOT PRODUCED" in report
        assert "in-sample by another name" in report
        # No numbers are printed for a record that was refused.
        assert "Expectancy per trade" not in report


class TestRuntimeProposalsAreUnaffected:
    """The line this change must not cross.

    The record is produced for review. Populating a live proposal with it is a
    behaviour change belonging to a separate change, so these tests assert -
    structurally, over the whole application - that it has not happened.
    """

    def _app_modules(self):
        for path in sorted(_APP_DIR.rglob("*.py")):
            if "__pycache__" in path.parts or path.suffix != ".py":
                continue
            yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_only_the_validation_package_constructs_an_edge_estimate(self):
        offenders = []
        for path, tree in self._app_modules():
            if _VALIDATION_DIR in path.parents:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "EdgeEstimate"
                ):
                    offenders.append(f"{path.relative_to(_APP_DIR)}:{node.lineno}")
        assert not offenders, (
            "Only the validation tooling may construct an EdgeEstimate: " f"{offenders}"
        )

    def test_no_application_code_passes_a_non_none_expected_edge_estimate(self):
        offenders = []
        for path, tree in self._app_modules():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if keyword.arg != "expected_edge_estimate":
                        continue
                    value = keyword.value
                    is_none = isinstance(value, ast.Constant) and value.value is None
                    # A pass-through of an existing (already None) field is not
                    # a population of one - the committee's read-only view does
                    # exactly this.
                    is_passthrough = (
                        isinstance(value, ast.Attribute)
                        and value.attr == "expected_edge_estimate"
                    )
                    if not is_none and not is_passthrough:
                        offenders.append(
                            f"{path.relative_to(_APP_DIR)}:{node.lineno}"
                        )
        assert not offenders, (
            "expected_edge_estimate must stay None at runtime until a separate "
            f"change wires it: {offenders}"
        )

    def test_the_proposal_schema_still_defaults_the_estimate_to_none(self):
        """A strategy that does not explicitly supply one - and per the guard
        above, none of them may - gets ``None``."""
        field = StrategyProposal.__dataclass_fields__["expected_edge_estimate"]
        assert field.default is None

    def test_the_framework_still_rejects_a_self_computed_estimate(self):
        """The record this milestone produces is legitimate precisely because
        the source marker cannot be forged - confirm that gate still holds."""
        with pytest.raises(InvalidProposalError):
            EdgeEstimate(
                expectancy=1.0, win_rate=0.5, profit_factor=2.0,
                sample_size=100, source="computed_by_the_strategy",
            )

    @pytest.mark.asyncio
    async def test_producing_a_record_does_not_touch_any_proposal(self):
        result = await _walk_forward([[10.0, -5.0], [8.0, -4.0], [6.0], [-3.0]])
        record = build_validated_edge_record(result)
        assert record is not None

        # The record is a standalone object; nothing links it to a proposal.
        assert not hasattr(record, "proposal")
        assert not isinstance(record, StrategyProposal)
