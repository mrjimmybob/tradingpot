"""Out-of-sample walk-forward measurement of a fixed configuration.

Two kinds of test live here, deliberately.

*Scripted-engine* tests substitute a stand-in for ``BacktestEngine`` that
returns exactly-known per-window results. They exist because the claims this
milestone has to prove are claims about *aggregation and reporting* - "every
window used identical parameters", "a sign flip is reported, not repaired" -
and those are only checkable when the per-window numbers are known exactly.
Coaxing a real strategy into a specific pattern of wins and losses would test
the strategy, not the walk-forward layer, and would break the moment the
strategy changed.

*Real-engine* tests then run the whole path through the unmodified
``BacktestEngine`` over synthetic candles, proving the integration holds:
determinism, window coverage, and metric-for-metric agreement with a directly
run backtest.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import List

import pytest

from app.backtesting.candle import Candle
from app.backtesting.data_provider import CsvHistoricalDataProvider
from app.backtesting.engine import BacktestEngine
from app.backtesting.execution_model import BacktestExecutionModel
from app.backtesting.results import TradeRecord, compute_result
from app.backtesting.validation import (
    FixedConfig,
    MeasurementSpan,
    format_walk_forward_report,
    plan_windows,
    resolve_span,
    run_walk_forward,
)
from app.backtesting.validation.walk_forward import (
    MIN_TRADES_FOR_CONFIDENCE,
    MIN_WINDOWS_FOR_CONFIDENCE,
    MS_PER_DAY,
)

DAY = MS_PER_DAY
START_MS = 1704067200000  # 2024-01-01 UTC
CANDLE_STEP_MS = DAY // 4  # 6h candles keep fixtures small but multi-day


# ---------------------------------------------------------------------------
# Fixtures and the scripted engine
# ---------------------------------------------------------------------------


def _candles(count: int, step_ms: int = CANDLE_STEP_MS, start_ms: int = START_MS) -> List[Candle]:
    """A deterministic oscillating series - no randomness anywhere."""
    out = []
    for i in range(count):
        price = 100.0 * (1 + 0.05 * math.sin(i / 12.0))
        out.append(Candle(
            timestamp=start_ms + i * step_ms,
            datetime="d", symbol="TESTUSD",
            open=price, high=price * 1.001, low=price * 0.999, close=price,
            base_volume=1.0, quote_volume=price, trade_count=1,
        ))
    return out


def _scripted_result(pnls: List[float], starting_balance: float, ts: int):
    """A BacktestResult with exactly the per-trade P&L asked for.

    Built through the engine's own ``compute_result`` so the metrics are
    internally coherent (win rate, profit factor and expectancy all derived
    from one trade list) rather than hand-set to values that could not
    co-occur.
    """
    trades = [
        TradeRecord(
            entry_timestamp=ts + i, exit_timestamp=ts + i + 1, strategy="scripted",
            entry_price=100.0, exit_price=100.0 + pnl, fees=0.0,
            gross_pnl=pnl, net_pnl=pnl, exit_reason="scripted",
        )
        for i, pnl in enumerate(pnls)
    ]
    equity = [(ts, starting_balance)]
    running = starting_balance
    for i, pnl in enumerate(pnls):
        running += pnl
        equity.append((ts + i + 1, running))
    return compute_result(
        starting_balance=starting_balance,
        ending_balance=starting_balance + sum(pnls),
        trades=trades,
        equity_curve=equity,
        total_fees_paid=0.0,
        buy_and_hold_return_pct=0.0,
    )


class _ScriptedEngine:
    """Stand-in for ``BacktestEngine`` returning pre-agreed per-window results.

    Records every call so the tests can assert what the walk-forward layer
    actually handed the engine - which is the only way to prove "the identical
    parameters were measured on every window" rather than assume it.
    """

    def __init__(self, pnls_per_window: List[List[float]]):
        self._script = list(pnls_per_window)
        self.calls: List[dict] = []

    async def run_candles(
        self, candles, trading_pair, strategy, strategy_params, starting_balance, quiet=True,
    ):
        self.calls.append({
            "candles": list(candles),
            "trading_pair": trading_pair,
            "strategy": strategy,
            "params": strategy_params,
            "starting_balance": starting_balance,
        })
        index = len(self.calls) - 1
        pnls = self._script[index] if index < len(self._script) else []
        return _scripted_result(pnls, starting_balance, candles[0].timestamp)


_MEAN_REVERSION_PARAMS = {
    "bar_interval_seconds": 0, "regime_filter_enabled": False,
    "bollinger_period": 10, "atr_period": 10, "cooldown_seconds": 0,
    "decision_score_threshold": 0.0,
}


def _config(**overrides) -> FixedConfig:
    params = dict(_MEAN_REVERSION_PARAMS)
    params.update(overrides)
    return FixedConfig("mean_reversion", "TEST/USD", params)


def _real_engine(fee_pct: float = 0.1) -> BacktestEngine:
    return BacktestEngine(
        data_provider=CsvHistoricalDataProvider(),
        execution_model=BacktestExecutionModel(fee_pct=fee_pct),
    )


# ---------------------------------------------------------------------------
# Rolling window planning
# ---------------------------------------------------------------------------


class TestRollingWindowsCoverTheRange:
    def test_successive_windows_tile_the_range_with_no_gap_or_overhang(self):
        span = MeasurementSpan(start_ms=0, end_ms=100 * DAY - 1)
        windows = plan_windows(span, window_ms=25 * DAY)

        assert len(windows) == 4
        assert windows[0].start_ms == span.start_ms
        assert windows[-1].end_ms == span.end_ms
        for earlier, later in zip(windows, windows[1:]):
            # Spans are inclusive on both ends, so contiguity means the next
            # window starts exactly one millisecond after the previous ended.
            assert later.start_ms == earlier.end_ms + 1

    def test_default_step_equals_window_so_samples_are_independent(self):
        span = MeasurementSpan(start_ms=0, end_ms=60 * DAY - 1)
        assert plan_windows(span, window_ms=20 * DAY) == plan_windows(
            span, window_ms=20 * DAY, step_ms=20 * DAY
        )

    def test_final_window_is_clipped_to_the_requested_end(self):
        span = MeasurementSpan(start_ms=0, end_ms=70 * DAY - 1)
        windows = plan_windows(span, window_ms=30 * DAY)

        assert len(windows) == 3
        assert windows[-1].end_ms == span.end_ms
        # A short tail window is kept, not silently dropped - dropping it would
        # quietly stop covering the range the operator asked for.
        assert windows[-1].end_ms - windows[-1].start_ms < 30 * DAY - 1

    def test_a_smaller_step_produces_overlapping_windows(self):
        span = MeasurementSpan(start_ms=0, end_ms=60 * DAY - 1)
        windows = plan_windows(span, window_ms=30 * DAY, step_ms=10 * DAY)

        assert len(windows) > 2
        assert windows[1].start_ms < windows[0].end_ms  # they share candles

    def test_a_step_wider_than_the_window_is_rejected(self):
        """It would leave unmeasured gaps, so the windows would no longer
        cover the requested range."""
        span = MeasurementSpan(start_ms=0, end_ms=100 * DAY)
        with pytest.raises(ValueError, match="gaps"):
            plan_windows(span, window_ms=10 * DAY, step_ms=20 * DAY)

    def test_an_unbounded_span_cannot_be_planned(self):
        with pytest.raises(ValueError, match="bounded span"):
            plan_windows(MeasurementSpan(), window_ms=DAY)

    def test_non_positive_sizes_are_rejected(self):
        span = MeasurementSpan(start_ms=0, end_ms=DAY)
        with pytest.raises(ValueError, match="window_ms"):
            plan_windows(span, window_ms=0)
        with pytest.raises(ValueError, match="step_ms"):
            plan_windows(span, window_ms=DAY, step_ms=0)

    def test_resolve_span_fills_open_ends_from_the_candle_series(self):
        candles = _candles(10)
        resolved = resolve_span(candles, MeasurementSpan())
        assert resolved.start_ms == candles[0].timestamp
        assert resolved.end_ms == candles[-1].timestamp

        partly = resolve_span(candles, MeasurementSpan(start_ms=candles[3].timestamp))
        assert partly.start_ms == candles[3].timestamp
        assert partly.end_ms == candles[-1].timestamp

    def test_resolve_span_rejects_an_empty_series(self):
        with pytest.raises(ValueError, match="empty candle series"):
            resolve_span([], MeasurementSpan())


# ---------------------------------------------------------------------------
# The same fixed parameters on every window
# ---------------------------------------------------------------------------


class TestSameParametersOnEveryWindow:
    @pytest.mark.asyncio
    async def test_every_window_receives_identical_parameters(self):
        engine = _ScriptedEngine([[1.0], [-1.0], [1.0], [-1.0]])
        config = _config()
        result = await run_walk_forward(
            engine, _candles(400), config, window_ms=25 * DAY,
        )

        assert len(engine.calls) == result.num_windows > 1
        for call in engine.calls:
            assert call["params"] == dict(config.params)
            assert call["strategy"] == config.strategy
            assert call["trading_pair"] == config.trading_pair
            assert call["starting_balance"] == config.starting_balance

    @pytest.mark.asyncio
    async def test_each_window_gets_its_own_copy_of_the_parameters(self):
        """Sharing one dict across windows would let a strategy that mutated it
        mid-replay change what later windows measure."""
        engine = _ScriptedEngine([[1.0]] * 4)
        await run_walk_forward(engine, _candles(400), _config(), window_ms=25 * DAY)

        seen = [id(call["params"]) for call in engine.calls]
        assert len(set(seen)) == len(seen)

    @pytest.mark.asyncio
    async def test_mutating_one_windows_params_cannot_reach_the_next(self):
        engine = _ScriptedEngine([[1.0]] * 4)
        config = _config()

        original = _ScriptedEngine.run_candles

        async def _mutating_run(self, candles, pair, strategy, params, balance, quiet=True):
            params["bollinger_period"] = 999  # a strategy misbehaving mid-replay
            return await original(self, candles, pair, strategy, params, balance, quiet)

        engine.run_candles = _mutating_run.__get__(engine, _ScriptedEngine)
        await run_walk_forward(engine, _candles(400), config, window_ms=25 * DAY)

        assert config.params["bollinger_period"] == 10
        assert all(call["params"]["bollinger_period"] == 999 for call in engine.calls), (
            "sanity: the mutation really did happen inside each call"
        )

    @pytest.mark.asyncio
    async def test_every_window_reports_the_same_parameter_fingerprint(self):
        engine = _ScriptedEngine([[1.0], [-2.0], [3.0], [-4.0]])
        config = _config()
        result = await run_walk_forward(engine, _candles(400), config, window_ms=25 * DAY)

        assert result.every_window_used_one_configuration
        assert {w.params_fingerprint for w in result.windows} == {config.params_fingerprint}


# ---------------------------------------------------------------------------
# An inconsistent edge is shown, never "fixed"
# ---------------------------------------------------------------------------


class TestInconsistentEdgeIsShownNotFixed:
    @pytest.mark.asyncio
    async def test_an_edge_in_one_window_and_absent_in_others_reads_as_mixed(self):
        engine = _ScriptedEngine([
            [10.0, 12.0, 8.0],      # window 1: a clear edge
            [-9.0, -7.0, -11.0],    # window 2: gone
            [-6.0, -8.0, -5.0],     # window 3: still gone
            [-4.0, -9.0, -6.0],     # window 4: still gone
        ])
        config = _config()
        result = await run_walk_forward(engine, _candles(400), config, window_ms=25 * DAY)

        assert result.consistency == "mixed"
        assert not result.has_positive_edge_in_every_traded_window
        assert len(result.profitable_windows) == 1
        assert len(result.losing_windows) == 3

        # The point of the whole change: nothing was adjusted to make windows
        # 2-4 look better. Every window measured the same fixed configuration.
        assert result.every_window_used_one_configuration
        assert all(call["params"] == dict(config.params) for call in engine.calls)

        report = format_walk_forward_report(result)
        assert "MIXED" in report
        assert "1 of 4 trading windows had positive expectancy" in report
        # Every window's own numbers are visible side by side, not averaged away.
        for window in result.windows:
            assert window.span.label() in report

    @pytest.mark.asyncio
    async def test_all_positive_windows_read_as_consistently_positive(self):
        engine = _ScriptedEngine([[5.0, 6.0], [4.0, 7.0], [8.0, 3.0], [5.0, 5.0]])
        result = await run_walk_forward(engine, _candles(400), _config(), window_ms=25 * DAY)

        assert result.consistency == "consistently_positive"
        assert result.has_positive_edge_in_every_traded_window
        report = format_walk_forward_report(result)
        assert "CONSISTENTLY POSITIVE" in report
        assert "not proof of an edge" in report

    @pytest.mark.asyncio
    async def test_all_negative_windows_read_as_consistently_negative_not_inconsistent(self):
        """A strategy that loses in every window is perfectly consistent. Calling
        that "inconsistent" would misdescribe it, and calling it "consistent"
        without a sign would read as good news."""
        engine = _ScriptedEngine([[-5.0], [-6.0], [-4.0], [-7.0]])
        result = await run_walk_forward(engine, _candles(400), _config(), window_ms=25 * DAY)

        assert result.consistency == "consistently_negative"
        assert not result.has_positive_edge_in_every_traded_window
        assert "CONSISTENTLY NEGATIVE" in format_walk_forward_report(result)

    @pytest.mark.asyncio
    async def test_fewer_than_two_trading_windows_is_not_assessable(self):
        engine = _ScriptedEngine([[5.0], [], [], []])
        result = await run_walk_forward(engine, _candles(400), _config(), window_ms=25 * DAY)

        assert result.consistency == "not_assessable"
        assert "NOT ASSESSABLE" in format_walk_forward_report(result)

    @pytest.mark.asyncio
    async def test_windows_without_trades_are_excluded_from_consistency_not_counted_as_losses(self):
        engine = _ScriptedEngine([[5.0], [], [6.0], []])
        result = await run_walk_forward(engine, _candles(400), _config(), window_ms=25 * DAY)

        assert result.num_windows == 4
        assert len(result.windows_with_trades) == 2
        assert result.consistency == "consistently_positive"


# ---------------------------------------------------------------------------
# Honest reporting of what the measurement cannot support
# ---------------------------------------------------------------------------


class TestLimitationsAreStated:
    async def _measure(self, script, window_days=25, step_days=None, candle_count=400):
        engine = _ScriptedEngine(script)
        return await run_walk_forward(
            engine, _candles(candle_count), _config(),
            window_ms=window_days * DAY,
            step_ms=step_days * DAY if step_days else None,
        )

    @pytest.mark.asyncio
    async def test_too_few_windows_is_called_out(self):
        result = await self._measure([[1.0]] * 4)
        assert result.num_windows < MIN_WINDOWS_FOR_CONFIDENCE
        assert any("too few independent samples" in note for note in result.limitations())

    @pytest.mark.asyncio
    async def test_too_few_trades_is_called_out(self):
        result = await self._measure([[1.0]] * 4)
        assert result.total_trades < MIN_TRADES_FOR_CONFIDENCE
        assert any("dominated by noise" in note for note in result.limitations())

    @pytest.mark.asyncio
    async def test_windows_without_trades_are_called_out(self):
        result = await self._measure([[1.0], [], [1.0], []])
        assert any("no trades at all" in note for note in result.limitations())

    @pytest.mark.asyncio
    async def test_overlapping_windows_are_flagged_as_not_independent(self):
        result = await self._measure([[1.0]] * 12, window_days=30, step_days=10)
        assert result.windows_overlap
        assert any("NOT independent" in note for note in result.limitations())
        assert "OVERLAPPING" in format_walk_forward_report(result)

    @pytest.mark.asyncio
    async def test_non_overlapping_windows_are_not_flagged(self):
        result = await self._measure([[1.0]] * 4)
        assert not result.windows_overlap
        assert not any("NOT independent" in note for note in result.limitations())

    @pytest.mark.asyncio
    async def test_unrealised_return_without_closed_trades_is_called_out(self):
        """A window can show a large Return% with zero trades - an accumulating
        strategy still holding at the window's end. Reported as unrealised so a
        reader cannot mistake a mark-to-market move for realised profit."""
        candles = _candles(400)
        result = await run_walk_forward(
            _real_engine(), candles, _config(), window_ms=25 * DAY,
        )
        # Substitute a window that held through to the end with no round trip.
        held = replace(result.windows[0], trades=(), num_trades=0, total_return_pct=12.5)
        held_result = replace(result, windows=(held,) + result.windows[1:])
        assert any("unrealised" in note for note in held_result.limitations())

    @pytest.mark.asyncio
    async def test_fully_realised_windows_do_not_get_the_unrealised_note(self):
        result = await self._measure([[1.0]] * 4)
        assert not any("unrealised" in note for note in result.limitations())

    @pytest.mark.asyncio
    async def test_warm_up_and_non_claim_notes_are_always_present(self):
        result = await self._measure([[1.0] * 20] * 8, window_days=12)
        notes = " ".join(result.limitations())
        assert "warm-up" in notes
        assert "not a profitability" in notes

    @pytest.mark.asyncio
    async def test_the_report_carries_its_own_limitations(self):
        """The numbers should be hard to copy anywhere without the caveats."""
        result = await self._measure([[1.0]] * 4)
        report = format_walk_forward_report(result)
        assert "Limitations:" in report
        for note in result.limitations():
            assert note in report


class TestUnmeasurableWindowsAreReported:
    @pytest.mark.asyncio
    async def test_windows_beyond_the_candle_data_are_skipped_and_disclosed(self):
        """Silently dropping them would shrink the denominator and present a
        partial run as a complete one."""
        candles = _candles(60)  # 15 days of 6h candles
        engine = _ScriptedEngine([[1.0]] * 10)
        span = MeasurementSpan(start_ms=candles[0].timestamp, end_ms=candles[0].timestamp + 40 * DAY)

        result = await run_walk_forward(
            engine, candles, _config(), window_ms=10 * DAY, span=span,
        )

        assert result.skipped, "windows past the end of the data must be recorded"
        assert all(s.num_candles < 2 for s in result.skipped)
        assert any("could not be measured" in note for note in result.limitations())
        report = format_walk_forward_report(result)
        assert "SKIPPED" in report


class TestReportRendering:
    @pytest.mark.asyncio
    async def test_report_shows_provenance_and_every_window(self):
        engine = _ScriptedEngine([[2.0, -1.0], [-3.0, 4.0], [1.0], [-2.0]])
        config = _config()
        result = await run_walk_forward(engine, _candles(400), config, window_ms=25 * DAY)
        report = format_walk_forward_report(result)

        assert "mean_reversion" in report and "TEST/USD" in report
        assert config.params_fingerprint in report
        assert "Trades" in report and "Expectancy" in report and "MaxDD%" in report
        assert report.count("2024-") >= result.num_windows
        assert f"Total closed trades: {result.total_trades}" in report

    @pytest.mark.asyncio
    async def test_infinite_profit_factor_renders_readably(self):
        """The engine reports ``inf`` for wins-and-no-losses; a bare float would
        render as "inf" anyway, but formatting it must not crash the report."""
        engine = _ScriptedEngine([[5.0, 6.0], [7.0, 8.0]])
        result = await run_walk_forward(engine, _candles(200), _config(), window_ms=25 * DAY)

        assert all(w.profit_factor == float("inf") for w in result.windows)
        assert "inf" in format_walk_forward_report(result)


# ---------------------------------------------------------------------------
# End to end through the real, unmodified BacktestEngine
# ---------------------------------------------------------------------------


class TestRealEngineWalkForward:
    @pytest.mark.asyncio
    async def test_windows_partition_the_candles_and_share_one_configuration(self):
        candles = _candles(400)
        config = _config()
        result = await run_walk_forward(
            _real_engine(), candles, config, window_ms=25 * DAY,
        )

        assert result.num_windows == 4
        assert not result.skipped
        # Non-overlapping windows: every candle measured exactly once.
        assert sum(w.num_candles for w in result.windows) == len(candles)
        assert result.every_window_used_one_configuration
        assert result.params_fingerprint == config.params_fingerprint

    @pytest.mark.asyncio
    async def test_walk_forward_is_deterministic_run_to_run(self):
        candles = _candles(400)
        config = _config()
        first = await run_walk_forward(_real_engine(), candles, config, window_ms=25 * DAY)
        second = await run_walk_forward(_real_engine(), candles, config, window_ms=25 * DAY)

        assert [w.num_trades for w in first.windows] == [w.num_trades for w in second.windows]
        for a, b in zip(first.windows, second.windows):
            assert a.span == b.span
            assert a.expectancy_per_trade == pytest.approx(b.expectancy_per_trade)
            assert a.ending_balance == pytest.approx(b.ending_balance)
        assert first.consistency == second.consistency

    @pytest.mark.asyncio
    async def test_a_single_full_range_window_equals_a_directly_run_backtest(self):
        """The walk-forward layer adds windowing and provenance and nothing
        else - one window over the whole range must reproduce the plain
        backtest of the same fixed parameters exactly."""
        candles = _candles(200)
        config = _config()

        direct = await _real_engine().run_candles(
            candles, "TEST/USD", "mean_reversion", dict(_MEAN_REVERSION_PARAMS),
            10_000.0, quiet=True,
        )
        result = await run_walk_forward(
            _real_engine(), candles, config, window_ms=10_000 * DAY,
        )

        assert result.num_windows == 1
        window = result.windows[0]
        assert window.num_trades == direct.num_trades
        assert window.expectancy_per_trade == pytest.approx(direct.expectancy_per_trade)
        assert window.win_rate_pct == pytest.approx(direct.win_rate)
        assert window.max_drawdown_pct == pytest.approx(direct.max_drawdown_pct)
        assert window.ending_balance == pytest.approx(direct.ending_balance)

    @pytest.mark.asyncio
    async def test_per_window_results_are_reported_side_by_side(self):
        candles = _candles(400)
        result = await run_walk_forward(
            _real_engine(), candles, _config(), window_ms=25 * DAY,
        )
        report = format_walk_forward_report(result)

        for window in result.windows:
            assert window.span.label() in report
        assert "Limitations:" in report

    @pytest.mark.asyncio
    async def test_the_progress_hook_cannot_influence_a_measurement(self):
        candles = _candles(200)
        seen = []

        with_hook = await run_walk_forward(
            _real_engine(), candles, _config(), window_ms=25 * DAY,
            on_window=lambda i, total, m: seen.append((i, total, m.num_trades)),
        )
        without_hook = await run_walk_forward(
            _real_engine(), candles, _config(), window_ms=25 * DAY,
        )

        assert len(seen) == with_hook.num_windows
        assert [w.num_trades for w in with_hook.windows] == [
            w.num_trades for w in without_hook.windows
        ]
