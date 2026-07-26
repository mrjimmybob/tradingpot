"""Measuring every strategy's shipped defaults over one range, in one pass."""
from __future__ import annotations

from typing import List

import pytest

from app.backtesting.candle import Candle
from app.backtesting.data_provider import CsvHistoricalDataProvider
from app.backtesting.engine import BacktestEngine
from app.backtesting.execution_model import BacktestExecutionModel
from app.backtesting.results import TradeRecord, compute_result
from app.backtesting.validation.baseline import (
    BASELINE_STRATEGIES,
    BaselineEntry,
    format_baseline_summary,
    measure_baseline,
)

DAY = 86_400_000
START_MS = 1704067200000


def _candles(count: int, step_ms: int = DAY // 4) -> List[Candle]:
    return [
        Candle(
            timestamp=START_MS + i * step_ms, datetime="d", symbol="TESTUSD",
            open=100.0, high=100.1, low=99.9, close=100.0,
            base_volume=1.0, quote_volume=100.0, trade_count=1,
        )
        for i in range(count)
    ]


class _RecordingEngine:
    """Records what each strategy was measured with, and returns a fixed result."""

    def __init__(self, pnls_per_call: List[List[float]] | None = None):
        self.calls: List[dict] = []
        self._script = pnls_per_call

    async def run_candles(self, candles, trading_pair, strategy, params, balance, quiet=True):
        self.calls.append({
            "strategy": strategy, "params": params, "trading_pair": trading_pair,
            "balance": balance, "first_ts": candles[0].timestamp,
            "last_ts": candles[-1].timestamp, "num_candles": len(candles),
        })
        index = len(self.calls) - 1
        pnls = (
            self._script[index] if self._script and index < len(self._script) else [1.0, -0.5]
        )
        ts = candles[0].timestamp
        trades = [
            TradeRecord(
                entry_timestamp=ts + i, exit_timestamp=ts + i + 1, strategy=strategy,
                entry_price=100.0, exit_price=100.0 + p, fees=0.0,
                gross_pnl=p, net_pnl=p, exit_reason="scripted",
            )
            for i, p in enumerate(pnls)
        ]
        return compute_result(
            balance, balance + sum(pnls), trades,
            [(ts, balance), (ts + 1, balance + sum(pnls))], 0.0, 0.0,
        )


class TestBaselineCoversEveryStrategy:
    def test_the_six_concrete_strategies_are_the_baseline_set(self):
        """auto_mode is deliberately absent: it selects among these rather than
        trading a thesis of its own, so measuring it would double-count
        whichever ones it picked."""
        assert len(BASELINE_STRATEGIES) == 6
        assert "auto_mode" not in BASELINE_STRATEGIES
        assert set(BASELINE_STRATEGIES) == {
            "dca_accumulator", "adaptive_grid", "mean_reversion",
            "trend_following", "volatility_breakout", "dip_recovery",
        }

    def test_every_baseline_strategy_is_dispatchable(self):
        """A name here that TradingEngine cannot dispatch would fail the
        baseline run only at the moment someone tried to use it."""
        from app.services.trading_engine import TradingEngine

        engine = TradingEngine()
        for strategy in BASELINE_STRATEGIES:
            assert engine._get_strategy_executor(strategy) is not None, strategy

    @pytest.mark.asyncio
    async def test_each_strategy_is_measured_over_the_identical_windows(self):
        engine = _RecordingEngine()
        entries = await measure_baseline(
            engine, _candles(400), "TEST/USD", window_ms=25 * DAY,
        )

        assert len(entries) == len(BASELINE_STRATEGIES)
        assert [e.strategy for e in entries] == list(BASELINE_STRATEGIES)

        windows_per_strategy = {}
        for call in engine.calls:
            windows_per_strategy.setdefault(call["strategy"], []).append(
                (call["first_ts"], call["last_ts"])
            )
        distinct = {tuple(v) for v in windows_per_strategy.values()}
        assert len(distinct) == 1, "strategies were measured over different windows"

    @pytest.mark.asyncio
    async def test_a_strategy_without_overrides_is_measured_with_its_own_defaults(self):
        """Passing ``{}`` is what makes this a measurement of the shipped
        defaults: the strategy falls back to its internal ones, exactly as a
        freshly created bot would."""
        engine = _RecordingEngine()
        await measure_baseline(engine, _candles(200), "TEST/USD", window_ms=25 * DAY)

        assert all(call["params"] == {} for call in engine.calls)

    @pytest.mark.asyncio
    async def test_per_strategy_overrides_reach_only_that_strategy(self):
        engine = _RecordingEngine()
        await measure_baseline(
            engine, _candles(200), "TEST/USD", window_ms=25 * DAY,
            params_by_strategy={"mean_reversion": {"bollinger_period": 14}},
        )

        for call in engine.calls:
            expected = {"bollinger_period": 14} if call["strategy"] == "mean_reversion" else {}
            assert call["params"] == expected

    @pytest.mark.asyncio
    async def test_a_subset_of_strategies_can_be_measured(self):
        engine = _RecordingEngine()
        entries = await measure_baseline(
            engine, _candles(200), "TEST/USD", window_ms=25 * DAY,
            strategies=("mean_reversion", "trend_following"),
        )
        assert [e.strategy for e in entries] == ["mean_reversion", "trend_following"]

    @pytest.mark.asyncio
    async def test_progress_is_reported_per_strategy(self):
        seen = []
        await measure_baseline(
            _RecordingEngine(), _candles(200), "TEST/USD", window_ms=25 * DAY,
            on_strategy=lambda i, total, name: seen.append((i, total, name)),
        )
        assert [s[2] for s in seen] == list(BASELINE_STRATEGIES)
        assert all(s[1] == len(BASELINE_STRATEGIES) for s in seen)

    @pytest.mark.asyncio
    async def test_each_entry_carries_its_record_or_the_reason_there_is_none(self):
        entries = await measure_baseline(
            _RecordingEngine(), _candles(400), "TEST/USD", window_ms=25 * DAY,
        )
        for entry in entries:
            assert isinstance(entry, BaselineEntry)
            if entry.record is None:
                assert entry.blockers, "a missing record must come with its reason"
            else:
                assert not entry.blockers


class TestSummaryIsAComparisonNotARanking:
    async def _entries(self, script=None):
        return await measure_baseline(
            _RecordingEngine(script), _candles(400), "TEST/USD", window_ms=25 * DAY,
        )

    @pytest.mark.asyncio
    async def test_rows_stay_in_declaration_order_regardless_of_result(self):
        """A table sorted by expectancy is a recommendation wearing a table's
        clothes, and these sample sizes would not support one."""
        # Deliberately give the LAST strategy measured the best results.
        script = [[-5.0]] * 20 + [[50.0]] * 8
        entries = await measure_baseline(
            _RecordingEngine(script), _candles(400), "TEST/USD", window_ms=25 * DAY,
        )
        summary = format_baseline_summary(entries)

        positions = [summary.index(s) for s in BASELINE_STRATEGIES]
        assert positions == sorted(positions), "summary rows were reordered by result"

    @pytest.mark.asyncio
    async def test_the_summary_says_it_is_not_a_ranking(self):
        summary = format_baseline_summary(await self._entries())
        assert "NOT a ranking" in summary
        assert "NOT sorted by performance" in summary
        assert "does not recommend a strategy" in summary

    @pytest.mark.asyncio
    async def test_every_strategy_gets_a_row(self):
        summary = format_baseline_summary(await self._entries())
        for strategy in BASELINE_STRATEGIES:
            assert strategy in summary

    @pytest.mark.asyncio
    async def test_a_strategy_without_a_record_shows_a_blank_not_a_zero(self):
        """A zero would read as "measured, and it was zero"; this strategy was
        measured and did not meet the bar for a record at all."""
        entries = await measure_baseline(
            _RecordingEngine([[]] * 40), _candles(400), "TEST/USD", window_ms=25 * DAY,
        )
        summary = format_baseline_summary(entries)

        assert all(e.record is None for e in entries)
        assert "+0.00" not in summary

    @pytest.mark.asyncio
    async def test_benchmark_columns_are_counts_not_scores(self):
        """"8/13" invites reading the other five; a single ratio invites
        reading nothing else."""
        summary = format_baseline_summary(await self._entries())
        assert "> B&H" in summary and "> DCA" in summary
        assert "a count, not a score" in summary

    @pytest.mark.asyncio
    async def test_exposure_is_explained_as_the_reason_bandh_may_be_unfair(self):
        summary = format_baseline_summary(await self._entries())
        assert "less exposed than" in summary
        assert "DCA column is the fairer comparison" in summary

    @pytest.mark.asyncio
    async def test_a_zero_trade_row_points_at_the_benchmark_section(self):
        """The whole point of this change: a zero-trade strategy is no longer a
        dead end, it is measured elsewhere in the report."""
        entries = await measure_baseline(
            _RecordingEngine([[]] * 40), _candles(400), "TEST/USD", window_ms=25 * DAY,
        )
        summary = format_baseline_summary(entries)
        assert "benchmark-relative section instead" in summary

    @pytest.mark.asyncio
    async def test_a_zero_trade_row_is_not_presented_as_an_inactive_strategy(self):
        """Closed round trips are the denominator; a strategy that scales in and
        out without fully flattening trades all window and still shows 0."""
        summary = format_baseline_summary(await self._entries())
        assert "CLOSED round trips only" in summary
        assert "does NOT mean" in summary


class TestBaselineWithTheRealEngine:
    @pytest.mark.asyncio
    async def test_every_strategys_defaults_are_measurable_end_to_end(self):
        """The real dispatch path for all six, on a small synthetic series -
        the baseline run must not fall over on any strategy's defaults."""
        candles = [
            Candle(
                timestamp=START_MS + i * 3_600_000, datetime="d", symbol="TESTUSD",
                open=100.0 + i * 0.1, high=100.0 + i * 0.1 + 0.5,
                low=100.0 + i * 0.1 - 0.5, close=100.0 + i * 0.1,
                base_volume=1.0, quote_volume=100.0, trade_count=1,
            )
            for i in range(300)
        ]
        engine = BacktestEngine(
            data_provider=CsvHistoricalDataProvider(),
            execution_model=BacktestExecutionModel(fee_pct=0.1),
        )
        entries = await measure_baseline(
            engine, candles, "TEST/USD", window_ms=5 * DAY,
        )

        assert len(entries) == len(BASELINE_STRATEGIES)
        for entry in entries:
            assert entry.walk_forward.num_windows > 1
            assert entry.walk_forward.every_window_used_one_configuration
        assert "NOT a ranking" in format_baseline_summary(entries)
