"""Return and drawdown measured against benchmarks, plus realised exposure.

The point of this module is that it works where expectancy does not, so the
central tests are the ones proving a strategy with zero closed round trips is
fully measured rather than reported as absent or as zero.
"""
from __future__ import annotations

import math
from typing import List, Optional

import pytest

from app.backtesting.candle import Candle
from app.backtesting.execution_model import BacktestExecutionModel
from app.backtesting.validation.benchmark_relative import (
    MIN_POINTS_FOR_EXPOSURE,
    BenchmarkRelativeMeasurement,
    benchmark_limitations,
    count_windows_beating,
    estimate_exposure,
    format_benchmark_report,
    format_windows_benchmark_report,
    measure_against_benchmarks,
    measure_windows_against_benchmarks,
)
from app.backtesting.validation.benchmarks import BUY_AND_HOLD, PERIODIC_DCA
from app.backtesting.validation.measurement import Measurement, MeasurementSpan

START_MS = 1704067200000
HOUR_MS = 3_600_000
BALANCE = 10_000.0


def _candles(prices: List[float]) -> List[Candle]:
    return [
        Candle(
            timestamp=START_MS + i * HOUR_MS, datetime="d", symbol="TESTUSD",
            open=p, high=p, low=p, close=p,
            base_volume=1.0, quote_volume=p, trade_count=1,
        )
        for i, p in enumerate(prices)
    ]


def _rising(n: int = 240, start: float = 100.0, drift: float = 0.01) -> List[Candle]:
    """A rising series whose per-candle returns actually vary.

    The modulation is not decoration. A perfectly constant-drift series has
    identical returns, hence zero return variance, which makes beta undefined -
    exactly the degenerate case `estimate_exposure` withholds. Real price data
    always varies, so a fixture that does not would test nothing.
    """
    prices, p = [], start
    for i in range(n):
        prices.append(p)
        p *= (1 + drift * (1 + 0.5 * math.sin(i / 4.0)))
    return _candles(prices)


def _oscillating(n: int = 240, base: float = 100.0, amplitude: float = 0.02) -> List[Candle]:
    """Varies without net drift, so a half-deployed portfolio's asset share -
    and therefore its exposure - stays put instead of drifting toward 1.0."""
    return _candles([base * (1 + amplitude * math.sin(i / 5.0)) for i in range(n)])


def _measurement(
    candles: List[Candle],
    equity: List[float],
    num_trades: int = 0,
    strategy: str = "test_strategy",
    max_drawdown_pct: float = 0.0,
) -> Measurement:
    """A Measurement with a chosen equity curve - the input this module reads."""
    curve = tuple((c.timestamp, e) for c, e in zip(candles, equity))
    terminal = equity[-1] if equity else BALANCE
    return Measurement(
        strategy=strategy, trading_pair="TEST/USD", params_fingerprint="fp",
        span=MeasurementSpan(candles[0].timestamp, candles[-1].timestamp),
        num_candles=len(candles),
        first_candle_ms=candles[0].timestamp, last_candle_ms=candles[-1].timestamp,
        starting_balance=BALANCE, ending_balance=terminal,
        total_return_pct=(terminal - BALANCE) / BALANCE * 100.0,
        buy_and_hold_return_pct=0.0,
        num_trades=num_trades, win_rate_pct=0.0, profit_factor=0.0,
        expectancy_per_trade=0.0, max_drawdown_pct=max_drawdown_pct,
        total_fees_paid=0.0, trades=(), equity_curve=curve,
    )


def _model(fee_pct: float = 0.0) -> BacktestExecutionModel:
    return BacktestExecutionModel(fee_pct=fee_pct)


class TestExposure:
    def test_a_fully_invested_strategy_betas_to_one(self):
        candles = _rising()
        base = BALANCE / candles[0].close
        equity = [base * c.close for c in candles]
        assert estimate_exposure(
            [(c.timestamp, e) for c, e in zip(candles, equity)], candles
        ) == pytest.approx(1.0, abs=1e-6)

    def test_a_strategy_sitting_in_cash_betas_to_zero(self):
        candles = _rising()
        equity = [BALANCE] * len(candles)
        assert estimate_exposure(
            [(c.timestamp, e) for c, e in zip(candles, equity)], candles
        ) == pytest.approx(0.0, abs=1e-9)

    def test_a_half_deployed_strategy_betas_to_about_a_half(self):
        """The measure that keeps an under-deployed strategy from being read as
        a poor one."""
        candles = _oscillating()
        base = (BALANCE / 2) / candles[0].close
        equity = [BALANCE / 2 + base * c.close for c in candles]
        exposure = estimate_exposure(
            [(c.timestamp, e) for c, e in zip(candles, equity)], candles
        )
        assert exposure == pytest.approx(0.5, abs=0.05)

    def test_too_few_points_yields_unavailable_not_a_number(self):
        candles = _rising(n=MIN_POINTS_FOR_EXPOSURE - 1)
        curve = [(c.timestamp, BALANCE) for c in candles]
        assert estimate_exposure(curve, candles) is None

    def test_an_asset_that_did_not_move_yields_unavailable(self):
        """Beta against near-zero variance is an arbitrarily large number with
        no meaning, so it must be withheld."""
        candles = _candles([100.0] * 60)
        curve = [(c.timestamp, BALANCE) for c in candles]
        assert estimate_exposure(curve, candles) is None

    def test_a_perfectly_constant_return_series_yields_unavailable(self):
        """Regression: a constant-drift series has genuinely positive but
        meaningless return variance (~1e-17 of float noise). Guarding only on
        `variance > 0` let this through and produced a beta of -4.9e10."""
        prices, p = [], 100.0
        for _ in range(60):
            prices.append(p)
            p *= 1.01  # every return identical
        candles = _candles(prices)
        base = BALANCE / candles[0].close
        curve = [(c.timestamp, base * c.close) for c in candles]

        assert estimate_exposure(curve, candles) is None

    def test_an_empty_curve_yields_unavailable(self):
        assert estimate_exposure([], _rising()) is None


class TestMeasuredWithoutClosedTrades:
    def test_a_strategy_with_zero_round_trips_is_fully_measured(self):
        """The case this module exists for: expectancy is undefined, but return,
        drawdown, exposure and benchmark standing are all defined."""
        candles = _rising()
        base = BALANCE / candles[0].close
        equity = [base * c.close for c in candles]
        measurement = _measurement(candles, equity, num_trades=0, max_drawdown_pct=0.0)

        result = measure_against_benchmarks(measurement, candles, model=_model())

        assert result.has_no_closed_trades
        assert result.num_closed_trades == 0
        assert result.return_pct > 0
        assert result.exposure == pytest.approx(1.0, abs=1e-6)
        assert len(result.comparisons) == 2

    def test_the_report_says_expectancy_is_undefined_rather_than_zero(self):
        candles = _rising()
        measurement = _measurement(candles, [BALANCE] * len(candles), num_trades=0)
        report = format_benchmark_report(
            measure_against_benchmarks(measurement, candles, model=_model())
        )
        assert "closed NO round trips" in report
        assert "expectancy is undefined" in report

    def test_the_partial_exit_caveat_is_always_stated(self):
        """The documented portfolio simplification is why a scale-out strategy
        can realise P&L and still count zero trades."""
        candles = _rising()
        measurement = _measurement(candles, [BALANCE] * len(candles), num_trades=42)
        notes = benchmark_limitations(
            measure_against_benchmarks(measurement, candles, model=_model())
        )
        assert any("FULL close" in n for n in notes)


class TestBenchmarkComparison:
    def test_a_strategy_identical_to_buy_and_hold_shows_no_excess(self):
        candles = _rising()
        base = BALANCE / candles[0].open
        equity = [base * c.close for c in candles]
        measurement = _measurement(candles, equity)

        result = measure_against_benchmarks(measurement, candles, model=_model(fee_pct=0.0))
        buy_hold = next(c for c in result.comparisons if c.benchmark_label.startswith(BUY_AND_HOLD))
        assert buy_hold.excess_return_pct == pytest.approx(0.0, abs=1e-6)

    def test_a_gradual_strategy_trails_buy_and_hold_but_not_dca_in_a_rising_market(self):
        """Exactly the misreading the DCA benchmark exists to prevent."""
        candles = _rising()
        from app.backtesting.validation.benchmarks import periodic_dca_curve

        dca_curve = periodic_dca_curve(candles, BALANCE, _model(fee_pct=0.0), 24 * HOUR_MS)
        measurement = _measurement(candles, [p[1] for p in dca_curve.equity_curve])

        result = measure_against_benchmarks(
            measurement, candles, cadence_ms=24 * HOUR_MS, model=_model(fee_pct=0.0),
        )
        buy_hold = next(c for c in result.comparisons if c.benchmark_label.startswith(BUY_AND_HOLD))
        dca = next(c for c in result.comparisons if c.benchmark_label.startswith(PERIODIC_DCA))

        assert buy_hold.excess_return_pct < 0, "under-exposed vs a lump sum, as expected"
        assert dca.excess_return_pct == pytest.approx(0.0, abs=1e-6), "fair reference agrees"

    def test_both_return_and_drawdown_are_compared(self):
        candles = _rising()
        measurement = _measurement(
            candles, [BALANCE] * len(candles), max_drawdown_pct=5.0,
        )
        result = measure_against_benchmarks(measurement, candles, model=_model())
        for comparison in result.comparisons:
            assert comparison.excess_return_pct == pytest.approx(
                result.return_pct - comparison.benchmark_return_pct
            )
            assert comparison.drawdown_difference_pct == pytest.approx(
                result.max_drawdown_pct - comparison.benchmark_max_drawdown_pct
            )

    def test_lower_drawdown_than_the_benchmark_reads_as_beating_it(self):
        candles = _rising()
        measurement = _measurement(candles, [BALANCE] * len(candles), max_drawdown_pct=0.0)
        result = measure_against_benchmarks(candles=candles, measurement=measurement, model=_model())
        for comparison in result.comparisons:
            if comparison.benchmark_max_drawdown_pct > 0:
                assert comparison.beat_on_drawdown

    def test_return_per_drawdown_is_withheld_when_there_was_no_drawdown(self):
        """`inf` would read as an unbounded edge rather than as a span too
        benign to have tested the strategy."""
        candles = _rising()
        measurement = _measurement(candles, [BALANCE * 1.5] * len(candles), max_drawdown_pct=0.0)
        result = measure_against_benchmarks(measurement, candles, model=_model())
        assert result.return_per_unit_drawdown is None
        assert "-" in format_benchmark_report(result)

    def test_return_per_drawdown_is_computed_when_there_was_one(self):
        candles = _rising()
        measurement = _measurement(candles, [BALANCE * 1.2] * len(candles), max_drawdown_pct=10.0)
        result = measure_against_benchmarks(measurement, candles, model=_model())
        assert result.return_per_unit_drawdown == pytest.approx(20.0 / 10.0)


class TestLimitations:
    def _result(self, exposure: Optional[float] = 0.5) -> BenchmarkRelativeMeasurement:
        candles = _rising()
        measurement = _measurement(candles, [BALANCE] * len(candles), num_trades=5)
        result = measure_against_benchmarks(measurement, candles, model=_model())
        from dataclasses import replace

        return replace(result, exposure=exposure)

    def test_excess_return_is_never_presented_as_skill(self):
        notes = benchmark_limitations(self._result())
        assert any("NOT skill when exposure differs" in n for n in notes)

    def test_unavailable_exposure_is_explained(self):
        notes = benchmark_limitations(self._result(exposure=None))
        assert any("could not be estimated" in n for n in notes)

    def test_available_exposure_is_explained_as_a_proxy(self):
        notes = benchmark_limitations(self._result(exposure=0.5))
        assert any("proxy for average deployment" in n for n in notes)

    def test_benchmark_costs_are_declared_as_modelled(self):
        notes = benchmark_limitations(self._result())
        assert any("modelled with the same fee model" in n for n in notes)


class TestPerWindowMeasurement:
    def _windows(self, candles, chunks=4):
        """Split a series into successive Measurements, each holding its own
        slice of a fully-invested equity curve."""
        size = len(candles) // chunks
        measurements = []
        for i in range(chunks):
            window = candles[i * size:(i + 1) * size]
            base = BALANCE / window[0].close
            measurements.append(_measurement(window, [base * c.close for c in window]))
        return measurements

    def test_each_window_is_compared_to_a_benchmark_built_over_that_window(self):
        """A benchmark spanning the whole range would carry information from
        outside the window into its comparison, destroying independence."""
        candles = _rising()
        rows = measure_windows_against_benchmarks(
            self._windows(candles), candles, model=_model(fee_pct=0.0),
        )
        assert len(rows) == 4

        whole_range = measure_against_benchmarks(
            _measurement(candles, [BALANCE] * len(candles)), candles, model=_model(fee_pct=0.0),
        )
        window_bh = rows[0].relative.comparisons[0].benchmark_return_pct
        range_bh = whole_range.comparisons[0].benchmark_return_pct
        assert window_bh != pytest.approx(range_bh)

    def test_a_fully_invested_strategy_matches_buy_and_hold_in_every_window(self):
        candles = _rising()
        rows = measure_windows_against_benchmarks(
            self._windows(candles), candles, model=_model(fee_pct=0.0),
        )
        for row in rows:
            buy_hold = row.relative.comparisons[0]
            assert buy_hold.excess_return_pct == pytest.approx(0.0, abs=1e-6)

    def test_counting_windows_beaten_is_a_count_not_a_score(self):
        candles = _rising()
        rows = measure_windows_against_benchmarks(
            self._windows(candles), candles, model=_model(fee_pct=0.0),
        )
        beat, compared = count_windows_beating(rows, PERIODIC_DCA)
        assert compared == len(rows)
        assert 0 <= beat <= compared

    def test_the_report_shows_every_window_and_its_caveats(self):
        candles = _rising()
        rows = measure_windows_against_benchmarks(
            self._windows(candles), candles, model=_model(fee_pct=0.0),
        )
        report = format_windows_benchmark_report("test_strategy", rows)

        for row in rows:
            assert row.span_label in report
        assert "vs buy-and-hold" in report
        assert "vs periodic DCA" in report
        assert "excess return / drawdown difference" in report
        assert "Closed round trips across all windows: 0" in report

    def test_limitations_describe_the_whole_run_not_the_first_window(self):
        """A run whose first window closed a trade must not suppress the
        'expectancy is undefined here' note the other windows earned."""
        candles = _rising()
        measurements = self._windows(candles)
        measurements[0] = _measurement(
            candles[:len(candles) // 4], [BALANCE] * (len(candles) // 4), num_trades=3,
        )
        rows = measure_windows_against_benchmarks(
            measurements, candles, model=_model(fee_pct=0.0),
        )
        report = format_windows_benchmark_report("test_strategy", rows)

        assert "Closed round trips across all windows: 3" in report
        # Total is non-zero, so the no-trades note must NOT appear.
        assert "closed NO round trips" not in report

    def test_an_empty_window_list_reports_cleanly(self):
        report = format_windows_benchmark_report("test_strategy", ())
        assert "No windows to compare" in report
