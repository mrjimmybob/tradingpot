"""Benchmark equity curves: buy-and-hold and periodic DCA.

Correctness is asserted two ways. Hand-computable cases pin the arithmetic
(a flat price series has an exactly predictable terminal equity given the fee
model), and structural invariants pin the behaviour that actually matters -
gradual deployment must beat a lump sum into a falling market and lose to it in
a rising one, which is the whole reason the DCA benchmark exists.
"""
from __future__ import annotations

from typing import List

import pytest

from app.backtesting.candle import Candle
from app.backtesting.execution_model import BacktestExecutionModel
from app.backtesting.validation.benchmarks import (
    BUY_AND_HOLD,
    DEFAULT_DCA_CADENCE_MS,
    MS_PER_DAY,
    PERIODIC_DCA,
    build_benchmarks,
    buy_and_hold_curve,
    periodic_dca_curve,
)
from app.backtesting.validation.benchmarks import _instalment_indices

START_MS = 1704067200000
HOUR_MS = 3_600_000
BALANCE = 10_000.0


def _candles(prices: List[float], step_ms: int = HOUR_MS) -> List[Candle]:
    return [
        Candle(
            timestamp=START_MS + i * step_ms, datetime="d", symbol="TESTUSD",
            open=p, high=p, low=p, close=p,
            base_volume=1.0, quote_volume=p, trade_count=1,
        )
        for i, p in enumerate(prices)
    ]


def _flat(n: int = 240, price: float = 100.0) -> List[Candle]:
    return _candles([price] * n)


def _rising(n: int = 240, start: float = 100.0, drift: float = 0.01) -> List[Candle]:
    prices, p = [], start
    for _ in range(n):
        prices.append(p)
        p *= (1 + drift)
    return _candles(prices)


def _falling(n: int = 240, start: float = 100.0, drift: float = 0.01) -> List[Candle]:
    prices, p = [], start
    for _ in range(n):
        prices.append(p)
        p *= (1 - drift)
    return _candles(prices)


def _model(fee_pct: float = 0.1) -> BacktestExecutionModel:
    return BacktestExecutionModel(fee_pct=fee_pct)


class TestBuyAndHold:
    def test_flat_price_leaves_only_the_entry_fee(self):
        """Exactly hand-computable: spending B inclusive of a fee rate f buys
        B/(1+f) of notional, so a flat price ends at B/(1+f)."""
        curve = buy_and_hold_curve(_flat(), BALANCE, _model(fee_pct=0.1))
        assert curve.terminal_equity == pytest.approx(BALANCE / 1.001, rel=1e-9)
        assert curve.terminal_return_pct == pytest.approx(-0.0999, abs=1e-3)

    def test_zero_fees_leave_the_balance_untouched_on_a_flat_price(self):
        curve = buy_and_hold_curve(_flat(), BALANCE, _model(fee_pct=0.0))
        assert curve.terminal_equity == pytest.approx(BALANCE, rel=1e-9)

    def test_fees_reduce_the_benchmark(self):
        """A cost-free benchmark is a handicap, not a benchmark."""
        free = buy_and_hold_curve(_rising(), BALANCE, _model(fee_pct=0.0))
        costed = buy_and_hold_curve(_rising(), BALANCE, _model(fee_pct=0.5))
        assert costed.terminal_equity < free.terminal_equity

    def test_it_tracks_the_asset_from_the_first_candle(self):
        candles = _rising(n=50)
        curve = buy_and_hold_curve(candles, BALANCE, _model(fee_pct=0.0))
        price_ratio = candles[-1].close / candles[0].open
        assert curve.terminal_equity == pytest.approx(BALANCE * price_ratio, rel=1e-9)

    def test_one_equity_point_per_candle_so_curves_align(self):
        candles = _flat(n=37)
        curve = buy_and_hold_curve(candles, BALANCE, _model())
        assert len(curve.equity_curve) == len(candles)
        assert [p[0] for p in curve.equity_curve] == [c.timestamp for c in candles]

    def test_drawdown_is_reported_for_a_falling_market(self):
        curve = buy_and_hold_curve(_falling(n=100), BALANCE, _model(fee_pct=0.0))
        assert curve.max_drawdown_pct > 50.0

    def test_an_empty_series_yields_an_empty_curve_not_a_crash(self):
        curve = buy_and_hold_curve([], BALANCE, _model())
        assert curve.equity_curve == ()
        assert curve.terminal_equity == BALANCE
        assert curve.max_drawdown_pct == 0.0

    def test_it_is_deterministic(self):
        a = buy_and_hold_curve(_rising(), BALANCE, _model())
        b = buy_and_hold_curve(_rising(), BALANCE, _model())
        assert a.equity_curve == b.equity_curve


class TestPeriodicDca:
    def test_flat_price_matches_buy_and_hold_because_only_fees_differ(self):
        """With a flat price, when you buy is irrelevant - both benchmarks end
        at balance minus the same total fee rate."""
        dca = periodic_dca_curve(_flat(), BALANCE, _model(fee_pct=0.1), 24 * HOUR_MS)
        assert dca.terminal_equity == pytest.approx(BALANCE / 1.001, rel=1e-6)

    def test_it_loses_to_a_lump_sum_in_a_rising_market(self):
        """Deploying gradually means buying at higher average prices. This is
        exactly the exposure effect that makes buy-and-hold an unfair reference
        for an accumulator."""
        candles = _rising()
        lump = buy_and_hold_curve(candles, BALANCE, _model(fee_pct=0.0))
        dca = periodic_dca_curve(candles, BALANCE, _model(fee_pct=0.0), 24 * HOUR_MS)
        assert dca.terminal_equity < lump.terminal_equity

    def test_it_beats_a_lump_sum_in_a_falling_market(self):
        candles = _falling()
        lump = buy_and_hold_curve(candles, BALANCE, _model(fee_pct=0.0))
        dca = periodic_dca_curve(candles, BALANCE, _model(fee_pct=0.0), 24 * HOUR_MS)
        assert dca.terminal_equity > lump.terminal_equity

    def test_it_draws_down_less_than_a_lump_sum_while_still_deploying(self):
        candles = _falling()
        lump = buy_and_hold_curve(candles, BALANCE, _model(fee_pct=0.0))
        dca = periodic_dca_curve(candles, BALANCE, _model(fee_pct=0.0), 24 * HOUR_MS)
        assert dca.max_drawdown_pct < lump.max_drawdown_pct

    def test_it_never_sells(self):
        """Equity on a flat price can only move by fees, never by a disposal."""
        candles = _flat()
        dca = periodic_dca_curve(candles, BALANCE, _model(fee_pct=0.0), 24 * HOUR_MS)
        values = [point[1] for point in dca.equity_curve]
        assert max(values) - min(values) < 1e-6

    def test_capital_is_fully_deployed_by_the_end(self):
        """Instalments are sized from the span up front, so they sum to the
        starting balance rather than leaving a residue of idle cash."""
        candles = _flat(n=240)
        dca = periodic_dca_curve(candles, BALANCE, _model(fee_pct=0.0), 24 * HOUR_MS)
        # Flat price: fully deployed means terminal equity equals the balance.
        assert dca.terminal_equity == pytest.approx(BALANCE, rel=1e-9)

    def test_a_cadence_longer_than_the_span_degenerates_to_a_lump_sum(self):
        candles = _rising(n=24)  # 24 hours
        dca = periodic_dca_curve(candles, BALANCE, _model(fee_pct=0.0), 365 * MS_PER_DAY)
        lump = buy_and_hold_curve(candles, BALANCE, _model(fee_pct=0.0))
        assert dca.terminal_equity == pytest.approx(lump.terminal_equity, rel=1e-9)

    def test_a_shorter_cadence_deploys_more_often(self):
        candles = _flat(n=240)
        weekly = _instalment_indices(candles, 7 * MS_PER_DAY)
        daily = _instalment_indices(candles, MS_PER_DAY)
        assert len(daily) > len(weekly) >= 1
        assert daily[0] == 0 and weekly[0] == 0

    def test_a_data_gap_costs_one_instalment_not_a_burst(self):
        """Boundaries are measured from the start, so a gap must not queue up
        every missed instalment and fire them all on the next candle."""
        prices = [100.0] * 5
        candles = _candles(prices, step_ms=MS_PER_DAY)
        # Jump 10 days ahead after the 5th candle.
        candles.append(Candle(
            timestamp=candles[-1].timestamp + 10 * MS_PER_DAY, datetime="d",
            symbol="TESTUSD", open=100.0, high=100.0, low=100.0, close=100.0,
            base_volume=1.0, quote_volume=100.0, trade_count=1,
        ))
        indices = _instalment_indices(candles, MS_PER_DAY)
        assert indices.count(5) == 1

    def test_cadence_is_reported_on_the_curve(self):
        dca = periodic_dca_curve(_flat(), BALANCE, _model(), 7 * MS_PER_DAY)
        assert dca.parameters == "cadence 7d"
        assert "cadence 7d" in dca.label

    def test_a_non_positive_cadence_is_rejected(self):
        with pytest.raises(ValueError, match="cadence_ms"):
            periodic_dca_curve(_flat(), BALANCE, _model(), 0)

    def test_an_empty_series_yields_an_empty_curve(self):
        dca = periodic_dca_curve([], BALANCE, _model())
        assert dca.equity_curve == ()

    def test_it_is_deterministic(self):
        a = periodic_dca_curve(_rising(), BALANCE, _model(), 24 * HOUR_MS)
        b = periodic_dca_curve(_rising(), BALANCE, _model(), 24 * HOUR_MS)
        assert a.equity_curve == b.equity_curve


class TestBenchmarkSet:
    def test_the_standard_set_is_both_benchmarks_in_a_fixed_order(self):
        benchmarks = build_benchmarks(_rising(), BALANCE, _model())
        assert [b.name for b in benchmarks] == [BUY_AND_HOLD, PERIODIC_DCA]

    def test_the_default_cadence_is_weekly_and_documented(self):
        assert DEFAULT_DCA_CADENCE_MS == 7 * MS_PER_DAY
        benchmarks = build_benchmarks(_rising(), BALANCE, _model())
        assert benchmarks[1].parameters == "cadence 7d"

    def test_benchmarks_do_not_depend_on_any_strategy(self):
        """Pure functions of candles and costs - no engine, no dispatch. Two
        callers measuring different strategies over the same candles must get
        identical benchmark curves."""
        candles = _rising()
        first = build_benchmarks(candles, BALANCE, _model())
        second = build_benchmarks(candles, BALANCE, _model())
        assert [b.equity_curve for b in first] == [b.equity_curve for b in second]
