"""Regime-conditioned reporting: bucketing measured trades by market regime.

The claims under test are that the labels come from the canonical detector
``_strategy_auto`` uses (not a second, ad hoc heuristic), that they are causal
(no lookahead), and that the bucketing arithmetic is right. Bucketing
correctness is proved against a synthetic series with known, deliberately
constructed regime transitions plus a stub detector whose output is fully
determined - so a failure points at this module rather than at a change in the
detector's thresholds.
"""
from __future__ import annotations

import math
from typing import List, Optional

import pytest

from app.backtesting.candle import Candle
from app.backtesting.data_provider import CsvHistoricalDataProvider
from app.backtesting.engine import BacktestEngine
from app.backtesting.execution_model import BacktestExecutionModel
from app.backtesting.results import TradeRecord
from app.backtesting.validation import (
    UNCLASSIFIED,
    FixedConfig,
    bucket_trades_by_regime,
    build_regime_timeline,
    canonical_regime_detector,
    format_regime_report,
    measure_fixed_config,
    regime_label_full,
    regime_label_trend,
)
from app.backtesting.validation.regime import (
    CANONICAL_MAX_BARS,
    CANONICAL_MIN_BARS,
    RegimeTimeline,
    _contiguous_runs,
)

START_MS = 1704067200000
STEP_MS = 3_600_000  # 1h


def _candles(prices: List[float], start_ms: int = START_MS, step_ms: int = STEP_MS):
    return [
        Candle(
            timestamp=start_ms + i * step_ms, datetime="d", symbol="TESTUSD",
            open=p, high=p * 1.002, low=p * 0.998, close=p,
            base_volume=1.0, quote_volume=p, trade_count=1,
        )
        for i, p in enumerate(prices)
    ]


def _trade(entry_ts: int, net_pnl: float, exit_ts: Optional[int] = None) -> TradeRecord:
    return TradeRecord(
        entry_timestamp=entry_ts, exit_timestamp=exit_ts or entry_ts + STEP_MS,
        strategy="test", entry_price=100.0, exit_price=100.0 + net_pnl, fees=0.0,
        gross_pnl=net_pnl, net_pnl=net_pnl, exit_reason="test",
    )


class _ScriptedDetector:
    """A detector returning a pre-agreed regime per call.

    Lets the bucketing tests state exactly which candle sits in which regime,
    so they test the bucketing rather than the live detector's thresholds
    (which are covered by the detector's own tests, and which would otherwise
    make these assertions fragile).
    """

    def __init__(self, trend_by_call: List[str]):
        self._script = list(trend_by_call)
        self.calls: List[tuple] = []

    def __call__(self, bars: list, current: Optional[dict]) -> dict:
        self.calls.append((len(bars), current))
        index = min(len(self.calls) - 1, len(self._script) - 1)
        return {
            "trend_state": self._script[index],
            "volatility_state": "medium",
            "liquidity_state": "normal",
            "persistence_bars": 0,
        }


class TestCanonicalDetectorIsUsed:
    def test_the_default_detector_is_the_one_strategy_auto_uses(self):
        """Spec: regime classification uses the same detector ``_strategy_auto``
        uses for live strategy selection, not a separate ad hoc one."""
        from app.services.trading_engine import TradingEngine

        detector = canonical_regime_detector()
        assert detector.__name__ == "_detect_market_regime_bar_based"
        assert detector.__func__ is TradingEngine._detect_market_regime_bar_based

    def test_the_detector_is_driven_the_way_strategy_auto_drives_it(self):
        """Same rolling window, same minimum bar count, same carried-forward
        regime - otherwise the hysteresis would see a different input sequence
        and produce different labels from the live path."""
        detector = _ScriptedDetector(["up"])
        candles = _candles([100.0 + i for i in range(CANONICAL_MAX_BARS + 30)])
        build_regime_timeline(candles, detector=detector)

        assert detector.calls, "detector was never called"
        # Never asked before it has the minimum number of bars.
        assert detector.calls[0][0] == CANONICAL_MIN_BARS
        # Never given more than the canonical rolling window.
        assert max(call[0] for call in detector.calls) == CANONICAL_MAX_BARS
        # The previous regime is carried forward so persistence/hysteresis works.
        assert detector.calls[0][1] is None
        assert all(call[1] is not None for call in detector.calls[1:])

    def test_the_real_detector_produces_labels_over_a_real_series(self):
        prices = [100.0 * (1 + 0.05 * math.sin(i / 12.0)) for i in range(200)]
        timeline = build_regime_timeline(_candles(prices))

        labels = {regime_label_trend(r) for r in timeline.regimes}
        assert labels - {UNCLASSIFIED}, "the canonical detector labelled nothing"
        for regime in timeline.regimes:
            if regime is not None:
                assert regime["trend_state"] in ("up", "down", "flat")
                assert regime["volatility_state"] in ("low", "medium", "high")
                assert regime["liquidity_state"] in ("low", "normal", "high")


class TestTimelineIsCausal:
    def test_candles_before_the_minimum_bar_count_are_unclassified(self):
        """They are not "flat" - labelling them with a real regime would invent
        data the detector never produced."""
        candles = _candles([100.0 + i for i in range(40)])
        timeline = build_regime_timeline(candles, detector=_ScriptedDetector(["up"]))

        assert all(r is None for r in timeline.regimes[: CANONICAL_MIN_BARS - 1])
        assert timeline.regimes[CANONICAL_MIN_BARS - 1] is not None
        assert regime_label_trend(timeline.regimes[0]) == UNCLASSIFIED

    def test_a_label_never_depends_on_a_later_candle(self):
        """Truncating the series must not change any label that was already
        assigned - the definition of no lookahead."""
        prices = [100.0 * (1 + 0.05 * math.sin(i / 9.0)) for i in range(160)]
        full = build_regime_timeline(_candles(prices))
        truncated = build_regime_timeline(_candles(prices[:120]))

        assert truncated.timestamps == full.timestamps[:120]
        for i in range(120):
            assert truncated.regimes[i] == full.regimes[i], f"label at candle {i} changed"

    def test_lookup_returns_the_regime_in_force_at_that_moment(self):
        candles = _candles([100.0 + i for i in range(40)])
        timeline = build_regime_timeline(candles, detector=_ScriptedDetector(["up"]))
        target = candles[30]

        assert timeline.at(target.timestamp) is timeline.regimes[30]
        # Between candles, the previous candle's regime is still in force.
        assert timeline.at(target.timestamp + 5) is timeline.regimes[30]

    def test_a_timestamp_before_the_series_is_unclassified_not_back_projected(self):
        candles = _candles([100.0 + i for i in range(40)])
        timeline = build_regime_timeline(candles, detector=_ScriptedDetector(["up"]))
        assert timeline.at(candles[0].timestamp - 1) is None

    def test_an_empty_series_yields_an_empty_timeline(self):
        timeline = build_regime_timeline([])
        assert timeline.timestamps == () and timeline.regimes == ()
        assert timeline.at(START_MS) is None


class TestBucketingCorrectness:
    """Known regime transitions, known trades, hand-checkable arithmetic."""

    def _timeline_with_known_transitions(self, count=60):
        # First 20 candles unclassified (warm-up), then "up" until candle 39,
        # then "down" from candle 40 onwards.
        script = ["up"] * 20 + ["down"] * 40
        candles = _candles([100.0 + i for i in range(count)])
        return candles, build_regime_timeline(candles, detector=_ScriptedDetector(script))

    def test_trades_land_in_the_bucket_of_their_entry_regime(self):
        candles, timeline = self._timeline_with_known_transitions()
        assert regime_label_trend(timeline.regimes[25]) == "up"
        assert regime_label_trend(timeline.regimes[50]) == "down"

        trades = [
            _trade(candles[25].timestamp, 10.0),
            _trade(candles[30].timestamp, 20.0),
            _trade(candles[50].timestamp, -5.0),
            _trade(candles[5].timestamp, 1.0),  # during warm-up
        ]
        breakdown = bucket_trades_by_regime(trades, timeline, label_fn=regime_label_trend)
        by_label = {b.label: b for b in breakdown.buckets}

        assert by_label["up"].num_trades == 2
        assert by_label["up"].expectancy_per_trade == pytest.approx(15.0)
        assert by_label["up"].total_net_pnl == pytest.approx(30.0)
        assert by_label["up"].win_rate_pct == pytest.approx(100.0)

        assert by_label["down"].num_trades == 1
        assert by_label["down"].expectancy_per_trade == pytest.approx(-5.0)
        assert by_label["down"].win_rate_pct == pytest.approx(0.0)

        assert by_label[UNCLASSIFIED].num_trades == 1
        assert breakdown.total_trades == 4

    def test_a_trade_is_attributed_to_its_entry_not_its_exit_regime(self):
        candles, timeline = self._timeline_with_known_transitions()
        # Enters while "up" at candle 30, exits deep into "down" at candle 55.
        trade = _trade(candles[30].timestamp, -50.0, exit_ts=candles[55].timestamp)

        breakdown = bucket_trades_by_regime([trade], timeline, label_fn=regime_label_trend)
        by_label = {b.label: b for b in breakdown.buckets}

        assert by_label["up"].num_trades == 1
        assert by_label["down"].num_trades == 0

    def test_regimes_the_market_entered_but_the_strategy_did_not_trade_appear_empty(self):
        """"This strategy never entered during a downtrend" must be visible as
        an empty row, not as a missing one."""
        candles, timeline = self._timeline_with_known_transitions()
        breakdown = bucket_trades_by_regime(
            [_trade(candles[25].timestamp, 5.0)], timeline, label_fn=regime_label_trend,
        )
        by_label = {b.label: b for b in breakdown.buckets}

        assert "down" in by_label
        assert by_label["down"].num_trades == 0
        note = next(n for n in breakdown.limitations() if "absence of evidence" in n)
        assert "down" in note
        # The warm-up bucket is not a regime, so listing it here would only
        # dilute the real finding.
        assert UNCLASSIFIED not in note

    def test_exposure_counts_the_candles_each_regime_held(self):
        candles, timeline = self._timeline_with_known_transitions(count=60)
        breakdown = bucket_trades_by_regime([], timeline, label_fn=regime_label_trend)
        by_label = {b.label: b for b in breakdown.buckets}

        assert by_label[UNCLASSIFIED].num_candles == CANONICAL_MIN_BARS - 1
        assert by_label["up"].num_candles == 20
        assert by_label["down"].num_candles == 60 - 20 - (CANONICAL_MIN_BARS - 1)
        assert sum(b.num_candles for b in breakdown.buckets) == 60
        assert sum(b.share_of_candles_pct for b in breakdown.buckets) == pytest.approx(100.0)

    def test_full_and_trend_labels_describe_the_same_trades_at_different_grain(self):
        candles, timeline = self._timeline_with_known_transitions()
        trades = [_trade(candles[i].timestamp, 1.0) for i in (25, 30, 50, 55)]

        trend = bucket_trades_by_regime(trades, timeline, label_fn=regime_label_trend)
        full = bucket_trades_by_regime(trades, timeline, label_fn=regime_label_full)

        assert trend.total_trades == full.total_trades == 4
        assert {b.label for b in full.buckets} == {"up/medium/normal", "down/medium/normal",
                                                   UNCLASSIFIED}

    def test_small_samples_are_flagged(self):
        candles, timeline = self._timeline_with_known_transitions()
        breakdown = bucket_trades_by_regime(
            [_trade(candles[25].timestamp, 1.0)], timeline, label_fn=regime_label_trend,
        )
        by_label = {b.label: b for b in breakdown.buckets}

        assert by_label["up"].is_small_sample
        assert any("not a finding" in n for n in breakdown.limitations())


class TestPerRegimeDrawdown:
    def test_a_peak_never_leaks_from_one_stretch_of_a_regime_into_a_later_one(self):
        """Concatenating disjoint stretches would carry the first stretch's peak
        into the second's trough and report a drawdown that never happened."""
        candles = _candles([100.0] * 60)
        # "up" for candles 19-29, "down" for 30-39, "up" again for 40-59.
        script = ["up"] * 11 + ["down"] * 10 + ["up"] * 30
        timeline = build_regime_timeline(candles, detector=_ScriptedDetector(script))

        # Equity climbs to 187 through the first "up" stretch, collapses to 50
        # on the way into "down", then climbs again through the second "up".
        equity = []
        for i, candle in enumerate(candles):
            if i < 30:
                value = 100.0 + i * 3.0
            elif i < 40:
                value = 50.0
            else:
                value = 50.0 + (i - 40)
            equity.append((candle.timestamp, value))

        breakdown = bucket_trades_by_regime(
            [], timeline, equity_curve=equity, label_fn=regime_label_trend,
        )
        by_label = {b.label: b for b in breakdown.buckets}

        # Equity rose monotonically inside both "up" stretches. Only a leaked
        # peak (187 from the first stretch, against 50 in the second) could
        # produce a drawdown here.
        assert by_label["up"].max_drawdown_pct == pytest.approx(0.0)

    def test_a_drop_on_the_way_into_a_regime_is_attributed_to_that_regime(self):
        """The collapse happens between the last "up" candle and the first
        "down" one. Measuring each stretch only from its own first mark would
        drop that move into the crack between the two and count it nowhere."""
        candles = _candles([100.0] * 60)
        script = ["up"] * 11 + ["down"] * 10 + ["up"] * 30
        timeline = build_regime_timeline(candles, detector=_ScriptedDetector(script))

        equity = []
        for i, candle in enumerate(candles):
            value = 200.0 if i < 30 else 50.0
            equity.append((candle.timestamp, value))

        breakdown = bucket_trades_by_regime(
            [], timeline, equity_curve=equity, label_fn=regime_label_trend,
        )
        by_label = {b.label: b for b in breakdown.buckets}

        assert by_label["down"].max_drawdown_pct == pytest.approx(75.0)

    def test_a_drawdown_inside_one_stretch_is_captured(self):
        candles = _candles([100.0] * 40)
        timeline = build_regime_timeline(candles, detector=_ScriptedDetector(["down"]))
        equity = []
        for i, candle in enumerate(candles):
            value = 200.0 if i < 30 else 100.0  # 50% drawdown inside the stretch
            equity.append((candle.timestamp, value))

        breakdown = bucket_trades_by_regime(
            [], timeline, equity_curve=equity, label_fn=regime_label_trend,
        )
        by_label = {b.label: b for b in breakdown.buckets}
        assert by_label["down"].max_drawdown_pct == pytest.approx(50.0)

    def test_no_equity_curve_reports_zero_rather_than_failing(self):
        candles = _candles([100.0] * 40)
        timeline = build_regime_timeline(candles, detector=_ScriptedDetector(["up"]))
        breakdown = bucket_trades_by_regime([], timeline, label_fn=regime_label_trend)
        assert all(b.max_drawdown_pct == 0.0 for b in breakdown.buckets)

    def test_contiguous_runs_finds_every_stretch(self):
        assert _contiguous_runs(["a", "a", "b", "a"], "a") == [(0, 1), (3, 3)]
        assert _contiguous_runs(["b", "b"], "a") == []
        assert _contiguous_runs(["a", "a"], "a") == [(0, 1)]
        assert _contiguous_runs([], "a") == []


class TestRegimeReport:
    def test_report_shows_every_bucket_with_its_exposure_and_caveats(self):
        candles = _candles([100.0 + i for i in range(60)])
        timeline = build_regime_timeline(
            candles, detector=_ScriptedDetector(["up"] * 20 + ["down"] * 40),
        )
        trades = [_trade(candles[25].timestamp, 5.0), _trade(candles[50].timestamp, -3.0)]
        report = format_regime_report(
            bucket_trades_by_regime(trades, timeline, label_fn=regime_label_trend)
        )

        assert "up" in report and "down" in report
        assert "Exposure" in report and "Expectancy" in report and "MaxDD%" in report
        assert "Total trades bucketed: 2" in report
        assert "entered entirely" not in report
        assert "at its ENTRY" in report  # the attribution caveat travels with the table
        assert "60-second bars" in report  # the bar-size caveat does too

    def test_report_survives_an_empty_breakdown(self):
        report = format_regime_report(bucket_trades_by_regime([], RegimeTimeline((), ())))
        assert "Total trades bucketed: 0" in report


class TestEndToEndWithTheRealEngine:
    @pytest.mark.asyncio
    async def test_a_real_measurements_trades_all_land_in_a_bucket(self):
        prices = [100.0 * (1 + 0.05 * math.sin(i / 12.0)) for i in range(300)]
        candles = _candles(prices)
        engine = BacktestEngine(
            data_provider=CsvHistoricalDataProvider(),
            execution_model=BacktestExecutionModel(fee_pct=0.1),
        )
        config = FixedConfig("mean_reversion", "TEST/USD", {
            "bar_interval_seconds": 0, "regime_filter_enabled": False,
            "bollinger_period": 10, "atr_period": 10, "cooldown_seconds": 0,
            "decision_score_threshold": 0.0,
        })
        measurement = await measure_fixed_config(engine, candles, config)
        assert measurement.num_trades > 0

        timeline = build_regime_timeline(candles)
        breakdown = bucket_trades_by_regime(
            measurement.trades, timeline, measurement.equity_curve, regime_label_full,
        )

        # Every trade is accounted for exactly once, and the per-bucket P&L
        # reconstructs the measurement's own total.
        assert breakdown.total_trades == measurement.num_trades
        assert sum(b.total_net_pnl for b in breakdown.buckets) == pytest.approx(
            sum(t.net_pnl for t in measurement.trades)
        )

    @pytest.mark.asyncio
    async def test_bucketing_does_not_touch_the_measurement_it_reads(self):
        prices = [100.0 * (1 + 0.05 * math.sin(i / 12.0)) for i in range(300)]
        candles = _candles(prices)
        engine = BacktestEngine(
            data_provider=CsvHistoricalDataProvider(),
            execution_model=BacktestExecutionModel(fee_pct=0.1),
        )
        config = FixedConfig("mean_reversion", "TEST/USD", {
            "bar_interval_seconds": 0, "regime_filter_enabled": False,
            "bollinger_period": 10, "atr_period": 10, "cooldown_seconds": 0,
            "decision_score_threshold": 0.0,
        })
        measurement = await measure_fixed_config(engine, candles, config)
        before_trades = list(measurement.trades)
        before_curve = list(measurement.equity_curve)

        timeline = build_regime_timeline(candles)
        bucket_trades_by_regime(
            measurement.trades, timeline, measurement.equity_curve, regime_label_full,
        )

        assert list(measurement.trades) == before_trades
        assert list(measurement.equity_curve) == before_curve
