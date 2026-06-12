"""Tests for the funding-rate diagnostic and exchange funding-rate access.

Covers:
- Pure funding statistics (compute_funding_stats)
- ExchangeService funding-rate methods (parsing, capability gating, failures)
- FundingRateDiagnostic.analyze (net-of-cost analysis, viability, edge cases)

All CCXT interactions are mocked - no real exchange calls.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import ccxt.async_support as ccxt

from app.services.exchange import ExchangeService, FundingRate
from app.services.funding_diagnostic import (
    compute_funding_stats,
    FundingRateDiagnostic,
)


# ============================================================================
# Pure statistics
# ============================================================================


class TestComputeFundingStats:
    def test_basic_statistics(self):
        rates = [0.0001, 0.0002, -0.0001, 0.0003]
        stats = compute_funding_stats(rates, interval_hours=8.0)

        assert stats.count == 4
        assert stats.mean_rate == pytest.approx(0.000125)
        assert stats.min_rate == -0.0001
        assert stats.max_rate == 0.0003
        assert stats.positive_pct == pytest.approx(75.0)
        assert stats.negative_pct == pytest.approx(25.0)
        # 8h interval -> 3/day -> 1095/year
        assert stats.periods_per_year == pytest.approx(1095.0)
        assert stats.annualized_mean_pct == pytest.approx(0.000125 * 1095 * 100)

    def test_empty_series_is_all_zero(self):
        stats = compute_funding_stats([], interval_hours=8.0)
        assert stats.count == 0
        assert stats.mean_rate == 0.0
        assert stats.annualized_mean_pct == 0.0
        assert stats.positive_pct == 0.0

    def test_single_value(self):
        stats = compute_funding_stats([0.0005], interval_hours=8.0)
        assert stats.count == 1
        assert stats.mean_rate == 0.0005
        assert stats.stdev_rate == 0.0  # pstdev of one element

    def test_zero_interval_defaults_to_eight_hours(self):
        stats = compute_funding_stats([0.0001], interval_hours=0.0)
        assert stats.interval_hours == 8.0


# ============================================================================
# Exchange funding-rate access
# ============================================================================


def _connected_funding_exchange(history_rows, supports=True):
    """Build a connected ExchangeService with mocked ccxt funding methods."""
    mock_exchange = AsyncMock()
    mock_exchange.load_markets = AsyncMock()
    mock_exchange.has = {
        "fetchFundingRateHistory": supports,
        "fetchFundingRate": supports,
    }
    mock_exchange.fetch_funding_rate_history = AsyncMock(return_value=history_rows)
    return mock_exchange


class TestToSwapSymbol:
    def test_spot_to_linear_perp(self):
        assert ExchangeService.to_swap_symbol("BTC/USDT") == "BTC/USDT:USDT"

    def test_already_swap_unchanged(self):
        assert ExchangeService.to_swap_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"

    def test_non_pair_unchanged(self):
        assert ExchangeService.to_swap_symbol("BTCUSDT") == "BTCUSDT"


class TestFundingRateHistory:
    @pytest.mark.asyncio
    async def test_history_parsed(self):
        rows = [
            {"symbol": "BTC/USDT:USDT", "fundingRate": 0.0001,
             "timestamp": int(datetime(2026, 1, 1).timestamp() * 1000), "interval": "8h"},
            {"symbol": "BTC/USDT:USDT", "fundingRate": -0.0002,
             "timestamp": int(datetime(2026, 1, 2).timestamp() * 1000), "interval": "8h"},
        ]
        mock_exchange = _connected_funding_exchange(rows)
        with patch("app.services.exchange.ccxt") as mock_ccxt:
            mock_ccxt.mexc = Mock(return_value=mock_exchange)
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                history = await service.get_funding_rate_history("BTC/USDT:USDT", limit=2)

        assert len(history) == 2
        assert isinstance(history[0], FundingRate)
        assert history[0].funding_rate == 0.0001
        assert history[0].interval_hours == 8.0
        assert history[1].funding_rate == -0.0002
        mock_exchange.fetch_funding_rate_history.assert_awaited_once_with(
            "BTC/USDT:USDT", None, 2
        )

    @pytest.mark.asyncio
    async def test_unsupported_exchange_returns_empty(self):
        mock_exchange = _connected_funding_exchange([], supports=False)
        with patch("app.services.exchange.ccxt") as mock_ccxt:
            mock_ccxt.mexc = Mock(return_value=mock_exchange)
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                assert service.supports_funding_rates() is False
                history = await service.get_funding_rate_history("BTC/USDT:USDT")

        assert history == []
        mock_exchange.fetch_funding_rate_history.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_not_connected_returns_empty(self):
        service = ExchangeService(exchange_id="mexc")
        assert await service.get_funding_rate_history("BTC/USDT:USDT") == []

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_empty(self):
        mock_exchange = _connected_funding_exchange([])
        mock_exchange.fetch_funding_rate_history = AsyncMock(
            side_effect=ccxt.ExchangeError("boom")
        )
        with patch("app.services.exchange.ccxt") as mock_ccxt:
            mock_ccxt.mexc = Mock(return_value=mock_exchange)
            mock_ccxt.ExchangeError = ccxt.ExchangeError
            mock_ccxt.NetworkError = ccxt.NetworkError
            mock_ccxt.RateLimitExceeded = ccxt.RateLimitExceeded
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                history = await service.get_funding_rate_history("BTC/USDT:USDT")

        assert history == []

    @pytest.mark.asyncio
    async def test_get_funding_rate_parses_interval(self):
        row = {"symbol": "BTC/USDT:USDT", "fundingRate": 0.0003,
               "timestamp": int(datetime(2026, 1, 1).timestamp() * 1000), "interval": "4h"}
        mock_exchange = _connected_funding_exchange([])
        mock_exchange.fetch_funding_rate = AsyncMock(return_value=row)
        with patch("app.services.exchange.ccxt") as mock_ccxt:
            mock_ccxt.mexc = Mock(return_value=mock_exchange)
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                rate = await service.get_funding_rate("BTC/USDT:USDT")

        assert rate is not None
        assert rate.funding_rate == 0.0003
        assert rate.interval_hours == 4.0


# ============================================================================
# Diagnostic analysis
# ============================================================================


class FakeExchange:
    """Minimal exchange stand-in for diagnostic tests."""

    def __init__(self, history):
        self._history = history

    def to_swap_symbol(self, symbol):
        return ExchangeService.to_swap_symbol(symbol)

    async def get_funding_rate_history(self, symbol, limit=200, since=None):
        return list(self._history)


def _history(rates, interval_hours=8.0):
    return [
        FundingRate("BTC/USDT:USDT", r, datetime(2026, 1, 1 + i), interval_hours)
        for i, r in enumerate(rates)
    ]


class TestFundingDiagnostic:
    @pytest.mark.asyncio
    async def test_no_data_returns_none(self):
        diag = FundingRateDiagnostic(FakeExchange([]), exchange_fee_pct=0.1)
        assert await diag.analyze("BTC/USDT") is None

    @pytest.mark.asyncio
    async def test_low_funding_not_viable(self):
        diag = FundingRateDiagnostic(
            FakeExchange(_history([0.0001, 0.0002, 0.0001, 0.0003])),
            exchange_fee_pct=0.1,
        )
        report = await diag.analyze("BTC/USDT", assumed_holding_periods=3, notional_usd=1000.0)

        assert report is not None
        assert report.swap_symbol == "BTC/USDT:USDT"
        # 0.1% per side -> 0.2% round trip
        assert report.roundtrip_cost_pct == pytest.approx(0.2)
        # mean 0.000175 -> 0.0175%/period; *3 = 0.0525% gross < 0.2% cost
        assert report.net_funding_pct < 0
        assert report.viable is False
        assert report.profitable_window_pct == pytest.approx(0.0)
        assert report.best_period_rate == 0.0003
        assert report.worst_period_rate == 0.0001

    @pytest.mark.asyncio
    async def test_high_funding_viable(self):
        diag = FundingRateDiagnostic(
            FakeExchange(_history([0.001, 0.001, 0.001, 0.001])),
            exchange_fee_pct=0.1,
        )
        report = await diag.analyze("BTC/USDT", assumed_holding_periods=3, notional_usd=1000.0)

        assert report is not None
        # gross = 0.1% * 3 = 0.3% > 0.2% round-trip cost
        assert report.gross_funding_pct == pytest.approx(0.3)
        assert report.net_funding_pct == pytest.approx(0.1)
        assert report.viable is True
        assert report.profitable_window_pct == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_summary_is_readable(self):
        diag = FundingRateDiagnostic(
            FakeExchange(_history([0.001, 0.001, 0.001])), exchange_fee_pct=0.1
        )
        report = await diag.analyze("BTC/USDT")
        text = report.summary()
        assert "Funding diagnostic for BTC/USDT" in text
        assert "Average funding" in text
        assert "Verdict" in text
