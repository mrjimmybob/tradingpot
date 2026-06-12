"""Tests for the low-impact fixes: L-A, L-B, L-C.

- L-A: simulated order history is capped (bounded memory over long runs).
- L-B: epoch->datetime conversions are UTC (consistent on non-UTC servers).
- L-C: a shared ticker cache deduplicates public-API polls across dry-run bots.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.services.exchange import SimulatedExchangeService, OrderSide
from app.services.trading_engine import TradingEngine
from app.utils import ms_to_utc, utcnow


def _ticker_payload(ts=0):
    return {
        "symbol": "BTC/USDT", "bid": 100.0, "ask": 100.0, "last": 100.0,
        "baseVolume": 1.0, "timestamp": ts,
    }


def _connected_sim(ttl=0.0, ticker_cache=None, balance=10_000_000.0, ts=0):
    sim = SimulatedExchangeService(
        initial_balance=balance, ticker_cache_ttl=ttl, ticker_cache=ticker_cache
    )
    sim._connected = True
    client = AsyncMock()
    client.fetch_ticker = AsyncMock(return_value=_ticker_payload(ts))
    sim.exchange = client
    return sim, client


# ============================================================================
# L-A - Bounded simulated order history
# ============================================================================


class TestSimulatedOrderCap:
    @pytest.mark.asyncio
    async def test_oldest_orders_evicted_over_cap(self, monkeypatch):
        sim, _ = _connected_sim(ttl=0.0)
        monkeypatch.setattr(SimulatedExchangeService, "MAX_SIMULATED_ORDERS", 3)

        for _ in range(5):
            await sim.place_market_order("BTC/USDT", OrderSide.BUY, 0.0001)

        assert len(sim._simulated_orders) == 3
        # The two oldest are gone; the most recent are retained.
        assert "sim_1" not in sim._simulated_orders
        assert "sim_2" not in sim._simulated_orders
        assert "sim_5" in sim._simulated_orders

    @pytest.mark.asyncio
    async def test_recent_orders_still_queryable(self, monkeypatch):
        sim, _ = _connected_sim(ttl=0.0)
        monkeypatch.setattr(SimulatedExchangeService, "MAX_SIMULATED_ORDERS", 2)

        for _ in range(4):
            await sim.place_market_order("BTC/USDT", OrderSide.BUY, 0.0001)

        recent = await sim.get_recent_orders("BTC/USDT")
        assert len(recent) == 2
        assert await sim.get_order("sim_4", "BTC/USDT") is not None
        assert await sim.get_order("sim_1", "BTC/USDT") is None


# ============================================================================
# L-B - UTC timestamp conversion
# ============================================================================


class TestUtcConversion:
    def test_ms_to_utc_is_timezone_independent(self):
        # 1735689600000 ms == 2025-01-01T00:00:00Z, regardless of local tz.
        assert ms_to_utc(1735689600000) == datetime(2025, 1, 1, 0, 0, 0)
        assert ms_to_utc(0) == datetime(1970, 1, 1, 0, 0, 0)

    def test_utcnow_matches_utc_clock(self):
        now = utcnow()
        ref = datetime.now(timezone.utc).replace(tzinfo=None)
        assert abs((ref - now).total_seconds()) < 5
        assert now.tzinfo is None

    @pytest.mark.asyncio
    async def test_ticker_timestamp_parsed_as_utc(self):
        sim, _ = _connected_sim(ttl=0.0, ts=1735689600000)
        ticker = await sim.get_ticker("BTC/USDT")
        assert ticker.timestamp == datetime(2025, 1, 1, 0, 0, 0)


# ============================================================================
# L-C - Shared ticker cache
# ============================================================================


class TestSharedTickerCache:
    def test_engine_wires_shared_cache(self):
        engine = TradingEngine()
        e1 = engine._make_simulated_exchange(1000.0)
        e2 = engine._make_simulated_exchange(2000.0)
        assert e1._ticker_cache is e2._ticker_cache
        assert e1._ticker_cache is engine._shared_ticker_cache

    @pytest.mark.asyncio
    async def test_shared_cache_dedupes_polls_across_bots(self):
        shared = {}
        s1, c1 = _connected_sim(ttl=60.0, ticker_cache=shared)
        s2, c2 = _connected_sim(ttl=60.0, ticker_cache=shared)

        t1 = await s1.get_ticker("BTC/USDT")   # fetches, populates shared cache
        t2 = await s2.get_ticker("BTC/USDT")   # served from shared cache

        assert c1.fetch_ticker.await_count == 1
        assert c2.fetch_ticker.await_count == 0  # no duplicate poll
        assert t1.last == t2.last

    @pytest.mark.asyncio
    async def test_unshared_caches_are_independent(self):
        s1, c1 = _connected_sim(ttl=60.0)  # own cache
        s2, c2 = _connected_sim(ttl=60.0)  # own cache

        await s1.get_ticker("BTC/USDT")
        await s2.get_ticker("BTC/USDT")

        assert c1.fetch_ticker.await_count == 1
        assert c2.fetch_ticker.await_count == 1  # each polls independently
