"""Regression tests for the MEXC websocket subscription lifecycle.

Covers the four defects behind the "received 1005 / Subscription limit
exceeded: 36 > 30 / Current Price 0" report:

1. MEXC native symbols must be slash-less ("BTCUSDT"), or MEXC rejects the
   channel and drops the socket (close code 1005).
2. Inbound messages (native symbol) must map back to the unified symbol so the
   price cache is keyed the way the frontend looks it up.
3. The per-connection subscription count must reset when a new socket is
   established, or a reconnect's resubscribe re-adds the full set on top of a
   stale count and falsely trips MAX_SUBSCRIPTIONS.
4. subscribe_symbol must be race-safe (no double-subscribe) and must never send
   the "*" broadcast sentinel to the exchange.
"""

import asyncio

import pytest

from app.services.websocket.mexc import MEXCWebSocketConnector
from app.services.websocket.manager import WebSocketManager


class TestSymbolNormalization:
    def test_native_symbol_strips_slash_and_uppercases(self):
        c = MEXCWebSocketConnector()
        assert c._native_symbol("BTC/USDT") == "BTCUSDT"
        assert c._native_symbol("eth/usdt") == "ETHUSDT"

    def test_channels_use_native_symbol(self):
        c = MEXCWebSocketConnector()
        assert c._get_depth_channel("BTC/USDT") == "spot@public.limit.depth.v3.api@BTCUSDT@20"
        assert c._get_trade_channel("BTC/USDT") == "spot@public.deals.v3.api@BTCUSDT"
        assert c._get_ticker_channel("BTC/USDT") == "spot@public.bookTicker.v3.api@BTCUSDT"
        assert c._get_kline_channel("BTC/USDT", "1m") == "spot@public.kline.v3.api@BTCUSDT@Min1"
        # No channel may carry the slash that makes MEXC close the socket (1005).
        for ch in (
            c._get_depth_channel("BTC/USDT"),
            c._get_trade_channel("BTC/USDT"),
            c._get_kline_channel("BTC/USDT", "5m"),
        ):
            assert "/" not in ch


class TestInboundSymbolMapping:
    def test_native_symbol_maps_back_to_unified(self):
        c = MEXCWebSocketConnector()
        c._register_symbol("BTC/USDT")  # done by subscribe_* in production
        msg = c._parse_message(
            '{"c":"spot@public.deals.v3.api@BTCUSDT","s":"BTCUSDT",'
            '"d":{"deals":[{"p":"50000","v":"0.1","t":1700000000000,"S":1}]}}'
        )
        assert msg is not None
        assert msg.symbol == "BTC/USDT"  # frontend keys on the unified form

    def test_unknown_symbol_falls_back_to_raw(self):
        c = MEXCWebSocketConnector()  # nothing registered
        msg = c._parse_message(
            '{"c":"spot@public.deals.v3.api@ETHUSDT","s":"ETHUSDT",'
            '"d":{"deals":[{"p":"3000","v":"1","t":1700000000000,"S":1}]}}'
        )
        assert msg is not None
        assert msg.symbol == "ETHUSDT"  # no crash, raw value preserved


class TestSubscriptionCountReset:
    def test_reset_clears_stale_count(self):
        c = MEXCWebSocketConnector()
        c._subscription_count = 30  # leftover from a previous socket
        c._reset_subscription_state()
        assert c._subscription_count == 0

    @pytest.mark.asyncio
    async def test_resubscribe_after_reset_does_not_trip_limit(self):
        """Six channels resubscribed after a reset stay well under the limit,
        instead of stacking onto a stale count toward 36 > 30."""
        c = MEXCWebSocketConnector()

        sent = []

        class FakeWS:
            async def send(self, payload):
                sent.append(payload)

        c._ws = FakeWS()
        # Simulate the original six-channel subscription set surviving a drop.
        channels = [
            c._get_depth_channel("BTC/USDT"),
            c._get_trade_channel("BTC/USDT"),
            c._get_ticker_channel("BTC/USDT"),
            c._get_kline_channel("BTC/USDT", "1m"),
            c._get_kline_channel("BTC/USDT", "5m"),
            c._get_kline_channel("BTC/USDT", "15m"),
        ]
        for ch in channels:
            c.state.subscriptions.add(ch)

        # Drive several reconnect cycles: each resets the count (as connect()
        # does) then resubscribes the whole set.
        for _ in range(5):
            c._reset_subscription_state()
            await c._resubscribe()
            assert c._subscription_count == len(channels) <= c.MAX_SUBSCRIPTIONS


class TestSubscribeSymbolSafety:
    @pytest.mark.asyncio
    async def test_concurrent_same_symbol_subscribes_once(self):
        m = WebSocketManager()

        class FakeConnector:
            def __init__(self):
                self.calls = 0

            async def subscribe_all(self, symbol):
                self.calls += 1
                await asyncio.sleep(0.01)  # force interleaving across the await
                return True

        fc = FakeConnector()
        m._connectors["MEXC"] = fc

        await asyncio.gather(*[m.subscribe_symbol("BTC/USDT") for _ in range(5)])
        assert fc.calls == 1
        assert m._tracked_symbols == {"BTC/USDT"}

    @pytest.mark.asyncio
    async def test_star_sentinel_not_sent_to_exchange(self):
        m = WebSocketManager()

        class FakeConnector:
            def __init__(self):
                self.calls = 0

            async def subscribe_all(self, symbol):
                self.calls += 1
                return True

        fc = FakeConnector()
        m._connectors["MEXC"] = fc

        assert await m.subscribe_symbol("*") is True
        assert fc.calls == 0
        assert "*" not in m._tracked_symbols
