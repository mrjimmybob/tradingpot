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
from datetime import datetime

import pytest

from app.services.websocket.mexc import MEXCWebSocketConnector
from app.services.websocket.manager import WebSocketManager
from app.services.exchange import Ticker


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

    @pytest.mark.asyncio
    async def test_rest_mode_subscribe_tracks_without_connector(self):
        """With no stream connector (REST feed), subscribing just tracks the
        symbol so the poll loop can pick it up."""
        m = WebSocketManager()  # no connectors registered
        assert await m.subscribe_symbol("BTC/USDT") is True
        assert "BTC/USDT" in m._tracked_symbols

    @pytest.mark.asyncio
    @pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
    async def test_blank_symbol_is_not_registered(self, blank):
        """A blank symbol must be rejected at the registration chokepoint, so it
        can never reach the REST poll loop (which called fetch_ticker('')
        repeatedly: 'mexc does not have market symbol')."""
        m = WebSocketManager()
        assert await m.subscribe_symbol(blank) is False
        assert blank not in m._tracked_symbols
        assert m._tracked_symbols == set()

    @pytest.mark.asyncio
    async def test_blank_symbol_from_client_message_not_tracked(self):
        """An empty symbol in a frontend WS subscribe payload is dropped and
        never tracked (the production entry point for the empty symbol)."""
        import json

        m = WebSocketManager()

        class FakeWS:
            async def send_json(self, *_a, **_k):
                pass

        ws = FakeWS()
        m._client_subscriptions[ws] = set()
        await m.handle_client_message(
            ws, json.dumps({"action": "subscribe", "symbols": ["", "BTC/USDT"]})
        )
        assert "" not in m._tracked_symbols
        assert "BTC/USDT" in m._tracked_symbols
        assert "" not in m._client_subscriptions[ws]


class TestRestPriceFeed:
    @pytest.mark.asyncio
    async def test_poll_loop_broadcasts_and_caches_price(self):
        m = WebSocketManager()

        class FakeRest:
            async def get_ticker(self, symbol):
                return Ticker(
                    symbol=symbol, bid=63851.0, ask=63852.0, last=63851.5,
                    volume=10.0, timestamp=datetime.utcnow(),
                )

        m._rest_exchange = FakeRest()
        m._tracked_symbols.add("BTC/USDT")
        m._running = True

        seen = []

        async def fake_broadcast(msg):
            seen.append(msg)
        m.broadcast = fake_broadcast

        task = asyncio.create_task(m._price_poll_loop())
        await asyncio.sleep(0.2)
        m._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        prices = [s for s in seen if s.get("type") == "price_update"]
        assert prices, "no price_update broadcast"
        assert prices[0]["symbol"] == "BTC/USDT"
        assert prices[0]["price"] == 63851.5
        assert m._price_cache["BTC/USDT"].price == 63851.5

    @pytest.mark.asyncio
    async def test_poll_loop_never_fetches_blank_symbol(self):
        """Defense in depth: a blank symbol present in _tracked_symbols must not
        be passed to get_ticker (it would error every tick)."""
        m = WebSocketManager()

        fetched = []

        class FakeRest:
            async def get_ticker(self, symbol):
                fetched.append(symbol)
                return Ticker(symbol=symbol, bid=1.0, ask=1.0, last=1.0,
                              volume=0.0, timestamp=datetime.utcnow())

        m._rest_exchange = FakeRest()
        m._tracked_symbols.update({"", "   ", "*", "BTC/USDT"})  # blanks + sentinel
        m._running = True

        async def fake_broadcast(_msg):
            pass
        m.broadcast = fake_broadcast

        task = asyncio.create_task(m._price_poll_loop())
        await asyncio.sleep(0.2)
        m._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "" not in fetched and "   " not in fetched and "*" not in fetched
        assert fetched and set(fetched) == {"BTC/USDT"}


class TestSubscriptionResponseHandling:
    def test_blocked_rejection_is_not_treated_as_success(self, caplog):
        """MEXC returns code:0 even for rejections; a 'Blocked!' response must
        be logged as rejected, not silently confirmed."""
        import logging

        c = MEXCWebSocketConnector()
        blocked = (
            '{"id":0,"code":0,"msg":"Not Subscribed successfully! '
            '[spot@public.bookTicker.v3.api@BTCUSDT].  Reason： Blocked! "}'
        )
        with caplog.at_level(logging.WARNING):
            result = c._parse_message(blocked)
        assert result is None  # control message, not market data
        assert any("rejected" in r.message.lower() for r in caplog.records)
