"""Tests for live execution hardening.

Covers fill accuracy (partial fills, exchange-reported fee currency/cost)
and live balance reconciliation. Order pre-flight tests live in
test_exchange_wrapper.py.
"""

import pytest
from contextlib import ExitStack
from unittest.mock import Mock, AsyncMock, patch

from app.services.trading_engine import TradingEngine, TradeSignal
from app.services.exchange import Balance
from app.models import Bot, BotStatus, Trade, TradeSide


@pytest.fixture
def engine():
    return TradingEngine()


@pytest.fixture
def mock_bot():
    bot = Mock(spec=Bot)
    bot.id = 1
    bot.name = "Test Bot"
    bot.trading_pair = "BTC/USDT"
    bot.strategy = "test_strategy"
    bot.budget = 10000.0
    bot.current_balance = 10000.0
    bot.is_dry_run = True
    bot.status = BotStatus.RUNNING
    bot.exchange_fee = 0.001
    bot.is_simulated = True
    return bot


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.add = Mock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    mock_result = Mock()
    mock_result.scalar_one_or_none = Mock(return_value=None)
    mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[])))
    session.execute = AsyncMock(return_value=mock_result)
    return session


@pytest.fixture
def mock_services():
    services = {}

    risk_check = Mock()
    risk_check.ok = True
    risk_check.action = "allow"
    portfolio_risk = AsyncMock()
    portfolio_risk.check_portfolio_risk = AsyncMock(return_value=risk_check)
    services["portfolio_risk"] = portfolio_risk

    capacity_check = Mock()
    capacity_check.ok = True
    capacity_check.adjusted_amount = None
    strategy_capacity = AsyncMock()
    strategy_capacity.check_capacity_for_trade = AsyncMock(return_value=capacity_check)
    services["strategy_capacity"] = strategy_capacity

    mock_trade = Mock(spec=Trade)
    mock_trade.id = 1
    mock_trade.base_amount = 0.01
    mock_trade.get_cost_basis_per_unit = Mock(return_value=50000.0)
    trade_recorder = AsyncMock()
    trade_recorder.record_trade = AsyncMock(return_value=mock_trade)
    services["trade_recorder"] = trade_recorder

    tax_engine = AsyncMock()
    tax_engine.process_buy = AsyncMock()
    tax_engine.process_sell = AsyncMock(return_value=[])
    services["tax_engine"] = tax_engine

    invariant_validator = AsyncMock()
    invariant_validator.validate_trade = AsyncMock()
    services["invariant_validator"] = invariant_validator

    wallet = AsyncMock()
    wallet.record_trade_result = AsyncMock(return_value=(True, "Success"))
    services["wallet"] = wallet

    return services


def service_patches(mock_services):
    """Patches for all service dependencies of _execute_trade."""
    return [
        patch("app.services.trading_engine.PortfolioRiskService",
              return_value=mock_services["portfolio_risk"]),
        patch("app.services.trading_engine.StrategyCapacityService",
              return_value=mock_services["strategy_capacity"]),
        patch("app.services.trading_engine.TradeRecorderService",
              return_value=mock_services["trade_recorder"]),
        patch("app.services.trading_engine.FIFOTaxEngine",
              return_value=mock_services["tax_engine"]),
        patch("app.services.trading_engine.LedgerInvariantService",
              return_value=mock_services["invariant_validator"]),
        patch("app.services.trading_engine.VirtualWalletService",
              return_value=mock_services["wallet"]),
        patch("app.services.trading_engine.CSVExportService"),
    ]


class FakeExchange:
    """Exchange returning a configurable order result."""

    def __init__(self, filled_ratio: float = 1.0, fee_currency: str = None):
        self.filled_ratio = filled_ratio
        self.fee_currency = fee_currency

    async def place_market_order(self, trading_pair, side, amount):
        filled = amount * self.filled_ratio
        order = Mock()
        order.id = "live_1"
        order.amount = amount
        order.filled = filled
        order.price = 50000.0
        order.cost = filled * 50000.0
        order.fee = filled * 50000.0 * 0.001
        order.fee_currency = self.fee_currency
        order.status = "closed" if self.filled_ratio == 1.0 else "partial"
        return order


class TestLiveFillAccuracy:
    """Trades, ledger entries, and positions must reflect actual execution
    results (filled amount, exchange-reported cost and fee currency)."""

    @pytest.mark.asyncio
    async def test_partial_fill_uses_filled_amount(
        self, mock_bot, mock_session, engine, mock_services
    ):
        """Trade record and position update use filled, not requested, amount."""
        exchange = FakeExchange(filled_ratio=0.5)
        signal = TradeSignal(action="buy", amount=1000.0, order_type="market")

        requested = 1000.0 / 50000.0  # 0.02
        expected_filled = requested * 0.5

        with ExitStack() as stack:
            for p in service_patches(mock_services):
                stack.enter_context(p)
            open_position = stack.enter_context(
                patch.object(engine, "_open_or_add_position", new=AsyncMock())
            )
            order = await engine._execute_trade(
                mock_bot, exchange, signal, 50000.0, mock_session
            )

        assert order is not None
        record_kwargs = mock_services["trade_recorder"].record_trade.call_args.kwargs
        assert record_kwargs["base_amount"] == pytest.approx(expected_filled)
        assert record_kwargs["quote_amount"] == pytest.approx(expected_filled * 50000.0)
        open_position.assert_awaited_once()
        assert open_position.await_args.args[2] == pytest.approx(expected_filled)

    @pytest.mark.asyncio
    async def test_fee_currency_from_exchange_response(
        self, mock_bot, mock_session, engine, mock_services
    ):
        """Fee asset comes from the exchange response, not assumed quote."""
        exchange = FakeExchange(fee_currency="BTC")
        signal = TradeSignal(action="buy", amount=1000.0, order_type="market")

        with ExitStack() as stack:
            for p in service_patches(mock_services):
                stack.enter_context(p)
            stack.enter_context(
                patch.object(engine, "_open_or_add_position", new=AsyncMock())
            )
            order = await engine._execute_trade(
                mock_bot, exchange, signal, 50000.0, mock_session
            )

        assert order is not None
        record_kwargs = mock_services["trade_recorder"].record_trade.call_args.kwargs
        assert record_kwargs["fee_asset"] == "BTC"

    @pytest.mark.asyncio
    async def test_fee_currency_falls_back_to_quote(
        self, mock_bot, mock_session, engine, mock_services
    ):
        """Missing fee currency in the response falls back to the quote asset."""
        exchange = FakeExchange(fee_currency=None)
        signal = TradeSignal(action="buy", amount=1000.0, order_type="market")

        with ExitStack() as stack:
            for p in service_patches(mock_services):
                stack.enter_context(p)
            stack.enter_context(
                patch.object(engine, "_open_or_add_position", new=AsyncMock())
            )
            await engine._execute_trade(
                mock_bot, exchange, signal, 50000.0, mock_session
            )

        record_kwargs = mock_services["trade_recorder"].record_trade.call_args.kwargs
        assert record_kwargs["fee_asset"] == "USDT"


class TestBalanceReconciliation:
    """Exchange balances must cover live bots' aggregate expectations."""

    def _make_session(self, live_bots, positions):
        session = AsyncMock()
        session.add = Mock()
        session.commit = AsyncMock()
        bots_result = Mock()
        bots_result.scalars = Mock(return_value=Mock(all=Mock(return_value=live_bots)))
        pos_result = Mock()
        pos_result.scalars = Mock(return_value=Mock(all=Mock(return_value=positions)))
        session.execute = AsyncMock(side_effect=[bots_result, pos_result])
        return session

    def _make_live_bot(self, bot_id=1, balance=1000.0, pair="BTC/USDT"):
        bot = Mock(spec=Bot)
        bot.id = bot_id
        bot.current_balance = balance
        bot.trading_pair = pair
        bot.is_dry_run = False
        return bot

    def _make_position(self, pair="BTC/USDT", amount=0.01):
        position = Mock()
        position.trading_pair = pair
        position.amount = amount
        return position

    def _make_exchange(self, balances: dict):
        exchange = Mock()

        async def get_balance(currency):
            total = balances.get(currency)
            if total is None:
                return None
            return Balance(currency=currency, free=total, used=0, total=total)

        exchange.get_balance = AsyncMock(side_effect=get_balance)
        return exchange

    @pytest.mark.asyncio
    async def test_sufficient_balances_no_alert(self, engine):
        session = self._make_session(
            [self._make_live_bot(balance=1000.0)],
            [self._make_position(amount=0.01)],
        )
        exchange = self._make_exchange({"USDT": 1500.0, "BTC": 0.02})

        await engine._reconcile_live_account(exchange, session)

        session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_quote_shortfall_creates_alert(self, engine):
        session = self._make_session([self._make_live_bot(balance=1000.0)], [])
        exchange = self._make_exchange({"USDT": 500.0})

        await engine._reconcile_live_account(exchange, session)

        session.add.assert_called_once()
        alert = session.add.call_args.args[0]
        assert alert.alert_type == "balance_reconciliation"
        assert "USDT" in alert.message
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_base_asset_shortfall_creates_alert(self, engine):
        session = self._make_session(
            [self._make_live_bot(balance=100.0)],
            [self._make_position(pair="BTC/USDT", amount=0.05)],
        )
        # Quote covered; BTC position only half-covered
        exchange = self._make_exchange({"USDT": 1000.0, "BTC": 0.025})

        await engine._reconcile_live_account(exchange, session)

        session.add.assert_called_once()
        alert = session.add.call_args.args[0]
        assert "BTC" in alert.message

    @pytest.mark.asyncio
    async def test_within_tolerance_no_alert(self, engine):
        """A shortfall inside the 1% tolerance does not alert."""
        session = self._make_session([self._make_live_bot(balance=1000.0)], [])
        exchange = self._make_exchange({"USDT": 995.0})  # -0.5%

        await engine._reconcile_live_account(exchange, session)

        session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_live_bots_no_exchange_calls(self, engine):
        session = self._make_session([], [])
        exchange = self._make_exchange({})

        await engine._reconcile_live_account(exchange, session)

        exchange.get_balance.assert_not_awaited()
        session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_throttled_within_interval(self, engine):
        session = self._make_session([self._make_live_bot(balance=1000.0)], [])
        exchange = self._make_exchange({"USDT": 1500.0})

        await engine._reconcile_live_account(exchange, session)
        first_calls = session.execute.await_count

        # Second call inside the interval must be a no-op
        await engine._reconcile_live_account(exchange, session)

        assert session.execute.await_count == first_calls
