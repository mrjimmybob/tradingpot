"""The backtest replay loop.

Flow (per Task 3):

    historical candle -> market state -> strategy decision -> simulated
    execution -> portfolio update

Reuses the exact production strategy decision code via
``TradingEngine._get_strategy_executor`` - no strategy logic is duplicated.
Strict no-lookahead: a decision made while "at" candle N only ever sees
candles 0..N; any resulting order fills at candle N+1's open.

Position/Order rows are written to a fresh, isolated in-memory database using
the real ORM models, so a strategy's own internal session queries (e.g.
``_get_bot_positions``) see exactly the state production code would produce -
not a parallel, potentially-drifting in-memory mirror.
"""
from __future__ import annotations

from typing import List, Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.models import Base, Bot, BotStatus, Order, OrderType, OrderStatus
from app.services.trading_engine import TradingEngine

from .candle import Candle, ms_to_naive_utc
from .clock import BacktestClock
from .data_provider import HistoricalDataProvider, _TIMEFRAME_SECONDS
from .execution_model import BacktestExecutionModel
from .portfolio import BacktestPortfolio
from .resampling import resample
from .results import BacktestResult, compute_result


class BacktestEngine:
    def __init__(
        self,
        data_provider: HistoricalDataProvider,
        execution_model: Optional[BacktestExecutionModel] = None,
    ):
        self.data_provider = data_provider
        self.execution_model = execution_model or BacktestExecutionModel()

    async def run(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        strategy: str,
        strategy_params: Optional[dict] = None,
        starting_balance: float = 10_000.0,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> BacktestResult:
        candles = self._load_candles(exchange, symbol, timeframe, start, end)
        if len(candles) < 2:
            raise ValueError(
                "Need at least 2 candles to backtest (one to decide on, one to fill against)"
            )
        return await self.run_candles(
            candles, symbol, strategy, strategy_params or {}, starting_balance,
        )

    def _load_candles(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: Optional[int],
        end: Optional[int],
    ) -> List[Candle]:
        """Load candles at the requested timeframe, generating it from the
        finest stored timeframe via resampling if it isn't stored directly
        (e.g. requesting "1h" when only "1m" CSVs are on disk)."""
        available = self.data_provider.list_timeframes(exchange, symbol)
        if timeframe in available:
            return self.data_provider.get_candles(exchange, symbol, timeframe, start=start, end=end)

        if timeframe not in _TIMEFRAME_SECONDS:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r}; supported: {sorted(_TIMEFRAME_SECONDS)}"
            )
        generatable_from = [tf for tf in available if _TIMEFRAME_SECONDS.get(tf, float("inf")) < _TIMEFRAME_SECONDS[timeframe]]
        if not generatable_from:
            raise ValueError(
                f"No stored timeframe finer than {timeframe!r} available for "
                f"{exchange}/{symbol} to generate it from (have: {available})"
            )
        finest = min(generatable_from, key=lambda tf: _TIMEFRAME_SECONDS[tf])
        base_candles = self.data_provider.get_candles(exchange, symbol, finest, start=start, end=end)
        return resample(base_candles, timeframe)

    async def run_candles(
        self,
        candles: List[Candle],
        trading_pair: str,
        strategy: str,
        strategy_params: dict,
        starting_balance: float,
    ) -> BacktestResult:
        """Same as ``run`` but takes an already-loaded candle list directly -
        the entry point unit tests use to assert no-lookahead / determinism
        against a small hand-built fixture without touching the filesystem."""
        # Dependency-injected clock, not a global patch: this TradingEngine
        # instance is the only thing that ever reads it. A live or dry-run
        # TradingEngine elsewhere in the same process holds its own SystemClock
        # and is completely unaffected, however many backtests run alongside it.
        clock = BacktestClock(candles[0].timestamp)
        engine = TradingEngine(clock=clock)
        portfolio = BacktestPortfolio(starting_balance)

        db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async_session = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

        try:
            async with async_session() as session:
                bot = Bot(
                    name="backtest", trading_pair=trading_pair, strategy=strategy,
                    strategy_params=strategy_params, budget=starting_balance,
                    current_balance=starting_balance, is_dry_run=True,
                    status=BotStatus.RUNNING,
                )
                session.add(bot)
                await session.flush()

                for i in range(len(candles) - 1):
                    # === decision: sees only candle i (and, through the
                    # DB session, whatever the strategy itself persisted
                    # on candles 0..i-1) - candle i+1 is not read at all
                    # until after the executor has already returned. ===
                    decision_candle = candles[i]

                    clock.set(decision_candle.timestamp)
                    bot.current_balance = portfolio.cash

                    executor = engine._get_strategy_executor(strategy)
                    if executor is None:
                        raise ValueError(f"Unknown strategy: {strategy!r}")

                    signal = await executor(
                        bot, decision_candle.close, strategy_params, session,
                    )

                    if signal and signal.action in ("buy", "sell"):
                        fill_candle = candles[i + 1]  # first access, decision already made
                        await self._apply_signal(
                            engine, session, bot, portfolio, strategy,
                            signal, fill_candle,
                        )
                        await session.flush()

                    portfolio.mark_to_market(decision_candle.timestamp, decision_candle.close)

                # Final mark-to-market on the last candle (no decision made on it -
                # it only ever serves as the last fill candle above).
                portfolio.mark_to_market(candles[-1].timestamp, candles[-1].close)
        finally:
            await db_engine.dispose()

        buy_and_hold_pct = (
            (candles[-1].close - candles[0].open) / candles[0].open * 100.0
            if candles[0].open else 0.0
        )

        return compute_result(
            starting_balance=starting_balance,
            ending_balance=portfolio.equity(candles[-1].close),
            trades=portfolio.trades,
            equity_curve=portfolio.equity_curve,
            total_fees_paid=portfolio.total_fees_paid,
            buy_and_hold_return_pct=buy_and_hold_pct,
        )

    async def _apply_signal(
        self,
        engine: TradingEngine,
        session: AsyncSession,
        bot: Bot,
        portfolio: BacktestPortfolio,
        strategy: str,
        signal,
        fill_candle: Candle,
    ) -> None:
        if signal.action == "buy":
            # Strategies compute a concrete USD notional for every buy signal
            # (see e.g. _strategy_dca's amount_usd, _strategy_mean_reversion's
            # order_size_usd) - there is no "buy with sentinel 0" convention
            # (unlike sell, where 0 means "close the entire position").
            notional_usd = min(signal.amount or 0.0, portfolio.cash)
            if notional_usd <= 0:
                return
            amount_base = notional_usd / fill_candle.open
            fill = self.execution_model.fill("buy", amount_base, fill_candle.open)

            order = Order(
                bot_id=bot.id, order_type=OrderType.MARKET_BUY, trading_pair=bot.trading_pair,
                amount=amount_base, price=fill.price, fees=fill.fee_usd,
                status=OrderStatus.FILLED, strategy_used=strategy, reason=signal.reason,
                is_simulated=True,
                created_at=ms_to_naive_utc(fill_candle.timestamp),
                filled_at=ms_to_naive_utc(fill_candle.timestamp),
            )
            session.add(order)
            await session.flush()

            portfolio.apply_buy(fill_candle.timestamp, amount_base, fill)

            owning_strategy = engine._resolve_owning_strategy(bot, signal.reason)
            await engine._open_or_add_position(
                bot.id, bot.trading_pair, amount_base, fill.price, session,
                owning_strategy=owning_strategy, entry_reason=signal.reason,
            )
            bot.current_balance = portfolio.cash

        else:  # sell
            if not portfolio.has_position:
                return
            amount_base = (
                portfolio.base_amount if not signal.amount or signal.amount <= 0
                else min(signal.amount / fill_candle.open, portfolio.base_amount)
            )
            if amount_base <= 0:
                return
            fill = self.execution_model.fill("sell", amount_base, fill_candle.open)

            order = Order(
                bot_id=bot.id, order_type=OrderType.MARKET_SELL, trading_pair=bot.trading_pair,
                amount=amount_base, price=fill.price, fees=fill.fee_usd,
                status=OrderStatus.FILLED, strategy_used=strategy, reason=signal.reason,
                is_simulated=True,
                created_at=ms_to_naive_utc(fill_candle.timestamp),
                filled_at=ms_to_naive_utc(fill_candle.timestamp),
            )
            session.add(order)
            await session.flush()

            portfolio.apply_sell(
                fill_candle.timestamp, amount_base, fill, strategy, signal.reason or "",
            )

            await engine._close_or_reduce_position(
                bot.id, bot.trading_pair, amount_base, fill.price, session,
                _NullWallet(),
            )
            bot.current_balance = portfolio.cash


class _NullWallet:
    """Minimal stand-in for VirtualWalletService: _close_or_reduce_position
    only uses it to record realized P&L for the production wallet ledger,
    which the backtest portfolio (BacktestPortfolio.apply_sell) already
    tracks independently - this just needs to not raise."""
    async def record_trade_result(self, *args, **kwargs) -> None:
        return None
