"""
Market data indicators service for real-time analysis.

Computes continuously:
- Orderbook imbalance ratio
- Volume delta (aggressive buy vs sell)
- ATR compression/expansion
- Spread volatility
- Short-term realized volatility
- Liquidity metrics
"""
import asyncio
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Deque, Callable, Awaitable

from .base import (
    DepthUpdate,
    TradeUpdate,
    KlineUpdate,
    TickerUpdate,
    DepthLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class OrderbookImbalance:
    """Orderbook imbalance metrics."""
    timestamp: datetime
    symbol: str

    # Bid/Ask imbalance (-1 to +1, positive = more bid pressure)
    imbalance_ratio: float

    # Total volumes
    bid_volume: float
    ask_volume: float

    # Depth-weighted imbalance (top N levels)
    weighted_imbalance: float

    # Liquidity vacuum detection (sudden book thinning)
    bid_depth_change: float  # Percent change from recent average
    ask_depth_change: float

    @property
    def is_bid_heavy(self) -> bool:
        return self.imbalance_ratio > 0.2

    @property
    def is_ask_heavy(self) -> bool:
        return self.imbalance_ratio < -0.2

    @property
    def has_liquidity_vacuum(self) -> bool:
        return self.bid_depth_change < -0.3 or self.ask_depth_change < -0.3


@dataclass
class VolumeDelta:
    """Volume delta metrics (buy vs sell aggression)."""
    timestamp: datetime
    symbol: str

    # Cumulative delta over window
    buy_volume: float
    sell_volume: float
    delta: float  # buy - sell

    # Normalized delta (-1 to +1)
    normalized_delta: float

    # Trade count
    buy_count: int
    sell_count: int

    # Average trade size
    avg_buy_size: float
    avg_sell_size: float

    @property
    def is_buy_aggressive(self) -> bool:
        return self.normalized_delta > 0.3

    @property
    def is_sell_aggressive(self) -> bool:
        return self.normalized_delta < -0.3


@dataclass
class SpreadMetrics:
    """Bid-ask spread metrics."""
    timestamp: datetime
    symbol: str

    # Current spread
    spread: float
    spread_percent: float

    # Spread statistics over window
    avg_spread: float
    spread_volatility: float  # Std dev of spread
    spread_widening: float  # Current vs average (1.0 = normal)

    # Best bid/ask
    best_bid: float
    best_ask: float

    @property
    def is_spread_wide(self) -> bool:
        return self.spread_widening > 1.5


@dataclass
class VolatilityMetrics:
    """Price volatility metrics."""
    timestamp: datetime
    symbol: str

    # ATR (Average True Range)
    atr: float
    atr_percent: float  # ATR as percent of price

    # ATR compression/expansion
    atr_ratio: float  # Current ATR vs longer-term ATR (< 1 = compression)

    # Realized volatility (short-term)
    realized_vol_1m: float  # 1-minute
    realized_vol_5m: float  # 5-minute
    realized_vol_15m: float  # 15-minute

    # Volatility regime
    vol_regime: str  # "low", "normal", "high", "extreme"

    @property
    def is_compressed(self) -> bool:
        return self.atr_ratio < 0.7

    @property
    def is_expanded(self) -> bool:
        return self.atr_ratio > 1.3


@dataclass
class MarketIndicators:
    """Combined market indicators for a symbol."""
    timestamp: datetime
    symbol: str

    orderbook: Optional[OrderbookImbalance] = None
    volume_delta: Optional[VolumeDelta] = None
    spread: Optional[SpreadMetrics] = None
    volatility: Optional[VolatilityMetrics] = None

    # Current price
    last_price: float = 0.0

    # Market sentiment score (-1 to +1)
    sentiment_score: float = 0.0

    # Risk level (0-1)
    risk_score: float = 0.5

    # Recommended action
    signal: str = "neutral"  # "bullish", "bearish", "neutral", "avoid"


# Type alias for indicator handlers
IndicatorHandler = Callable[[MarketIndicators], Awaitable[None]]


class MarketDataService:
    """
    Service for computing market indicators from WebSocket data.

    Maintains rolling windows of:
    - Depth snapshots for orderbook analysis
    - Trades for volume delta
    - Prices for volatility calculation
    - Spreads for spread analysis
    """

    def __init__(
        self,
        depth_window: int = 100,  # Number of depth updates to keep
        trade_window: int = 500,  # Number of trades to keep
        price_window: int = 300,  # Number of price points for volatility
        spread_window: int = 100,  # Number of spread samples
        kline_window: int = 50,  # Number of klines for ATR
        update_interval: float = 1.0,  # Indicator update interval in seconds
    ):
        self.depth_window = depth_window
        self.trade_window = trade_window
        self.price_window = price_window
        self.spread_window = spread_window
        self.kline_window = kline_window
        self.update_interval = update_interval

        # Data storage per symbol
        self._depth_history: Dict[str, Deque[DepthUpdate]] = {}
        self._trade_history: Dict[str, Deque[TradeUpdate]] = {}
        self._price_history: Dict[str, Deque[tuple]] = {}  # (timestamp, price)
        self._spread_history: Dict[str, Deque[tuple]] = {}  # (timestamp, spread)
        self._kline_history: Dict[str, Dict[str, Deque[KlineUpdate]]] = {}  # symbol -> interval -> klines

        # Current indicators per symbol
        self._indicators: Dict[str, MarketIndicators] = {}

        # Indicator handlers
        self._handlers: List[IndicatorHandler] = []

        # Background task
        self._update_task: Optional[asyncio.Task] = None
        self._running = False

    def on_indicators(self, handler: IndicatorHandler) -> None:
        """Register handler for indicator updates."""
        self._handlers.append(handler)

    async def start(self) -> None:
        """Start indicator computation."""
        if self._running:
            return

        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("MarketDataService started")

    async def stop(self) -> None:
        """Stop indicator computation."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            self._update_task = None
        logger.info("MarketDataService stopped")

    # Data ingestion handlers (to be connected to WebSocket)
    async def handle_depth(self, depth: DepthUpdate) -> None:
        """Handle depth update from WebSocket."""
        symbol = depth.symbol

        if symbol not in self._depth_history:
            self._depth_history[symbol] = deque(maxlen=self.depth_window)

        self._depth_history[symbol].append(depth)

        # Update spread history
        if depth.spread is not None:
            if symbol not in self._spread_history:
                self._spread_history[symbol] = deque(maxlen=self.spread_window)
            self._spread_history[symbol].append((depth.timestamp, depth.spread))

    async def handle_trade(self, trade: TradeUpdate) -> None:
        """Handle trade update from WebSocket."""
        symbol = trade.symbol

        if symbol not in self._trade_history:
            self._trade_history[symbol] = deque(maxlen=self.trade_window)

        self._trade_history[symbol].append(trade)

        # Update price history
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.price_window)
        self._price_history[symbol].append((trade.timestamp, trade.price))

    async def handle_kline(self, kline: KlineUpdate) -> None:
        """Handle kline update from WebSocket."""
        symbol = kline.symbol
        interval = kline.interval

        if symbol not in self._kline_history:
            self._kline_history[symbol] = {}

        if interval not in self._kline_history[symbol]:
            self._kline_history[symbol][interval] = deque(maxlen=self.kline_window)

        # Only keep closed klines, update last if not closed
        klines = self._kline_history[symbol][interval]
        if kline.is_closed:
            klines.append(kline)
        elif klines and not klines[-1].is_closed:
            klines[-1] = kline
        else:
            klines.append(kline)

    async def handle_ticker(self, ticker: TickerUpdate) -> None:
        """Handle ticker update from WebSocket."""
        symbol = ticker.symbol

        # Update price history from ticker
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.price_window)
        self._price_history[symbol].append((ticker.timestamp, ticker.last_price))

    async def _update_loop(self) -> None:
        """Background loop to compute and broadcast indicators."""
        while self._running:
            try:
                # Compute indicators for all tracked symbols
                symbols = set(self._depth_history.keys()) | set(self._trade_history.keys())

                for symbol in symbols:
                    indicators = await self._compute_indicators(symbol)
                    if indicators:
                        self._indicators[symbol] = indicators

                        # Broadcast to handlers
                        for handler in self._handlers:
                            try:
                                await handler(indicators)
                            except Exception as e:
                                logger.error(f"Indicator handler error: {e}")

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Indicator update error: {e}")
                await asyncio.sleep(self.update_interval)

    async def _compute_indicators(self, symbol: str) -> Optional[MarketIndicators]:
        """Compute all indicators for a symbol."""
        now = datetime.utcnow()

        # Get latest price
        last_price = 0.0
        if symbol in self._price_history and self._price_history[symbol]:
            last_price = self._price_history[symbol][-1][1]

        indicators = MarketIndicators(
            timestamp=now,
            symbol=symbol,
            last_price=last_price,
        )

        # Compute individual indicators
        indicators.orderbook = self._compute_orderbook_imbalance(symbol)
        indicators.volume_delta = self._compute_volume_delta(symbol)
        indicators.spread = self._compute_spread_metrics(symbol)
        indicators.volatility = self._compute_volatility(symbol, last_price)

        # Compute composite scores
        indicators.sentiment_score = self._compute_sentiment(indicators)
        indicators.risk_score = self._compute_risk(indicators)
        indicators.signal = self._compute_signal(indicators)

        return indicators

    def _compute_orderbook_imbalance(self, symbol: str) -> Optional[OrderbookImbalance]:
        """Compute orderbook imbalance metrics."""
        if symbol not in self._depth_history or not self._depth_history[symbol]:
            return None

        depths = list(self._depth_history[symbol])
        latest = depths[-1]

        # Calculate total bid/ask volumes
        bid_volume = sum(level.quantity for level in latest.bids[:10])
        ask_volume = sum(level.quantity for level in latest.asks[:10])
        total_volume = bid_volume + ask_volume

        if total_volume == 0:
            return None

        # Simple imbalance ratio
        imbalance_ratio = (bid_volume - ask_volume) / total_volume

        # Weighted imbalance (closer levels weighted more)
        weighted_bid = sum(
            level.quantity * (1 / (i + 1))
            for i, level in enumerate(latest.bids[:10])
        )
        weighted_ask = sum(
            level.quantity * (1 / (i + 1))
            for i, level in enumerate(latest.asks[:10])
        )
        weighted_total = weighted_bid + weighted_ask
        weighted_imbalance = (
            (weighted_bid - weighted_ask) / weighted_total
            if weighted_total > 0 else 0
        )

        # Liquidity vacuum detection (compare to recent average)
        bid_depth_change = 0.0
        ask_depth_change = 0.0

        if len(depths) >= 10:
            recent_depths = depths[-10:-1]
            avg_bid_vol = sum(
                sum(d.bids[i].quantity for i in range(min(10, len(d.bids))))
                for d in recent_depths
            ) / len(recent_depths)
            avg_ask_vol = sum(
                sum(d.asks[i].quantity for i in range(min(10, len(d.asks))))
                for d in recent_depths
            ) / len(recent_depths)

            if avg_bid_vol > 0:
                bid_depth_change = (bid_volume - avg_bid_vol) / avg_bid_vol
            if avg_ask_vol > 0:
                ask_depth_change = (ask_volume - avg_ask_vol) / avg_ask_vol

        return OrderbookImbalance(
            timestamp=latest.timestamp,
            symbol=symbol,
            imbalance_ratio=imbalance_ratio,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            weighted_imbalance=weighted_imbalance,
            bid_depth_change=bid_depth_change,
            ask_depth_change=ask_depth_change,
        )

    def _compute_volume_delta(self, symbol: str) -> Optional[VolumeDelta]:
        """Compute volume delta metrics."""
        if symbol not in self._trade_history or not self._trade_history[symbol]:
            return None

        trades = list(self._trade_history[symbol])
        now = datetime.utcnow()

        # Filter to last minute
        recent_trades = [
            t for t in trades
            if (now - t.timestamp).total_seconds() < 60
        ]

        if not recent_trades:
            return None

        # Separate buy/sell
        buy_trades = [t for t in recent_trades if t.side == "buy"]
        sell_trades = [t for t in recent_trades if t.side == "sell"]

        buy_volume = sum(t.value for t in buy_trades)
        sell_volume = sum(t.value for t in sell_trades)
        delta = buy_volume - sell_volume
        total = buy_volume + sell_volume

        normalized_delta = delta / total if total > 0 else 0

        avg_buy_size = buy_volume / len(buy_trades) if buy_trades else 0
        avg_sell_size = sell_volume / len(sell_trades) if sell_trades else 0

        return VolumeDelta(
            timestamp=now,
            symbol=symbol,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            delta=delta,
            normalized_delta=normalized_delta,
            buy_count=len(buy_trades),
            sell_count=len(sell_trades),
            avg_buy_size=avg_buy_size,
            avg_sell_size=avg_sell_size,
        )

    def _compute_spread_metrics(self, symbol: str) -> Optional[SpreadMetrics]:
        """Compute spread metrics."""
        if symbol not in self._depth_history or not self._depth_history[symbol]:
            return None

        latest = self._depth_history[symbol][-1]
        if not latest.best_bid or not latest.best_ask:
            return None

        spread = latest.spread
        spread_percent = latest.spread_percent

        # Spread statistics
        spreads = self._spread_history.get(symbol, [])
        spread_values = [s[1] for s in spreads]

        if len(spread_values) < 5:
            avg_spread = spread
            spread_volatility = 0
            spread_widening = 1.0
        else:
            avg_spread = sum(spread_values) / len(spread_values)
            spread_volatility = (
                sum((s - avg_spread) ** 2 for s in spread_values) / len(spread_values)
            ) ** 0.5
            spread_widening = spread / avg_spread if avg_spread > 0 else 1.0

        return SpreadMetrics(
            timestamp=latest.timestamp,
            symbol=symbol,
            spread=spread,
            spread_percent=spread_percent,
            avg_spread=avg_spread,
            spread_volatility=spread_volatility,
            spread_widening=spread_widening,
            best_bid=latest.best_bid.price,
            best_ask=latest.best_ask.price,
        )

    def _compute_volatility(
        self, symbol: str, current_price: float
    ) -> Optional[VolatilityMetrics]:
        """Compute volatility metrics."""
        # ATR from klines
        atr = 0.0
        atr_ratio = 1.0

        klines_data = self._kline_history.get(symbol, {})
        if "1m" in klines_data and len(klines_data["1m"]) >= 14:
            klines = list(klines_data["1m"])

            # Calculate ATR (14-period)
            true_ranges = []
            for i in range(1, len(klines)):
                high = klines[i].high
                low = klines[i].low
                prev_close = klines[i - 1].close

                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close),
                )
                true_ranges.append(tr)

            if true_ranges:
                atr = sum(true_ranges[-14:]) / min(14, len(true_ranges))

                # ATR ratio (current vs longer term)
                if len(true_ranges) >= 28:
                    long_atr = sum(true_ranges[-28:-14]) / 14
                    atr_ratio = atr / long_atr if long_atr > 0 else 1.0

        atr_percent = (atr / current_price * 100) if current_price > 0 else 0

        # Realized volatility from price history
        prices = self._price_history.get(symbol, [])
        price_list = [p[1] for p in prices]

        realized_vol_1m = self._calc_realized_vol(price_list, 60)  # ~60 ticks/min
        realized_vol_5m = self._calc_realized_vol(price_list, 300)
        realized_vol_15m = self._calc_realized_vol(price_list, 900)

        # Determine volatility regime
        if atr_percent < 0.5:
            vol_regime = "low"
        elif atr_percent < 1.5:
            vol_regime = "normal"
        elif atr_percent < 3.0:
            vol_regime = "high"
        else:
            vol_regime = "extreme"

        return VolatilityMetrics(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            atr=atr,
            atr_percent=atr_percent,
            atr_ratio=atr_ratio,
            realized_vol_1m=realized_vol_1m,
            realized_vol_5m=realized_vol_5m,
            realized_vol_15m=realized_vol_15m,
            vol_regime=vol_regime,
        )

    def _calc_realized_vol(self, prices: List[float], window: int) -> float:
        """Calculate realized volatility from price series."""
        if len(prices) < 2:
            return 0.0

        # Use most recent prices up to window
        recent = prices[-window:] if len(prices) > window else prices

        # Calculate log returns
        returns = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 0:
                log_return = math.log(recent[i] / recent[i - 1])
                returns.append(log_return)

        if not returns:
            return 0.0

        # Standard deviation of returns
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return variance ** 0.5

    def _compute_sentiment(self, indicators: MarketIndicators) -> float:
        """Compute overall market sentiment score (-1 to +1)."""
        scores = []

        # Orderbook imbalance
        if indicators.orderbook:
            scores.append(indicators.orderbook.imbalance_ratio)

        # Volume delta
        if indicators.volume_delta:
            scores.append(indicators.volume_delta.normalized_delta)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def _compute_risk(self, indicators: MarketIndicators) -> float:
        """Compute risk score (0 to 1, higher = riskier)."""
        risk = 0.5  # Base risk

        # Spread widening increases risk
        if indicators.spread and indicators.spread.is_spread_wide:
            risk += 0.15

        # Liquidity vacuum increases risk
        if indicators.orderbook and indicators.orderbook.has_liquidity_vacuum:
            risk += 0.2

        # High volatility increases risk
        if indicators.volatility:
            if indicators.volatility.vol_regime == "high":
                risk += 0.1
            elif indicators.volatility.vol_regime == "extreme":
                risk += 0.25

        return min(1.0, max(0.0, risk))

    def _compute_signal(self, indicators: MarketIndicators) -> str:
        """Compute trading signal based on indicators."""
        # High risk = avoid
        if indicators.risk_score > 0.8:
            return "avoid"

        sentiment = indicators.sentiment_score

        if sentiment > 0.4:
            return "bullish"
        elif sentiment < -0.4:
            return "bearish"

        return "neutral"

    def get_indicators(self, symbol: str) -> Optional[MarketIndicators]:
        """Get current indicators for a symbol."""
        return self._indicators.get(symbol)

    def get_all_indicators(self) -> Dict[str, MarketIndicators]:
        """Get indicators for all tracked symbols."""
        return dict(self._indicators)
