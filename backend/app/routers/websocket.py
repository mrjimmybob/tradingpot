"""WebSocket router for real-time updates."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging

from ..services.websocket import ws_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Messages from client:
    - {"action": "subscribe", "symbols": ["BTCUSDT", "ETHUSDT"]}
    - {"action": "unsubscribe", "symbols": ["BTCUSDT"]}
    - {"action": "ping"}

    Messages to client:
    - {"type": "price_update", "symbol": "BTCUSDT", "price": 50000, ...}
    - {"type": "indicator_update", "symbol": "BTCUSDT", "sentiment": 0.5, ...}
    - {"type": "bot_update", "bot_id": 1, "status": "running", "pnl": 100, ...}
    - {"type": "stats_update", "total_bots": 5, "running_bots": 3, ...}
    - {"type": "pong"}
    """
    await ws_manager.connect_client(websocket)

    try:
        while True:
            message = await websocket.receive_text()
            await ws_manager.handle_client_message(websocket, message)

    except WebSocketDisconnect:
        await ws_manager.disconnect_client(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.disconnect_client(websocket)


@router.post("/ws/subscribe/{symbol}")
async def subscribe_symbol(symbol: str):
    """Subscribe to market data for a symbol via REST."""
    success = await ws_manager.subscribe_symbol(symbol)
    return {"success": success, "symbol": symbol}


@router.post("/ws/unsubscribe/{symbol}")
async def unsubscribe_symbol(symbol: str):
    """Unsubscribe from market data for a symbol via REST."""
    success = await ws_manager.unsubscribe_symbol(symbol)
    return {"success": success, "symbol": symbol}


@router.get("/ws/indicators/{symbol}")
async def get_indicators(symbol: str):
    """Get current market indicators for a symbol."""
    indicators = ws_manager.get_market_indicators(symbol)
    if not indicators:
        return {"error": "No indicators available for symbol", "symbol": symbol}

    return {
        "symbol": symbol,
        "timestamp": indicators.timestamp.isoformat(),
        "last_price": indicators.last_price,
        "sentiment_score": indicators.sentiment_score,
        "risk_score": indicators.risk_score,
        "signal": indicators.signal,
        "orderbook": {
            "imbalance_ratio": indicators.orderbook.imbalance_ratio,
            "bid_volume": indicators.orderbook.bid_volume,
            "ask_volume": indicators.orderbook.ask_volume,
            "has_liquidity_vacuum": indicators.orderbook.has_liquidity_vacuum,
        } if indicators.orderbook else None,
        "volume_delta": {
            "buy_volume": indicators.volume_delta.buy_volume,
            "sell_volume": indicators.volume_delta.sell_volume,
            "delta": indicators.volume_delta.delta,
            "normalized_delta": indicators.volume_delta.normalized_delta,
        } if indicators.volume_delta else None,
        "spread": {
            "spread": indicators.spread.spread,
            "spread_percent": indicators.spread.spread_percent,
            "spread_widening": indicators.spread.spread_widening,
        } if indicators.spread else None,
        "volatility": {
            "atr": indicators.volatility.atr,
            "atr_percent": indicators.volatility.atr_percent,
            "atr_ratio": indicators.volatility.atr_ratio,
            "vol_regime": indicators.volatility.vol_regime,
        } if indicators.volatility else None,
    }


@router.get("/ws/indicators")
async def get_all_indicators():
    """Get market indicators for all tracked symbols."""
    all_indicators = ws_manager.get_all_market_indicators()

    return {
        "count": len(all_indicators),
        "indicators": {
            symbol: {
                "timestamp": ind.timestamp.isoformat(),
                "last_price": ind.last_price,
                "sentiment_score": ind.sentiment_score,
                "risk_score": ind.risk_score,
                "signal": ind.signal,
            }
            for symbol, ind in all_indicators.items()
        },
    }
