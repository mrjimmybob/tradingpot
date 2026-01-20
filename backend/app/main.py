"""TradingBot FastAPI Application.

Includes alerts router for alert management.
WebSocket support for real-time market data and UI updates.
"""

import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import init_db
from .routers import bots, health, stats, reports, config as config_router, alerts
from .routers import websocket as ws_router
from .routers import data_sources, portfolio, ledger
from .services.websocket import ws_manager
from .services.config import config_service, ConfigValidationException
from .services import trading_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: Validate configuration
    try:
        config = config_service.load_and_validate()
        print("Configuration validated successfully")
    except ConfigValidationException as e:
        print(f"FATAL: {e}")
        print("Server cannot start with invalid configuration.")
        sys.exit(1)
    except Exception as e:
        print(f"WARNING: Could not load config file: {e}")
        print("Using default configuration")

    # Initialize database
    await init_db()
    print("Database initialized")

    # Start WebSocket manager
    await ws_manager.start()
    print("WebSocket manager started")

    # Auto-resume bots that were running when server stopped
    try:
        resumed_count = await trading_engine.trading_engine.resume_bots_on_startup()
        if resumed_count > 0:
            print(f"Resumed {resumed_count} bot(s) from previous session")
    except Exception as e:
        print(f"WARNING: Failed to resume bots: {e}")

    yield

    # Shutdown: Save state and stop bots gracefully
    print("Initiating graceful shutdown...")

    # Graceful shutdown of trading engine (saves state)
    try:
        shutdown_count = await trading_engine.trading_engine.graceful_shutdown()
        if shutdown_count > 0:
            print(f"Saved state for {shutdown_count} bot(s)")
    except Exception as e:
        print(f"WARNING: Error during trading engine shutdown: {e}")

    # Stop WebSocket manager
    await ws_manager.stop()
    print("WebSocket manager stopped")

    print("Graceful shutdown complete")


app = FastAPI(
    title="TradingBot API",
    description="Crypto Trading Bot Management API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(bots.router, prefix="/api/bots", tags=["Bots"])
app.include_router(stats.router, prefix="/api", tags=["Stats"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])
app.include_router(config_router.router, prefix="/api/config", tags=["Config"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(ws_router.router, prefix="/api", tags=["WebSocket"])
app.include_router(data_sources.router, prefix="/api/data-sources", tags=["Data Sources"])
app.include_router(portfolio.router, prefix="/api", tags=["Portfolio"])
app.include_router(ledger.router, prefix="/api", tags=["Ledger"])


@app.get("/")
async def root():
    """Root endpoint redirect to docs."""
    return {"message": "TradingBot API", "docs": "/docs"}
