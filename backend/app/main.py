"""TradingBot FastAPI Application.

Includes alerts router for alert management.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import init_db
from .routers import bots, health, stats, reports, config as config_router, alerts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await init_db()
    print("Database initialized")
    # TODO: Auto-resume running bots
    yield
    # Shutdown
    print("Shutting down...")
    # TODO: Graceful shutdown of running bots


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


@app.get("/")
async def root():
    """Root endpoint redirect to docs."""
    return {"message": "TradingBot API", "docs": "/docs"}
