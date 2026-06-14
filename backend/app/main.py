"""TradingBot FastAPI Application.

Includes alerts router for alert management.
WebSocket support for real-time market data and UI updates.
"""

import secrets
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import init_db
from .routers import bots, health, stats, reports, config as config_router, alerts
from .routers import websocket as ws_router
from .routers import data_sources, portfolio, ledger
from .services.websocket import ws_manager
from .services.config import config_service, get_api_token, ConfigValidationException
from .services import trading_engine
from .services.db_backup import db_backup_service

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def binding_failsafe_error(host: str, token: str) -> str:
    """Return an error message when the configuration would expose an
    unauthenticated API beyond loopback, else an empty string."""
    if host not in LOOPBACK_HOSTS and not token:
        return (
            f"server.host is '{host}' (non-loopback) but no API token is set. "
            "Set TRADINGBOT_API_TOKEN (or server.api_token in config.yaml) "
            "before exposing the API beyond localhost."
        )
    return ""


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

    # Fail-safe: never expose an unauthenticated API beyond loopback
    host = config_service.get("server.host") or "127.0.0.1"
    failsafe_error = binding_failsafe_error(host, get_api_token())
    if failsafe_error:
        print(f"FATAL: {failsafe_error}")
        sys.exit(1)
    if get_api_token():
        print("API authentication: enabled (bearer token)")
    else:
        print("API authentication: disabled (loopback-only mode)")

    # Initialize database
    await init_db()
    print("Database initialized")

    # M-2: take a startup backup and start periodic backups.
    try:
        await db_backup_service.backup_once()
        db_backup_service.start()
    except Exception as e:
        print(f"WARNING: database backup could not be started: {e}")

    # Start WebSocket manager
    await ws_manager.start()
    print("WebSocket manager started")

    # Auto-resume bots that were running when server stopped
    try:
        resumed_count = await trading_engine.resume_bots_on_startup()
        if resumed_count > 0:
            print(f"Resumed {resumed_count} bot(s) from previous session")
    except Exception as e:
        print(f"WARNING: Failed to resume bots: {e}")

    yield

    # Shutdown: Save state and stop bots gracefully
    print("Initiating graceful shutdown...")

    # Graceful shutdown of trading engine (saves state)
    try:
        shutdown_count = await trading_engine.graceful_shutdown()
        if shutdown_count > 0:
            print(f"Saved state for {shutdown_count} bot(s)")
    except Exception as e:
        print(f"WARNING: Error during trading engine shutdown: {e}")

    # Stop the periodic DB backup task
    try:
        await db_backup_service.stop()
    except Exception as e:
        print(f"WARNING: error stopping DB backup task: {e}")

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

# CORS middleware. Origins are configurable (server.cors_origins) so a live
# deployment can scope them; default to the local dev frontends. (M-5)
# Best-effort load so config is available at import time; the lifespan handler
# performs the authoritative load + validation (and exits on a bad config).
try:
    config_service.load_and_validate()
except Exception:
    pass
_cors_origins = config_service.get("server.cors_origins") or [
    "http://localhost:5173",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths reachable without a token (liveness checks and docs UI; API calls
# made from the docs UI still require the token)
AUTH_EXEMPT_PATHS = {"/", "/api/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def auth_middleware(request, call_next):
    """Require a bearer token on /api routes when a token is configured."""
    token = get_api_token()
    path = request.url.path
    if (
        token
        and request.method != "OPTIONS"  # CORS preflight carries no auth header
        and path.startswith("/api")
        and path not in AUTH_EXEMPT_PATHS
    ):
        auth_header = request.headers.get("authorization", "")
        provided = (
            auth_header[7:].strip()
            if auth_header.lower().startswith("bearer ")
            else ""
        )
        if not provided or not secrets.compare_digest(provided, token):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API token"},
            )
    return await call_next(request)

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


def _resolve_frontend_dist(backend_dir=None):
    """Resolve the frontend build directory to serve, or None for API-only.

    The web GUI is part of the product, so it is served BY DEFAULT: when
    ``server.frontend_dist`` is not configured, the conventional build location
    ``<repo>/frontend/dist`` is auto-discovered. This makes the UI work without
    manual config edits and survives CI/CD ``git reset`` of tracked files.

    The path stays configurable via ``server.frontend_dist`` (or the
    ``TRADINGBOT__SERVER__FRONTEND_DIST`` env override). Relative configured
    paths are anchored to the backend directory, so resolution is independent
    of the current working directory. Returns the directory when it contains
    ``index.html``; otherwise logs a warning and returns None (API-only mode).

    Args:
        backend_dir: Override for the backend directory (testing only).
    """
    from pathlib import Path

    if backend_dir is None:
        backend_dir = Path(__file__).resolve().parent.parent
    backend_dir = Path(backend_dir)

    dist_setting = config_service.get("server.frontend_dist")
    if dist_setting:
        dist = Path(dist_setting)
        if not dist.is_absolute():
            dist = (backend_dir / dist_setting).resolve()
    else:
        # Auto-discover the conventional build location (CWD-independent).
        dist = (backend_dir.parent / "frontend" / "dist").resolve()

    if not (dist / "index.html").is_file():
        print(f"WARNING: no frontend build at {dist}; serving API only")
        return None
    return dist


def _mount_frontend() -> bool:
    """Mount the resolved frontend build (SPA) on the API, if one exists.

    API/docs/ws routes are registered earlier and take precedence; unmatched
    GET paths fall back to index.html for client-side routing.
    """
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    dist = _resolve_frontend_dist()
    if dist is None:
        return False

    index = dist / "index.html"
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        # API/ws/docs are matched by earlier routes; everything else is the SPA.
        candidate = (dist / full_path).resolve()
        if full_path and candidate.is_file() and dist in candidate.parents:
            return FileResponse(str(candidate))
        return FileResponse(str(index))

    print(f"Serving frontend from {dist}")
    return True


if not _mount_frontend():
    @app.get("/")
    async def root():
        """Root endpoint redirect to docs."""
        return {"message": "TradingBot API", "docs": "/docs"}
