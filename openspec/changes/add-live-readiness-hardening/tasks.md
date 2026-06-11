# Tasks: add-live-readiness-hardening

## 1. Exchange credentials (foundation — others depend on it)

- [x] 1.1 Add env-var credential resolution to `ExchangeService._load_config()` (`<EXCHANGE_ID>_API_KEY`/`_API_SECRET` override YAML) with placeholder (`YOUR_*`) and empty values treated as unset; expose `has_credentials()` on the service
- [x] 1.2 Unit tests: env overrides YAML, placeholder rejected, empty rejected (`tests/test_exchange_wrapper.py`)

## 2. Dry-run real market data

- [x] 2.1 Rework `SimulatedExchangeService`: `connect()` creates the real public ccxt client via `super().connect()`; `get_ticker()` delegates to the real implementation; remove the hardcoded `mock_prices` table; keep balances/orders simulated
- [x] 2.2 Add per-symbol ticker cache with configurable TTL (default 2s) used by `get_ticker()`
- [x] 2.3 Unit tests with mocked ccxt client: delegation, cache hit within TTL, `None` (no fallback price) when the public API fails (`tests/test_exchange_wrapper.py`)

## 3. Bot start safety

- [x] 3.1 In `trading_engine.start_bot` (and the resume path `resume_bots_on_startup`): check `await exchange.connect()` result; on failure do not start the loop, set bot status back with a clear error, return/raise so the API reports the reason
- [x] 3.2 Guard: live (non-dry-run) bots refuse to start when `has_credentials()` is false, with explicit error message
- [x] 3.3 Tests: connect-failure aborts start; live start without credentials rejected; dry-run start without credentials succeeds (`tests/test_trading_engine.py`)

## 4. API authentication and binding

- [x] 4.1 Add auth middleware in `backend/app/main.py`: bearer token from `TRADINGBOT_API_TOKEN` (fallback `server.api_token` in `config.yaml`); protect `/api/*` except `/api/health`; 401 on missing/invalid token
- [x] 4.2 WebSocket endpoint: validate `token` query param before accepting when a token is configured
- [x] 4.3 Startup fail-safe in lifespan: abort with clear error if `server.host` is non-loopback and no token configured
- [x] 4.4 Change defaults: `backend/config.yaml` `server.host` → `127.0.0.1`; `backend/run.ps1` → no `0.0.0.0`
- [x] 4.5 Tests: 401 without token, 200 with token, health open, WS refusal, non-loopback + no token aborts startup (new `tests/test_auth.py`)

## 5. Frontend token support

- [x] 5.1 Create `frontend/src/lib/api.ts` with `apiFetch` (same signature as `fetch`) attaching `Authorization` from `localStorage('tradingbot_api_token')`; surface 401 with a clear toast/message
- [x] 5.2 Replace direct `fetch(` API call sites (~15 files) with `apiFetch`
- [x] 5.3 Append `?token=` to the WebSocket URL when a token is stored (`WebSocketContext`/`useWebSocket`)
- [x] 5.4 Settings page: field to view/set the stored API token
- [x] 5.5 Frontend lint + build pass (`npm run lint`, `npm run build`)

## 6. Docs and verification

- [x] 6.1 README: document `TRADINGBOT_API_TOKEN`, env-var credentials, new localhost default, dry-run-needs-internet note, and the dry-run → live testing path
- [x] 6.2 Run full backend suite (expect all green; baseline 428 passed)
- [x] 6.3 Manual smoke test: start server, create dry-run bot, confirm ticker prices match live MEXC prices and change over time
