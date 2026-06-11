# Design: Live-readiness hardening

## Context

The rollout goal is: (1) dry-run bots against **real** market data to validate strategies, then (2) run a small live-money test. Today dry-run prices are hardcoded constants, the API is wide open on `0.0.0.0`, and credential handling is YAML-only with no validation. The 428-test suite mocks `get_ticker` at the call site, which leaves room to change the simulator's data source without breaking tests.

## Goals / Non-Goals

**Goals**
- Dry-run bots see the same prices a live bot would see, with no code-path divergence in strategy logic.
- A machine on the LAN cannot reach the API by default; remote access requires a token.
- Credentials come from env vars first; placeholders never reach the exchange; live bots cannot start without usable credentials.

**Non-Goals**
- Realistic limit-order fill simulation (slippage/partial fills) — pending `add-trading-safety-boundaries` covers execution cost modeling.
- Multi-user accounts, sessions, RBAC, TLS termination.

## Decisions

### 1. Dry-run market data: delegate to the public REST API via the existing base class

`SimulatedExchangeService` already inherits from `ExchangeService`. Decision: its `connect()` calls `super().connect()` to create a real ccxt client (public endpoints work with empty credentials — `load_markets` and `fetch_ticker` are unauthenticated), and `get_ticker()` delegates to the inherited real implementation. Balances, order placement, and order lookup stay simulated.

- **No fake fallback.** If the public API is unreachable, `get_ticker()` returns `None`. The bot loop already treats a missing ticker as "skip iteration, retry in 5s" (`trading_engine._run_bot_loop`), which is the fail-safe we want. Silently reverting to mock prices would recreate the original problem.
- **Ticker cache with short TTL (default 2s), keyed by symbol, per service instance.** Bot loops run at 1s intervals; without a cache, N bots on the same pair generate N REST calls/second. ccxt's `enableRateLimit` already throttles, but the cache keeps us comfortably inside MEXC's public limits.
- **Simulated limit orders keep filling immediately** — now at the real current price. Documented limitation; fill realism is out of scope here.

Alternative considered: reuse the existing `MEXCWebSocketConnector` real-time feed. Rejected for now — it couples the simulator to the WS manager lifecycle and adds failure modes; REST + cache is simpler and sufficient at 1s granularity. The WS feed can replace the REST fetch later without touching the simulator's interface.

Test impact: tests that exercise strategies mock `get_ticker` with `AsyncMock` and are unaffected. New simulator tests mock the underlying ccxt client, so the suite stays offline-deterministic.

### 2. API auth: single static bearer token, fail-safe binding

Single-operator system → a single static token is appropriate; OAuth/JWT is overkill.

- Token source: `TRADINGBOT_API_TOKEN` env var (falls back to `server.api_token` in `config.yaml` for convenience, env wins).
- Enforcement: ASGI middleware on `/api/*` requiring `Authorization: Bearer <token>`. Exempt: `/` and `/api/health` (liveness checks), `/docs` + `/openapi.json` stay reachable but every API call from docs still needs the token.
- WebSocket: token via `?token=` query parameter, checked before `accept()`.
- **Fail-safe rule:** if the configured token is empty, auth is disabled but the server refuses to start when `server.host` is not loopback (`127.0.0.1`/`localhost`/`::1`). This keeps the zero-config local workflow ("clone, run, open UI") working while making remote exposure impossible without a token.
- Defaults flipped: `config.yaml` `server.host` → `127.0.0.1`; `run.ps1` drops `--host 0.0.0.0`.

### 3. Frontend: one API client module instead of 15 raw `fetch` call sites

New `frontend/src/lib/api.ts` exporting `apiFetch(path, init?)` that attaches `Authorization` from `localStorage('tradingbot_api_token')` (settable on the Settings page) and handles 401 with a clear toast. All direct `fetch(` call sites switch to it — mechanical replacement. WS context appends `?token=` when a token is stored. With no token stored (default local setup), behavior is unchanged.

### 4. Credentials: env-var override + placeholder rejection + live-start guard

- `ExchangeService._load_config()` resolves `api_key`/`api_secret` as: env var (`MEXC_API_KEY`/`MEXC_API_SECRET`, uppercased exchange id) → YAML value → empty.
- Values matching the placeholder pattern (`YOUR_*`) or empty are treated as **unset**.
- `trading_engine.start_bot`: for a non-dry-run bot, if credentials are unset → bot does not start, status set with a clear error message. For all bots, if `await exchange.connect()` returns `False` → bot does not start (today the result is ignored).

## Risks / Trade-offs

- **Public REST dependency in dry-run**: MEXC outages now stall dry-run bots. Acceptable — stalling is the correct behavior for a system whose purpose is to validate against real markets.
- **Static token, no TLS**: adequate for localhost/LAN single-operator use; anyone deploying beyond that needs a reverse proxy with TLS (documented in README).
- **`fetch` → `apiFetch` sweep touches many files**: mechanical but wide; mitigated by keeping the wrapper signature identical to `fetch`.

## Migration

No DB changes. Existing local setups keep working with zero new configuration (localhost binding, no token, dry-run now needs internet access). Live trading newly *requires* real credentials — previously it silently started with placeholders and failed downstream.
