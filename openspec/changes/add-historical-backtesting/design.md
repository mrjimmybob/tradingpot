## Context

There is no backtesting engine in the repository today. The only existing artifact resembling one,
`backend/tests/test_backtest_determinism.py`, does not call any real strategy: it hardcodes an
alternating buy/sell signal and only checks that `TradingEngine._execute_trade` itself is deterministic once
seven of its collaborating services (`PortfolioRiskService`, `StrategyCapacityService`, `TradeRecorderService`,
`FIFOTaxEngine`, `LedgerInvariantService`, `VirtualWalletService`, `CSVExportService`) are patched out. That
test is left as-is; it is not the reusable foundation this change builds, and this change does not touch it.

`data/backtest/` already exists with real minute-bar CSVs for `binance/BTCUSDT/1m/` (2020–2026) and
`binance/SOLUSDT/1m/` (2021–2026), but nothing in the codebase reads them yet.

Strategy decision logic lives entirely as methods on `TradingEngine` (`backend/app/services/trading_engine.py`,
8,673 lines) — there is no separate `strategies/` package (`backend/app/strategies/__init__.py` is a one-line
stub). Every strategy is dispatched through one seam:

```python
_get_strategy_executor(strategy_name) -> Callable[[Bot, float, dict, AsyncSession], Awaitable[Optional[TradeSignal]]]
```

(trading_engine.py:1111-1147, dispatch map at :1127-1136). This is the exact reuse point: the backtester
calls this same function, with the same signature, for every historical candle. No strategy method is
copied or reimplemented.

Two things stand between "call the same function" and "get a valid historical replay":

1. **Wall-clock dependency.** `trading_engine.py` calls `datetime.utcnow()` 63 times (bar-close detection,
   cooldown timers, entry/exit timestamps) via a single import, `from datetime import datetime, timedelta`
   (trading_engine.py:6). There is no other time source (`grep` for `time.time()`, `time.monotonic()`, and
   `datetime.now()` in the file returns nothing). Left alone, a backtest replaying 2020–2026 candles would
   have every cooldown and bar-close check compare against *today's* real wall-clock time, producing
   nonsensical results (e.g., cooldowns permanently "not yet expired" because the historical entry timestamp
   and the real `utcnow()` are years apart, or bars appearing to "close" every real-time second regardless of
   candle spacing). This is the one concrete blocker to historical replay found in this audit
   (see "Tick vs. Candle Audit" below for what was checked and found *not* to block replay).
2. **DB-backed position/order state.** Strategy methods query `Position`/`Order` through the `session`
   argument (e.g. `_get_bot_positions`, trading_engine.py:8221-8256) to know whether they currently hold a
   position. A backtest needs the same tables, populated the same way, so this state is correct on every
   subsequent candle — not a parallel in-memory shadow structure that could drift from what the strategy
   code actually reads.

Also relevant: `backend/app/services/execution_cost_model.py` already provides a deterministic, side-effect-free
`ExecutionCostModel.estimate_cost(...)` (fee + spread + slippage, spot-only, "Deterministic (no randomness)"
by its own docstring). This is reused for the backtest execution model rather than re-deriving fee/spread/
slippage math a second time.

Finally, this proposal depends on the companion proposal `fix-strategy-integrity`: the `Position.owning_strategy`
field it adds is the same field the backtester's simulated position bookkeeping populates, per that
proposal's requirement that any future backtesting mode reuse the same ownership mechanism rather than a
parallel one. `funding_carry` must already be gone from `_get_strategy_executor` by the time this change's
strategy-support tests run.

### Tick vs. Candle Audit (Task 6 — report only, not fixed here)

Per instructions, strategies are not rewritten in this change. Findings, for the record:

| Location | Behavior | Blocks replay? |
|---|---|---|
| `trading_engine.py` `datetime.utcnow()` (59 call sites) | Wall-clock reads for cooldowns/bar-close/entry-time | **Yes** — fixed via constructor-injected `Clock` below |
| `_strategy_trend_following` EMA, `calculate_ema(price_history, period)` (~2920), fed by `price_history.append(current_price)` | Computed over whatever values are appended per call, one per candle in backtest | No — one call per historical candle is a faithful replay unit; EMA is simply computed over candle closes instead of sub-candle ticks, same as live-on-fast-loop vs live-on-slow-loop |
| `_strategy_trend_following` ATR/trailing stop, `_calc_bar_atr` (~2968) | Already bar-based | No |
| `_strategy_dip_recovery` (whole strategy, ~4285), `_calc_price_atr_proxy` (~2830-2852) | Tick-based ATR proxy by design ("no OHLC available for tick data") | No — receives one `current_price` per backtest candle exactly as it does one per live tick; behaves correctly, just at candle-close granularity |
| `_strategy_grid`, `_strategy_mean_reversion`, `_strategy_volatility_breakout` Bollinger/ATR | Already bar-aggregated (`bar_interval_seconds`) | No, provided the injected clock is in place so bar-close detection uses candle timestamps |
| Trailing stop ratchets in trend_following/volatility_breakout | Updated once per call, monotonic | No |

No indicator math is changed in this proposal. The single fix required for correct historical replay is the
clock, made available via dependency injection rather than a global patch (see below).

## Goals / Non-Goals
- Goals:
  - Deterministic replay of any discovered symbol/timeframe/date-range through any existing strategy or
    Auto Mode, using the exact production decision code.
  - Realistic execution: fees, configurable spread/slippage, strict no-lookahead fill timing.
  - Standard performance metrics, trade-level export, and a buy-and-hold benchmark for comparison.
  - Symbol/exchange/timeframe-agnostic data loading with zero hardcoded names.
- Non-Goals:
  - Rewriting strategy alpha or indicator math.
  - Adding new strategies, tuning existing parameters, or optimizing for any specific dataset.
  - Downloading data from exchanges; the filesystem under `data/backtest/` is the only source of truth.
  - Reusing the full live `_execute_trade` pipeline (portfolio risk caps across bots, wallet ledger, tax
    lots, CSV export, email alerts) — that pipeline exists to operate real capital under real accounting
    constraints across concurrently-running bots, which is out of scope for measuring one strategy's
    historical expectancy. `test_backtest_determinism.py` already demonstrates that reusing it requires
    patching out seven collaborating services just to execute one trade — the backtester instead uses a
    purpose-built, minimal `SimulatedExecutionModel` + portfolio ledger (Task 5/3), while still reusing the
    real `Position`/`Order`/`Trade` ORM tables (via an isolated database) so strategies' own position queries
    stay correct.

## Decisions

- Decision: **Reuse seam is `TradingEngine._get_strategy_executor(name)`**, called once per historical candle
  with `(synthetic_bot, candle.close, bot.strategy_params, backtest_session)`. `auto_mode` is supported the
  same way — it is just another entry in the dispatch map, and it exercises the Auto Mode ownership rules
  from `fix-strategy-integrity` unmodified.
  - Why: this is the only public dispatch seam in the codebase; anything lower-level (calling individual
    `_strategy_*` methods directly) would require the backtester to know which strategy maps to which method,
    duplicating the dispatch map instead of the strategy logic.

- Decision: **Isolated per-run database**, not a shared shadow structure. Each backtest run creates a fresh
  SQLite database (file or `:memory:`), runs `Base.metadata.create_all` for the full model set (so
  `Position`/`Order`/`Trade`/`Bot` exist with the same schema as production, including `owning_strategy` from
  the companion proposal), and inserts one synthetic `Bot` row configured with the requested strategy,
  parameters, and starting balance. All `_get_bot_positions`/order-insert calls made by strategy code during
  the run operate on this database exactly as they would in production.
  - Why: strategies read position state through `session` queries, not through arguments the backtester
    controls directly (trading_engine.py:8221-8256 `_get_bot_positions`). The only way to guarantee a
    strategy method sees consistent state across candles is to actually persist it the same way production
    does — a parallel in-memory mirror could silently drift from what the query returns.

- Decision: **Clock seam via constructor dependency injection, not monkeypatching.** `TradingEngine.__init__`
  takes an optional `clock: Clock` (`app/services/clock.py`, an ABC with `.now()`), defaulting to `SystemClock`
  (real `datetime.utcnow()`). Every one of the 59 `datetime.utcnow()` call sites inside `TradingEngine` was
  replaced with `self.clock.now()`. The backtest engine constructs its own `TradingEngine(clock=BacktestClock(...))`
  per run; `BacktestClock` (`app/backtesting/clock.py`) is a plain stateful object with `.now()`/`.set()`, advanced
  as candles are processed.
  - Why (revised from the original module-attribute-patch decision above, replaced after a post-implementation
    audit found it unsafe): patching `app.services.trading_engine.datetime` mutates process-global state. The
    backtest loop's own `await` points (strategy calls, DB flushes) are places asyncio can switch to any other
    coroutine in the same process — including a live/dry-run bot's `_run_bot_loop`, which would then read the
    frozen historical time for as long as the patch was active. Nothing prevented a future caller (e.g. an API
    endpoint) from invoking the backtest engine from inside the live server process, which would have made this
    a real, not just theoretical, leak. Constructor injection removes the shared state entirely: each
    `TradingEngine` instance owns its own clock reference, so a backtest and a live/dry-run engine literally
    cannot see each other's clock, regardless of process topology or interleaving.
  - No further scoping/context-management is needed (a plain object, not a global patch, needs no restore-on-exit).

- Decision: **No-lookahead is structural, not a rule the strategy has to obey.** The engine loop processes
  candle `N`, exposes `_get_bot_positions`/current price built only from candles `0..N`, invokes the strategy
  executor, and if it returns a `TradeSignal`, the signal is queued and only converted into a filled
  `Order`/`Trade`/`Position` update using candle `N+1`'s open price (fee/spread/slippage applied per Task 5)
  before candle `N+1` is itself presented to the strategy. The strategy process never receives candle `N+1`
  or later while deciding on candle `N` — there is no argument path by which it could.

- Decision: **Reuse `ExecutionCostModel.estimate_cost`** for fee/spread/slippage rather than a new cost
  formula, configured with the backtest's fee/spread/slippage parameters (defaults: 0.1% fee per side,
  spread and slippage both configurable, default 0).
  - Why: keeps live and backtest cost semantics consistent, and the model is already deterministic
    (dataclass, no randomness) by design.

- Decision: **CSV provider is pure filesystem discovery, no manifest.** `list_exchanges()`/`list_symbols()`/
  `list_timeframes()` are directory listings (`os.scandir`) under `data/backtest/`; nothing is hardcoded.
  Column mapping detects the two `Volume <ASSET>` columns positionally (first = base volume, second = quote
  volume) rather than by asset name, per the required format. Multiple files in one
  `{exchange}/{symbol}/{timeframe}/` directory are all loaded, concatenated, deduplicated by timestamp
  (last-write-wins if a timestamp appears in more than one file, logged), sorted ascending, and validated for
  gaps (a gap is reported, not silently dropped or invented — the provider raises/reports rather than
  interpolating, since Task 4 does not ask for synthetic candle generation).
  - Known data-quality issue found in the shipped fixtures: `data/backtest/binance/BTCUSDT/1m/BTCUSDT_2024_minute.csv`
    has ~44,600 rows (out of ~527,000) where the `Unix` column is encoded in a different unit (16-digit,
    consistent with nanoseconds) than the rest of the file (13-digit milliseconds), producing bogus 1970-dated
    rows if divided by the wrong scale. The provider must detect inconsistent timestamp magnitude within a
    file/column and treat those rows as a validation failure (reported, not silently misinterpreted), rather
    than assuming a single fixed unit for the whole dataset. This is a real, present issue in the shipped test
    fixtures, not a hypothetical.

- Decision: **Resampling is a pure function over already-loaded 1m candles** (open=first, high=max, low=min,
  close=last, volume=sum, tradecount=sum), grouped by `floor(timestamp / timeframe_seconds)`. No separate
  code path per target timeframe.

## Risks / Trade-offs
- An isolated per-run SQLite database adds setup overhead per backtest compared to a pure-Python loop; this
  is deliberate (see Non-Goals) to guarantee strategy position-state fidelity. Long backtests (multi-year,
  1-minute candles) may be slow; this proposal does not commit to a performance target, only correctness.
- Reusing `_get_strategy_executor` means any bug in live strategy decision code (outside the scope of this
  audit's fixes) is faithfully reproduced in backtests too — this is intentional (it's what "no lookahead,
  reuse production logic" means) but worth stating: this proposal does not fix strategy bugs, only measures
  their historical behavior.
- Residual risk with the injected-clock approach: any *future* code added inside `TradingEngine` that calls
  `datetime.utcnow()` directly instead of `self.clock.now()` would silently reintroduce a wall-clock
  dependency invisible to backtests (correct live behavior, wrong/non-deterministic backtest behavior). There
  is no automated guard against this beyond code review; a lint rule banning `datetime.utcnow()` inside
  `app/services/trading_engine.py` would close it if this recurs.

## Migration Plan
1. Add `backend/app/backtesting/` with `data_provider.py` (`HistoricalDataProvider` interface +
   `CsvHistoricalDataProvider`), `resampling.py`, `execution_model.py` (wraps `ExecutionCostModel`),
   `portfolio.py` (balance/position/equity-curve bookkeeping + metrics), `engine.py` (the replay loop,
   constructing `TradingEngine(clock=BacktestClock(...))`), `clock.py` (`BacktestClock`), `run.py` (CLI
   entry point), and `results.py` (metrics/trade export dataclasses).
2. Add `app/services/clock.py` (`Clock` ABC, `SystemClock`) and give `TradingEngine.__init__` an optional
   `clock` parameter (default `SystemClock`), replacing all internal `datetime.utcnow()` calls with
   `self.clock.now()`.
3. Wire `owning_strategy` population into the backtester's simulated position-open path once
   `fix-strategy-integrity` lands (shared dependency, sequence that proposal first or in parallel with a
   short-lived stub field).
4. Add tests (see tasks.md) proving determinism, no-lookahead, fee impact, CSV discovery/merge/resample, a
   losing strategy showing negative expectancy alongside a buy-and-hold comparison, and clock isolation
   between a backtest engine and a live/dry-run engine (including under concurrent async execution in the
   same process).
5. No production code path is altered for live trading; live and dry-run both get `SystemClock` by default
   (`TradingEngine()` with no arguments, unchanged call sites), so behavior is identical to before this
   change for both.

## Open Questions
- None blocking for this phase. Whether/how to persist backtest results (a report file vs. a DB table) is
  left to implementation — Task 3 requires the metrics and trade export to exist and be computed correctly,
  not a specific storage format.
