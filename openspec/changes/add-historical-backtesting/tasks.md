## 1. CSV historical data provider
- [x] 1.1 Define `HistoricalDataProvider` interface (`get_candles(exchange, symbol, timeframe, start=None,
      end=None)`, `list_exchanges()`, `list_symbols(exchange)`, `list_timeframes(exchange, symbol)`) in
      `backend/app/backtesting/data_provider.py`.
- [x] 1.2 Implement `CsvHistoricalDataProvider`: directory discovery, dynamic column mapping (positional
      volume-column detection), multi-file load/merge/sort/dedupe, duplicate-timestamp and gap validation,
      inconsistent-timestamp-magnitude detection, start/end filtering.
- [x] 1.3 Implement `backend/app/backtesting/resampling.py`: 1m → 5m/15m/1h/4h/1d aggregation
      (open=first, high=max, low=min, close=last, volume=sum, tradecount=sum).

## 2. Backtest execution model
- [x] 2.1 Implement `backend/app/backtesting/execution_model.py` wrapping
      `ExecutionCostModel.estimate_cost` with a default 0.1% per-side fee and configurable spread/slippage.
- [x] 2.2 Enforce next-candle-open fill timing (no lookahead) structurally in the engine loop (task 3).

## 3. Backtest engine
- [x] 3.1 Implement the clock seam via constructor dependency injection: `app/services/clock.py` (`Clock`
      ABC, `SystemClock`), `TradingEngine(clock: Optional[Clock] = None)` defaulting to `SystemClock`, all
      internal `datetime.utcnow()` calls replaced with `self.clock.now()`. `BacktestClock`
      (`app/backtesting/clock.py`) is a plain stateful object advanced via `.set()`, injected via
      `TradingEngine(clock=BacktestClock(...))` — no global/module-attribute patching. (Revised from an
      earlier module-attribute-patch implementation after a post-implementation audit found it unsafe to run
      alongside live/dry-run bots in the same process; see design.md.)
- [x] 3.2 Implement per-run isolated database setup (`Base.metadata.create_all` against a fresh SQLite
      file/`:memory:`) and a synthetic `Bot` row for the requested strategy/parameters/starting balance.
- [x] 3.3 Implement the replay loop: candle N → strategy executor call (via `_get_strategy_executor`) →
      queued signal → simulated fill at candle N+1 open (task 2) → `Position`/`Order`/`Trade` rows updated
      (populating `owning_strategy`/`entry_reason` per `fix-strategy-integrity`) → portfolio/equity update.
- [x] 3.4 Implement metrics computation (starting/ending balance, total return %, trade count, win rate,
      average win/loss, largest win/loss, max drawdown, profit factor, expectancy per trade, total fees,
      buy-and-hold comparison).
- [x] 3.5 Implement per-trade export (entry/exit timestamp, strategy, entry/exit price, fees, gross/net P&L,
      exit reason).
- [x] 3.6 Add a CLI entry point (`backend/app/backtesting/run.py`, `python -m app.backtesting.run` or
      `python -m backend.app.backtesting.run`) exercising: full-range backtest for a discovered symbol,
      date-limited backtest, specific-strategy backtest, and `auto_mode` backtest. No exchange connection, no
      `tradingbot.db` access. Prints return %, buy-and-hold return %, trade count, win rate, profit factor,
      expectancy, max drawdown, fees paid.
- [x] 3.7 Wire timeframe generation into candle loading (`BacktestEngine._load_candles`): if the requested
      timeframe isn't stored on disk, resample it from the finest stored timeframe that is (e.g. `1h` from
      `1m`) rather than requiring every timeframe to exist as its own directory.

## 4. Tests
- [x] 4.1 CSV provider: temporary fake exchange/symbol folders, discovery, dynamic symbol loading, multi-file
      merge, correct sort, duplicate/gap detection, resampling correctness.
- [x] 4.2 Engine: candle replay correctness against a small hand-computed fixture.
- [x] 4.3 Fees reduce returns (fee-on vs fee-off comparison over identical trades).
- [x] 4.4 A known-losing strategy/parameter combination shows negative expectancy, unmodified.
- [x] 4.5 Buy-and-hold benchmark computed correctly against a hand-computed fixture.
- [x] 4.6 No future candles are visible during decisions (assert the strategy call for candle N cannot
      observe candle N+1's data; e.g. via a data provider stub that raises if asked for future indices).
- [x] 4.7 Same backtest input produces identical output across repeated runs (metrics, trade sequence,
      trade fields), including with the dependency-injected clock and a cooldown-sensitive strategy.
- [x] 4.8 Backtest clock cannot affect a live/dry-run engine's clock, including under concurrent async
      execution in the same process/event loop.
- [x] 4.9 CLI executes a backtest successfully (subprocess invocation, real argv parsing, exit codes).

## 5. Validation
- [x] 5.1 Run the full backend test suite; fix regressions.
- [x] 5.2 `openspec validate add-historical-backtesting --strict --no-interactive`.
