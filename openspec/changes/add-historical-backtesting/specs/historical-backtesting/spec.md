## ADDED Requirements

### Requirement: Strategy Decision Reuse
The backtesting engine SHALL invoke the same production strategy decision code used by live trading — via
`TradingEngine._get_strategy_executor(strategy_name)` — for every historical candle, without duplicating or
reimplementing any strategy's decision logic.

#### Scenario: Backtest calls the live dispatch seam
- **WHEN** a backtest is run for a given strategy name
- **THEN** the engine resolves the strategy's decision function through the same lookup used by live
  trading and calls it with the same argument signature `(bot, current_price, params, session)`

#### Scenario: Auto Mode is backtestable the same way
- **WHEN** a backtest is run with strategy `auto_mode`
- **THEN** the engine dispatches through the same seam and Auto Mode's position-ownership rules (see the
  `fix-strategy-integrity` change) apply during the replay exactly as they do live

### Requirement: No-Lookahead Execution
The engine SHALL ensure a strategy's decision for candle N is made using only candles `0..N`, and any
resulting order fills at candle `N+1`'s open price. The strategy process SHALL have no access path to
candle `N+1` or later while deciding on candle N.

#### Scenario: Decision uses only past and current candle
- **WHEN** the engine processes candle N
- **THEN** the strategy executor is invoked with market state derived only from candles `0..N`

#### Scenario: Fill happens on the next candle's open
- **WHEN** a strategy decision on candle N produces a market order
- **THEN** the simulated fill price is candle `N+1`'s open price, not candle N's close or any later price

### Requirement: Deterministic Replay
Running a backtest with identical data, strategy, parameters, and date range SHALL always produce identical
output.

#### Scenario: Repeated run produces identical results
- **WHEN** the same backtest (same data, strategy, parameters, date range) is run twice
- **THEN** every output metric, the trade sequence, and the trade-by-trade fields are identical between runs

### Requirement: Performance Metrics
The engine SHALL compute and report: starting balance, ending balance, total return percent, number of
trades, win rate, average win, average loss, largest win, largest loss, max drawdown, profit factor,
expectancy per trade, total fees paid, and a buy-and-hold comparison over the same date range.

#### Scenario: Metrics computed for a completed backtest
- **WHEN** a backtest run completes
- **THEN** the result includes all of the listed metrics, computed from the recorded trade and equity history

#### Scenario: Losing strategy shows negative expectancy
- **WHEN** a backtest is run against a strategy/parameter combination that loses money over the period
- **THEN** the reported expectancy per trade and total return are negative, and the result is not adjusted,
  hidden, or filtered to look otherwise

#### Scenario: Buy-and-hold benchmark is reported alongside strategy results
- **WHEN** a backtest completes for a symbol and date range
- **THEN** the result includes the return of simply holding the symbol over that same range, for comparison

### Requirement: Trade Export
The engine SHALL export individual trades with: entry timestamp, exit timestamp, strategy, entry price, exit
price, fees, gross P&L, net P&L, and exit reason.

#### Scenario: Every closed trade is exported
- **WHEN** a backtest closes a position
- **THEN** a trade record is added to the export list with all required fields populated
