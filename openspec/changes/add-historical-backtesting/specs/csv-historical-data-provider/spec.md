## ADDED Requirements

### Requirement: Dynamic Discovery
`CsvHistoricalDataProvider` SHALL discover available exchanges, symbols, and timeframes by listing
directories under `data/backtest/` at runtime, with no hardcoded exchange, symbol, year, filename, or date
range.

#### Scenario: New symbol works without code changes
- **WHEN** a new folder `data/backtest/{exchange}/{symbol}/{timeframe}/*.csv` is added following the
  directory convention
- **THEN** `list_exchanges()`, `list_symbols(exchange)`, and `list_timeframes(exchange, symbol)` reflect it
  without any code change, and `get_candles(exchange, symbol, timeframe)` loads it successfully

### Requirement: CSV Column Mapping
The provider SHALL parse the header `Unix,Date,Symbol,Open,High,Low,Close,Volume <ASSET>,Volume <ASSET>,tradecount`
and map columns by position/role rather than by hardcoded asset name: `Unix`→timestamp, `Date`→datetime,
`Symbol`→symbol, `Open`/`High`/`Low`/`Close`→OHLC, the first `Volume *` column→base volume, the second
`Volume *` column→quote volume, `tradecount`→trade count.

#### Scenario: Volume columns detected regardless of asset name
- **WHEN** a CSV has columns `Volume BTC` and `Volume USDT`, or `Volume SOL` and `Volume USDT`
- **THEN** the first is mapped to base volume and the second to quote volume without the asset name being
  referenced anywhere in code

### Requirement: Multi-File Merge and Validation
The provider SHALL load all CSV files in a requested `{exchange}/{symbol}/{timeframe}/` directory, merge
them, sort candles oldest to newest, and validate for duplicate and missing timestamps.

#### Scenario: Multiple files merge into one sorted series
- **WHEN** a directory contains multiple yearly CSV files, each ordered newest-first internally
- **THEN** `get_candles(...)` returns a single series sorted strictly oldest to newest across all files

#### Scenario: Duplicate timestamps are detected
- **WHEN** the same timestamp appears more than once across the loaded files
- **THEN** the provider detects and reports the duplicate rather than silently keeping an arbitrary one

#### Scenario: Missing candles are detected
- **WHEN** there is a gap between consecutive candle timestamps larger than one timeframe interval
- **THEN** the provider detects and reports the gap rather than silently proceeding as if the data were
  continuous

#### Scenario: Inconsistent timestamp encoding is detected
- **WHEN** rows within a loaded file encode `Unix` at a different numeric magnitude than the rest of the file
  (e.g. nanoseconds mixed with milliseconds, as found in the shipped `BTCUSDT_2024_minute.csv` fixture)
- **THEN** the provider flags the affected rows as invalid rather than silently misinterpreting them into an
  incorrect date

### Requirement: Start/End Filtering
`get_candles` SHALL support optional `start` and `end` bounds, returning only candles within that range.

#### Scenario: Date-limited request
- **WHEN** `get_candles(exchange, symbol, timeframe, start=X, end=Y)` is called
- **THEN** only candles with timestamps in `[X, Y]` are returned

### Requirement: Timeframe Resampling
The provider SHALL generate larger candles from 1-minute candles on demand (5m, 15m, 1h, 4h, 1d), using
open=first, high=max, low=min, close=last, volume=sum, tradecount=sum.

#### Scenario: Resample 1m to 1h
- **WHEN** 1-minute candles for a full hour are resampled to `1h`
- **THEN** the resulting candle's open equals the first minute's open, close equals the last minute's close,
  high/low are the max/min across the hour, and volume/tradecount are summed
