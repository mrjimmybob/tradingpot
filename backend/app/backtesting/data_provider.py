"""Historical market data access.

``HistoricalDataProvider`` is the interface the backtest engine depends on.
``CsvHistoricalDataProvider`` is the only implementation: it discovers and
loads local CSV files under a root directory. No exchange, symbol, year,
filename, or date range is ever hardcoded - everything is discovered from
the filesystem at call time, so any new ``{exchange}/{symbol}/{timeframe}/``
folder that follows the convention works without a code change.
"""
from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .candle import Candle

logger = logging.getLogger(__name__)

# Plausible bounds for a unix-milliseconds crypto-market timestamp. Rows
# outside this range are almost certainly a units mistake (e.g. nanoseconds
# mixed into a milliseconds column - see ValidationReport.invalid_timestamp_rows)
# rather than a real candle, and are excluded rather than silently misread.
_MIN_PLAUSIBLE_YEAR = 2009  # before Bitcoin existed
_MAX_PLAUSIBLE_YEAR = 2100


class DataIntegrityError(ValueError):
    """Raised when requested historical data cannot be loaded safely."""


@dataclass
class ValidationReport:
    """What CsvHistoricalDataProvider found while merging/validating candles."""
    duplicate_timestamps: List[int] = field(default_factory=list)
    gaps: List[tuple] = field(default_factory=list)  # (after_ts, before_ts) in ms
    invalid_timestamp_rows: int = 0

    @property
    def is_clean(self) -> bool:
        return not self.duplicate_timestamps and not self.gaps and not self.invalid_timestamp_rows


class HistoricalDataProvider(ABC):
    """Interface the backtest engine depends on. Symbol/exchange-agnostic."""

    @abstractmethod
    def list_exchanges(self) -> List[str]:
        ...

    @abstractmethod
    def list_symbols(self, exchange: str) -> List[str]:
        ...

    @abstractmethod
    def list_timeframes(self, exchange: str, symbol: str) -> List[str]:
        ...

    @abstractmethod
    def get_candles(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> List[Candle]:
        ...


def _plausible_ms_timestamp(ts_ms: int) -> bool:
    try:
        year = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).year
    except (ValueError, OSError, OverflowError):
        return False
    return _MIN_PLAUSIBLE_YEAR <= year <= _MAX_PLAUSIBLE_YEAR


class CsvHistoricalDataProvider(HistoricalDataProvider):
    """Discovers and loads candle CSVs under ``data/backtest/`` (default root).

    Directory convention::

        {root}/{exchange}/{symbol}/{timeframe}/*.csv

    CSV header (asset name in the volume columns varies, e.g. "Volume BTC",
    "Volume SOL" - detected positionally, never by name)::

        Unix,Date,Symbol,Open,High,Low,Close,Volume <ASSET>,Volume <ASSET>,tradecount
    """

    def __init__(self, root: str | Path = "data/backtest"):
        self.root = Path(root)
        self.last_validation: Optional[ValidationReport] = None

    # --- discovery -----------------------------------------------------

    def _subdirs(self, path: Path) -> List[str]:
        if not path.is_dir():
            return []
        return sorted(p.name for p in path.iterdir() if p.is_dir())

    def list_exchanges(self) -> List[str]:
        return self._subdirs(self.root)

    def list_symbols(self, exchange: str) -> List[str]:
        return self._subdirs(self.root / exchange)

    def list_timeframes(self, exchange: str, symbol: str) -> List[str]:
        return self._subdirs(self.root / exchange / symbol)

    # --- loading ---------------------------------------------------------

    def get_candles(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> List[Candle]:
        directory = self.root / exchange / symbol / timeframe
        if not directory.is_dir():
            raise DataIntegrityError(f"No data directory: {directory}")

        csv_files = sorted(directory.glob("*.csv"))
        if not csv_files:
            raise DataIntegrityError(f"No CSV files found in {directory}")

        raw_by_timestamp: dict[int, Candle] = {}
        duplicates: List[int] = []
        invalid_rows = 0

        for file_path in csv_files:
            for candle, is_valid in self._parse_file(file_path, symbol):
                if not is_valid:
                    invalid_rows += 1
                    continue
                if candle.timestamp in raw_by_timestamp:
                    duplicates.append(candle.timestamp)
                # Last file processed (sorted by filename) wins on conflict -
                # deterministic, not "whichever the OS happened to list first".
                raw_by_timestamp[candle.timestamp] = candle

        candles = sorted(raw_by_timestamp.values(), key=lambda c: c.timestamp)

        gaps = self._find_gaps(candles, timeframe)
        self.last_validation = ValidationReport(
            duplicate_timestamps=sorted(set(duplicates)),
            gaps=gaps,
            invalid_timestamp_rows=invalid_rows,
        )
        if duplicates:
            logger.warning(
                f"{exchange}/{symbol}/{timeframe}: {len(set(duplicates))} duplicate "
                f"timestamp(s) across source files; kept the last file's row."
            )
        if gaps:
            logger.warning(
                f"{exchange}/{symbol}/{timeframe}: {len(gaps)} gap(s) larger than "
                f"one {timeframe} interval."
            )
        if invalid_rows:
            logger.warning(
                f"{exchange}/{symbol}/{timeframe}: {invalid_rows} row(s) with an "
                f"implausible timestamp were excluded (see ValidationReport)."
            )

        if start is not None:
            candles = [c for c in candles if c.timestamp >= start]
        if end is not None:
            candles = [c for c in candles if c.timestamp <= end]

        return candles

    def validate(self, candles: List[Candle], timeframe: str) -> ValidationReport:
        """Standalone validation entry point (also run internally by
        get_candles) - useful for asserting on merge/gap/duplicate behavior
        directly in tests without re-parsing files."""
        seen: dict[int, int] = {}
        duplicates = []
        for c in candles:
            seen[c.timestamp] = seen.get(c.timestamp, 0) + 1
        duplicates = sorted(ts for ts, count in seen.items() if count > 1)
        gaps = self._find_gaps(sorted(candles, key=lambda c: c.timestamp), timeframe)
        return ValidationReport(duplicate_timestamps=duplicates, gaps=gaps)

    def _find_gaps(self, sorted_candles: List[Candle], timeframe: str) -> List[tuple]:
        interval_ms = _TIMEFRAME_SECONDS.get(timeframe, 60) * 1000
        gaps = []
        for prev, cur in zip(sorted_candles, sorted_candles[1:]):
            delta = cur.timestamp - prev.timestamp
            if delta > interval_ms:
                gaps.append((prev.timestamp, cur.timestamp))
        return gaps

    def _parse_file(self, file_path: Path, symbol: str):
        """Yields (Candle, is_valid) pairs. Column mapping is positional /
        role-based, never by hardcoded asset name (Volume columns detected by
        the "Volume" prefix, in header order: first = base, second = quote)."""
        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            volume_cols = [c for c in fieldnames if c.startswith("Volume")]
            base_vol_col = volume_cols[0] if len(volume_cols) > 0 else None
            quote_vol_col = volume_cols[1] if len(volume_cols) > 1 else None

            for row in reader:
                try:
                    ts_ms = int(float(row["Unix"]))
                except (KeyError, ValueError, TypeError):
                    yield None, False
                    continue

                if not _plausible_ms_timestamp(ts_ms):
                    yield None, False
                    continue

                dt_iso = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()

                try:
                    candle = Candle(
                        timestamp=ts_ms,
                        datetime=dt_iso,
                        symbol=row.get("Symbol") or symbol,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        base_volume=float(row[base_vol_col]) if base_vol_col else 0.0,
                        quote_volume=float(row[quote_vol_col]) if quote_vol_col else 0.0,
                        trade_count=int(float(row["tradecount"])) if row.get("tradecount") else 0,
                    )
                except (KeyError, ValueError, TypeError):
                    yield None, False
                    continue

                yield candle, True


# Shared with resampling.py (kept here too to avoid a circular import for
# _find_gaps's interval lookup).
_TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
}
