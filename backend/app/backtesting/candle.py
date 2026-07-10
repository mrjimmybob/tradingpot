"""Candle data structure shared by the data provider, resampler, and engine."""
from dataclasses import dataclass
from datetime import datetime, timezone


def ms_to_naive_utc(ts_ms: int) -> datetime:
    """Convert a unix-milliseconds timestamp to a naive UTC datetime, matching
    the naive-``datetime.utcnow()`` convention the rest of the codebase uses
    (avoids the deprecated ``datetime.utcfromtimestamp``)."""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class Candle:
    """One OHLCV candle.

    ``timestamp`` is the authoritative unix-milliseconds close/open marker
    (matches the CSV ``Unix`` column). ``datetime`` is a human-readable ISO
    string derived from it, kept only for display/debugging - all ordering,
    deduplication, and gap logic uses ``timestamp``.
    """
    timestamp: int  # unix epoch milliseconds
    datetime: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    base_volume: float
    quote_volume: float
    trade_count: int
