"""Generate larger candles from smaller ones (e.g. 1m -> 5m/15m/1h/4h/1d).

Pure function, no I/O: aggregation only. Works on any input granularity,
not just 1m - candles are grouped by flooring their timestamp to the target
bucket size, so resampling 1m->1h and resampling 5m->1h produce the same
result for the same underlying market data.
"""
from __future__ import annotations

from typing import List

from .candle import Candle
from .data_provider import _TIMEFRAME_SECONDS

SUPPORTED_TIMEFRAMES = tuple(_TIMEFRAME_SECONDS.keys())


def resample(candles: List[Candle], target_timeframe: str) -> List[Candle]:
    if target_timeframe not in _TIMEFRAME_SECONDS:
        raise ValueError(
            f"Unsupported timeframe {target_timeframe!r}; supported: {SUPPORTED_TIMEFRAMES}"
        )
    if not candles:
        return []

    bucket_ms = _TIMEFRAME_SECONDS[target_timeframe] * 1000
    buckets: dict[int, List[Candle]] = {}
    for c in sorted(candles, key=lambda c: c.timestamp):
        bucket_key = (c.timestamp // bucket_ms) * bucket_ms
        buckets.setdefault(bucket_key, []).append(c)

    resampled = []
    for bucket_ts in sorted(buckets):
        group = buckets[bucket_ts]
        resampled.append(
            Candle(
                timestamp=bucket_ts,
                datetime=group[0].datetime,
                symbol=group[0].symbol,
                open=group[0].open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                base_volume=sum(c.base_volume for c in group),
                quote_volume=sum(c.quote_volume for c in group),
                trade_count=sum(c.trade_count for c in group),
            )
        )
    return resampled
