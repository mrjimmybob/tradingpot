"""Tests for CsvHistoricalDataProvider (add-historical-backtesting).

Uses temporary fake exchange/symbol/timeframe folders so discovery and
loading are proven dynamic - no exchange, symbol, or asset name is ever
hardcoded in the provider, and these tests use invented names (never BTC/
USDT/binance) to prove it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.backtesting.data_provider import CsvHistoricalDataProvider, DataIntegrityError
from app.backtesting.resampling import resample


def _write_csv(path: Path, rows: list[tuple], base_asset: str, quote_asset: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"Unix,Date,Symbol,Open,High,Low,Close,Volume {base_asset},Volume {quote_asset},tradecount\n"
    lines = [header]
    for unix_ms, date_str, symbol, o, h, l, c, base_vol, quote_vol, trades in rows:
        lines.append(f"{unix_ms},{date_str},{symbol},{o},{h},{l},{c},{base_vol},{quote_vol},{trades}\n")
    path.write_text("".join(lines))


@pytest.fixture
def fake_root(tmp_path) -> Path:
    return tmp_path / "data" / "backtest"


class TestDynamicDiscovery:
    def test_discovers_exchange_symbol_timeframe(self, fake_root):
        _write_csv(
            fake_root / "zaplex" / "FOOBAR" / "1m" / "FOOBAR_2024.csv",
            [(1704067200000, "2024-01-01 00:00:00", "FOOBAR", 1.0, 1.1, 0.9, 1.05, 10.0, 10.5, 5)],
            base_asset="FOO", quote_asset="BAR",
        )
        provider = CsvHistoricalDataProvider(root=fake_root)
        assert provider.list_exchanges() == ["zaplex"]
        assert provider.list_symbols("zaplex") == ["FOOBAR"]
        assert provider.list_timeframes("zaplex", "FOOBAR") == ["1m"]

    def test_new_symbol_works_without_code_changes(self, fake_root):
        """Two different invented symbols with different (invented) asset
        names in their volume columns both load through the identical code
        path - proves no symbol/asset-name special-casing exists."""
        _write_csv(
            fake_root / "zaplex" / "FOOBAR" / "1m" / "a.csv",
            [(1704067200000, "2024-01-01 00:00:00", "FOOBAR", 1.0, 1.1, 0.9, 1.05, 10.0, 10.5, 5)],
            base_asset="FOO", quote_asset="BAR",
        )
        _write_csv(
            fake_root / "zaplex" / "QUXNIX" / "1m" / "a.csv",
            [(1704067200000, "2024-01-01 00:00:00", "QUXNIX", 2.0, 2.2, 1.8, 2.1, 20.0, 42.0, 7)],
            base_asset="QUX", quote_asset="NIX",
        )
        provider = CsvHistoricalDataProvider(root=fake_root)
        assert sorted(provider.list_symbols("zaplex")) == ["FOOBAR", "QUXNIX"]

        foobar = provider.get_candles("zaplex", "FOOBAR", "1m")
        quxnix = provider.get_candles("zaplex", "QUXNIX", "1m")
        assert foobar[0].base_volume == 10.0 and foobar[0].quote_volume == 10.5
        assert quxnix[0].base_volume == 20.0 and quxnix[0].quote_volume == 42.0

    def test_missing_directory_raises_clear_error(self, fake_root):
        provider = CsvHistoricalDataProvider(root=fake_root)
        with pytest.raises(DataIntegrityError):
            provider.get_candles("nope", "NOPE", "1m")


class TestMultiFileMergeAndSort:
    def test_merges_and_sorts_multiple_files(self, fake_root):
        # Two files, each newest-first internally (matches the real fixture
        # convention), non-overlapping ranges, deliberately supplied out of
        # chronological file order.
        _write_csv(
            fake_root / "zaplex" / "FOOBAR" / "1m" / "2025.csv",
            [
                (1704067320000, "d", "FOOBAR", 3, 3, 3, 3, 1, 1, 1),
                (1704067260000, "d", "FOOBAR", 2, 2, 2, 2, 1, 1, 1),
            ],
            base_asset="FOO", quote_asset="BAR",
        )
        _write_csv(
            fake_root / "zaplex" / "FOOBAR" / "1m" / "2024.csv",
            [
                (1704067200000, "d", "FOOBAR", 1, 1, 1, 1, 1, 1, 1),
            ],
            base_asset="FOO", quote_asset="BAR",
        )
        provider = CsvHistoricalDataProvider(root=fake_root)
        candles = provider.get_candles("zaplex", "FOOBAR", "1m")
        assert [c.timestamp for c in candles] == [1704067200000, 1704067260000, 1704067320000]
        assert [c.open for c in candles] == [1, 2, 3]

    def test_duplicate_timestamps_detected(self, fake_root):
        _write_csv(
            fake_root / "zaplex" / "FOOBAR" / "1m" / "a.csv",
            [(1704067200000, "d", "FOOBAR", 1, 1, 1, 1, 1, 1, 1)],
            base_asset="FOO", quote_asset="BAR",
        )
        _write_csv(
            fake_root / "zaplex" / "FOOBAR" / "1m" / "b.csv",
            [(1704067200000, "d", "FOOBAR", 999, 999, 999, 999, 1, 1, 1)],
            base_asset="FOO", quote_asset="BAR",
        )
        provider = CsvHistoricalDataProvider(root=fake_root)
        candles = provider.get_candles("zaplex", "FOOBAR", "1m")
        assert len(candles) == 1  # deduped, not duplicated
        assert provider.last_validation.duplicate_timestamps == [1704067200000]

    def test_gap_detected(self, fake_root):
        _write_csv(
            fake_root / "zaplex" / "FOOBAR" / "1m" / "a.csv",
            [
                (1704067200000, "d", "FOOBAR", 1, 1, 1, 1, 1, 1, 1),
                (1704067500000, "d", "FOOBAR", 2, 2, 2, 2, 1, 1, 1),  # +300s, a 4-interval gap
            ],
            base_asset="FOO", quote_asset="BAR",
        )
        provider = CsvHistoricalDataProvider(root=fake_root)
        candles = provider.get_candles("zaplex", "FOOBAR", "1m")
        assert provider.last_validation.gaps == [(1704067200000, 1704067500000)]

    def test_implausible_timestamp_rows_excluded(self, fake_root):
        """Reproduces the real defect found in the shipped BTCUSDT fixture:
        a Unix column value at the wrong magnitude (nanoseconds mixed into a
        milliseconds file) must be excluded and counted, not silently
        misread into a bogus 1970-something date."""
        nanosecond_scale_value = 1704067200000 * 1_000_000  # would decode to ~1970 if treated as ms
        _write_csv(
            fake_root / "zaplex" / "FOOBAR" / "1m" / "a.csv",
            [
                (1704067200000, "d", "FOOBAR", 1, 1, 1, 1, 1, 1, 1),
                (nanosecond_scale_value, "d", "FOOBAR", 2, 2, 2, 2, 1, 1, 1),
            ],
            base_asset="FOO", quote_asset="BAR",
        )
        provider = CsvHistoricalDataProvider(root=fake_root)
        candles = provider.get_candles("zaplex", "FOOBAR", "1m")
        assert len(candles) == 1
        assert candles[0].timestamp == 1704067200000
        assert provider.last_validation.invalid_timestamp_rows == 1

    def test_start_end_filtering(self, fake_root):
        _write_csv(
            fake_root / "zaplex" / "FOOBAR" / "1m" / "a.csv",
            [
                (1704067200000, "d", "FOOBAR", 1, 1, 1, 1, 1, 1, 1),
                (1704067260000, "d", "FOOBAR", 2, 2, 2, 2, 1, 1, 1),
                (1704067320000, "d", "FOOBAR", 3, 3, 3, 3, 1, 1, 1),
            ],
            base_asset="FOO", quote_asset="BAR",
        )
        provider = CsvHistoricalDataProvider(root=fake_root)
        candles = provider.get_candles(
            "zaplex", "FOOBAR", "1m", start=1704067260000, end=1704067260000,
        )
        assert [c.open for c in candles] == [2]


class TestResampling:
    def _minute_candles(self):
        from app.backtesting.candle import Candle
        base_ms = 1704067200000
        candles = []
        for i in range(10):
            candles.append(Candle(
                timestamp=base_ms + i * 60_000,
                datetime="d", symbol="FOOBAR",
                open=100.0 + i, high=100.0 + i + 0.5, low=100.0 + i - 0.5,
                close=100.0 + i + 0.2, base_volume=1.0, quote_volume=100.0,
                trade_count=1,
            ))
        return candles

    def test_resample_1m_to_5m(self):
        candles = self._minute_candles()
        resampled = resample(candles, "5m")
        assert len(resampled) == 2
        first = resampled[0]
        bucket = candles[:5]
        assert first.open == bucket[0].open
        assert first.close == bucket[-1].close
        assert first.high == max(c.high for c in bucket)
        assert first.low == min(c.low for c in bucket)
        assert first.base_volume == sum(c.base_volume for c in bucket)
        assert first.trade_count == sum(c.trade_count for c in bucket)

    def test_resample_unsupported_timeframe_raises(self):
        with pytest.raises(ValueError):
            resample(self._minute_candles(), "3m")
