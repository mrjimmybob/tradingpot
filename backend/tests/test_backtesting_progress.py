"""Tests for CLI progress reporting during CSV loading and backtest replay
(progress-reporting change). Purely about visibility - none of these assert
on backtest results or candle parsing, only on what gets printed to stdout.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.backtesting.data_provider import CsvHistoricalDataProvider
from app.backtesting.engine import BacktestEngine
from app.backtesting.execution_model import BacktestExecutionModel


def _write_minute_csv(path: Path, n: int, base_ms: int = 1704067200000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Unix,Date,Symbol,Open,High,Low,Close,Volume FOO,Volume BAR,tradecount\n"]
    for i in range(n):
        price = 100.0 * (1 + 0.02 * math.sin(i / 12.0))
        ts = base_ms + i * 60_000
        lines.append(f"{ts},d,FOOBAR,{price},{price * 1.001},{price * 0.999},{price},1,{price},1\n")
    path.write_text("".join(lines))


@pytest.fixture
def fake_root(tmp_path) -> Path:
    return tmp_path / "data" / "backtest"


_MEAN_REVERSION_PARAMS = {
    "bar_interval_seconds": 0, "regime_filter_enabled": False,
    "bollinger_period": 10, "atr_period": 10, "cooldown_seconds": 0,
}


class TestCsvLoadProgress:
    def test_progress_output_appears_during_csv_loading(self, fake_root, capsys):
        csv_path = fake_root / "zaplex" / "FOOBAR" / "1m" / "a.csv"
        _write_minute_csv(csv_path, 200)

        provider = CsvHistoricalDataProvider(root=fake_root)
        candles = provider.get_candles("zaplex", "FOOBAR", "1m")

        out = capsys.readouterr().out
        assert len(candles) == 200
        assert "Loading:" in out
        assert csv_path.name in out
        assert "Rows: 200" in out
        assert "[" in out and "]" in out and "%" in out
        assert "Finished:" in out
        assert "Loaded: 200 candles" in out
        assert "Skipped: 0 rows" in out
        assert "Time:" in out

    def test_quiet_suppresses_csv_progress(self, fake_root, capsys):
        csv_path = fake_root / "zaplex" / "FOOBAR" / "1m" / "a.csv"
        _write_minute_csv(csv_path, 200)

        provider = CsvHistoricalDataProvider(root=fake_root, quiet=True)
        candles = provider.get_candles("zaplex", "FOOBAR", "1m")

        out = capsys.readouterr().out
        assert len(candles) == 200
        assert out == ""

    def test_large_csv_does_not_generate_excessive_output(self, fake_root, capsys):
        csv_path = fake_root / "zaplex" / "FOOBAR" / "1m" / "a.csv"
        _write_minute_csv(csv_path, 50_000)

        provider = CsvHistoricalDataProvider(root=fake_root)
        candles = provider.get_candles("zaplex", "FOOBAR", "1m")

        out = capsys.readouterr().out
        assert len(candles) == 50_000
        # A handful of progress updates, nowhere near one line/write per row.
        assert out.count("%") < 200


class TestBacktestReplayProgress:
    def _engine(self, root: Path, quiet_provider: bool = False) -> BacktestEngine:
        return BacktestEngine(
            data_provider=CsvHistoricalDataProvider(root=root, quiet=quiet_provider),
            execution_model=BacktestExecutionModel(fee_pct=0.1),
        )

    @pytest.mark.asyncio
    async def test_progress_output_appears_during_backtest_replay(self, fake_root, capsys):
        _write_minute_csv(fake_root / "zaplex" / "FOOBAR" / "1m" / "a.csv", 200)
        engine = self._engine(fake_root, quiet_provider=True)

        await engine.run(
            exchange="zaplex", symbol="FOOBAR", timeframe="1m",
            strategy="mean_reversion", strategy_params=_MEAN_REVERSION_PARAMS,
            starting_balance=10_000.0,
        )

        out = capsys.readouterr().out
        assert "Starting backtest:" in out
        assert "Symbol:\nFOOBAR" in out
        assert "Strategy:\nmean_reversion" in out
        assert "Candles:\n200" in out
        assert "Range:" in out
        assert "[" in out and "]" in out and "%" in out
        assert "Date:" in out
        assert "Trades:" in out
        assert "Equity:" in out

    @pytest.mark.asyncio
    async def test_quiet_suppresses_backtest_progress(self, fake_root, capsys):
        _write_minute_csv(fake_root / "zaplex" / "FOOBAR" / "1m" / "a.csv", 200)
        engine = self._engine(fake_root, quiet_provider=True)

        await engine.run(
            exchange="zaplex", symbol="FOOBAR", timeframe="1m",
            strategy="mean_reversion", strategy_params=_MEAN_REVERSION_PARAMS,
            starting_balance=10_000.0,
            quiet=True,
        )

        out = capsys.readouterr().out
        assert out == ""

    @pytest.mark.asyncio
    async def test_large_backtest_does_not_generate_excessive_output(self, fake_root, capsys):
        _write_minute_csv(fake_root / "zaplex" / "FOOBAR" / "1m" / "a.csv", 5_000)
        engine = self._engine(fake_root, quiet_provider=True)

        result = await engine.run(
            exchange="zaplex", symbol="FOOBAR", timeframe="1m",
            strategy="mean_reversion", strategy_params=_MEAN_REVERSION_PARAMS,
            starting_balance=10_000.0,
        )

        out = capsys.readouterr().out
        assert result.num_trades >= 0
        # Roughly 50-100 updates target (Task 2) - assert well under one per candle.
        assert out.count("Date:") < 150
