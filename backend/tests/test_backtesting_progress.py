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
        # Phase announcements (Loading data is emitted by the provider, since
        # quiet_provider=True here it's the engine's own phases we check).
        assert "Preparing backtest..." in out
        assert "Running strategy replay..." in out
        assert "Computing metrics..." in out
        assert "Starting backtest:" in out
        assert "Symbol:\nFOOBAR" in out
        assert "Strategy:\nmean_reversion" in out
        assert "Candles:\n200" in out
        assert "Range:" in out
        assert "[" in out and "]" in out and "%" in out
        assert "Candles: " in out and " / 199" in out  # processed / total decisions
        assert "Date:" in out
        assert "Trades:" in out
        assert "Equity:" in out
        assert "Elapsed:" in out
        assert "ETA:" in out

    @pytest.mark.asyncio
    async def test_first_update_prints_immediately_without_waiting_for_throttle_window(
        self, fake_root, capsys,
    ):
        """Regression test for the reported bug: for a run large enough that
        update_every > 1, the very first replay update must still appear at
        candle 1, not only once update_every candles have been processed -
        otherwise a long run looks frozen at the start."""
        _write_minute_csv(fake_root / "zaplex" / "FOOBAR" / "1m" / "a.csv", 500)
        engine = self._engine(fake_root, quiet_provider=True)

        await engine.run(
            exchange="zaplex", symbol="FOOBAR", timeframe="1m",
            strategy="mean_reversion", strategy_params=_MEAN_REVERSION_PARAMS,
            starting_balance=10_000.0,
        )

        out = capsys.readouterr().out
        assert "Candles: 1 / 499" in out

    @pytest.mark.asyncio
    async def test_long_backtest_shows_intermediate_progress_before_completion(
        self, fake_root, capsys,
    ):
        """A long-enough run must show more than just a first and a final
        block - there should be genuine progress in between, proving updates
        are throttled (not skipped) across the run rather than only firing
        at the very start and very end."""
        _write_minute_csv(fake_root / "zaplex" / "FOOBAR" / "1m" / "a.csv", 8_000)
        engine = self._engine(fake_root, quiet_provider=True)

        await engine.run(
            exchange="zaplex", symbol="FOOBAR", timeframe="1m",
            strategy="mean_reversion", strategy_params=_MEAN_REVERSION_PARAMS,
            starting_balance=10_000.0,
        )

        out = capsys.readouterr().out
        num_updates = out.count("Candles: ")
        assert num_updates > 2  # more than just "first" and "final"
        assert "Candles: 1 / " in out
        assert "100%" in out  # the final update always completes the bar

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
        # ~100 updates target - assert well under one per candle.
        assert out.count("Date:") < 150

    @pytest.mark.asyncio
    async def test_progress_reporting_does_not_change_backtest_results(self, fake_root):
        """Purely visual: running the same backtest with and without
        progress output must produce byte-identical results."""
        _write_minute_csv(fake_root / "zaplex" / "FOOBAR" / "1m" / "a.csv", 300)

        verbose_result = await self._engine(fake_root, quiet_provider=True).run(
            exchange="zaplex", symbol="FOOBAR", timeframe="1m",
            strategy="mean_reversion", strategy_params=_MEAN_REVERSION_PARAMS,
            starting_balance=10_000.0, quiet=False,
        )
        quiet_result = await self._engine(fake_root, quiet_provider=True).run(
            exchange="zaplex", symbol="FOOBAR", timeframe="1m",
            strategy="mean_reversion", strategy_params=_MEAN_REVERSION_PARAMS,
            starting_balance=10_000.0, quiet=True,
        )

        assert verbose_result.ending_balance == quiet_result.ending_balance
        assert verbose_result.num_trades == quiet_result.num_trades
        assert verbose_result.equity_curve == quiet_result.equity_curve
        assert [(t.entry_timestamp, t.exit_timestamp, t.net_pnl) for t in verbose_result.trades] == \
               [(t.entry_timestamp, t.exit_timestamp, t.net_pnl) for t in quiet_result.trades]
