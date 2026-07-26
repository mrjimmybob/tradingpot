"""The validation CLI: it measures, prints, and changes nothing.

Uses invented exchange/symbol names against a temporary data root - never
binance/BTC/USDT - so nothing here can accidentally depend on the real
``data/backtest`` tree or on any symbol being special-cased.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.backtesting.validation import cli

HOUR_MS = 3_600_000
START_MS = 1704067200000  # 2024-01-01 UTC


def _write_hourly_csv(path: Path, count: int, base_asset: str, quote_asset: str, symbol: str) -> None:
    """A continuous, gap-free hourly series in the provider's CSV format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"Unix,Date,Symbol,Open,High,Low,Close,"
        f"Volume {base_asset},Volume {quote_asset},tradecount\n"
    )
    rows = [header]
    for i in range(count):
        ts = START_MS + i * HOUR_MS
        price = 100.0 * (1 + 0.05 * math.sin(i / 12.0))
        rows.append(
            f"{ts},2024-01-01 00:00:00,{symbol},{price},{price * 1.001},"
            f"{price * 0.999},{price},1.0,{price},1\n"
        )
    path.write_text("".join(rows))


@pytest.fixture
def data_root(tmp_path) -> Path:
    root = tmp_path / "data" / "backtest"
    _write_hourly_csv(
        root / "zaplex" / "FOOBAR" / "1h" / "FOOBAR_2024.csv",
        count=24 * 12,  # 12 days
        base_asset="FOO", quote_asset="BAR", symbol="FOOBAR",
    )
    return root


def _args(data_root: Path, **overrides):
    argv = [
        "--exchange", "zaplex",
        "--symbol", "FOOBAR",
        "--timeframe", "1h",
        "--strategy", "mean_reversion",
        "--data-root", str(data_root),
        "--window-days", "3",
        "--quiet",
        "--params", '{"bar_interval_seconds": 0, "regime_filter_enabled": false, '
                    '"bollinger_period": 10, "atr_period": 10, "cooldown_seconds": 0, '
                    '"decision_score_threshold": 0.0}',
    ]
    for flag, value in overrides.items():
        argv += [f"--{flag.replace('_', '-')}", str(value)]
    return cli._build_arg_parser().parse_args(argv)


def _tree_snapshot(root: Path):
    """Every file under ``root`` with its size and mtime - enough to detect any
    write, rewrite, or deletion the CLI might perform."""
    return {
        p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*")) if p.is_file()
    }


class TestArgumentParsing:
    def test_defaults_are_non_overlapping_windows(self):
        args = cli._build_arg_parser().parse_args([
            "--exchange", "zaplex", "--symbol", "FOOBAR",
            "--timeframe", "1h", "--strategy", "mean_reversion",
        ])
        assert args.window_days == cli._DEFAULT_WINDOW_DAYS
        assert args.step_days is None, (
            "step defaults to the window size, i.e. independent windows"
        )
        assert args.params == "{}"

    def test_there_is_no_optimisation_flag(self):
        """A CLI is where a "just try a few values" flag would appear first.

        Checked against the parser's actual option names, not its help prose -
        the description deliberately *does* contain "tunes" and "searches", in
        the sentence saying it never does either.
        """
        flags = [
            option.lower()
            for action in cli._build_arg_parser()._actions
            for option in action.option_strings
        ]
        for word in ("optimi", "tune", "sweep", "grid", "search", "candidate", "best"):
            assert not any(word in flag for flag in flags), f"CLI exposes a {word!r} flag"


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_unknown_exchange_fails_cleanly(self, data_root, capsys):
        args = _args(data_root)
        args.exchange = "nosuchexchange"
        assert await cli._run(args) == 1
        assert "Unknown exchange" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_unknown_symbol_fails_cleanly(self, data_root, capsys):
        args = _args(data_root)
        args.symbol = "NOSUCHPAIR"
        assert await cli._run(args) == 1
        assert "Unknown symbol" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_unknown_strategy_fails_cleanly(self, data_root, capsys):
        args = _args(data_root)
        args.strategy = "no_such_strategy"
        assert await cli._run(args) == 1
        assert "Unknown strategy" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_malformed_params_json_fails_cleanly(self, data_root, capsys):
        args = _args(data_root)
        args.params = "{not json"
        assert await cli._run(args) == 1
        assert "not valid JSON" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_params_must_be_an_object(self, data_root, capsys):
        args = _args(data_root)
        args.params = "[1, 2, 3]"
        assert await cli._run(args) == 1
        assert "must be a JSON object" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_non_positive_window_fails_cleanly(self, data_root, capsys):
        args = _args(data_root)
        args.window_days = 0
        assert await cli._run(args) == 1
        assert "--window-days must be positive" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_non_positive_step_fails_cleanly(self, data_root, capsys):
        args = _args(data_root)
        args.step_days = 0
        assert await cli._run(args) == 1
        assert "--step-days must be positive" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_a_step_wider_than_the_window_is_refused(self, data_root, capsys):
        args = _args(data_root)
        args.step_days = 9  # window is 3
        assert await cli._run(args) == 1
        assert "gaps" in capsys.readouterr().err


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_prints_a_walk_forward_report_and_succeeds(self, data_root, capsys):
        assert await cli._run(_args(data_root)) == 0

        out = capsys.readouterr().out
        assert "Walk-forward measurement: mean_reversion on FOOBAR" in out
        assert "Windows:" in out
        assert "Parameters: fixed, fingerprint" in out
        assert "Limitations:" in out
        assert "not a profitability claim" in out
        # 12 days of data measured in 3-day windows.
        assert "4 x 3d" in out

    @pytest.mark.asyncio
    async def test_the_run_writes_nothing_to_the_data_root(self, data_root):
        """The read-only claim, checked against the filesystem rather than
        inferred from the absence of a write call."""
        before = _tree_snapshot(data_root)
        assert await cli._run(_args(data_root)) == 0
        assert _tree_snapshot(data_root) == before

    @pytest.mark.asyncio
    async def test_no_stray_files_are_created_anywhere_under_tmp(self, data_root, tmp_path):
        before = _tree_snapshot(tmp_path)
        assert await cli._run(_args(data_root)) == 0
        assert _tree_snapshot(tmp_path) == before

    @pytest.mark.asyncio
    async def test_the_reported_fingerprint_matches_the_parameters_supplied(
        self, data_root, capsys
    ):
        """The operator can verify from the report alone that the parameters
        they passed are the parameters that were measured."""
        import json

        from app.backtesting.validation import FixedConfig

        args = _args(data_root)
        expected = FixedConfig(
            args.strategy, args.symbol, json.loads(args.params), args.starting_balance,
        ).params_fingerprint

        assert await cli._run(args) == 0
        assert expected in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_repeated_runs_produce_an_identical_report(self, data_root, capsys):
        assert await cli._run(_args(data_root)) == 0
        first = capsys.readouterr().out
        assert await cli._run(_args(data_root)) == 0
        second = capsys.readouterr().out
        assert first == second

    @pytest.mark.asyncio
    async def test_a_narrower_date_range_measures_fewer_windows(self, data_root, capsys):
        args = _args(data_root)
        args.start = "2024-01-01"
        args.end = "2024-01-06"
        assert await cli._run(args) == 0
        assert "2 x 3d" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_overlapping_windows_are_flagged_in_the_output(self, data_root, capsys):
        args = _args(data_root)
        args.step_days = 1
        assert await cli._run(args) == 0
        out = capsys.readouterr().out
        assert "OVERLAPPING" in out
        assert "NOT independent" in out

    @pytest.mark.asyncio
    async def test_per_regime_breakdown_is_printed(self, data_root, capsys):
        assert await cli._run(_args(data_root)) == 0
        out = capsys.readouterr().out

        assert "Performance by trend regime (rollup)" in out
        assert "Performance by full regime (trend/volatility/liquidity)" in out
        assert "Exposure" in out
        assert "Total trades bucketed:" in out
        assert "at its ENTRY" in out

    @pytest.mark.asyncio
    async def test_the_regime_report_can_be_skipped(self, data_root, capsys):
        args = _args(data_root)
        args.skip_regime_report = True
        assert await cli._run(args) == 0
        out = capsys.readouterr().out

        assert "Walk-forward measurement" in out
        assert "Performance by trend regime" not in out

    @pytest.mark.asyncio
    async def test_overlapping_windows_disclose_double_counted_trades(self, data_root, capsys):
        args = _args(data_root)
        args.step_days = 1
        assert await cli._run(args) == 0
        assert "counted more than once" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_the_regime_report_writes_nothing_either(self, data_root):
        before = _tree_snapshot(data_root)
        args = _args(data_root)
        assert args.skip_regime_report is False
        assert await cli._run(args) == 0
        assert _tree_snapshot(data_root) == before

    @pytest.mark.asyncio
    async def test_insufficient_candles_fails_cleanly(self, tmp_path, capsys):
        root = tmp_path / "data" / "backtest"
        _write_hourly_csv(
            root / "zaplex" / "FOOBAR" / "1h" / "tiny.csv",
            count=1, base_asset="FOO", quote_asset="BAR", symbol="FOOBAR",
        )
        assert await cli._run(_args(root)) == 1
        assert "need at least 2" in capsys.readouterr().err
