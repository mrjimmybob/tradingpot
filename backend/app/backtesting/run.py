"""Command-line backtest runner.

Examples:

    python -m app.backtesting.run --exchange binance --symbol BTCUSDT \\
        --timeframe 1h --strategy mean_reversion \\
        --start 2020-01-01 --end 2026-01-01

    # Also invocable from the repo root:
    python -m backend.app.backtesting.run --exchange binance --symbol SOLUSDT \\
        --timeframe 1d --strategy auto_mode

Any exchange/symbol under data/backtest/ works with no code change (dynamic
discovery via CsvHistoricalDataProvider). Any of the generated timeframes
(1m/5m/15m/1h/4h/1d) works even if only a finer one is stored on disk - see
BacktestEngine._load_candles. Any strategy TradingEngine can dispatch to
(including auto_mode) works.

Does not connect to any exchange and does not touch tradingbot.db - see
app/backtesting/engine.py (isolated in-memory DB per run) and
app/backtesting/data_provider.py (local CSV only).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make this runnable both as `python -m app.backtesting.run` (cwd=backend/,
# the convention the rest of this codebase uses) and as
# `python -m backend.app.backtesting.run` (cwd=repo root) by ensuring
# backend/ is on sys.path before any `app.*` import below resolves.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import argparse
import asyncio
import json
from datetime import datetime, timezone

from app.backtesting.data_provider import CsvHistoricalDataProvider, DataIntegrityError
from app.backtesting.engine import BacktestEngine
from app.backtesting.execution_model import BacktestExecutionModel
from app.backtesting.results import BacktestResult

_REPO_ROOT = _BACKEND_DIR.parent
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "backtest"


def _parse_date(value: str) -> int:
    """"YYYY-MM-DD" -> unix milliseconds (UTC midnight)."""
    dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a historical backtest against local CSV candle data. "
                     "No exchange connection, no production database access.",
    )
    parser.add_argument("--exchange", required=True, help="e.g. binance (any discovered exchange)")
    parser.add_argument("--symbol", required=True, help="e.g. BTCUSDT (any discovered symbol)")
    parser.add_argument("--timeframe", required=True, help="e.g. 1m, 5m, 15m, 1h, 4h, 1d")
    parser.add_argument("--strategy", required=True, help="e.g. mean_reversion, trend_following, auto_mode, ...")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--fee-pct", type=float, default=0.1, help="Per-side fee, default 0.1%%")
    parser.add_argument("--spread-pct", type=float, default=0.0)
    parser.add_argument("--slippage-pct", type=float, default=0.0)
    parser.add_argument(
        "--params", default="{}",
        help="JSON object of strategy parameters, e.g. '{\"bollinger_period\": 14}'",
    )
    parser.add_argument(
        "--data-root", default=str(_DEFAULT_DATA_ROOT),
        help=f"Root of the data/backtest directory tree (default: {_DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Disable progress output (useful for tests, scripts, automation)",
    )
    return parser


def _print_result(result: BacktestResult, exchange: str, symbol: str, timeframe: str, strategy: str) -> None:
    print(f"\nBacktest: {strategy} on {exchange}/{symbol} [{timeframe}]")
    print("-" * 60)
    print(f"{'Starting balance':<28}${result.starting_balance:,.2f}")
    print(f"{'Ending balance':<28}${result.ending_balance:,.2f}")
    print(f"{'Return %':<28}{result.total_return_pct:+.2f}%")
    print(f"{'Buy-and-hold return %':<28}{result.buy_and_hold_return_pct:+.2f}%")
    print(f"{'Trades':<28}{result.num_trades}")
    print(f"{'Win rate':<28}{result.win_rate:.1f}%")
    print(f"{'Profit factor':<28}{result.profit_factor:.2f}")
    print(f"{'Expectancy per trade':<28}${result.expectancy_per_trade:+.2f}")
    print(f"{'Max drawdown %':<28}{result.max_drawdown_pct:.2f}%")
    print(f"{'Total fees paid':<28}${result.total_fees_paid:,.2f}")
    print("-" * 60)
    print("This is a measurement, not a profitability claim.")


async def _run(args: argparse.Namespace) -> int:
    provider = CsvHistoricalDataProvider(root=args.data_root, quiet=args.quiet)

    exchanges = provider.list_exchanges()
    if args.exchange not in exchanges:
        print(f"Unknown exchange {args.exchange!r}. Discovered: {exchanges}", file=sys.stderr)
        return 1

    symbols = provider.list_symbols(args.exchange)
    if args.symbol not in symbols:
        print(f"Unknown symbol {args.symbol!r} for {args.exchange!r}. Discovered: {symbols}", file=sys.stderr)
        return 1

    try:
        strategy_params = json.loads(args.params)
    except json.JSONDecodeError as e:
        print(f"--params is not valid JSON: {e}", file=sys.stderr)
        return 1

    engine = BacktestEngine(
        data_provider=provider,
        execution_model=BacktestExecutionModel(
            fee_pct=args.fee_pct, spread_pct=args.spread_pct, slippage_pct=args.slippage_pct,
        ),
    )

    from app.services.trading_engine import TradingEngine
    if TradingEngine()._get_strategy_executor(args.strategy) is None:
        print(f"Unknown strategy {args.strategy!r}.", file=sys.stderr)
        return 1

    start_ms = _parse_date(args.start) if args.start else None
    end_ms = _parse_date(args.end) if args.end else None

    try:
        result = await engine.run(
            exchange=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            strategy=args.strategy,
            strategy_params=strategy_params,
            starting_balance=args.starting_balance,
            start=start_ms,
            end=end_ms,
            quiet=args.quiet,
        )
    except (DataIntegrityError, ValueError) as e:
        print(f"Backtest could not run: {e}", file=sys.stderr)
        return 1

    _print_result(result, args.exchange, args.symbol, args.timeframe, args.strategy)
    return 0


def main() -> int:
    args = _build_arg_parser().parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
