"""Command-line entry point for read-only strategy validation measurement.

Examples::

    python -m app.backtesting.validation.cli --exchange binance --symbol BTCUSDT \\
        --timeframe 1h --strategy dca_accumulator \\
        --start 2020-01-01 --end 2026-01-01 --window-days 180

    # Also invocable from the repo root:
    python -m backend.app.backtesting.validation.cli --exchange binance \\
        --symbol SOLUSDT --timeframe 1d --strategy mean_reversion

This prints a report and changes nothing. It does not connect to an exchange,
does not touch ``tradingbot.db``, and - unlike anything that could be called an
optimiser - it has no way to alter the parameters it was given. The parameters
you pass in ``--params`` are the parameters every window is measured with; the
report prints their fingerprint so that is checkable rather than trusted.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Runnable both as `python -m app.backtesting.validation.cli` (cwd=backend/,
# the convention the rest of this codebase uses) and as
# `python -m backend.app.backtesting.validation.cli` (cwd=repo root), matching
# app/backtesting/run.py.
_BACKEND_DIR = Path(__file__).resolve().parents[3]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import argparse
import asyncio
import json
from datetime import datetime, timezone

from app.backtesting.data_provider import CsvHistoricalDataProvider, DataIntegrityError
from app.backtesting.engine import BacktestEngine
from app.backtesting.execution_model import BacktestExecutionModel
from app.backtesting.validation.baseline import (
    BASELINE_STRATEGIES,
    BaselineEntry,
    format_baseline_summary,
)
from app.backtesting.validation.edge_record import (
    build_validated_edge_record,
    edge_record_blockers,
    format_edge_record_report,
)
from app.backtesting.validation.measurement import FixedConfig, MeasurementSpan
from app.backtesting.validation.regime import (
    bucket_trades_by_regime,
    build_regime_timeline,
    format_regime_report,
    regime_label_full,
    regime_label_trend,
)
from app.backtesting.validation.walk_forward import (
    MS_PER_DAY,
    format_walk_forward_report,
    run_walk_forward,
)

_REPO_ROOT = _BACKEND_DIR.parent
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "backtest"

# 180 days: over the ~6 years of history on disk this yields ~12 non-overlapping
# windows - enough to see whether a result survives being asked repeatedly,
# while each window is still long enough that indicator warm-up is a small
# fraction of it. A run-time choice, not a contract; override with --window-days.
_DEFAULT_WINDOW_DAYS = 180

# --strategy all: measure every concrete strategy over one range in one pass,
# loading the candle series only once.
ALL_STRATEGIES = "all"


def _parse_date(value: str) -> int:
    """"YYYY-MM-DD" -> unix milliseconds (UTC midnight)."""
    dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a FIXED strategy configuration out-of-sample across rolling "
            "windows. Read-only: measures and reports, never tunes, searches, or "
            "writes back any parameter."
        ),
    )
    parser.add_argument("--exchange", required=True, help="e.g. binance")
    parser.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    parser.add_argument("--timeframe", required=True, help="e.g. 1m, 5m, 15m, 1h, 4h, 1d")
    parser.add_argument(
        "--strategy", required=True,
        help="e.g. dca_accumulator, mean_reversion, ... or 'all' to measure every "
             "concrete strategy over the same range in one pass (loading candles once) "
             "and print a cross-strategy comparison",
    )
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--window-days", type=int, default=_DEFAULT_WINDOW_DAYS,
        help=f"Length of each out-of-sample window (default: {_DEFAULT_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--step-days", type=int, default=None,
        help="Distance between window starts (default: equal to --window-days, i.e. "
             "non-overlapping, independent windows). A smaller step yields more, "
             "OVERLAPPING windows; the report flags them as non-independent.",
    )
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--fee-pct", type=float, default=0.1, help="Per-side fee, default 0.1%%")
    parser.add_argument("--spread-pct", type=float, default=0.0)
    parser.add_argument("--slippage-pct", type=float, default=0.0)
    parser.add_argument(
        "--params", default="{}",
        help="JSON object of the FIXED strategy parameters to measure, e.g. "
             "'{\"bollinger_period\": 14}'. These are measured as given on every "
             "window; nothing here alters them.",
    )
    parser.add_argument(
        "--data-root", default=str(_DEFAULT_DATA_ROOT),
        help=f"Root of the data/backtest directory tree (default: {_DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--skip-regime-report", action="store_true",
        help="Skip the per-regime breakdown. Regime labelling costs roughly one "
             "detector pass per candle, which is noticeable on minute-resolution "
             "multi-year ranges.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-window progress (the report itself is still printed)",
    )
    return parser


def _load_candles(engine: BacktestEngine, args: argparse.Namespace, start_ms, end_ms):
    """Load the whole requested range once, up front.

    Uses the engine's own loader so the timeframe-resampling fallback (asking
    for 1h when only 1m CSVs are stored) behaves identically to a hand-run
    backtest. Reading candles is the only thing this does - windowing happens
    afterwards, in memory, against this one series.
    """
    return engine._load_candles(args.exchange, args.symbol, args.timeframe, start_ms, end_ms)


async def _run(args: argparse.Namespace) -> int:
    provider = CsvHistoricalDataProvider(root=args.data_root, quiet=args.quiet)

    exchanges = provider.list_exchanges()
    if args.exchange not in exchanges:
        print(f"Unknown exchange {args.exchange!r}. Discovered: {exchanges}", file=sys.stderr)
        return 1

    symbols = provider.list_symbols(args.exchange)
    if args.symbol not in symbols:
        print(f"Unknown symbol {args.symbol!r} for {args.exchange!r}. Discovered: {symbols}",
              file=sys.stderr)
        return 1

    try:
        strategy_params = json.loads(args.params)
    except json.JSONDecodeError as e:
        print(f"--params is not valid JSON: {e}", file=sys.stderr)
        return 1
    if not isinstance(strategy_params, dict):
        print("--params must be a JSON object of strategy parameters", file=sys.stderr)
        return 1

    if args.window_days <= 0:
        print("--window-days must be positive", file=sys.stderr)
        return 1
    if args.step_days is not None and args.step_days <= 0:
        print("--step-days must be positive", file=sys.stderr)
        return 1

    strategies = _resolve_strategies(args.strategy)
    from app.services.trading_engine import TradingEngine
    trading_engine = TradingEngine()
    for name in strategies:
        if trading_engine._get_strategy_executor(name) is None:
            print(f"Unknown strategy {name!r}.", file=sys.stderr)
            return 1
    if args.strategy == ALL_STRATEGIES and strategy_params:
        # One --params dict cannot mean the same thing to six different
        # strategies, and silently applying it to all of them would produce a
        # baseline of something nobody configured.
        print(
            f"--params cannot be combined with --strategy {ALL_STRATEGIES}; measure a "
            "single strategy to supply explicit parameters.",
            file=sys.stderr,
        )
        return 1

    engine = BacktestEngine(
        data_provider=provider,
        execution_model=BacktestExecutionModel(
            fee_pct=args.fee_pct, spread_pct=args.spread_pct, slippage_pct=args.slippage_pct,
        ),
    )

    start_ms = _parse_date(args.start) if args.start else None
    end_ms = _parse_date(args.end) if args.end else None

    try:
        candles = _load_candles(engine, args, start_ms, end_ms)
    except (DataIntegrityError, ValueError) as e:
        print(f"Could not load candles: {e}", file=sys.stderr)
        return 1
    if len(candles) < 2:
        print(f"Loaded {len(candles)} candle(s); need at least 2 to measure.", file=sys.stderr)
        return 1

    def _report_window(index: int, total: int, measurement) -> None:
        print(
            f"  window {index}/{total}  {measurement.span.label()}  "
            f"{measurement.num_trades} trade(s)",
            flush=True,
        )

    span = MeasurementSpan(start_ms=start_ms, end_ms=end_ms)
    entries = []
    for position, name in enumerate(strategies, start=1):
        if len(strategies) > 1:
            print(f"\n{'=' * 79}\n[{position}/{len(strategies)}] {name}\n{'=' * 79}", flush=True)

        config = FixedConfig(
            strategy=name,
            trading_pair=args.symbol,
            params=strategy_params,
            starting_balance=args.starting_balance,
        )
        try:
            result = await run_walk_forward(
                engine,
                candles,
                config,
                window_ms=args.window_days * MS_PER_DAY,
                step_ms=args.step_days * MS_PER_DAY if args.step_days else None,
                span=span,
                quiet=True,  # the engine's own per-candle bar would drown the run
                on_window=None if args.quiet else _report_window,
            )
        except ValueError as e:
            print(f"Measurement could not run: {e}", file=sys.stderr)
            return 1

        print(format_walk_forward_report(result))
        if not args.skip_regime_report:
            _print_regime_reports(candles, result, quiet=args.quiet)
        print(format_edge_record_report(result))

        entries.append(BaselineEntry(
            strategy=name,
            walk_forward=result,
            record=build_validated_edge_record(result),
            blockers=edge_record_blockers(result),
        ))

    if len(entries) > 1:
        print(format_baseline_summary(entries))
    return 0


def _resolve_strategies(requested: str) -> list:
    """``all`` expands to the six concrete strategies; anything else is itself."""
    if requested == ALL_STRATEGIES:
        return list(BASELINE_STRATEGIES)
    return [requested]


def _print_regime_reports(candles, result, quiet: bool) -> None:
    """Per-regime breakdown over every trade the walk-forward run produced.

    Pools trades across windows because a per-window *and* per-regime split
    would leave a handful of trades in each cell - too few to read. Pooling is
    only sound while the windows are disjoint, so overlapping runs say plainly
    that some trades are counted more than once rather than presenting a
    double-counted table as if it were clean.
    """
    trades = [trade for window in result.windows for trade in window.trades]
    if not trades:
        print("\nNo closed trades to break down by regime.")
        return

    if not quiet:
        print("\nClassifying market regimes...", flush=True)
    timeline = build_regime_timeline(candles)
    equity_curve = [point for window in result.windows for point in window.equity_curve]

    if result.windows_overlap:
        print(
            "\nNOTE: windows overlap, so trades from shared candles appear in more than "
            "one window and are counted more than once below."
        )

    print(format_regime_report(
        bucket_trades_by_regime(
            trades, timeline, equity_curve, regime_label_trend, label_kind="trend regime",
        ),
        title="Performance by trend regime (rollup)",
    ))
    print(format_regime_report(
        bucket_trades_by_regime(
            trades, timeline, equity_curve, regime_label_full,
            label_kind="full regime (trend/volatility/liquidity)",
        ),
        title="Performance by full regime (trend/volatility/liquidity)",
    ))


def main() -> int:
    args = _build_arg_parser().parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
