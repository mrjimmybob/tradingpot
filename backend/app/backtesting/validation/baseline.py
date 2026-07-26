"""Measure every strategy's shipped defaults over one range, in one pass.

Every strategy default in this codebase was set by hand and never checked
against data. This runs the same out-of-sample measurement over all of them so
that gap can be closed once, comparably: identical range, identical windows,
identical execution model, each strategy measured with the parameters it
actually ships with.

The cross-strategy summary is a **comparison, not a ranking**. Rows stay in a
fixed declaration order and are never sorted by performance, because a table
sorted by expectancy is a recommendation wearing a table's clothes - and the
sample sizes available over six years of history do not support one. Choosing
between strategies on these numbers would be exactly the selection step this
change is scoped to stay out of.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from app.backtesting.candle import Candle

from .edge_record import (
    ValidatedEdgeRecord,
    build_validated_edge_record,
    edge_record_blockers,
)
from .measurement import FixedConfig, MeasurementSpan
from .walk_forward import WalkForwardMeasurement, run_walk_forward

# The six concrete strategies TradingEngine can dispatch to. ``auto_mode`` is
# deliberately absent: it selects among these rather than trading a thesis of
# its own, so measuring it here would double-count whichever ones it picked.
BASELINE_STRATEGIES: Tuple[str, ...] = (
    "dca_accumulator",
    "adaptive_grid",
    "mean_reversion",
    "trend_following",
    "volatility_breakout",
    "dip_recovery",
)


@dataclass(frozen=True)
class BaselineEntry:
    """One strategy's measurement, and the record it did or did not support."""

    strategy: str
    walk_forward: WalkForwardMeasurement
    record: Optional[ValidatedEdgeRecord]
    blockers: Tuple[str, ...]


async def measure_baseline(
    engine,
    candles: Sequence[Candle],
    trading_pair: str,
    window_ms: int,
    strategies: Sequence[str] = BASELINE_STRATEGIES,
    step_ms: Optional[int] = None,
    starting_balance: float = 10_000.0,
    params_by_strategy: Optional[Dict[str, dict]] = None,
    span: Optional[MeasurementSpan] = None,
    quiet: bool = True,
    on_strategy: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[BaselineEntry, ...]:
    """Measure each strategy over the same candles, windows, and execution model.

    ``params_by_strategy`` supplies fixed parameters per strategy; a strategy
    absent from it is measured with ``{}``, which is what makes this a
    measurement of the *shipped defaults* - the strategy falls back to its own
    internal ones, exactly as a freshly created bot would.
    """
    overrides = params_by_strategy or {}
    entries: List[BaselineEntry] = []

    for index, strategy in enumerate(strategies, start=1):
        if on_strategy is not None:
            on_strategy(index, len(strategies), strategy)
        config = FixedConfig(
            strategy=strategy,
            trading_pair=trading_pair,
            params=overrides.get(strategy, {}),
            starting_balance=starting_balance,
        )
        result = await run_walk_forward(
            engine, candles, config, window_ms=window_ms, step_ms=step_ms,
            span=span, quiet=quiet,
        )
        entries.append(BaselineEntry(
            strategy=strategy,
            walk_forward=result,
            record=build_validated_edge_record(result),
            blockers=edge_record_blockers(result),
        ))

    return tuple(entries)


def format_baseline_summary(entries: Sequence[BaselineEntry]) -> str:
    """One row per strategy, in declaration order - never sorted by result."""
    lines: List[str] = []
    lines.append("")
    lines.append("Cross-strategy summary (comparison, NOT a ranking)")

    header = (
        f"{'Strategy':<22}{'Windows':>9}{'Traded':>8}{'Trades':>8}"
        f"{'Consistency':>24}{'Pooled exp.':>14}{'Record':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for entry in entries:
        result = entry.walk_forward
        pooled = f"{entry.record.estimate.expectancy:+.2f}" if entry.record else "-"
        lines.append(
            f"{entry.strategy:<22}"
            f"{result.num_windows:>9}"
            f"{len(result.windows_with_trades):>8}"
            f"{result.total_trades:>8}"
            f"{result.consistency:>24}"
            f"{pooled:>14}"
            f"{('yes' if entry.record else 'no'):>10}"
        )
    lines.append("-" * len(header))
    lines.append("")
    lines.append(
        "Rows are in declaration order and are NOT sorted by performance. This table "
        "compares what was measured; it does not recommend a strategy, and the sample "
        "sizes here would not support one. 'Record' is whether the measurement met the "
        "bar for a validated out-of-sample record - see each strategy's section for why "
        "not, where it did not."
    )
    lines.append(
        "'Pooled exp.' is expectancy per closed trade in quote currency, pooled across "
        "out-of-sample windows, and is blank where no record could be produced."
    )
    lines.append(
        "IMPORTANT: 'Trades' counts CLOSED round trips only. A strategy that scales in "
        "and out without ever fully flattening - an accumulator, or a grid - can trade "
        "throughout a window and still show 0 here. A zero row therefore does NOT mean "
        "the strategy was inactive: read its Return%/MaxDD% columns above, which mark "
        "open positions to market. Expectancy per closed trade is simply not a "
        "meaningful measure for those strategies, which is why no record is produced "
        "for them rather than a misleading one."
    )
    lines.append("")
    return "\n".join(lines)
