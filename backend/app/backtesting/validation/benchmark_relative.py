"""Return and drawdown measured against benchmarks - the instrument for
strategies that close no round trips.

Expectancy, win rate, and profit factor are all denominated in *closed round
trips*, so they are undefined for a strategy that never closes one. Three of
this project's six strategies never do: ``dca_accumulator`` never sells by
design, and ``adaptive_grid``/``dip_recovery`` scale out partially while the
backtest portfolio only records a closed ``TradeRecord`` on a **full** close
(``portfolio.py:65-73``). Those strategies are not inactive - their equity moves,
substantially - they are just invisible to that instrument.

Everything here is derived from the equity curve a measurement already recorded,
so it is defined whether or not a single trade closed.

The one thing that must not be read carelessly is excess return. A strategy that
deploys capital gradually is under-exposed by construction and will trail
buy-and-hold in a rising market regardless of how well it chose its entries.
Exposure is therefore reported on the same row as the comparison, and where it
cannot be estimated honestly it is reported as unavailable rather than as a
number.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from app.backtesting.candle import Candle
from app.backtesting.results import compute_result

from .benchmarks import BenchmarkCurve, build_benchmarks

# Below this many equity points, a beta estimate is noise dressed as a number.
MIN_POINTS_FOR_EXPOSURE = 10

# Minimum standard deviation of the asset's per-candle returns before a beta is
# worth computing. A real asset's per-candle return deviation is on the order of
# 1e-3; a synthetic series with perfectly constant returns has a deviation of
# ~1e-17, which is float noise, and dividing a near-zero covariance by it yields
# arbitrarily large nonsense (a constant-drift fixture produced -4.9e10 during
# development). Testing `variance > 0` does not catch that - the variance is
# genuinely positive, just meaningless.
MIN_RETURN_STDDEV_FOR_EXPOSURE = 1e-9


@dataclass(frozen=True)
class BenchmarkComparison:
    """One strategy measured against one benchmark."""

    benchmark_label: str
    benchmark_return_pct: float
    benchmark_max_drawdown_pct: float
    excess_return_pct: float
    drawdown_difference_pct: float

    @property
    def beat_on_return(self) -> bool:
        return self.excess_return_pct > 0

    @property
    def beat_on_drawdown(self) -> bool:
        """Lower drawdown than the benchmark. Negative difference is better."""
        return self.drawdown_difference_pct < 0


@dataclass(frozen=True)
class BenchmarkRelativeMeasurement:
    """A strategy's own return/drawdown plus its standing against each benchmark."""

    strategy: str
    return_pct: float
    max_drawdown_pct: float
    exposure: Optional[float]
    num_closed_trades: int
    comparisons: Tuple[BenchmarkComparison, ...] = ()

    @property
    def return_per_unit_drawdown(self) -> Optional[float]:
        """Return per unit of worst drawdown suffered to get it.

        ``None`` rather than infinity when there was no drawdown: a strategy
        that never drew down over the measured span has an undefined ratio, and
        printing ``inf`` would read as an unbounded edge rather than as a span
        too benign to have tested it.
        """
        if self.max_drawdown_pct <= 0:
            return None
        return self.return_pct / self.max_drawdown_pct

    @property
    def has_no_closed_trades(self) -> bool:
        """The case this module exists for: measurable here, not by expectancy."""
        return self.num_closed_trades == 0


def _drawdown_pct(equity_curve: Sequence[Tuple[int, float]], starting_balance: float) -> float:
    if not equity_curve:
        return 0.0
    return compute_result(
        starting_balance=starting_balance,
        ending_balance=equity_curve[-1][1],
        trades=[],
        equity_curve=list(equity_curve),
        total_fees_paid=0.0,
        buy_and_hold_return_pct=0.0,
    ).max_drawdown_pct


def _returns(values: Sequence[float]) -> List[float]:
    out = []
    for earlier, later in zip(values, values[1:]):
        out.append((later - earlier) / earlier if earlier else 0.0)
    return out


def estimate_exposure(
    equity_curve: Sequence[Tuple[int, float]], candles: Sequence[Candle]
) -> Optional[float]:
    """Realised exposure to the asset, as the beta of equity returns to price returns.

    A proxy, not a measured cash/base split: the portfolio records only total
    equity (``portfolio.py:94``), so the split is not recoverable after the fact
    and recovering it would mean modifying the engine. For a long-only spot
    strategy, beta to the asset is a faithful stand-in for average fractional
    deployment - roughly 1.0 for something always fully invested, roughly 0.0
    for something sitting in cash.

    Returns ``None`` where the basis is degenerate (too few points, or an asset
    that did not move), because a beta computed against near-zero variance is an
    arbitrarily large number with no meaning.
    """
    if len(equity_curve) < MIN_POINTS_FOR_EXPOSURE or len(candles) < MIN_POINTS_FOR_EXPOSURE:
        return None

    # Align on the overlapping prefix: the engine marks equity once per candle,
    # but a caller may hand in a curve and series of slightly different length.
    length = min(len(equity_curve), len(candles))
    equity_returns = _returns([point[1] for point in equity_curve[:length]])
    price_returns = _returns([candle.close for candle in candles[:length]])
    if len(equity_returns) < MIN_POINTS_FOR_EXPOSURE - 1:
        return None

    n = len(price_returns)
    mean_price = sum(price_returns) / n
    variance = sum((r - mean_price) ** 2 for r in price_returns) / n
    if variance ** 0.5 < MIN_RETURN_STDDEV_FOR_EXPOSURE:
        return None

    mean_equity = sum(equity_returns) / n
    covariance = sum(
        (e - mean_equity) * (p - mean_price)
        for e, p in zip(equity_returns, price_returns)
    ) / n
    return covariance / variance


def measure_against_benchmarks(
    measurement,
    candles: Sequence[Candle],
    benchmarks: Optional[Sequence[BenchmarkCurve]] = None,
    cadence_ms: Optional[int] = None,
    model=None,
) -> BenchmarkRelativeMeasurement:
    """Compare a :class:`Measurement` to each benchmark on return and drawdown.

    ``benchmarks`` may be supplied to reuse curves across many strategies
    measured over the same candles - the curves depend only on the candles and
    the cost model, never on the strategy, so building them once is not a
    shortcut but the correct thing to do.
    """
    if benchmarks is None:
        kwargs = {} if cadence_ms is None else {"cadence_ms": cadence_ms}
        benchmarks = build_benchmarks(
            candles, measurement.starting_balance, model, **kwargs,
        )

    strategy_drawdown = measurement.max_drawdown_pct
    comparisons = tuple(
        BenchmarkComparison(
            benchmark_label=benchmark.label,
            benchmark_return_pct=benchmark.terminal_return_pct,
            benchmark_max_drawdown_pct=benchmark.max_drawdown_pct,
            excess_return_pct=measurement.total_return_pct - benchmark.terminal_return_pct,
            drawdown_difference_pct=strategy_drawdown - benchmark.max_drawdown_pct,
        )
        for benchmark in benchmarks
    )

    return BenchmarkRelativeMeasurement(
        strategy=measurement.strategy,
        return_pct=measurement.total_return_pct,
        max_drawdown_pct=strategy_drawdown,
        exposure=estimate_exposure(measurement.equity_curve, candles),
        num_closed_trades=measurement.num_trades,
        comparisons=comparisons,
    )


@dataclass(frozen=True)
class WindowBenchmarkRow:
    """One out-of-sample window's benchmark-relative standing."""

    span_label: str
    relative: BenchmarkRelativeMeasurement


def measure_windows_against_benchmarks(
    measurements: Sequence,
    candles: Sequence[Candle],
    cadence_ms: Optional[int] = None,
    model=None,
) -> Tuple[WindowBenchmarkRow, ...]:
    """Compare each out-of-sample window to benchmarks built over *that window*.

    Benchmarks are rebuilt per window rather than once over the whole range, so
    each window stays an independent sample: a benchmark spanning the full range
    would carry information from outside the window into its comparison.
    """
    from .measurement import select_candles

    rows: List[WindowBenchmarkRow] = []
    for measurement in measurements:
        window_candles = select_candles(candles, measurement.span)
        if not window_candles:
            continue
        rows.append(WindowBenchmarkRow(
            span_label=measurement.span.label(),
            relative=measure_against_benchmarks(
                measurement, window_candles, cadence_ms=cadence_ms, model=model,
            ),
        ))
    return tuple(rows)


def count_windows_beating(
    rows: Sequence[WindowBenchmarkRow], benchmark_label: str
) -> Tuple[int, int]:
    """(windows with higher return than the benchmark, windows compared).

    A count, deliberately, not a score: "8 of 13" invites reading the other five,
    where a single ratio would invite reading nothing else.
    """
    compared = 0
    beat = 0
    for row in rows:
        for comparison in row.relative.comparisons:
            if not comparison.benchmark_label.startswith(benchmark_label):
                continue
            compared += 1
            if comparison.beat_on_return:
                beat += 1
    return beat, compared


def format_windows_benchmark_report(strategy: str, rows: Sequence[WindowBenchmarkRow]) -> str:
    """Per-window return/drawdown against each benchmark, side by side."""
    lines: List[str] = ["", f"Benchmark-relative measurement by window: {strategy}"]
    if not rows:
        lines.append("  No windows to compare.")
        lines.append("")
        return "\n".join(lines)

    benchmark_labels = [c.benchmark_label for c in rows[0].relative.comparisons]
    header = (
        f"{'Window':<26}{'Return%':>10}{'MaxDD%':>9}{'Ret/DD':>9}{'Expo':>7}"
        + "".join(f"{('vs ' + label.split(' (')[0]):>22}" for label in benchmark_labels)
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        relative = row.relative
        cells = "".join(
            f"{(f'{c.excess_return_pct:+.1f}% / {c.drawdown_difference_pct:+.1f}dd'):>22}"
            for c in relative.comparisons
        )
        lines.append(
            f"{row.span_label:<26}"
            f"{relative.return_pct:>+10.2f}"
            f"{relative.max_drawdown_pct:>9.2f}"
            f"{_fmt_optional(relative.return_per_unit_drawdown, '.2f'):>9}"
            f"{_fmt_optional(relative.exposure, '.2f'):>7}"
            + cells
        )
    lines.append("-" * len(header))

    for label in benchmark_labels:
        beat, compared = count_windows_beating(rows, label.split(" (")[0])
        lines.append(f"Windows with higher return than {label}: {beat} of {compared}")
    total_trades = sum(r.relative.num_closed_trades for r in rows)
    lines.append(f"Closed round trips across all windows: {total_trades}")
    lines.append("")
    lines.append(
        "Columns after Expo are 'excess return / drawdown difference' versus that "
        "benchmark. Negative drawdown difference means the strategy drew down LESS "
        "than the benchmark."
    )
    lines.append("")
    # Limitations describe the run as a whole, so they are derived from an
    # aggregate view rather than from whichever window happened to be first -
    # otherwise a run whose first window closed a trade would suppress the
    # "expectancy is undefined here" note that the other twelve earned.
    exposures = [r.relative.exposure for r in rows if r.relative.exposure is not None]
    aggregate = replace(
        rows[0].relative,
        num_closed_trades=total_trades,
        exposure=(sum(exposures) / len(exposures)) if exposures else None,
    )
    for note in benchmark_limitations(aggregate):
        lines.append(f"  - {note}")
    lines.append("")
    return "\n".join(lines)


def benchmark_limitations(result: BenchmarkRelativeMeasurement) -> List[str]:
    """What must travel with these numbers for them to be read correctly."""
    notes: List[str] = []

    if result.has_no_closed_trades:
        notes.append(
            "This strategy closed NO round trips over the measured span, so expectancy, "
            "win rate and profit factor are undefined for it. The return and drawdown "
            "above are the measurement - not a fallback, and not evidence it was idle."
        )
    notes.append(
        "Closed-trade counts understate strategies that scale out without ever fully "
        "flattening: the backtest portfolio records a closed trade only on a FULL close "
        "(portfolio.py:65-73). Realised P&L from partial exits is in the equity curve "
        "above but not in the trade count."
    )
    if result.exposure is None:
        notes.append(
            "Exposure could not be estimated for this span (too few equity points, or an "
            "asset that did not move), so it is reported as unavailable rather than as a "
            "number that would not mean anything."
        )
    else:
        notes.append(
            f"Exposure ({result.exposure:.2f}) is the beta of equity returns to the "
            "asset's - a proxy for average deployment, not a measured cash/asset split. "
            "A strategy below ~1.0 was less exposed than buy-and-hold, so part of any "
            "shortfall against it is exposure rather than selection."
        )
    notes.append(
        "Excess return is NOT skill when exposure differs. Compare a gradually deploying "
        "strategy against the periodic-DCA benchmark, which deploys on a comparable "
        "schedule, rather than against buy-and-hold."
    )
    notes.append(
        "Benchmark costs are modelled with the same fee model the strategy paid, not "
        "observed. Benchmarks pay fees; they are not frictionless ideals."
    )
    return notes


def _fmt_optional(value: Optional[float], spec: str = "+.2f") -> str:
    return "-" if value is None else format(value, spec)


def format_benchmark_report(
    result: BenchmarkRelativeMeasurement, title: str = ""
) -> str:
    """The strategy's own return/drawdown, then its standing against each benchmark."""
    lines: List[str] = ["", title or f"Benchmark-relative measurement: {result.strategy}"]

    header = f"{'':<26}{'Return%':>11}{'MaxDD%':>10}{'Ret/DD':>10}{'Exposure':>11}"
    lines.append(header)
    lines.append("-" * len(header))
    lines.append(
        f"{result.strategy:<26}"
        f"{result.return_pct:>+11.2f}"
        f"{result.max_drawdown_pct:>10.2f}"
        f"{_fmt_optional(result.return_per_unit_drawdown, '.2f'):>10}"
        f"{_fmt_optional(result.exposure, '.2f'):>11}"
    )
    for comparison in result.comparisons:
        lines.append(
            f"{'  ' + comparison.benchmark_label:<26}"
            f"{comparison.benchmark_return_pct:>+11.2f}"
            f"{comparison.benchmark_max_drawdown_pct:>10.2f}"
            f"{'':>10}{'':>11}"
        )
        lines.append(
            f"{'    vs ' + comparison.benchmark_label:<26}"
            f"{comparison.excess_return_pct:>+11.2f}"
            f"{comparison.drawdown_difference_pct:>+10.2f}"
            f"{'':>10}{'':>11}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"Closed round trips: {result.num_closed_trades}"
        + ("  (expectancy is undefined for this strategy)" if result.has_no_closed_trades else "")
    )
    lines.append("")
    for note in benchmark_limitations(result):
        lines.append(f"  - {note}")
    lines.append("")
    return "\n".join(lines)
