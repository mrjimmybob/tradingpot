"""Out-of-sample measurement of ONE fixed configuration across rolling windows.

"Walk-forward" here does **not** mean train-then-optimise-then-validate. There
is no training step, because nothing is being chosen: every window measures the
identical operator-supplied parameters. The window split exists for exactly one
reason - to turn a single backtest number into a *sequence* of independent
samples, so that a strategy which happened to profit in one market phase cannot
present that as an edge.

This is the difference the whole change turns on. A single 2020-2026 backtest
of ``dca_accumulator`` cannot distinguish "this strategy has an edge" from "this
window was a bull market". Six successive windows can - not by proving the edge,
but by showing whether it survives being asked the same question six times.

What this module does when the answer is ugly is: nothing. An edge present in
one window and absent in five is reported as an edge present in one window and
absent in five. There is no code path here that reaches for better parameters,
and adding one would be a different change (see the package docstring).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

from app.backtesting.candle import Candle

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.backtesting.engine import BacktestEngine

from .measurement import (
    FixedConfig,
    Measurement,
    MeasurementSpan,
    measure_fixed_config,
    select_candles,
)

MS_PER_DAY = 86_400_000

# Below this many measured windows, per-window agreement is not evidence of
# anything - three coin flips landing heads is not a biased coin. Reports say so
# out loud rather than letting a reader infer confidence from a tidy table.
MIN_WINDOWS_FOR_CONFIDENCE = 5

# The conventional floor for treating a mean of per-trade outcomes as more than
# anecdote. Well short of a real power calculation, and labelled as such.
MIN_TRADES_FOR_CONFIDENCE = 30


@dataclass(frozen=True)
class SkippedWindow:
    """A planned window that could not be measured - reported, never hidden.

    Silently dropping these would quietly shrink the denominator: a run where
    half the windows had no data would otherwise present as a clean, complete
    result over the surviving half.
    """

    span: MeasurementSpan
    num_candles: int
    reason: str


@dataclass(frozen=True)
class WalkForwardMeasurement:
    """Per-window measurements of one fixed configuration, side by side."""

    strategy: str
    trading_pair: str
    params_fingerprint: str
    span: MeasurementSpan
    window_ms: int
    step_ms: int
    windows: Tuple[Measurement, ...]
    skipped: Tuple[SkippedWindow, ...] = ()

    # --- what was measured -------------------------------------------------

    @property
    def num_windows(self) -> int:
        return len(self.windows)

    @property
    def windows_with_trades(self) -> Tuple[Measurement, ...]:
        return tuple(w for w in self.windows if w.num_trades > 0)

    @property
    def total_trades(self) -> int:
        return sum(w.num_trades for w in self.windows)

    @property
    def windows_overlap(self) -> bool:
        """Overlapping windows share candles, so they are not independent
        samples - the honest default is ``step == window``."""
        return self.step_ms < self.window_ms

    @property
    def every_window_used_one_configuration(self) -> bool:
        """The central claim of this change, checked rather than asserted.

        Every :class:`Measurement` carries the fingerprint of the parameters it
        was actually run with. If they all match the configuration's own
        fingerprint, no window can have been measured with anything else.
        """
        return all(w.params_fingerprint == self.params_fingerprint for w in self.windows)

    # --- what the measurements show ----------------------------------------

    @property
    def profitable_windows(self) -> Tuple[Measurement, ...]:
        return tuple(w for w in self.windows_with_trades if w.expectancy_per_trade > 0)

    @property
    def losing_windows(self) -> Tuple[Measurement, ...]:
        return tuple(w for w in self.windows_with_trades if w.expectancy_per_trade <= 0)

    @property
    def consistency(self) -> str:
        """How the sign of per-window expectancy behaves across windows.

        One of ``"not_assessable"`` (fewer than two windows traded at all),
        ``"consistently_positive"``, ``"consistently_negative"``, or ``"mixed"``.

        Not a boolean, and deliberately so. "Consistent" is ambiguous for a
        strategy that loses in every single window - that result is perfectly
        consistent and perfectly bad - so a lone ``True``/``False`` would read
        as a verdict in whichever direction the reader was hoping for. It is
        also not a score: a stable sign across four windows is a weak
        observation, and a number invites treating it as a strong one.
        """
        traded = self.windows_with_trades
        if len(traded) < 2:
            return "not_assessable"
        if len(self.profitable_windows) == len(traded):
            return "consistently_positive"
        if not self.profitable_windows:
            return "consistently_negative"
        return "mixed"

    @property
    def has_positive_edge_in_every_traded_window(self) -> bool:
        """The narrow, unambiguous question. Not evidence of an edge on its
        own - see :meth:`limitations`."""
        return self.consistency == "consistently_positive"

    def limitations(self) -> List[str]:
        """Everything that should stop a reader over-reading the table above.

        Emitted with the report, not buried in documentation, because the whole
        purpose of this tool is to prevent a number from being trusted more
        than it deserves.
        """
        notes: List[str] = []

        traded = self.windows_with_trades
        if self.num_windows < MIN_WINDOWS_FOR_CONFIDENCE:
            notes.append(
                f"Only {self.num_windows} window(s) measured (< {MIN_WINDOWS_FOR_CONFIDENCE}). "
                "This is too few independent samples to distinguish an edge from luck; "
                "treat agreement between windows as suggestive at most."
            )
        if self.total_trades < MIN_TRADES_FOR_CONFIDENCE:
            notes.append(
                f"Only {self.total_trades} trade(s) across all windows "
                f"(< {MIN_TRADES_FOR_CONFIDENCE}). Per-trade expectancy over a sample "
                "this small is dominated by noise."
            )
        if len(traded) < self.num_windows:
            notes.append(
                f"{self.num_windows - len(traded)} of {self.num_windows} window(s) produced "
                "no trades at all. A window with no trades is neither evidence for nor "
                "against an edge; it is excluded from the consistency check."
            )
        if self.skipped:
            notes.append(
                f"{len(self.skipped)} planned window(s) could not be measured (insufficient "
                "candle data) and are listed separately; the range is not fully covered."
            )
        if self.windows_overlap:
            notes.append(
                f"Windows overlap (step {self.step_ms / MS_PER_DAY:.0f}d < window "
                f"{self.window_ms / MS_PER_DAY:.0f}d), so they share candles and are NOT "
                "independent samples. Agreement between overlapping windows is partly an "
                "artefact of the overlap."
            )
        if any(w.num_trades == 0 and w.total_return_pct != 0.0 for w in self.windows):
            notes.append(
                "Some window(s) show a non-zero Return%/MaxDD% with zero trades. Those "
                "columns mark an open position to market at the window's end, whereas "
                "Trades/Win%/PF/Expectancy count only CLOSED round trips - an "
                "accumulating strategy holding through a window shows its unrealised "
                "move here, which is not realised profit."
            )
        notes.append(
            "Each window is measured cold: indicator and regime warm-up happens inside "
            "the window, so the earliest bars of every window are effectively "
            "non-trading. Shorter windows lose proportionally more of their span to it."
        )
        notes.append(
            "Fees and slippage are modelled, not observed. Live fills, funding, and "
            "partial fills will differ."
        )
        notes.append(
            "This is a measurement of a fixed configuration, not a profitability "
            "claim and not a recommendation to change any parameter."
        )
        return notes


def plan_windows(
    span: MeasurementSpan, window_ms: int, step_ms: Optional[int] = None
) -> List[MeasurementSpan]:
    """Successive rolling windows tiling ``span``.

    ``step_ms`` defaults to ``window_ms`` - contiguous, non-overlapping windows,
    which is the only arrangement that yields genuinely independent samples.
    A smaller step is permitted (more windows from limited history) but the
    resulting non-independence is reported as a limitation rather than being
    absorbed silently.

    The final window is clipped to ``span.end_ms``, so the windows together
    cover the requested range exactly with no overhang past the end.
    """
    if span.start_ms is None or span.end_ms is None:
        raise ValueError(
            "plan_windows needs a bounded span; resolve it against the candle "
            "series before planning windows"
        )
    if window_ms <= 0:
        raise ValueError(f"window_ms must be positive (got {window_ms})")
    step = window_ms if step_ms is None else step_ms
    if step <= 0:
        raise ValueError(f"step_ms must be positive (got {step})")
    if step > window_ms:
        # A step wider than the window leaves unmeasured gaps between windows,
        # so the windows would no longer cover the requested range - and a
        # range that is only partly measured is exactly the kind of quiet
        # sampling choice this tool exists to prevent.
        raise ValueError(
            f"step_ms ({step}) must not exceed window_ms ({window_ms}); a larger step "
            "would leave gaps in the measured range"
        )

    windows: List[MeasurementSpan] = []
    start = span.start_ms
    while start <= span.end_ms:
        end = min(start + window_ms - 1, span.end_ms)
        windows.append(MeasurementSpan(start_ms=start, end_ms=end))
        if end >= span.end_ms:
            break
        start += step
    return windows


def resolve_span(candles: Sequence[Candle], span: Optional[MeasurementSpan]) -> MeasurementSpan:
    """Fill in either open end of ``span`` from the candle series itself, so a
    caller can say "the whole history" without knowing its bounds."""
    if not candles:
        raise ValueError("Cannot resolve a measurement span over an empty candle series")
    span = span or MeasurementSpan()
    return MeasurementSpan(
        start_ms=span.start_ms if span.start_ms is not None else candles[0].timestamp,
        end_ms=span.end_ms if span.end_ms is not None else candles[-1].timestamp,
    )


async def run_walk_forward(
    engine: "BacktestEngine",
    candles: Sequence[Candle],
    config: FixedConfig,
    window_ms: int,
    step_ms: Optional[int] = None,
    span: Optional[MeasurementSpan] = None,
    quiet: bool = True,
    on_window: Optional[Callable[[int, int, Measurement], None]] = None,
) -> WalkForwardMeasurement:
    """Measure ``config`` - unchanged - on every window of ``span``.

    ``on_window(index, total, measurement)`` is an optional progress hook for
    long runs; it is observability only and cannot influence a measurement.
    """
    resolved = resolve_span(candles, span)
    planned = plan_windows(resolved, window_ms, step_ms)

    measured: List[Measurement] = []
    skipped: List[SkippedWindow] = []

    for index, window_span in enumerate(planned, start=1):
        available = select_candles(candles, window_span)
        if len(available) < 2:
            skipped.append(SkippedWindow(
                span=window_span,
                num_candles=len(available),
                reason="fewer than 2 candles in range",
            ))
            continue
        # Hand over the already-selected slice rather than the full series:
        # measure_fixed_config filters by the same span, so the result is
        # identical, but the whole run stays O(total candles) instead of
        # re-scanning every candle once per window.
        measurement = await measure_fixed_config(
            engine, available, config, window_span, quiet=quiet,
        )
        measured.append(measurement)
        if on_window is not None:
            on_window(index, len(planned), measurement)

    return WalkForwardMeasurement(
        strategy=config.strategy,
        trading_pair=config.trading_pair,
        params_fingerprint=config.params_fingerprint,
        span=resolved,
        window_ms=window_ms,
        step_ms=window_ms if step_ms is None else step_ms,
        windows=tuple(measured),
        skipped=tuple(skipped),
    )


def _fmt_profit_factor(value: float) -> str:
    """``inf`` is what the engine reports for "wins, no losses" - which over a
    handful of trades is a small sample, not an infinite edge."""
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


def format_walk_forward_report(result: WalkForwardMeasurement) -> str:
    """A per-window table with its own caveats attached.

    The table and the limitations are produced together, by one function, on
    purpose: the numbers should be hard to copy somewhere without the reasons
    they might not mean what they appear to.
    """
    lines: List[str] = []
    lines.append("")
    lines.append(
        f"Walk-forward measurement: {result.strategy} on {result.trading_pair}"
    )
    lines.append(f"Range:      {result.span.label()}")
    lines.append(
        f"Windows:    {result.num_windows} x {result.window_ms / MS_PER_DAY:.0f}d "
        f"(step {result.step_ms / MS_PER_DAY:.0f}d)"
        + ("  [OVERLAPPING - not independent]" if result.windows_overlap else "")
    )
    lines.append(f"Parameters: fixed, fingerprint {result.params_fingerprint}")

    header = (
        f"{'Window':<26}{'Candles':>9}{'Trades':>8}{'Win%':>8}"
        f"{'PF':>8}{'Expectancy':>13}{'MaxDD%':>9}{'Return%':>10}{'B&H%':>10}"
    )
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    for window in result.windows:
        lines.append(
            f"{window.span.label():<26}"
            f"{window.num_candles:>9,}"
            f"{window.num_trades:>8}"
            f"{window.win_rate_pct:>8.1f}"
            f"{_fmt_profit_factor(window.profit_factor):>8}"
            f"{window.expectancy_per_trade:>+13.2f}"
            f"{window.max_drawdown_pct:>9.2f}"
            f"{window.total_return_pct:>+10.2f}"
            f"{window.buy_and_hold_return_pct:>+10.2f}"
        )
    lines.append("-" * len(header))

    for skip in result.skipped:
        lines.append(f"{skip.span.label():<26}  SKIPPED ({skip.reason}, {skip.num_candles} candles)")

    lines.append("")
    lines.append(
        f"Windows measured: {result.num_windows}   "
        f"with trades: {len(result.windows_with_trades)}   "
        f"positive expectancy: {len(result.profitable_windows)}   "
        f"non-positive: {len(result.losing_windows)}"
    )
    lines.append(f"Total closed trades: {result.total_trades}")

    traded_count = len(result.windows_with_trades)
    positive_count = len(result.profitable_windows)
    consistency = result.consistency
    if consistency == "not_assessable":
        lines.append(
            "Consistency: NOT ASSESSABLE - fewer than two windows produced any trades."
        )
    elif consistency == "consistently_positive":
        lines.append(
            f"Consistency: CONSISTENTLY POSITIVE - all {traded_count} trading window(s) "
            "had positive expectancy. That is consistency across the windows measured, "
            "not proof of an edge."
        )
    elif consistency == "consistently_negative":
        lines.append(
            f"Consistency: CONSISTENTLY NEGATIVE - none of the {traded_count} trading "
            "window(s) had positive expectancy. The result is stable, and stably bad."
        )
    else:
        lines.append(
            f"Consistency: MIXED - {positive_count} of {traded_count} trading windows had "
            "positive expectancy and the rest did not. The strategy's result depends on "
            "which period it is measured over, so no single-window number describes it."
        )

    if not result.every_window_used_one_configuration:
        # Unreachable by construction; reported rather than asserted so that if
        # it ever became reachable, the report would say so instead of quietly
        # presenting incomparable windows side by side.
        lines.append(
            "WARNING: not every window was measured with the same parameters - "
            "these windows are NOT comparable."
        )

    lines.append("")
    lines.append("Limitations:")
    for note in result.limitations():
        lines.append(f"  - {note}")
    lines.append("")
    return "\n".join(lines)
