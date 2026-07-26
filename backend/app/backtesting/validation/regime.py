"""Per-regime breakdown of measured trades - post-hoc bucketing, not replay.

A single expectancy figure hides the question that actually matters about a
strategy: *when* does it work? A mean-reversion strategy that earns in chop and
bleeds in trends has an average that describes neither state. This module
answers that by labelling every candle with the market regime in force at it,
then bucketing the trades a measurement already produced.

Two properties make the labels trustworthy rather than decorative:

1. **The labels come from the canonical detector.** Regimes are classified by
   ``TradingEngine._detect_market_regime_bar_based`` - the same function
   ``_strategy_auto`` uses to pick strategies live - driven exactly the way
   ``_strategy_auto`` drives it: bar by bar, carrying the previous regime
   forward so the 3-bar persistence hysteresis behaves identically, with the
   same 100-bar window and the same 20-bar minimum before any regime is
   reported at all. A second, ad hoc "is it trending?" heuristic here would
   produce a report about a market that no part of the system trades in.

2. **The labels are causal.** The regime at candle *i* is computed from
   candles ``0..i`` only. Labelling a trade with a regime that could only be
   known later would quietly reintroduce lookahead into a tool whose entire
   purpose is to keep measurement honest.

Nothing here re-runs the engine or influences a decision. It reads a finished
measurement's trades and equity curve and groups them.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from app.backtesting.candle import Candle
from app.backtesting.results import TradeRecord, compute_result

# Mirrors _strategy_auto exactly: it keeps the last 100 bars and only asks for
# a regime once it holds at least 20 of them. Changing either here would make
# these labels describe a different classifier from the live one.
CANONICAL_MAX_BARS = 100
CANONICAL_MIN_BARS = 20

# Trades entered before the detector has enough bars are not "flat" - they are
# unclassified. Labelling them with a real regime would invent data.
UNCLASSIFIED = "unclassified (warm-up)"


def canonical_regime_detector() -> Callable[[list, Optional[dict]], dict]:
    """The exact bound method ``_strategy_auto`` calls to classify regimes.

    Returned rather than imported at module scope so a test can assert the
    identity of the function actually used, and so constructing a
    ``TradingEngine`` (which builds live-trading state) only happens when a
    caller genuinely wants regime labels.
    """
    from app.services.trading_engine import TradingEngine

    return TradingEngine()._detect_market_regime_bar_based


def regime_label_full(regime: Optional[dict]) -> str:
    """The whole regime as the detector defines it: ``trend/volatility/liquidity``.

    This is the regime ``_is_strategy_eligible`` actually gates on, so it is
    the honest primary bucketing - at the cost of splitting trades across up to
    27 buckets, which is why :func:`regime_label_trend` exists alongside it.
    """
    if not regime:
        return UNCLASSIFIED
    return (
        f"{regime.get('trend_state', '?')}/"
        f"{regime.get('volatility_state', '?')}/"
        f"{regime.get('liquidity_state', '?')}"
    )


def regime_label_trend(regime: Optional[dict]) -> str:
    """Trend direction only - a coarser rollup that keeps per-bucket sample
    sizes large enough to be worth reading."""
    if not regime:
        return UNCLASSIFIED
    return str(regime.get("trend_state", "?"))


@dataclass(frozen=True)
class RegimeTimeline:
    """The regime in force at each candle, in ascending timestamp order."""

    timestamps: Tuple[int, ...]
    regimes: Tuple[Optional[dict], ...]

    def at(self, timestamp_ms: int) -> Optional[dict]:
        """The regime in force at ``timestamp_ms``.

        Uses the most recent label at or before the timestamp - the regime a
        live bot would have been acting under at that moment. A timestamp
        before the first labelled candle returns ``None`` (unclassified), never
        the first known regime projected backwards.
        """
        index = bisect_right(self.timestamps, timestamp_ms) - 1
        if index < 0:
            return None
        return self.regimes[index]

    def label_counts(self, label_fn: Callable[[Optional[dict]], str]) -> Dict[str, int]:
        """How many candles carried each label - a strategy's *exposure* to a
        regime, which is what makes "it loses in downtrends" meaningful (or
        not: three candles of downtrend is not a finding)."""
        counts: Dict[str, int] = {}
        for regime in self.regimes:
            label = label_fn(regime)
            counts[label] = counts.get(label, 0) + 1
        return counts


def build_regime_timeline(
    candles: Sequence[Candle],
    detector: Optional[Callable[[list, Optional[dict]], dict]] = None,
) -> RegimeTimeline:
    """Label every candle with the regime in force at it.

    Walks the series once, forward, maintaining the same rolling bar history
    and carried-forward regime that ``_strategy_auto`` maintains, so the
    detector's 3-bar persistence hysteresis sees the identical sequence of
    inputs it would see live. Because the walk is forward-only and the regime
    at candle *i* is computed before candle *i+1* is ever touched, the labels
    contain no lookahead.
    """
    detect = detector or canonical_regime_detector()

    timestamps: List[int] = []
    regimes: List[Optional[dict]] = []
    bars: List[dict] = []
    current: Optional[dict] = None

    for candle in candles:
        bars.append({
            "open": candle.open, "high": candle.high,
            "low": candle.low, "close": candle.close,
        })
        if len(bars) > CANONICAL_MAX_BARS:
            del bars[0]
        if len(bars) >= CANONICAL_MIN_BARS:
            current = detect(bars, current)
        timestamps.append(candle.timestamp)
        regimes.append(current)

    return RegimeTimeline(timestamps=tuple(timestamps), regimes=tuple(regimes))


@dataclass(frozen=True)
class RegimeBucket:
    """Measured performance of the trades entered under one regime label."""

    label: str
    num_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy_per_trade: float
    total_net_pnl: float
    max_drawdown_pct: float
    num_candles: int
    share_of_candles_pct: float

    @property
    def is_small_sample(self) -> bool:
        """Fewer than 10 trades: reported, but not a basis for any claim."""
        return self.num_trades < 10


@dataclass(frozen=True)
class RegimeBreakdown:
    label_kind: str
    buckets: Tuple[RegimeBucket, ...] = ()
    total_trades: int = 0
    total_candles: int = 0

    def limitations(self) -> List[str]:
        notes: List[str] = []
        small = [b.label for b in self.buckets if b.is_small_sample and b.num_trades]
        if small:
            notes.append(
                f"Small samples ({', '.join(small)}): fewer than 10 trades each. "
                "Per-regime expectancy over so few trades is not a finding."
            )
        # The warm-up bucket is excluded here: it is not a regime the market
        # was in, so "no trades were entered under it" is a tautology that only
        # dilutes the real finding about regimes the strategy genuinely sat out.
        thin = [
            b.label for b in self.buckets
            if b.num_trades == 0 and b.label != UNCLASSIFIED
        ]
        if thin:
            notes.append(
                f"No trades were entered under: {', '.join(thin)}. That is an absence "
                "of evidence about those regimes, not evidence the strategy avoids them."
            )
        unclassified = next(
            (b for b in self.buckets if b.label == UNCLASSIFIED and b.num_trades), None
        )
        if unclassified is not None:
            notes.append(
                f"{unclassified.num_trades} trade(s) were entered before the detector had "
                f"{CANONICAL_MIN_BARS} bars and carry no regime. They are shown separately "
                "rather than folded into a real regime."
            )
        notes.append(
            "Regimes are classified from the measured candles' own timeframe. The live "
            "detector runs on 60-second bars, so labels here describe the same classifier "
            "applied at a coarser bar size, not a replay of live regime state."
        )
        notes.append(
            "A trade is attributed entirely to the regime at its ENTRY. A trade held "
            "across a regime change is not split between them."
        )
        return notes


def _contiguous_runs(labels: Sequence[str], target: str) -> List[Tuple[int, int]]:
    """Index ranges over which ``labels`` continuously equals ``target``."""
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, label in enumerate(labels):
        if label == target and start is None:
            start = i
        elif label != target and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(labels) - 1))
    return runs


def _worst_drawdown_while_in_regime(
    equity_curve: Sequence[Tuple[int, float]],
    timestamps: Sequence[int],
    labels: Sequence[str],
    target: str,
) -> float:
    """The worst peak-to-trough drawdown suffered *within* one stretch of this
    regime, taken across all such stretches.

    Computed per contiguous stretch rather than over every candle carrying the
    label: concatenating disjoint periods would carry a peak from March into a
    trough in September and report a drawdown that never happened.

    Each stretch is anchored on the equity mark immediately *preceding* it, so
    a drop that occurs on the move into a regime is attributed to that regime
    rather than falling into the crack between two stretches and being counted
    nowhere at all. The anchor is one point, not a running peak, so the peaks
    of earlier disjoint stretches still cannot leak into a later one.
    """
    if not equity_curve:
        return 0.0
    curve_times = [point[0] for point in equity_curve]
    worst = 0.0
    for start_index, end_index in _contiguous_runs(labels, target):
        start_ts, end_ts = timestamps[start_index], timestamps[end_index]
        lo = max(0, bisect_right(curve_times, start_ts - 1) - 1)
        hi = bisect_right(curve_times, end_ts)
        segment = list(equity_curve[lo:hi])
        if len(segment) < 2:
            continue
        # The engine's own metric function, over just this segment - so a
        # per-regime drawdown is computed exactly the way the headline
        # drawdown is, never by a second implementation that could drift.
        result = compute_result(
            starting_balance=segment[0][1] or 1.0,
            ending_balance=segment[-1][1],
            trades=[],
            equity_curve=segment,
            total_fees_paid=0.0,
            buy_and_hold_return_pct=0.0,
        )
        worst = max(worst, result.max_drawdown_pct)
    return worst


def bucket_trades_by_regime(
    trades: Sequence[TradeRecord],
    timeline: RegimeTimeline,
    equity_curve: Sequence[Tuple[int, float]] = (),
    label_fn: Callable[[Optional[dict]], str] = regime_label_full,
    label_kind: str = "regime",
) -> RegimeBreakdown:
    """Group ``trades`` by the regime at each trade's ENTRY and measure each group.

    Buckets are created for every label the *market* spent time in, not only
    those the strategy traded in - so "this strategy never entered during a
    downtrend" is visible as an empty bucket rather than as a missing row.
    """
    candle_labels = [label_fn(regime) for regime in timeline.regimes]
    candle_counts = timeline.label_counts(label_fn)
    total_candles = len(timeline.regimes)

    grouped: Dict[str, List[TradeRecord]] = {label: [] for label in candle_counts}
    for trade in trades:
        label = label_fn(timeline.at(trade.entry_timestamp))
        grouped.setdefault(label, []).append(trade)

    buckets: List[RegimeBucket] = []
    for label in sorted(grouped, key=lambda l: (l == UNCLASSIFIED, l)):
        bucket_trades = grouped[label]
        # Trade-only metrics (win rate, profit factor, expectancy) depend on
        # the trade list alone; the balances passed here are placeholders that
        # no field read below is derived from. Routing them through the
        # engine's compute_result rather than recomputing them keeps every
        # number in this report defined exactly once, in one place.
        metrics = compute_result(
            starting_balance=1.0, ending_balance=1.0, trades=list(bucket_trades),
            equity_curve=[], total_fees_paid=0.0, buy_and_hold_return_pct=0.0,
        )
        candles_in_regime = candle_counts.get(label, 0)
        buckets.append(RegimeBucket(
            label=label,
            num_trades=metrics.num_trades,
            win_rate_pct=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            expectancy_per_trade=metrics.expectancy_per_trade,
            total_net_pnl=sum(t.net_pnl for t in bucket_trades),
            max_drawdown_pct=_worst_drawdown_while_in_regime(
                equity_curve, timeline.timestamps, candle_labels, label,
            ),
            num_candles=candles_in_regime,
            share_of_candles_pct=(
                candles_in_regime / total_candles * 100.0 if total_candles else 0.0
            ),
        ))

    return RegimeBreakdown(
        label_kind=label_kind,
        buckets=tuple(buckets),
        total_trades=sum(b.num_trades for b in buckets),
        total_candles=total_candles,
    )


def _fmt_profit_factor(value: float) -> str:
    return "inf" if value == float("inf") else f"{value:.2f}"


def format_regime_report(breakdown: RegimeBreakdown, title: str = "") -> str:
    """A per-regime table, with the exposure that gives each row its weight."""
    lines: List[str] = []
    lines.append("")
    lines.append(title or f"Performance by {breakdown.label_kind}")

    header = (
        f"{'Regime':<26}{'Candles':>9}{'Exposure':>10}{'Trades':>8}{'Win%':>8}"
        f"{'PF':>8}{'Expectancy':>13}{'Net P&L':>12}{'MaxDD%':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for bucket in breakdown.buckets:
        flag = "  *" if bucket.is_small_sample and bucket.num_trades else ""
        lines.append(
            f"{bucket.label:<26}"
            f"{bucket.num_candles:>9,}"
            f"{bucket.share_of_candles_pct:>9.1f}%"
            f"{bucket.num_trades:>8}"
            f"{bucket.win_rate_pct:>8.1f}"
            f"{_fmt_profit_factor(bucket.profit_factor):>8}"
            f"{bucket.expectancy_per_trade:>+13.2f}"
            f"{bucket.total_net_pnl:>+12.2f}"
            f"{bucket.max_drawdown_pct:>9.2f}"
            f"{flag}"
        )
    lines.append("-" * len(header))
    lines.append(f"Total trades bucketed: {breakdown.total_trades}   (* = fewer than 10 trades)")
    lines.append("")
    for note in breakdown.limitations():
        lines.append(f"  - {note}")
    return "\n".join(lines)
