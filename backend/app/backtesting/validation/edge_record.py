"""The validated measurement record - produced and reported, never wired in.

``add-strategy-decision-framework`` defined :class:`EdgeEstimate` with a
mechanically enforced ``source`` field precisely so that expected-edge numbers
could never be self-computed by the strategy claiming them. Until now nothing
in this repository could legitimately construct one, so every
``StrategyProposal.expected_edge_estimate`` has been ``None``. This module is
the tool that contract was waiting for.

What it does **not** do is equally deliberate: it does not populate a live
proposal. Producing the record is measurement; feeding it into runtime
decisions is a behaviour change, and belongs to a separate change that can be
reviewed on its own terms. A guard test asserts that no code under ``app/``
outside this package constructs an ``EdgeEstimate`` or passes a non-``None``
``expected_edge_estimate``.

The record is refused outright unless it can honestly be called out-of-sample.
Producing a "validated" number from one window would launder a single backtest
into a stamp of approval - the exact failure this whole change exists to
prevent - so the blockers below are hard refusals, not warnings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.backtesting.results import compute_result
from app.services.strategy_framework.proposal import VALIDATED_EDGE_SOURCE, EdgeEstimate

from .walk_forward import MIN_TRADES_FOR_CONFIDENCE, MIN_WINDOWS_FOR_CONFIDENCE

# A record must rest on more than one out-of-sample window that actually
# traded; otherwise the "validated" estimate is a single backtest wearing a
# different name.
MIN_TRADING_WINDOWS_FOR_A_RECORD = 2


@dataclass(frozen=True)
class ValidatedEdgeRecord:
    """An :class:`EdgeEstimate` plus the provenance needed to judge it.

    ``EdgeEstimate`` carries four numbers and a source string - deliberately
    minimal, because it is a runtime contract. A human reading a measurement
    needs to know which configuration produced it, over what range, across how
    many windows, and what the numbers cannot support; all of that lives here,
    alongside the estimate rather than inside it.
    """

    strategy: str
    trading_pair: str
    params_fingerprint: str
    span_label: str
    num_windows: int
    num_trading_windows: int
    consistency: str
    estimate: EdgeEstimate
    caveats: Tuple[str, ...] = ()

    @property
    def is_validated_source(self) -> bool:
        return self.estimate.source == VALIDATED_EDGE_SOURCE


def edge_record_blockers(result) -> Tuple[str, ...]:
    """Reasons this measurement cannot yield a validated record.

    Hard refusals, not caveats. Each one describes a situation where the
    resulting number would be misleading in a way no footnote could repair.
    """
    blockers: List[str] = []

    if result.num_windows < MIN_TRADING_WINDOWS_FOR_A_RECORD:
        blockers.append(
            f"Only {result.num_windows} window(s) were measured. A validated record "
            "must rest on multiple out-of-sample windows; one window is a backtest, "
            "not corroboration."
        )
    trading_windows = len(result.windows_with_trades)
    if trading_windows == 0:
        blockers.append(
            "No window produced any trades, so there is nothing to estimate from."
        )
    elif trading_windows < MIN_TRADING_WINDOWS_FOR_A_RECORD:
        blockers.append(
            f"Only {trading_windows} window(s) produced any trades. An estimate resting "
            "on a single window's trades is in-sample by another name, however many "
            "windows were planned."
        )
    if result.windows_overlap:
        blockers.append(
            "Windows overlap, so trades from shared candles would be pooled more than "
            "once. Re-run with a step equal to the window size to produce a record."
        )
    if result.total_trades < 1:
        blockers.append("No closed trades were measured; there is nothing to estimate.")
    if not result.every_window_used_one_configuration:
        blockers.append(
            "Not every window was measured with the same parameters, so their trades "
            "do not describe one configuration and cannot be pooled."
        )
    return tuple(blockers)


def _caveats(result, estimate: EdgeEstimate, pooled_losses: int) -> Tuple[str, ...]:
    notes: List[str] = []
    if estimate.sample_size < MIN_TRADES_FOR_CONFIDENCE:
        notes.append(
            f"Sample size is {estimate.sample_size} trade(s), below the {MIN_TRADES_FOR_CONFIDENCE} "
            "conventionally treated as a floor for reading a mean of per-trade outcomes. "
            "The estimate is real; the confidence in it is low."
        )
    if result.num_windows < MIN_WINDOWS_FOR_CONFIDENCE:
        notes.append(
            f"Pooled from {result.num_windows} out-of-sample window(s), fewer than the "
            f"{MIN_WINDOWS_FOR_CONFIDENCE} at which per-window agreement starts to mean much."
        )
    if result.consistency == "mixed":
        notes.append(
            "Per-window expectancy changes sign across windows. The pooled expectancy "
            "below is an average over states the strategy behaves differently in, and "
            "describes none of them."
        )
    if pooled_losses == 0:
        notes.append(
            "No losing trades were recorded, so profit factor is infinite. That is a "
            "property of this sample, not an unbounded edge."
        )
    notes.append(
        "Produced for review only. This record is NOT wired into any live "
        "StrategyProposal; expected_edge_estimate remains None at runtime."
    )
    return tuple(notes)


def build_validated_edge_record(result) -> Optional[ValidatedEdgeRecord]:
    """Pool the out-of-sample windows into one validated measurement record.

    Returns ``None`` when :func:`edge_record_blockers` finds any reason the
    number would mislead. Pooling is over the windows' trades, which are
    disjoint because a record is refused for overlapping windows.
    """
    if edge_record_blockers(result):
        return None

    trades = [trade for window in result.windows for trade in window.trades]
    # Routed through the engine's own metric function so expectancy, win rate
    # and profit factor here are the same quantities, computed the same way, as
    # every other number this tooling reports.
    pooled = compute_result(
        starting_balance=1.0, ending_balance=1.0, trades=trades,
        equity_curve=[], total_fees_paid=0.0, buy_and_hold_return_pct=0.0,
    )

    estimate = EdgeEstimate(
        expectancy=pooled.expectancy_per_trade,
        # BacktestResult reports win rate as a percentage; EdgeEstimate
        # validates it as a fraction in [0, 1] and rejects anything else.
        win_rate=pooled.win_rate / 100.0,
        profit_factor=pooled.profit_factor,
        sample_size=pooled.num_trades,
        source=VALIDATED_EDGE_SOURCE,
    )

    return ValidatedEdgeRecord(
        strategy=result.strategy,
        trading_pair=result.trading_pair,
        params_fingerprint=result.params_fingerprint,
        span_label=result.span.label(),
        num_windows=result.num_windows,
        num_trading_windows=len(result.windows_with_trades),
        consistency=result.consistency,
        estimate=estimate,
        caveats=_caveats(result, estimate, len([t for t in trades if t.net_pnl < 0])),
    )


def format_edge_record_report(result) -> str:
    """The validated record, or a plain statement of why there isn't one."""
    lines: List[str] = ["", "Validated measurement record"]
    lines.append("-" * 60)

    blockers = edge_record_blockers(result)
    if blockers:
        lines.append("NOT PRODUCED. This measurement cannot support a validated record:")
        for blocker in blockers:
            lines.append(f"  - {blocker}")
        lines.append("")
        return "\n".join(lines)

    record = build_validated_edge_record(result)
    assert record is not None  # no blockers, so a record is always produced
    estimate = record.estimate

    lines.append(f"{'Strategy':<24}{record.strategy}")
    lines.append(f"{'Symbol':<24}{record.trading_pair}")
    lines.append(f"{'Parameters':<24}fixed, fingerprint {record.params_fingerprint}")
    lines.append(f"{'Range':<24}{record.span_label}")
    lines.append(
        f"{'Out-of-sample windows':<24}{record.num_windows} "
        f"({record.num_trading_windows} produced trades)"
    )
    lines.append(f"{'Per-window consistency':<24}{record.consistency}")
    lines.append("")
    lines.append(f"{'Expectancy per trade':<24}{estimate.expectancy:+.2f}")
    lines.append(f"{'Win rate':<24}{estimate.win_rate * 100:.1f}%")
    profit_factor = (
        "inf" if estimate.profit_factor == float("inf") else f"{estimate.profit_factor:.2f}"
    )
    lines.append(f"{'Profit factor':<24}{profit_factor}")
    lines.append(f"{'Sample size':<24}{estimate.sample_size} closed trades")
    lines.append(f"{'Source':<24}{estimate.source}")
    lines.append("")
    for caveat in record.caveats:
        lines.append(f"  - {caveat}")
    lines.append("")
    return "\n".join(lines)
