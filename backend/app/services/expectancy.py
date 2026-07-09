"""Closed-trade expectancy diagnostics.

Pure, dependency-free helpers for turning a list of closed trades into win
rate / average win / average loss / expectancy numbers, and for detecting
*structurally impossible* patterns in them (a take-profit exit that loses
money on average, a sign error, a formula mismatch) — not for judging whether
a strategy is profitable. Markets vary; profitability is not asserted here.

Not used by any strategy or by the trade viability gate — this is an
observability tool for auditing exit behavior after the fact (see
tests/test_expectancy_diagnostics.py), not a runtime trading decision.

Sign convention (documented once here to avoid the classic bug of mixing
signed and unsigned "loss" numbers): ``avg_loss`` is SIGNED and <= 0 (e.g.
-0.062, not 0.062). Expectancy is therefore an addition, not a subtraction:

    expectancy = win_rate * avg_win + loss_rate * avg_loss
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class ClosedTrade:
    """Minimal shape needed for expectancy diagnostics.

    Args:
        pnl: Realized profit/loss for one closed round-trip, in whatever
            consistent unit the caller uses (USD, %, etc.) — the formulas
            are unit-agnostic as long as all trades in a report use the same
            unit.
        exit_reason: Label attached to the exit (e.g. "mean reached",
            "Hard stop"). Used only to group results and to flag
            take-profit-labeled exits that lose money on average.
    """
    pnl: float
    exit_reason: str = ""


@dataclass
class ReasonStats:
    exit_reason: str
    n: int
    wins: int
    losses: int
    win_rate: float
    avg_pnl: float
    total_pnl: float


@dataclass
class ExpectancyReport:
    n: int
    wins: int
    losses: int
    breakeven: int
    win_rate: float
    loss_rate: float
    avg_win: float  # >= 0 by convention
    avg_loss: float  # <= 0 by convention (signed)
    expectancy: float  # win_rate * avg_win + loss_rate * avg_loss
    total_pnl: float
    by_reason: Dict[str, ReasonStats] = field(default_factory=dict)


def compute_expectancy(trades: Iterable[ClosedTrade]) -> ExpectancyReport:
    """Compute win rate / avg win / avg loss / expectancy from closed trades.

    A trade with pnl == 0 counts toward ``n`` but is neither a win nor a
    loss (breakeven) — it does not distort avg_win/avg_loss.
    """
    trades = list(trades)
    n = len(trades)

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    breakeven = [t for t in trades if t.pnl == 0]

    win_rate = len(wins) / n if n else 0.0
    loss_rate = len(losses) / n if n else 0.0
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
    expectancy = win_rate * avg_win + loss_rate * avg_loss
    total_pnl = sum(t.pnl for t in trades)

    grouped: Dict[str, List[ClosedTrade]] = {}
    for t in trades:
        grouped.setdefault(t.exit_reason, []).append(t)

    by_reason: Dict[str, ReasonStats] = {}
    for reason, group in grouped.items():
        g_wins = [t for t in group if t.pnl > 0]
        g_losses = [t for t in group if t.pnl < 0]
        g_n = len(group)
        by_reason[reason] = ReasonStats(
            exit_reason=reason,
            n=g_n,
            wins=len(g_wins),
            losses=len(g_losses),
            win_rate=(len(g_wins) / g_n) if g_n else 0.0,
            avg_pnl=(sum(t.pnl for t in group) / g_n) if g_n else 0.0,
            total_pnl=sum(t.pnl for t in group),
        )

    return ExpectancyReport(
        n=n, wins=len(wins), losses=len(losses), breakeven=len(breakeven),
        win_rate=win_rate, loss_rate=loss_rate,
        avg_win=avg_win, avg_loss=avg_loss, expectancy=expectancy,
        total_pnl=total_pnl, by_reason=by_reason,
    )


# Exit-reason substrings that mean "this was supposed to be a take-profit /
# target exit". Matched case-insensitively. Deliberately small and literal —
# a label-matching heuristic for diagnostics, not a new strategy/indicator.
TAKE_PROFIT_LABELS = (
    "target", "take profit", "take-profit", "mean reached", "band reached",
)

_FLOAT_TOL = 1e-9


def find_expectancy_red_flags(
    report: ExpectancyReport,
    take_profit_labels: Iterable[str] = TAKE_PROFIT_LABELS,
) -> List[str]:
    """Detect structurally impossible/self-contradictory patterns.

    Deliberately does NOT assert the strategy is profitable — expectancy can
    be legitimately negative in real markets. It flags patterns that
    indicate a bug rather than bad luck:

      - a take-profit-labeled exit that loses money on average (Finding 1's
        exact bug signature: a target exit is supposed to realize a gain by
        definition of "target reached")
      - win_rate/loss_rate summing to more than the whole population
      - a "win" with a negative average, or a "loss" with a positive average
        (sign-convention corruption)
      - expectancy not matching its own defining formula
    """
    flags: List[str] = []

    if report.win_rate + report.loss_rate > 1.0 + _FLOAT_TOL:
        flags.append(
            f"win_rate ({report.win_rate:.3f}) + loss_rate ({report.loss_rate:.3f}) "
            "> 1.0 — impossible for a partition of the same trade population"
        )

    if report.avg_win < 0:
        flags.append(f"avg_win is negative (${report.avg_win:.4f}) — a 'win' cannot lose money")
    if report.avg_loss > 0:
        flags.append(f"avg_loss is positive (${report.avg_loss:.4f}) — a 'loss' cannot make money")

    recomputed = report.win_rate * report.avg_win + report.loss_rate * report.avg_loss
    if abs(recomputed - report.expectancy) > _FLOAT_TOL:
        flags.append(
            f"expectancy (${report.expectancy:.4f}) does not match its own formula "
            f"win_rate*avg_win + loss_rate*avg_loss (${recomputed:.4f})"
        )

    tp_labels = tuple(s.lower() for s in take_profit_labels)
    for reason, stats in sorted(report.by_reason.items()):
        label = reason.lower()
        if stats.n > 0 and stats.avg_pnl < 0 and any(tp in label for tp in tp_labels):
            flags.append(
                f"take-profit exit '{reason}' has a negative average result "
                f"(${stats.avg_pnl:.4f} over {stats.n} trades, "
                f"{stats.win_rate * 100:.0f}% win rate) — a target-reached exit "
                "is structurally losing money"
            )

    return flags
