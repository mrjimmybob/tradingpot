"""Benchmark equity curves - what the strategy has to beat to justify itself.

Two references, both computed as pure deterministic functions over the same
candles a measurement ran on, and both paying the same execution costs the
strategy paid:

- **buy-and-hold** answers "could I have just held?" - the question any strategy
  must beat before its complexity is worth anything.
- **periodic DCA** answers it *fairly for an accumulator*. A strategy that
  deploys capital gradually is under-exposed by construction and will trail
  buy-and-hold in any rising market from exposure alone, regardless of how well
  it chose its entries. Comparing it against a benchmark that deploys on the
  same gradual schedule isolates timing and sizing from mere exposure. It is
  also already this project's stated long-term accumulation benchmark, so
  measuring against it makes an existing intent concrete.

Neither benchmark runs a strategy or the replay loop. A benchmark whose value
depended on the engine's replay would be exactly as hard to trust as the thing
it is supposed to benchmark.

The cost model is reused, never re-derived: the effective fee rate is probed
*from* ``BacktestExecutionModel`` rather than recomputed here, so a benchmark can
never drift from the fees the measured strategy actually paid. A cost-free
benchmark is not a benchmark, it is a handicap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from app.backtesting.candle import Candle
from app.backtesting.execution_model import BacktestExecutionModel
from app.backtesting.results import compute_result

MS_PER_DAY = 86_400_000

# Weekly. Frequent enough to track a gradual accumulator's deployment profile
# over a multi-month window, coarse enough that fees do not dominate the
# comparison. A documented, reported run-time parameter - never searched, and
# never varied to flatter a strategy (that would be optimisation).
DEFAULT_DCA_CADENCE_MS = 7 * MS_PER_DAY

BUY_AND_HOLD = "buy-and-hold"
PERIODIC_DCA = "periodic DCA"


@dataclass(frozen=True)
class BenchmarkCurve:
    """A benchmark's equity over time, plus the parameters that define it.

    ``parameters`` is carried on the curve rather than left to the caller
    because a benchmark is only interpretable alongside its own settings - a
    periodic-DCA result means nothing without its cadence.
    """

    name: str
    parameters: str
    equity_curve: Tuple[Tuple[int, float], ...]
    starting_balance: float

    @property
    def terminal_equity(self) -> float:
        return self.equity_curve[-1][1] if self.equity_curve else self.starting_balance

    @property
    def terminal_return_pct(self) -> float:
        if self.starting_balance <= 0:
            return 0.0
        return (self.terminal_equity - self.starting_balance) / self.starting_balance * 100.0

    @property
    def max_drawdown_pct(self) -> float:
        """Computed by the engine's own metric function, so a benchmark's
        drawdown is the same quantity, computed the same way, as a strategy's."""
        if not self.equity_curve:
            return 0.0
        return compute_result(
            starting_balance=self.starting_balance,
            ending_balance=self.terminal_equity,
            trades=[],
            equity_curve=list(self.equity_curve),
            total_fees_paid=0.0,
            buy_and_hold_return_pct=0.0,
        ).max_drawdown_pct

    @property
    def label(self) -> str:
        return f"{self.name} ({self.parameters})" if self.parameters else self.name


def _effective_fee_rate(model: BacktestExecutionModel, price: float) -> float:
    """The model's fee as a fraction of notional, probed from the model itself.

    Deriving it rather than re-implementing ``fee_pct / 100`` means a benchmark
    keeps paying whatever the execution model charges, including any future
    change to how the production cost model computes it.
    """
    probe = model.fill("buy", 1.0, price)
    if probe.notional_usd <= 0:
        return 0.0
    return probe.fee_usd / probe.notional_usd


def _buy_with_cash(
    model: BacktestExecutionModel, cash: float, price: float
) -> Tuple[float, float]:
    """Spend ``cash`` (inclusive of fees) at ``price``; return (base, spent).

    Solving ``notional * (1 + fee_rate) = cash`` is what makes the instalment
    sizes exact: sizing on notional alone would overspend by the fee on every
    purchase and quietly leave the benchmark short of capital.
    """
    if cash <= 0 or price <= 0:
        return 0.0, 0.0
    fee_rate = _effective_fee_rate(model, price)
    notional = cash / (1.0 + fee_rate)
    fill = model.fill("buy", notional / price, price)
    return notional / price, fill.notional_usd + fill.fee_usd


def buy_and_hold_curve(
    candles: Sequence[Candle],
    starting_balance: float = 10_000.0,
    model: Optional[BacktestExecutionModel] = None,
) -> BenchmarkCurve:
    """Deploy everything at the first candle's open and hold to the last.

    Marked at every candle's close, so the curve aligns point-for-point with a
    measurement's own equity curve and the two are directly comparable.
    """
    model = model or BacktestExecutionModel()
    if not candles:
        return BenchmarkCurve(BUY_AND_HOLD, "", (), starting_balance)

    base, spent = _buy_with_cash(model, starting_balance, candles[0].open)
    cash = starting_balance - spent
    curve = [(candle.timestamp, cash + base * candle.close) for candle in candles]
    return BenchmarkCurve(BUY_AND_HOLD, "", tuple(curve), starting_balance)


def periodic_dca_curve(
    candles: Sequence[Candle],
    starting_balance: float = 10_000.0,
    model: Optional[BacktestExecutionModel] = None,
    cadence_ms: int = DEFAULT_DCA_CADENCE_MS,
) -> BenchmarkCurve:
    """Deploy in equal instalments at ``cadence_ms``, never sell.

    The instalment count is fixed up front from the span so the instalments sum
    to exactly ``starting_balance``: sizing each instalment from the cash
    remaining would front-load the deployment and quietly turn this into a
    different, faster-deploying benchmark.

    A cadence longer than the span still deploys once, at the first candle -
    the degenerate case is a lump sum, not an empty benchmark.
    """
    model = model or BacktestExecutionModel()
    if cadence_ms <= 0:
        raise ValueError(f"cadence_ms must be positive (got {cadence_ms})")
    if not candles:
        return BenchmarkCurve(PERIODIC_DCA, _cadence_label(cadence_ms), (), starting_balance)

    buy_indices = _instalment_indices(candles, cadence_ms)
    instalment = starting_balance / len(buy_indices)

    cash = starting_balance
    base = 0.0
    pending = set(buy_indices)
    curve: List[Tuple[int, float]] = []
    for index, candle in enumerate(candles):
        if index in pending:
            bought, spent = _buy_with_cash(model, min(instalment, cash), candle.open)
            base += bought
            cash -= spent
        curve.append((candle.timestamp, cash + base * candle.close))

    return BenchmarkCurve(
        PERIODIC_DCA, _cadence_label(cadence_ms), tuple(curve), starting_balance,
    )


def _instalment_indices(candles: Sequence[Candle], cadence_ms: int) -> List[int]:
    """Candle indices at which an instalment is deployed.

    The first candle always buys; thereafter the first candle at or past each
    cadence boundary does. Boundaries are measured from the start rather than
    from the previous purchase, so a data gap delays one instalment instead of
    shifting every later one.
    """
    indices = [0]
    next_boundary = candles[0].timestamp + cadence_ms
    for index, candle in enumerate(candles[1:], start=1):
        if candle.timestamp >= next_boundary:
            indices.append(index)
            # Skip boundaries the gap jumped over, so a long gap costs one
            # instalment rather than firing a burst of them on the next candle.
            while next_boundary <= candle.timestamp:
                next_boundary += cadence_ms
    return indices


def _cadence_label(cadence_ms: int) -> str:
    if cadence_ms % MS_PER_DAY == 0:
        return f"cadence {cadence_ms // MS_PER_DAY}d"
    return f"cadence {cadence_ms}ms"


def build_benchmarks(
    candles: Sequence[Candle],
    starting_balance: float = 10_000.0,
    model: Optional[BacktestExecutionModel] = None,
    cadence_ms: int = DEFAULT_DCA_CADENCE_MS,
) -> Tuple[BenchmarkCurve, ...]:
    """The standard benchmark set, in a fixed order."""
    return (
        buy_and_hold_curve(candles, starting_balance, model),
        periodic_dca_curve(candles, starting_balance, model, cadence_ms),
    )
