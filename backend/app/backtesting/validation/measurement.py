"""Measurement of a single FIXED strategy configuration over a date range.

This is the measurement primitive the rest of the validation tooling is built
from: walk-forward measurement (rolling windows) and regime-conditioned
reporting both call :func:`measure_fixed_config` repeatedly with *the same*
:class:`FixedConfig`, varying only the span of candles it sees.

Two properties are load-bearing and deliberately structural rather than
conventional:

1. **The configuration is immutable.** :class:`FixedConfig` deep-copies the
   operator's parameter dict at construction and exposes it only through a
   read-only mapping, so no caller - present or future - can hand one
   :class:`FixedConfig` to N windows and quietly have window ``k`` measure
   something different from window ``0``. Every run gets a *fresh deep copy*
   (:meth:`FixedConfig.params_for_run`), so a strategy that mutated its params
   dict mid-replay could not leak that mutation into the next window either.

2. **The configuration is fingerprinted.** ``params_fingerprint`` is a stable
   hash of the canonical JSON form of the parameters, carried on every
   resulting :class:`Measurement`. A report over many windows can therefore
   *prove* - not assert in prose - that every window measured identical
   parameters, which is precisely the guarantee that distinguishes measurement
   from optimisation.

Nothing here decides anything. There is no "better", no ranking, no candidate
set, and no write path. See the package docstring for the binding scope.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence, Tuple

from app.backtesting.candle import Candle, ms_to_naive_utc
from app.backtesting.results import BacktestResult, TradeRecord

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cost
    from app.backtesting.engine import BacktestEngine


def _fingerprint(params: Mapping[str, Any]) -> str:
    """A stable, order-independent digest of a parameter mapping.

    ``sort_keys`` makes ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` produce
    the same fingerprint (they are the same configuration), while ``default=str``
    keeps the digest total: an exotic parameter *value* degrades to its repr
    rather than raising and taking down a measurement run.

    The ``dict()`` is not redundant. ``FixedConfig.params`` is a
    ``MappingProxyType``, which ``json`` does not recognise as an object - it
    would fall through to ``default=str`` and hash the mapping's *repr*, making
    the fingerprint depend on key insertion order and silently defeating the
    whole point of fingerprinting.
    """
    canonical = json.dumps(dict(params), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class MeasurementSpan:
    """A half-open-in-spirit date range, in unix milliseconds, inclusive of
    both endpoints (matching how the CSV data provider and the CLI's
    ``--start``/``--end`` already treat ranges).

    ``None`` on either side means "unbounded on that side" so a caller can ask
    for "everything up to X" without inventing a sentinel timestamp.
    """

    start_ms: Optional[int] = None
    end_ms: Optional[int] = None

    def __post_init__(self) -> None:
        if (
            self.start_ms is not None
            and self.end_ms is not None
            and self.end_ms < self.start_ms
        ):
            raise ValueError(
                f"MeasurementSpan end_ms ({self.end_ms}) is before start_ms ({self.start_ms})"
            )

    def contains(self, timestamp_ms: int) -> bool:
        if self.start_ms is not None and timestamp_ms < self.start_ms:
            return False
        if self.end_ms is not None and timestamp_ms > self.end_ms:
            return False
        return True

    def label(self) -> str:
        """Human-readable ``YYYY-MM-DD → YYYY-MM-DD`` for report rows."""
        start = ms_to_naive_utc(self.start_ms).date().isoformat() if self.start_ms is not None else "…"
        end = ms_to_naive_utc(self.end_ms).date().isoformat() if self.end_ms is not None else "…"
        return f"{start} → {end}"


@dataclass(frozen=True)
class FixedConfig:
    """A strategy configuration held fixed for the whole of a measurement run.

    The parameter dict handed in is deep-copied and wrapped read-only, so the
    operator's own dict can never be mutated by a measurement, and a
    ``FixedConfig`` can never be mutated by anything.
    """

    strategy: str
    trading_pair: str
    params: Mapping[str, Any] = field(default_factory=dict)
    starting_balance: float = 10_000.0

    def __post_init__(self) -> None:
        if not self.strategy:
            raise ValueError("FixedConfig.strategy must be a non-empty strategy name")
        if not self.trading_pair:
            raise ValueError("FixedConfig.trading_pair must be a non-empty symbol")
        if self.starting_balance <= 0:
            raise ValueError(
                f"FixedConfig.starting_balance must be positive (got {self.starting_balance})"
            )
        # Freeze the caller's params behind a deep copy. object.__setattr__ is
        # how a frozen dataclass normalises a field in __post_init__.
        object.__setattr__(
            self, "params", MappingProxyType(copy.deepcopy(dict(self.params)))
        )

    @property
    def params_fingerprint(self) -> str:
        return _fingerprint(self.params)

    def params_for_run(self) -> dict:
        """A fresh, mutable deep copy for one engine run.

        The engine hands this dict to production strategy code and stores it on
        an in-memory ``Bot`` row; handing out a copy per run means even a
        strategy that mutated it in place could not make window ``k+1`` measure
        different parameters than window ``k``.
        """
        return copy.deepcopy(dict(self.params))


@dataclass(frozen=True)
class Measurement:
    """The read-only result of measuring one fixed configuration over one span.

    A flattened, immutable projection of :class:`BacktestResult` plus the
    provenance needed to prove what was measured (which strategy, which
    parameters by fingerprint, over which candles). Trades and the equity curve
    are carried through because the later regime-conditioned and aggregate
    reporting is computed *from measurements*, never by re-running the engine.
    """

    strategy: str
    trading_pair: str
    params_fingerprint: str
    span: MeasurementSpan
    num_candles: int
    first_candle_ms: int
    last_candle_ms: int

    starting_balance: float
    ending_balance: float
    total_return_pct: float
    buy_and_hold_return_pct: float

    num_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy_per_trade: float
    max_drawdown_pct: float
    total_fees_paid: float

    trades: Tuple[TradeRecord, ...] = ()
    equity_curve: Tuple[Tuple[int, float], ...] = ()

    @property
    def is_empty(self) -> bool:
        """No round trip closed in this span. Reported as-is - a window with no
        trades is a real measurement outcome, not an error and not a zero."""
        return self.num_trades == 0


def select_candles(
    candles: Sequence[Candle], span: MeasurementSpan
) -> List[Candle]:
    """The candles of ``candles`` falling inside ``span``, order preserved.

    Slicing an already-loaded series (rather than re-reading CSVs per window)
    is what keeps a multi-window walk-forward run to a single load; the engine
    still only ever sees a plain ascending candle list, exactly as it does for
    a hand-run backtest.
    """
    return [c for c in candles if span.contains(c.timestamp)]


async def measure_fixed_config(
    engine: "BacktestEngine",
    candles: Sequence[Candle],
    config: FixedConfig,
    span: Optional[MeasurementSpan] = None,
    quiet: bool = True,
) -> Measurement:
    """Measure ``config`` - unchanged - over the candles of ``span``.

    Calls :meth:`BacktestEngine.run_candles` exactly as a manually run backtest
    would, so the no-lookahead guarantee proved for the engine holds here
    verbatim. The only thing this function adds is immutability of the
    configuration and provenance on the result.

    Raises ``ValueError`` if the span holds fewer than two candles - the engine
    needs one candle to decide on and one to fill against, and silently
    returning a zeroed "result" for an unmeasurable window would be a fabricated
    measurement.
    """
    span = span or MeasurementSpan()
    window = select_candles(candles, span)
    if len(window) < 2:
        raise ValueError(
            f"Span {span.label()} contains {len(window)} candle(s); at least 2 are "
            "required to measure (one to decide on, one to fill against)"
        )

    result: BacktestResult = await engine.run_candles(
        window,
        config.trading_pair,
        config.strategy,
        config.params_for_run(),
        config.starting_balance,
        quiet=quiet,
    )

    return Measurement(
        strategy=config.strategy,
        trading_pair=config.trading_pair,
        params_fingerprint=config.params_fingerprint,
        span=span,
        num_candles=len(window),
        first_candle_ms=window[0].timestamp,
        last_candle_ms=window[-1].timestamp,
        starting_balance=result.starting_balance,
        ending_balance=result.ending_balance,
        total_return_pct=result.total_return_pct,
        buy_and_hold_return_pct=result.buy_and_hold_return_pct,
        num_trades=result.num_trades,
        win_rate_pct=result.win_rate,
        profit_factor=result.profit_factor,
        expectancy_per_trade=result.expectancy_per_trade,
        max_drawdown_pct=result.max_drawdown_pct,
        total_fees_paid=result.total_fees_paid,
        trades=tuple(result.trades),
        equity_curve=tuple(tuple(point) for point in result.equity_curve),
    )
