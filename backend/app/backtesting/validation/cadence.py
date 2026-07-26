"""Is this measurement's candle interval fine enough for this strategy?

A strategy's time-based parameters are written for the cadence it runs at live -
roughly once a minute. Measure it on candles coarser than those constants and
its mechanisms stop working, not because the strategy is wrong but because it
never gets a second look at anything.

This is not hypothetical. ``dip_recovery`` reports ``setup_expiry_minutes: 240``:
a decline arms a setup, and the setup is abandoned if no reversal is confirmed
within 240 minutes. Measured on 4h candles - which are *exactly* 240 minutes -
every setup expired on the very next evaluation, and the strategy recorded zero
trades across six years. Measured on 1h candles the same code with the same
parameters opened 127 positions over six months. The 4h "zero" was an artefact
of the measurement, and it was initially read as a defect in the strategy.

Reporting that zero without this warning is the tooling failing at its one job:
a window with no trades is supposed to be reported as an absence of evidence,
not as evidence of absence.

Nothing here changes a parameter. It reads the declared parameter schema, infers
the measurement's candle interval from the candles themselves, and reports the
mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.backtesting.candle import Candle

MS_PER_MINUTE = 60_000

# Parameter-name suffixes that denote a duration, with their multiplier to ms.
# Matching on the declared naming convention rather than a hand-maintained list
# of parameter names means a duration added to any strategy later is covered the
# day it is added.
_DURATION_SUFFIXES: Tuple[Tuple[str, int], ...] = (
    ("_seconds", 1_000),
    ("_minutes", 60_000),
    ("_hours", 3_600_000),
)


@dataclass(frozen=True)
class CadenceWarning:
    """One strategy time constant that the measurement is too coarse to honour."""

    parameter: str
    value_ms: int
    candle_interval_ms: int
    description: str

    @property
    def evaluations_per_period(self) -> float:
        """How many times the strategy gets to look before this period elapses.

        At or below 1.0 the mechanism resolves inside a single bar and cannot
        behave as it would live.
        """
        if self.candle_interval_ms <= 0:
            return 0.0
        return self.value_ms / self.candle_interval_ms

    @property
    def is_fatal(self) -> bool:
        return self.evaluations_per_period <= 1.0

    def describe(self) -> str:
        value_min = self.value_ms / MS_PER_MINUTE
        interval_min = self.candle_interval_ms / MS_PER_MINUTE
        verdict = (
            "resolves within a SINGLE candle, so it cannot function at all"
            if self.is_fatal
            else f"gets only ~{self.evaluations_per_period:.1f} evaluations"
        )
        return (
            f"{self.parameter} = {value_min:g} min vs a {interval_min:g} min candle "
            f"interval: {verdict}. ({self.description})"
        )


def infer_candle_interval_ms(candles: Sequence[Candle]) -> Optional[int]:
    """The measurement's candle interval, taken from the candles themselves.

    The median gap rather than the first: a series with a data gap would
    otherwise report the gap as its interval. Returns ``None`` for a series too
    short to have an interval at all.
    """
    if len(candles) < 2:
        return None
    gaps = [
        later.timestamp - earlier.timestamp
        for earlier, later in zip(candles, candles[1:])
        if later.timestamp > earlier.timestamp
    ]
    if not gaps:
        return None
    return int(median(gaps))


def declared_parameters(strategy: str) -> Dict[str, Dict[str, Any]]:
    """The strategy's declared parameter schema.

    Imported lazily from the config router - the project's machine-readable
    source for parameter names, defaults, and descriptions - so this module
    stays importable without the API layer, and so a parameter added there is
    picked up here with no change.
    """
    from app.routers.config import STRATEGIES

    for info in STRATEGIES:
        if info.name == strategy:
            return dict(info.parameters)
    return {}


def duration_parameters(strategy: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Tuple[int, str]]:
    """Effective duration parameters as ``{name: (milliseconds, description)}``.

    Operator-supplied values override declared defaults, so the check applies to
    what was actually measured rather than to what the schema says.
    """
    overrides = params or {}
    out: Dict[str, Tuple[int, str]] = {}
    for name, spec in declared_parameters(strategy).items():
        multiplier = next(
            (m for suffix, m in _DURATION_SUFFIXES if name.endswith(suffix)), None
        )
        if multiplier is None:
            continue
        value = overrides.get(name, spec.get("default"))
        if not isinstance(value, (int, float)) or value <= 0:
            continue
        out[name] = (int(value * multiplier), str(spec.get("description", "")))
    return out


def check_cadence(
    strategy: str,
    candles: Sequence[Candle],
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[CadenceWarning, ...]:
    """Duration parameters this measurement's candle interval is too coarse for.

    Reported worst-first (fewest evaluations), because the first line a reader
    sees should be the one most likely to explain a surprising result.
    """
    interval_ms = infer_candle_interval_ms(candles)
    if not interval_ms:
        return ()

    warnings = [
        CadenceWarning(
            parameter=name, value_ms=value_ms,
            candle_interval_ms=interval_ms, description=description,
        )
        for name, (value_ms, description) in duration_parameters(strategy, params).items()
        # Fewer than ~4 evaluations means the mechanism is barely exercised;
        # below 1 it cannot run at all. Above that it is being measured fairly.
        if value_ms / interval_ms < 4.0
    ]
    return tuple(sorted(warnings, key=lambda w: w.evaluations_per_period))


def format_cadence_warning(
    strategy: str, warnings: Sequence[CadenceWarning], num_trades: int
) -> str:
    """A prominent block, or nothing at all when the measurement is sound."""
    if not warnings:
        return ""

    fatal = [w for w in warnings if w.is_fatal]
    lines: List[str] = ["", "!" * 79]
    lines.append(f"CADENCE WARNING: this timeframe is too coarse for {strategy}")
    lines.append("!" * 79)
    for warning in warnings:
        lines.append(f"  - {warning.describe()}")
    lines.append("")
    if fatal and num_trades == 0:
        lines.append(
            "This strategy recorded NO trades AND has a mechanism that cannot function "
            "at this candle interval. Do NOT read the zero as evidence about the "
            "strategy: re-measure on a finer timeframe before drawing any conclusion."
        )
    elif fatal:
        lines.append(
            "A mechanism above cannot function at this candle interval, so these "
            "results do not describe how the strategy would behave live."
        )
    else:
        lines.append(
            "The mechanisms above are barely exercised at this candle interval, so "
            "these results understate how they would behave live."
        )
    lines.append(
        "Strategy time constants are written for the ~60s cadence the engine runs at "
        "live; a backtest evaluates once per candle."
    )
    lines.append("")
    return "\n".join(lines)
