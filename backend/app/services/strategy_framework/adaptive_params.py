"""Pillar 4 — Parameter Adaptation: shared, deterministic parameter resolvers.

Every strategy's audit document must classify every parameter as ADAPTIVE
(cite the formula) or FIXED (document why fixed is correct) — no silent,
undocumented hardcoding is allowed post-certification (design.md, pillar
4). This module gives strategies shared, reusable resolvers to compute
adaptive parameters instead of each strategy inventing its own ad hoc
scaling formula, and is also the landing point for Strategy Edge
Management's Category B ("parameter mismatch") remediation — an adaptive
parameter pillar 7 determines is miscalibrated for current conditions is
adapted here, via ``adapt_for_edge_category_b``, mathematically justified
by the measured signal, never a blanket re-tune.

This module is infrastructure only — no strategy calls it yet (Phase 0).
It does not fix specific numeric bases/bounds for any strategy; those are
strategy-specific, certification-phase decisions (design.md Non-Goals).

Every resolver is a pure function of its arguments — same inputs, same
output, every time — and every result carries a human-readable ``formula``
string, so a resolved parameter is self-explaining (pillar 8) rather than
"trust me, it's adaptive."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class AdaptedParameter:
    """One resolved adaptive parameter, with its formula for pillar 8."""

    name: str
    value: float
    formula: str
    inputs: Dict[str, float]


def _clamp(value: float, lo: Optional[float], hi: Optional[float]) -> float:
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


class AdaptiveParameterResolver:
    """Shared, deterministic parameter resolvers strategies compose into."""

    @staticmethod
    def atr_percentile_scaled_multiplier(
        *,
        name: str,
        atr_percentile: float,
        base_multiplier: float,
        min_multiplier: float,
        max_multiplier: float,
    ) -> AdaptedParameter:
        """Scale a multiplier (e.g. a stop-distance ATR multiplier) by where
        current volatility sits relative to its own recent history.

        ``atr_percentile`` is the strategy's own ratio of current ATR to a
        recent average/percentile (the same convention
        ``_detect_market_regime_bar_based`` already computes internally) —
        this resolver does not compute ATR itself, it only applies the
        scaling once a strategy has that ratio.

        formula: ``clamp(base_multiplier * atr_percentile, min, max)``
        """
        if not isinstance(atr_percentile, (int, float)) or atr_percentile < 0:
            raise ValueError(
                f"atr_percentile must be a non-negative number, got {atr_percentile!r}"
            )
        if min_multiplier > max_multiplier:
            raise ValueError("min_multiplier must be <= max_multiplier")

        raw = base_multiplier * atr_percentile
        value = _clamp(raw, min_multiplier, max_multiplier)
        formula = (
            f"clamp({base_multiplier} * atr_percentile, "
            f"{min_multiplier}, {max_multiplier})"
        )
        return AdaptedParameter(
            name=name,
            value=value,
            formula=formula,
            inputs={
                "atr_percentile": float(atr_percentile),
                "base_multiplier": float(base_multiplier),
                "min_multiplier": float(min_multiplier),
                "max_multiplier": float(max_multiplier),
            },
        )

    @staticmethod
    def regime_scaled_lookback(
        *,
        name: str,
        regime: str,
        base_lookback: int,
        regime_multipliers: Dict[str, float],
        min_lookback: int,
        max_lookback: int,
    ) -> AdaptedParameter:
        """Scale a lookback window by the current market regime.

        ``regime_multipliers`` maps a regime tag (e.g. "trend_up") to a
        multiplier; a regime with no declared multiplier defaults to 1.0
        (no adaptation for that regime, not an error — a strategy may
        intentionally only adapt for regimes it cares about).

        formula: ``clamp(round(base_lookback * regime_multipliers[regime]), min, max)``
        """
        if min_lookback > max_lookback:
            raise ValueError("min_lookback must be <= max_lookback")

        multiplier = regime_multipliers.get(regime, 1.0)
        raw = round(base_lookback * multiplier)
        value = _clamp(raw, min_lookback, max_lookback)
        formula = (
            f"clamp(round({base_lookback} * regime_multipliers[{regime!r}]="
            f"{multiplier}), {min_lookback}, {max_lookback})"
        )
        return AdaptedParameter(
            name=name,
            value=float(value),
            formula=formula,
            inputs={
                "base_lookback": float(base_lookback),
                "regime_multiplier": float(multiplier),
                "min_lookback": float(min_lookback),
                "max_lookback": float(max_lookback),
            },
        )

    @staticmethod
    def adapt_for_edge_category_b(
        *,
        name: str,
        current_value: float,
        degradation_signal: str,
        adjustment_factor: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> AdaptedParameter:
        """Apply a Strategy Edge Management Category B ("parameter
        mismatch") adaptation: a mathematically justified adjustment to ONE
        specific parameter, never a blanket re-tune of unrelated parameters
        (design.md, pillar 7, Category B action).

        ``degradation_signal`` is a required, human-readable citation of the
        measured evidence that justified this specific adjustment (e.g.
        "ATR percentile 1.8x the value this multiplier was last calibrated
        against") — recorded on the result so the adaptation is auditable,
        not a guess.

        formula: ``clamp(current_value * adjustment_factor, min, max)``
        """
        if not isinstance(degradation_signal, str) or not degradation_signal.strip():
            raise ValueError(
                "degradation_signal is mandatory for a Category B adaptation — "
                "an adjustment with no cited evidence is a guess, not a "
                "mathematically justified adaptation"
            )
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError("min_value must be <= max_value")

        raw = current_value * adjustment_factor
        value = _clamp(raw, min_value, max_value)
        formula = (
            f"clamp({current_value} * adjustment_factor({adjustment_factor}), "
            f"{min_value}, {max_value}) [Category B: {degradation_signal}]"
        )
        return AdaptedParameter(
            name=name,
            value=value,
            formula=formula,
            inputs={
                "current_value": float(current_value),
                "adjustment_factor": float(adjustment_factor),
            },
        )
