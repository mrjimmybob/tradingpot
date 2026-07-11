"""Pillar 3 — Evidence-Based Decision Score (formerly "Trade Quality Score" /
"Confidence").

A trading system must never "feel confident." Every candidate trade's
score SHALL be entirely derived from measurable observations: a weighted
accumulator over a strategy-declared list of Evidence Items, each with a
documented, deterministic Measurement, a deterministic Normalization to a
bounded contribution range, a Weight, and a Reason cross-referencing the
strategy's Theory document (pillar 1). See ``design.md``'s "3. Evidence-
Based Decision Score" for the full specification this module implements,
including the required Evidence Report format.

This module is infrastructure only — no strategy calls it yet (Phase 0).
It provides no default Evidence Items, weights, or thresholds: those are
strategy-specific, certification-phase decisions (see design.md's
Non-Goals — calibrating actual numeric values is explicitly out of scope
for this framework and left to each strategy's certification, informed by
``add-strategy-validation-tooling``).

Determinism enforcement: a factor that "cannot be written as a Measurement
+ Normalization pair" is rejected in two ways —
  1. At ``EvidenceItem`` construction: ``measurement``/``normalization``
     must be callables, ``weight`` must be a positive number, and
     ``reason`` must be non-empty (a factor with no documented reason it
     contributes to edge is, by definition, not traceable to the Theory
     document pillar 1 requires).
  2. At scoring time: ``DecisionScoreEngine.score()`` calls every item's
     measurement and normalization TWICE against the same input data and
     raises ``NonDeterministicEvidenceItemError`` if the two calls
     disagree — an empirical, mechanical determinism check every score
     computation pays for automatically, rather than trusting a strategy
     author's intent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

Measurement = Callable[[Dict[str, Any]], float]
Normalization = Callable[[float], float]


class InvalidEvidenceItemError(ValueError):
    """Raised when an Evidence Item cannot be expressed as a deterministic
    Measurement + Normalization pair with a documented reason — the
    framework's mechanical enforcement of "the framework explicitly
    REJECTS subjective or unmeasurable factors" (design.md, pillar 3)."""


class NonDeterministicEvidenceItemError(RuntimeError):
    """Raised when an Evidence Item's Measurement or Normalization produces
    a different result for the same input across two calls in the same
    scoring pass — the item is not reproducible and therefore not
    admissible, regardless of what its author intended."""


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class EvidenceItem:
    """One measurable factor contributing to a strategy's Decision Score.

    Attributes:
        name: Human-readable factor name (e.g. "Trend strength").
        measurement: Deterministic computation producing a raw value from
            the data dict a strategy supplies at scoring time (e.g. "EMA(20)
            slope over the last 10 bars, in %").
        normalization: Deterministic mapping from the raw measurement to a
            bounded contribution range (e.g. ``clamp(slope_pct / 2.0, -1, 1)``).
            The engine additionally clamps the result to [-1, 1] as a
            defensive backstop — a Normalization is expected to already do
            this itself, per its own definition.
        weight: This item's maximum point contribution to the total score.
            Must be > 0. Weights SHOULD sum to <= 100 across a strategy's
            declared Evidence Items so the total naturally lands in the
            documented 0-100 range; the engine does not forcibly rescale
            (calibrating actual weights is a certification-phase decision,
            not fixed by this shared module).
        reason: Cross-reference to the strategy's Theory document (pillar 1)
            explaining WHY this measurable factor is evidence for or
            against the strategy's specific edge. Mandatory and non-empty —
            an Evidence Item with no documented reason is not traceable to
            a theory and is rejected at construction.
    """

    name: str
    measurement: Measurement
    normalization: Normalization
    weight: float
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidEvidenceItemError("EvidenceItem.name must be a non-empty string")
        if not callable(self.measurement):
            raise InvalidEvidenceItemError(
                f"EvidenceItem '{self.name}': measurement must be a deterministic "
                "callable, not a fixed/subjective value"
            )
        if not callable(self.normalization):
            raise InvalidEvidenceItemError(
                f"EvidenceItem '{self.name}': normalization must be a deterministic "
                "callable mapping the raw measurement to a bounded range"
            )
        if not isinstance(self.weight, (int, float)) or self.weight <= 0:
            raise InvalidEvidenceItemError(
                f"EvidenceItem '{self.name}': weight must be a positive number, "
                f"got {self.weight!r}"
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise InvalidEvidenceItemError(
                f"EvidenceItem '{self.name}': reason it contributes to edge is "
                "mandatory (must cross-reference the strategy's Theory document)"
            )


@dataclass(frozen=True)
class EvidenceContribution:
    """One Evidence Item's resolved contribution for a single scoring pass."""

    name: str
    raw_value: float
    normalized_value: float
    weight: float
    contribution: float
    reason: str
    passed: bool  # contribution >= 0 — used only for Evidence Report rendering


@dataclass(frozen=True)
class EvidenceReport:
    """The complete, human-readable rendering of one scoring pass.

    ``render()`` produces the exact structure design.md's pillar 3
    specifies: a checklist of signed per-item contributions, the total
    Decision Score, the minimum required threshold, and the approve/reject
    outcome.
    """

    strategy: str
    contributions: List[EvidenceContribution]
    total: float
    threshold: float
    approved: bool

    def render(self) -> str:
        lines = ["Evidence"]
        for c in self.contributions:
            mark = "✓" if c.passed else "✗"
            lines.append(f"{mark} {c.name:<32} {c.contribution:+.0f}")
        lines.append("")
        lines.append("Total Decision Score:")
        lines.append(f"{self.total:.0f} / 100")
        lines.append("")
        lines.append("Minimum required:")
        lines.append(f"{self.threshold:.0f}")
        lines.append("")
        lines.append("Trade approved." if self.approved else "Trade rejected.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "total": self.total,
            "threshold": self.threshold,
            "approved": self.approved,
            "contributions": [
                {
                    "name": c.name,
                    "raw_value": c.raw_value,
                    "normalized_value": c.normalized_value,
                    "weight": c.weight,
                    "contribution": c.contribution,
                    "reason": c.reason,
                }
                for c in self.contributions
            ],
        }


@dataclass(frozen=True)
class DecisionScoreResult:
    """The full result of one Decision Score evaluation."""

    total: float
    threshold: float
    approved: bool
    contributions: List[EvidenceContribution]
    evidence_report: EvidenceReport


class DecisionScoreEngine:
    """Weighted accumulator over a strategy-declared list of Evidence Items.

    Stateless: ``score()`` is a pure function of ``items``, ``data``, and
    ``threshold`` — the same inputs always produce the same
    ``DecisionScoreResult``, including the same ``EvidenceReport``.
    """

    def score(
        self,
        strategy: str,
        items: List[EvidenceItem],
        data: Dict[str, Any],
        threshold: float,
    ) -> DecisionScoreResult:
        if not items:
            raise InvalidEvidenceItemError(
                f"{strategy}: DecisionScoreEngine.score() called with zero "
                "Evidence Items — a Decision Score with no evidence is not "
                "a score, it is an unmeasured guess"
            )

        contributions: List[EvidenceContribution] = []
        for item in items:
            raw_first = item.measurement(data)
            raw_second = item.measurement(data)
            if raw_first != raw_second:
                raise NonDeterministicEvidenceItemError(
                    f"{strategy}: Evidence Item '{item.name}' measurement is "
                    f"non-deterministic ({raw_first!r} != {raw_second!r} for the "
                    "same input data) — rejected"
                )
            if not isinstance(raw_first, (int, float)):
                raise InvalidEvidenceItemError(
                    f"{strategy}: Evidence Item '{item.name}' measurement did not "
                    f"produce a number (got {type(raw_first).__name__})"
                )

            norm_first = item.normalization(raw_first)
            norm_second = item.normalization(raw_first)
            if norm_first != norm_second:
                raise NonDeterministicEvidenceItemError(
                    f"{strategy}: Evidence Item '{item.name}' normalization is "
                    f"non-deterministic ({norm_first!r} != {norm_second!r} for the "
                    "same raw value) — rejected"
                )
            if not isinstance(norm_first, (int, float)):
                raise InvalidEvidenceItemError(
                    f"{strategy}: Evidence Item '{item.name}' normalization did not "
                    f"produce a number (got {type(norm_first).__name__})"
                )

            normalized = _clamp(float(norm_first))
            contribution_value = normalized * item.weight
            contributions.append(
                EvidenceContribution(
                    name=item.name,
                    raw_value=float(raw_first),
                    normalized_value=normalized,
                    weight=item.weight,
                    contribution=contribution_value,
                    reason=item.reason,
                    passed=contribution_value >= 0,
                )
            )

        total = sum(c.contribution for c in contributions)
        approved = total >= threshold

        report = EvidenceReport(
            strategy=strategy,
            contributions=contributions,
            total=total,
            threshold=threshold,
            approved=approved,
        )

        return DecisionScoreResult(
            total=total,
            threshold=threshold,
            approved=approved,
            contributions=contributions,
            evidence_report=report,
        )
