"""Pillar 6 — Continuous Trade Management: a shared post-entry re-evaluation
hook pattern.

After entry, every strategy SHALL continuously re-evaluate its open
position every evaluation cycle — whether the original thesis still holds,
whether volatility has changed materially, and whether the stop, partial
profit, or full exit should be adjusted — rather than only checking a
fixed stop and target set at entry (design.md, pillar 6). Three of six
strategies already do a version of this ad hoc; this module formalizes the
pattern as shared infrastructure so it is consistent across all six,
rather than each strategy inventing (or omitting) its own re-evaluation
logic.

This module does not know any strategy's indicators. Thesis re-evaluation
((a) below) is delegated to the shared ``MarketSuitabilityGate`` (pillar
2) so "is the market still suitable" always means the same thing whether
checked at entry or mid-trade. The other three checks — volatility change,
stop tightening, partial profit — are strategy-specific by nature (what
counts as "volatility changed materially" or "the stop should tighten" is
different for a grid strategy than a trend-following one), so a strategy
supplies its own pure predicate functions; the monitor only orchestrates
and aggregates them deterministically.

This module is infrastructure only — no strategy calls it yet (Phase 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .market_suitability import MarketSuitabilityGate, MarketSuitabilityResult

# A hook returns (fired: bool, reason: str). ``reason`` SHOULD be empty when
# fired is False, but the monitor does not require it.
HookResult = Tuple[bool, str]
Hook = Callable[[], HookResult]


@dataclass(frozen=True)
class TradeManagementReport:
    """The full, explainable result of one post-entry re-evaluation cycle."""

    thesis_intact: bool
    suitability: MarketSuitabilityResult
    volatility_changed: bool
    stop_should_tighten: bool
    take_partial_profit: bool
    reasons: List[str] = field(default_factory=list)

    @property
    def should_exit(self) -> bool:
        """Thesis invalidation is the only condition this module treats as a
        full-exit signal — volatility change/stop-tighten/partial-profit are
        management adjustments, not exit triggers, and are left for the
        strategy's own exit logic to act on."""
        return not self.thesis_intact


class TradeManagementMonitor:
    """Shared hook pattern for continuous post-entry thesis re-evaluation.

    Stateless: ``evaluate()`` is a pure function of its arguments and the
    (strategy-supplied, ideally pure) hooks it calls — same inputs and
    hook behavior, same report, every time.
    """

    def __init__(self, suitability_gate: Optional[MarketSuitabilityGate] = None) -> None:
        self._suitability_gate = suitability_gate or MarketSuitabilityGate()

    def evaluate(
        self,
        *,
        current_regime: dict,
        allowed_regimes: Sequence[str],
        volatility_check: Optional[Hook] = None,
        stop_tighten_check: Optional[Hook] = None,
        partial_profit_check: Optional[Hook] = None,
    ) -> TradeManagementReport:
        """Re-evaluate an open position for one tick.

        Args:
            current_regime: The current bar-based regime dict (same shape
                ``MarketSuitabilityGate`` already consumes).
            allowed_regimes: This strategy's declared suitable regimes —
                the SAME list used at entry, so "is the thesis still
                intact" is judged by the identical standard entry was.
            volatility_check: Optional strategy-supplied predicate —
                (b) has volatility changed materially since entry.
            stop_tighten_check: Optional strategy-supplied predicate —
                (c) should the stop tighten beyond a pure trailing formula.
            partial_profit_check: Optional strategy-supplied predicate —
                (d) should partial profit be taken.

        A ``None`` hook means "this strategy does not implement this
        check" and is treated as not-fired, never as an error — pillar 6
        does not require every strategy to implement all four checks
        identically, only that each is explicitly considered.
        """
        suitability = self._suitability_gate.evaluate(current_regime, allowed_regimes)

        reasons: List[str] = []
        if not suitability.is_suitable:
            reasons.append(f"thesis: {suitability.reason}")

        volatility_changed, vol_reason = self._run_hook(volatility_check)
        if volatility_changed:
            reasons.append(f"volatility: {vol_reason}")

        stop_should_tighten, tighten_reason = self._run_hook(stop_tighten_check)
        if stop_should_tighten:
            reasons.append(f"stop: {tighten_reason}")

        take_partial_profit, partial_reason = self._run_hook(partial_profit_check)
        if take_partial_profit:
            reasons.append(f"partial profit: {partial_reason}")

        return TradeManagementReport(
            thesis_intact=suitability.is_suitable,
            suitability=suitability,
            volatility_changed=volatility_changed,
            stop_should_tighten=stop_should_tighten,
            take_partial_profit=take_partial_profit,
            reasons=reasons,
        )

    @staticmethod
    def _run_hook(hook: Optional[Hook]) -> HookResult:
        if hook is None:
            return False, ""
        fired, reason = hook()
        return bool(fired), str(reason or "")
