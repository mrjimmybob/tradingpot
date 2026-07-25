"""Pillar 7 — Strategy Edge Management (formerly "Failure Detection").

Detecting "I am losing" is insufficient; professional traders investigate
*why*. ``StrategyEdgeManager`` is a per-bot, per-strategy continuous
tracker of measurable degradation signals that classifies detected
degradation into exactly one of three categories, each with a specific,
bounded action (design.md, pillar 7):

  * **A — Temporary market mismatch**: conditions moved outside the
    strategy's suitable range, no evidence the edge itself is gone.
    Action: reduce activity / raise the Decision Score threshold / wait.
    NEVER a permanent stop.
  * **B — Parameter mismatch**: the thesis may still be valid, but current
    parameters no longer fit conditions. Action: adapt the specific
    parameter via ``adaptive_params.AdaptiveParameterResolver``,
    mathematically justified — never a blanket re-tune.
  * **C — Edge disappeared**: degradation persists after A/B responses, or
    measured evidence contradicts the strategy's Theory document. Action:
    stop trading; do not resume without a human-reviewed re-certification.

**This manager NEVER force-closes an open position.** It has no method
that touches a ``Position`` or places an order — structurally impossible,
not merely a documented rule (see ``test_strategy_framework_edge_
management.py``'s ``test_public_api_has_no_position_closing_capability``).
Mirrors how ``PortfolioRiskService``'s loss caps already only block NEW
trades, never liquidate open ones.

Scope boundary: this module tracks and classifies TRADE-OUTCOME-DERIVED
signals (win rate, expectancy, consecutive losses, reward:risk, holding
time, Decision Score trend) that require no strategy-specific indicator
knowledge. Regime/volatility-vs-assumptions and parameter-mismatch
evidence are supplied BY THE CALLER (the strategy, via the shared
``MarketSuitabilityGate`` for regime, and its own Theory-document-informed
judgement for parameter mismatch) — this manager does not compute or
interpret indicators itself, consistent with keeping indicator
interpretation permanently strategy-owned.

This module is infrastructure only — no strategy calls it yet (Phase 0).
It fixes the three-category *structure* and requires every numeric
threshold to be supplied (constructor defaults exist only so the class is
usable standalone/in tests) — calibrating real per-strategy threshold
values is a certification-phase decision (design.md Non-Goals), not fixed
here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Deque, Dict, List, Mapping, Optional, Tuple


class EdgeCategory(str, Enum):
    NONE = "none"
    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True)
class TradeOutcome:
    """One completed trade's measurable outcome, as reported by a strategy."""

    pnl: float
    win: bool
    reward_risk_realized: Optional[float] = None
    holding_seconds: Optional[float] = None
    at: Optional[datetime] = None


@dataclass(frozen=True)
class EdgeSignal:
    """One measurable signal that participated in a classification."""

    name: str
    value: float
    description: str


@dataclass(frozen=True)
class EdgeStatus:
    """The full, explainable result of one edge evaluation."""

    category: EdgeCategory
    action: str
    signals: List[EdgeSignal]
    reason: str
    can_adapt: bool
    should_wait: bool
    should_stop: bool
    evaluated_at: Optional[datetime] = None

    def explain(self) -> str:
        """Answers, from measurable evidence only: why (not) profitable,
        can I adapt, should I wait, should I stop — the framework's
        required continuous self-diagnostic (design.md, pillar 7)."""
        signal_text = "; ".join(f"{s.name}={s.value:.4f}" for s in self.signals)
        parts = [f"Category {self.category.value}: {self.reason}"]
        if signal_text:
            parts.append(f"Evidence: {signal_text}")
        parts.append(f"Action: {self.action}")
        parts.append(
            f"Can adapt: {self.can_adapt}. Should wait: {self.should_wait}. "
            f"Should stop: {self.should_stop}."
        )
        return " ".join(parts)


_Key = Tuple[int, str]


class StrategyEdgeManager:
    """Per-bot, per-strategy continuous edge tracker and classifier.

    All thresholds are constructor-configurable — there is no fixed,
    hardcoded "this strategy is degrading" cutoff baked into the class.
    Defaults are placeholders for standalone use/testing, not calibrated
    values for any real strategy.
    """

    def __init__(
        self,
        *,
        min_sample_size: int = 20,
        outcome_window: int = 50,
        expectancy_floor: float = 0.0,
        win_rate_floor: float = 0.35,
        decision_score_trend_window: int = 10,
        decision_score_trend_epsilon: float = 1.0,
    ) -> None:
        if min_sample_size < 1:
            raise ValueError("min_sample_size must be >= 1")
        if outcome_window < min_sample_size:
            raise ValueError("outcome_window must be >= min_sample_size")
        self.min_sample_size = min_sample_size
        self.outcome_window = outcome_window
        self.expectancy_floor = expectancy_floor
        self.win_rate_floor = win_rate_floor
        self.decision_score_trend_window = decision_score_trend_window
        self.decision_score_trend_epsilon = decision_score_trend_epsilon

        self._outcomes: Dict[_Key, Deque[TradeOutcome]] = {}
        self._decision_scores: Dict[_Key, Deque[float]] = {}

    # -- Recording -----------------------------------------------------

    def record_trade_outcome(
        self,
        bot_id: int,
        strategy: str,
        *,
        pnl: float,
        win: bool,
        reward_risk_realized: Optional[float] = None,
        holding_seconds: Optional[float] = None,
        at: Optional[datetime] = None,
    ) -> None:
        key = (bot_id, strategy)
        if key not in self._outcomes:
            self._outcomes[key] = deque(maxlen=self.outcome_window)
        self._outcomes[key].append(
            TradeOutcome(
                pnl=pnl,
                win=win,
                reward_risk_realized=reward_risk_realized,
                holding_seconds=holding_seconds,
                at=at,
            )
        )

    def record_decision_score(self, bot_id: int, strategy: str, score: float) -> None:
        key = (bot_id, strategy)
        if key not in self._decision_scores:
            self._decision_scores[key] = deque(maxlen=self.decision_score_trend_window * 2)
        self._decision_scores[key].append(score)

    def reset(self, bot_id: int, strategy: str) -> None:
        """Clear tracked history — e.g. after a Category C stop is
        followed by a human-reviewed re-certification and the strategy
        resumes with a clean slate."""
        self._outcomes.pop((bot_id, strategy), None)
        self._decision_scores.pop((bot_id, strategy), None)

    # -- Classification --------------------------------------------------

    def evaluate(
        self,
        bot_id: int,
        strategy: str,
        *,
        regime_outside_suitable_range: bool,
        parameter_mismatch_evidence: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> EdgeStatus:
        """Classify current degradation (if any) from tracked evidence.

        Args:
            regime_outside_suitable_range: caller-supplied (typically via
                ``MarketSuitabilityGate``) — True if current market
                conditions are outside this strategy's declared suitable
                range. This manager never computes this itself.
            parameter_mismatch_evidence: caller-supplied citation of a
                specific parameter no longer fitting current conditions
                (e.g. "stop ATR multiplier last calibrated at 0.6x current
                volatility"), or None if no such evidence exists.
        """
        outcomes = self._outcomes.get((bot_id, strategy), deque())
        evaluated_at = now

        if len(outcomes) < self.min_sample_size:
            return EdgeStatus(
                category=EdgeCategory.NONE,
                action="Insufficient trade history to classify - continue trading normally.",
                signals=[
                    EdgeSignal(
                        "sample_size", float(len(outcomes)),
                        f"{len(outcomes)} of {self.min_sample_size} minimum trades tracked",
                    )
                ],
                reason=(
                    f"only {len(outcomes)} trade(s) tracked, need "
                    f"{self.min_sample_size} to classify edge status"
                ),
                can_adapt=False,
                should_wait=False,
                should_stop=False,
                evaluated_at=evaluated_at,
            )

        stats = self._compute_stats(bot_id, strategy, outcomes)
        signals = self._signals_from_stats(stats)

        degrading, degrading_reasons = self._is_degrading(stats)

        if not degrading:
            return EdgeStatus(
                category=EdgeCategory.NONE,
                action="No degradation detected - continue trading normally.",
                signals=signals,
                reason=(
                    f"expectancy {stats['expectancy']:.4f} >= floor "
                    f"{self.expectancy_floor:.4f}, win rate {stats['win_rate']:.2%} "
                    f">= floor {self.win_rate_floor:.2%}, no adverse Decision "
                    "Score trend"
                ),
                can_adapt=False,
                should_wait=False,
                should_stop=False,
                evaluated_at=evaluated_at,
            )

        reason_text = "; ".join(degrading_reasons)

        if regime_outside_suitable_range:
            return EdgeStatus(
                category=EdgeCategory.A,
                action=(
                    "Reduce activity; raise the Decision Score threshold; wait. "
                    "Never a permanent stop."
                ),
                signals=signals,
                reason=f"degradation with regime outside suitable range: {reason_text}",
                can_adapt=False,
                should_wait=True,
                should_stop=False,
                evaluated_at=evaluated_at,
            )

        if parameter_mismatch_evidence:
            return EdgeStatus(
                category=EdgeCategory.B,
                action=(
                    "Adapt the specific parameter via AdaptiveParameterResolver, "
                    "mathematically justified by the measured signal."
                ),
                signals=signals,
                reason=(
                    f"degradation with parameter mismatch evidence "
                    f"({parameter_mismatch_evidence}): {reason_text}"
                ),
                can_adapt=True,
                should_wait=False,
                should_stop=False,
                evaluated_at=evaluated_at,
            )

        return EdgeStatus(
            category=EdgeCategory.C,
            action=(
                "Stop trading this strategy. Do not resume without a "
                "human-reviewed re-certification."
            ),
            signals=signals,
            reason=(
                f"degradation persists with no regime or parameter-mismatch "
                f"explanation: {reason_text}"
            ),
            can_adapt=False,
            should_wait=False,
            should_stop=True,
            evaluated_at=evaluated_at,
        )

    # -- Internal, deterministic statistics -----------------------------

    def _compute_stats(
        self, bot_id: int, strategy: str, outcomes: Deque[TradeOutcome]
    ) -> dict:
        n = len(outcomes)
        wins = sum(1 for o in outcomes if o.win)
        win_rate = wins / n
        expectancy = sum(o.pnl for o in outcomes) / n

        rr_values = [o.reward_risk_realized for o in outcomes if o.reward_risk_realized is not None]
        avg_reward_risk = sum(rr_values) / len(rr_values) if rr_values else None

        holding_values = [o.holding_seconds for o in outcomes if o.holding_seconds is not None]
        avg_holding_seconds = sum(holding_values) / len(holding_values) if holding_values else None

        consecutive_losses = 0
        for outcome in reversed(outcomes):
            if outcome.win:
                break
            consecutive_losses += 1

        decision_score_trend_delta = self._decision_score_trend(bot_id, strategy)

        return {
            "win_rate": win_rate,
            "expectancy": expectancy,
            "avg_reward_risk": avg_reward_risk,
            "avg_holding_seconds": avg_holding_seconds,
            "consecutive_losses": consecutive_losses,
            "decision_score_trend_delta": decision_score_trend_delta,
        }

    def _decision_score_trend(self, bot_id: int, strategy: str) -> Optional[float]:
        """Recent-half mean minus older-half mean of tracked Decision
        Scores. Negative -> trending down. None if not enough samples."""
        scores = self._decision_scores.get((bot_id, strategy))
        window = self.decision_score_trend_window
        if not scores or len(scores) < window * 2:
            return None
        values = list(scores)[-window * 2:]
        older_half = values[:window]
        recent_half = values[window:]
        return (sum(recent_half) / window) - (sum(older_half) / window)

    @staticmethod
    def _signals_from_stats(stats: dict) -> List[EdgeSignal]:
        signals = [
            EdgeSignal("win_rate", stats["win_rate"], "fraction of tracked trades that were winners"),
            EdgeSignal("expectancy", stats["expectancy"], "mean P&L per tracked trade"),
            EdgeSignal(
                "consecutive_losses", float(stats["consecutive_losses"]),
                "current losing streak length",
            ),
        ]
        if stats["avg_reward_risk"] is not None:
            signals.append(
                EdgeSignal("avg_reward_risk", stats["avg_reward_risk"], "mean realized reward:risk ratio")
            )
        if stats["avg_holding_seconds"] is not None:
            signals.append(
                EdgeSignal(
                    "avg_holding_seconds", stats["avg_holding_seconds"],
                    "mean holding time in seconds",
                )
            )
        if stats["decision_score_trend_delta"] is not None:
            signals.append(
                EdgeSignal(
                    "decision_score_trend_delta", stats["decision_score_trend_delta"],
                    "recent-half mean Decision Score minus older-half mean",
                )
            )
        return signals

    def _is_degrading(self, stats: dict) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        if stats["expectancy"] < self.expectancy_floor:
            reasons.append(
                f"expectancy {stats['expectancy']:.4f} < floor {self.expectancy_floor:.4f}"
            )
        if stats["win_rate"] < self.win_rate_floor:
            reasons.append(
                f"win rate {stats['win_rate']:.2%} < floor {self.win_rate_floor:.2%}"
            )
        trend = stats["decision_score_trend_delta"]
        if trend is not None and trend < -self.decision_score_trend_epsilon:
            reasons.append(
                f"Decision Score trending down by {trend:.4f} "
                f"(beyond epsilon {self.decision_score_trend_epsilon:.4f})"
            )
        return bool(reasons), reasons


# Objective, permanent invalidation conditions for a DCA accumulation thesis.
# These are the ONLY things that make a classic DCA "lose its edge". NONE of
# them is price-, trend-, or performance-based - that is the entire point.
DCA_THESIS_INVALIDATION_CONDITIONS: Tuple[str, ...] = (
    "operator_invalidated",       # operator manually declares the thesis dead
    "asset_delisted",             # asset permanently delisted / untradeable
    "fundamental_failure",        # fundamental project failure
    "regulatory_impossibility",   # holding/trading the asset is legally impossible
    "portfolio_withdrawn",        # portfolio policy permanently withdraws the asset
)


class DcaEdgeManager:
    """Strategy Edge Management (Pillar 7) for a classic, direction-agnostic DCA.

    Deliberately distinct from ``StrategyEdgeManager``: a classic DCA does NOT
    fail because the market falls, so this monitors the health of the
    ACCUMULATION PROCESS, never mark-to-market profitability. It records no
    trade outcomes and computes no performance statistics - there is
    intentionally **no price / pnl / trend input on this class at all**, so an
    ordinary drawdown can never be reinterpreted as loss of edge. It reuses the
    shared ``EdgeStatus`` / ``EdgeCategory`` / ``EdgeSignal`` output contract so
    it is wired to the framework identically to the other strategies; only the
    classification axis differs (see audits/dca_accumulator.md Phase 6.1/6.4 and
    design.md's "Pure-accumulator exception"). Stateless and fully deterministic.

    Categories:
      C    - objective, permanent invalidation of the accumulation thesis
             (operator invalidation, delisting, fundamental failure, regulatory
             impossibility, portfolio withdrawal). The ONLY thing that stops a
             classic DCA; requires human re-certification.
      A    - a TEMPORARY operational / portfolio-management pause (e.g. budget
             temporarily exhausted, portfolio exposure headroom reached, an
             opt-in operator overlay pausing). Wait; never a permanent stop.
      B    - an operational / portfolio PARAMETER adaptation (e.g. chunk size
             floored to the executable minimum or trimmed to portfolio
             headroom). Adapt; never a directional change.
      NONE - accumulation process healthy; continue on schedule.
    """

    def evaluate(
        self,
        *,
        thesis_invalidation: Optional[Mapping[str, bool]] = None,
        operational_pause: Optional[str] = None,
        operational_adaptation: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> EdgeStatus:
        """Classify accumulation-process health from operational/thesis inputs.

        Args:
            thesis_invalidation: mapping of objective, permanent invalidation
                conditions (keys from ``DCA_THESIS_INVALIDATION_CONDITIONS``) to
                booleans. Any True -> Category C. Unrecognised keys are ignored,
                so no ad-hoc (e.g. price/performance) condition can invalidate
                the thesis through this door.
            operational_pause: citation of a TEMPORARY operational/portfolio
                reason accumulation is paused this cycle -> Category A.
            operational_adaptation: citation of an operational/portfolio
                parameter being adapted this cycle -> Category B.
        """
        conditions = thesis_invalidation or {}
        active = [
            name for name in DCA_THESIS_INVALIDATION_CONDITIONS
            if bool(conditions.get(name))
        ]
        if active:
            return EdgeStatus(
                category=EdgeCategory.C,
                action=(
                    "Stop accumulating. Do not resume without a human-reviewed "
                    "re-certification of the investment thesis."
                ),
                signals=[
                    EdgeSignal(name, 1.0, "objective thesis-invalidation condition active")
                    for name in active
                ],
                reason="accumulation thesis objectively invalidated: " + ", ".join(active),
                can_adapt=False,
                should_wait=False,
                should_stop=True,
                evaluated_at=now,
            )

        if operational_pause:
            return EdgeStatus(
                category=EdgeCategory.A,
                action=(
                    "Temporarily reduce/pause accumulation for this operational "
                    "reason; resume automatically when it clears. Never a permanent stop."
                ),
                signals=[EdgeSignal("operational_pause", 1.0, operational_pause)],
                reason=f"temporary operational/portfolio pause: {operational_pause}",
                can_adapt=False,
                should_wait=True,
                should_stop=False,
                evaluated_at=now,
            )

        if operational_adaptation:
            return EdgeStatus(
                category=EdgeCategory.B,
                action=(
                    "Adapt the operational parameter (e.g. chunk size vs. the "
                    "executable minimum or portfolio headroom). Never a directional change."
                ),
                signals=[EdgeSignal("operational_adaptation", 1.0, operational_adaptation)],
                reason=f"operational/portfolio parameter adaptation: {operational_adaptation}",
                can_adapt=True,
                should_wait=False,
                should_stop=False,
                evaluated_at=now,
            )

        return EdgeStatus(
            category=EdgeCategory.NONE,
            action="Accumulation process healthy - continue on schedule.",
            signals=[],
            reason="thesis valid; no operational pause or adaptation in effect",
            can_adapt=False,
            should_wait=False,
            should_stop=False,
            evaluated_at=now,
        )
