"""Pillar 2 — Market Suitability: a shared, ACTUALLY-ENFORCED regime gate.

``add-strategy-decision-framework``'s audit found that market-suitability
checks exist in some form for most strategies, but are inconsistently
enforced — most acutely, ``volatility_breakout`` already computes a
suitability value that is never consulted by its own entry decision
(diagnostic-only). This module exists so "a strategy computed a
suitability value" and "a strategy is actually blocked from opening a new
position when unsuitable" can never again silently diverge: a strategy
calls ``MarketSuitabilityGate.gate(...)`` and treats ``False`` as a hard
stop, not a value to log and ignore.

This module does not invent a new regime detector. It wraps the existing,
canonical bar-based one (``TradingEngine._detect_market_regime_bar_based``,
made canonical by ``fix-regime-detection-consistency``) and reuses the
EXACT regime-tag convention ``TradingEngine._is_strategy_eligible`` already
uses for Auto Mode dispatch — ``f"trend_{trend_state}"``,
``f"volatility_{volatility_state}"``, ``f"volatility_{volatility_direction}"``,
``f"liquidity_{liquidity_state}"`` — so a strategy's standalone suitability
gate and its Auto-mode regime eligibility can never disagree about what
"trend_flat" or "volatility_expanding" means. See ``trading_engine.py``
around ``_is_strategy_eligible`` (~line 5362) and
``_get_strategy_capabilities`` (~line 6118) for the existing convention
this reuses.

Determinism: both public methods are pure functions of their arguments —
same ``regime``/``allowed_regimes`` in, same result out, always.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

# Regime dict keys produced by TradingEngine._detect_market_regime_bar_based.
_TREND_KEY = "trend_state"
_VOLATILITY_KEY = "volatility_state"
_VOLATILITY_DIRECTION_KEY = "volatility_direction"
_LIQUIDITY_KEY = "liquidity_state"

# Defaults match TradingEngine._detect_market_regime_bar_based's own
# "not enough data" neutral regime, so a gate evaluated before warm-up
# completes fails closed to a real (if uninteresting) regime rather than
# raising on a missing key.
_DEFAULT_TREND = "flat"
_DEFAULT_VOLATILITY = "medium"
_DEFAULT_VOLATILITY_DIRECTION = "stable"
_DEFAULT_LIQUIDITY = "normal"

# Sentinel a strategy declares to mean "suitable in every regime" (matches
# TradingEngine._get_strategy_capabilities's dca_accumulator convention).
ALL_REGIMES = "all"


@dataclass(frozen=True)
class MarketSuitabilityResult:
    """The full, explainable result of one suitability evaluation.

    ``is_suitable`` is the only field a hard gate needs to check, but the
    rest exists so the decision is reproducible and explainable (pillar 8):
    an operator (or a future ``StrategyProposal.market_suitability`` field)
    can see exactly which regime tags were computed, which were allowed,
    and which (if any) matched.
    """

    is_suitable: bool
    regime_tags: List[str]
    allowed_regimes: List[str]
    matched_tags: List[str]
    reason: str


class MarketSuitabilityGate:
    """Wraps the canonical regime detector's output as an enforceable gate.

    Stateless and side-effect free: every method is a pure function of its
    arguments, so the same regime dict and ``allowed_regimes`` list always
    produce the same result — required for the framework's determinism
    guarantee and for unit testing without a live regime detector.
    """

    @staticmethod
    def regime_tags(regime: dict) -> List[str]:
        """Build the regime-tag vocabulary from a raw regime dict.

        ``regime`` is the dict shape produced by
        ``TradingEngine._detect_market_regime_bar_based`` (keys
        ``trend_state``/``volatility_state``/``volatility_direction``/
        ``liquidity_state``). Missing keys fall back to the same neutral
        defaults that detector itself returns before warm-up completes.
        """
        trend = regime.get(_TREND_KEY, _DEFAULT_TREND)
        volatility = regime.get(_VOLATILITY_KEY, _DEFAULT_VOLATILITY)
        vol_direction = regime.get(_VOLATILITY_DIRECTION_KEY, _DEFAULT_VOLATILITY_DIRECTION)
        liquidity = regime.get(_LIQUIDITY_KEY, _DEFAULT_LIQUIDITY)
        return [
            f"trend_{trend}",
            f"volatility_{volatility}",
            f"volatility_{vol_direction}",
            f"liquidity_{liquidity}",
        ]

    def evaluate(
        self,
        regime: dict,
        allowed_regimes: Sequence[str],
    ) -> MarketSuitabilityResult:
        """Evaluate suitability without enforcing it — see ``gate()`` for the
        hard-stop entry point every strategy's entry decision must consult.
        """
        allowed = list(allowed_regimes or [])
        tags = self.regime_tags(regime)

        if not allowed:
            return MarketSuitabilityResult(
                is_suitable=False,
                regime_tags=tags,
                allowed_regimes=allowed,
                matched_tags=[],
                reason="no allowed_regimes declared for this strategy",
            )

        if ALL_REGIMES in allowed:
            return MarketSuitabilityResult(
                is_suitable=True,
                regime_tags=tags,
                allowed_regimes=allowed,
                matched_tags=list(tags),
                reason="strategy declares allowed_regimes=['all']",
            )

        matched = [tag for tag in tags if tag in allowed]
        is_suitable = bool(matched)
        reason = (
            f"regime matches {matched}"
            if is_suitable
            else f"current regime {tags} outside allowed {allowed}"
        )
        return MarketSuitabilityResult(
            is_suitable=is_suitable,
            regime_tags=tags,
            allowed_regimes=allowed,
            matched_tags=matched,
            reason=reason,
        )

    def gate(self, regime: dict, allowed_regimes: Sequence[str]) -> bool:
        """Hard gate: a strategy's entry decision MUST treat ``False`` here
        as "do not open a new position", per the framework's Market
        Suitability Gate requirement — computing a suitability value and
        not consulting it (the ``volatility_breakout`` finding this module
        exists to close) is exactly the violation this method prevents by
        being the single call site a strategy's entry branch depends on.
        """
        return self.evaluate(regime, allowed_regimes).is_suitable
