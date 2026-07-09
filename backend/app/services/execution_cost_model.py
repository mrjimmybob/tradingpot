"""Execution cost modeling service — risk and viability estimation.

Cost estimates are used at the execution layer for two purposes:
1. P&L tracking accuracy: costs are logged/recorded alongside each trade.
2. Trade viability gate: the central gate in _execute_trade uses round-trip cost
   estimates to block BUY signals where the expected price move cannot cover fees.
   This is an EXECUTION-layer decision, not a strategy-layer one — the gate lives
   in _execute_trade, not inside individual strategy functions.

Strategies MUST NOT use this model to score opportunities or rank signals.
auto_mode MUST NOT use cost estimates in its strategy scoring weights.

Design constraints:
- Spot trading only (no leverage, no margin)
- Deterministic (no randomness)
- Default values = 0 (preserves current behavior when fee is not configured)
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutionCostEstimate:
    """Execution cost estimate breakdown.

    All costs are in USD (quote currency).
    """
    exchange_fee: float  # Exchange commission
    spread_cost: float   # Bid-ask spread cost
    slippage_cost: float # Expected price slippage
    total_cost: float    # Sum of all costs

    # Metadata
    notional_usd: float
    side: str  # "buy" or "sell"


class ExecutionCostModel:
    """Service for estimating execution costs before trade execution.

    This is a RISK-ONLY model. It does not:
    - Predict future prices
    - Generate alpha signals
    - Influence strategy decisions
    - Feed into auto_mode scoring

    It only:
    - Estimates costs for risk calculations
    - Improves P&L tracking accuracy
    - Helps size orders more accurately
    """

    def __init__(
        self,
        exchange_fee_pct: float = 0.0,
        market_spread_pct: float = 0.0,
        slippage_pct: float = 0.0,
        impact_pct: float = 0.0,
    ):
        """Initialize execution cost model.

        Args:
            exchange_fee_pct: Exchange commission rate (e.g., 0.1 for 0.1%)
            market_spread_pct: Typical bid-ask spread (e.g., 0.05 for 0.05%)
            slippage_pct: Expected slippage (e.g., 0.02 for 0.02%)
            impact_pct: Market impact (e.g., 0.01 for 0.01%)

        Note:
            All default to 0.0, which preserves current behavior.
            These must be explicitly configured to enable cost modeling.
        """
        self.exchange_fee_pct = exchange_fee_pct
        self.market_spread_pct = market_spread_pct
        self.slippage_pct = slippage_pct
        self.impact_pct = impact_pct

        if exchange_fee_pct > 0 or market_spread_pct > 0 or slippage_pct > 0 or impact_pct > 0:
            logger.info(
                f"ExecutionCostModel initialized with: "
                f"exchange_fee={exchange_fee_pct}%, "
                f"spread={market_spread_pct}%, "
                f"slippage={slippage_pct}%, "
                f"impact={impact_pct}%"
            )

    def estimate_cost(
        self,
        side: str,
        notional_usd: float,
        price: float,
        exchange_fee_pct: Optional[float] = None,
        market_spread_pct: Optional[float] = None,
        slippage_pct: Optional[float] = None,
        impact_pct: Optional[float] = None,
    ) -> ExecutionCostEstimate:
        """Estimate execution cost for a trade.

        Formula:
            exchange_fee = notional * exchange_fee_pct / 100
            spread_cost = notional * market_spread_pct / 100
            slippage_cost = notional * slippage_pct / 100
            impact_cost = notional * impact_pct / 100 (not used in spot)
            total_cost = exchange_fee + spread_cost + slippage_cost

        Args:
            side: "buy" or "sell"
            notional_usd: Order size in USD
            price: Current market price
            exchange_fee_pct: Override default exchange fee
            market_spread_pct: Override default spread
            slippage_pct: Override default slippage
            impact_pct: Override default impact (unused for spot)

        Returns:
            ExecutionCostEstimate with breakdown

        Note:
            This is deterministic - no randomness, no prediction.
            Default behavior (all costs = 0) is preserved unless configured.
        """
        # Use provided overrides or instance defaults
        fee_pct = exchange_fee_pct if exchange_fee_pct is not None else self.exchange_fee_pct
        spread_pct = market_spread_pct if market_spread_pct is not None else self.market_spread_pct
        slip_pct = slippage_pct if slippage_pct is not None else self.slippage_pct
        imp_pct = impact_pct if impact_pct is not None else self.impact_pct

        # Calculate costs
        exchange_fee = notional_usd * (fee_pct / 100.0)
        spread_cost = notional_usd * (spread_pct / 100.0)
        slippage_cost = notional_usd * (slip_pct / 100.0)

        # Impact cost not used in spot trading (no slippage model for spot)
        # Kept for future extensibility but always 0 for spot
        impact_cost = 0.0

        total_cost = exchange_fee + spread_cost + slippage_cost + impact_cost

        # Log if costs are significant
        if total_cost > 0.01:  # More than $0.01
            cost_pct = (total_cost / notional_usd * 100) if notional_usd > 0 else 0
            logger.debug(
                f"Estimated execution cost for {side} ${notional_usd:.2f}: "
                f"${total_cost:.4f} ({cost_pct:.4f}%) "
                f"[fee=${exchange_fee:.4f}, spread=${spread_cost:.4f}, slip=${slippage_cost:.4f}]"
            )

        return ExecutionCostEstimate(
            exchange_fee=exchange_fee,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            notional_usd=notional_usd,
            side=side,
        )

    def estimate_roundtrip_cost(
        self,
        notional_usd: float,
        price: float,
    ) -> float:
        """Estimate total cost for a roundtrip (buy + sell).

        This is useful for estimating minimum profit needed to break even.

        Args:
            notional_usd: Order size in USD
            price: Current market price

        Returns:
            Total roundtrip cost in USD
        """
        buy_cost = self.estimate_cost("buy", notional_usd, price)
        sell_cost = self.estimate_cost("sell", notional_usd, price)

        return buy_cost.total_cost + sell_cost.total_cost


# Global default instance (all costs = 0, preserves current behavior)
default_cost_model = ExecutionCostModel()


def get_cost_model(
    exchange_fee_pct: Optional[float] = None,
    market_spread_pct: Optional[float] = None,
    slippage_pct: Optional[float] = None,
    impact_pct: Optional[float] = None,
) -> ExecutionCostModel:
    """Get execution cost model with specified parameters.

    If all parameters are None or 0, returns default model (all costs = 0).
    Otherwise creates a new model with specified parameters.

    Args:
        exchange_fee_pct: Exchange commission rate
        market_spread_pct: Typical bid-ask spread
        slippage_pct: Expected slippage
        impact_pct: Market impact (unused for spot)

    Returns:
        ExecutionCostModel instance
    """
    # Check if any non-zero costs specified
    has_costs = any([
        exchange_fee_pct and exchange_fee_pct > 0,
        market_spread_pct and market_spread_pct > 0,
        slippage_pct and slippage_pct > 0,
        impact_pct and impact_pct > 0,
    ])

    if not has_costs:
        return default_cost_model

    return ExecutionCostModel(
        exchange_fee_pct=exchange_fee_pct or 0.0,
        market_spread_pct=market_spread_pct or 0.0,
        slippage_pct=slippage_pct or 0.0,
        impact_pct=impact_pct or 0.0,
    )
