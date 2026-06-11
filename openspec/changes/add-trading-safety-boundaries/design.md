## Context
The system already enforces per-bot risk limits and a virtual wallet. It does not model execution costs
or enforce risk constraints across multiple bots and strategies.

## Goals / Non-Goals
- Goals:
  - Model execution costs (fees, spread, slippage/impact) consistently in sim and live paths.
  - Enforce portfolio-level risk caps across all bots owned by the operator.
  - Enforce strategy capacity limits to prevent over-allocation to a single strategy.
- Non-Goals:
  - Multi-user account management or permissions.
  - Exchange-level risk controls or margin trading.

## Decisions
- Decision: Introduce an ExecutionCostModel service used by trading_engine and simulated exchange.
  - Why: Centralized modeling avoids duplicated logic and keeps PnL consistent.
- Decision: Add portfolio caps as a configuration object loaded at runtime and enforced pre-trade.
  - Why: Portfolio caps are cross-cutting and should be enforced alongside per-bot checks.
- Decision: Apply strategy capacity limits in auto-mode selection and in final order sizing.
  - Why: Prevents auto-mode from picking an over-capacity strategy and enforces hard caps at execution.

## Risks / Trade-offs
- Cost modeling is only an estimate until fill data is available; add fields for modeled vs realized costs.
- Enforcing portfolio caps may pause bots more often; surface clear reasons in logs and UI.

## Migration Plan
1. Add new schema fields with defaults that preserve current behavior (no caps, zero slippage/impact).
2. Roll out cost modeling in dry-run first; validate PnL changes.
3. Enable portfolio caps and capacity limits once UI configuration is in place.

## Open Questions
- What default values should be used for spread/slippage/impact for MEXC spot?
- Should portfolio caps apply to dry-run bots or only live bots?
