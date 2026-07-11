"""Shared Strategy Decision Framework infrastructure (Phase 0).

Implements the shared modules specified by the OpenSpec change
``add-strategy-decision-framework`` (see ``openspec/changes/
add-strategy-decision-framework/design.md`` and ``tasks.md`` Phase 0).

This package is infrastructure ONLY: nothing here is wired into any of the
6 existing strategies yet (that is Phase 1-6, tracked separately). Every
module is independently unit tested against synthetic inputs.

Modules:
  * ``market_suitability`` — Pillar 2: ``MarketSuitabilityGate``.
  * ``decision_score`` — Pillar 3: ``DecisionScoreEngine``, ``EvidenceItem``,
    Evidence Report rendering.
  * ``adaptive_params`` — Pillar 4: ``AdaptiveParameterResolver``.
  * ``trade_management`` — Pillar 6: ``TradeManagementMonitor``.
  * ``edge_management`` — Pillar 7: ``StrategyEdgeManager``.
  * ``proposal`` — Pillar 10: ``StrategyProposal`` and its component types,
    plus the documented (not implemented) future ``CommitteeDecision``/
    ``TrustAdjustment`` contract.
  * ``standalone_adapter`` — translates a ``StrategyProposal`` into the
    existing, unchanged ``_execute_trade`` pipeline.
"""
