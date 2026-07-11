"""Documentation-only future-consumer contract for StrategyProposal.

NO CODE IN THIS FILE. It exists so a future Auto Mode implementation
starts from an already-agreed target instead of re-deriving it — per
``add-strategy-decision-framework``'s tasks.md 0.12: "Documentation-only
(no implementation)... Neither class is implemented, instantiated, or
wired to anything - Auto Mode remains explicitly out of scope for this
change." Do not add classes, functions, or imports to this file. If you
are implementing Auto Mode, this docstring is your starting spec — see
``openspec/changes/add-auto-mode-investment-committee/`` for the complete,
frozen architectural specification (Committee Process, ranking/tie-
breaking, certification) this summarizes only the data shapes of.

======================================================================
CommitteeDecision (future contract — specified, not implemented)
======================================================================

Auto Mode's future committee consumes ``StrategyProposal`` objects (this
module's ``proposal.StrategyProposal``) and produces a separate
``CommitteeDecision`` — proposals are NEVER mutated into a decision, they
are referenced by one (Proposal Immutability, see ``proposal.py``).

Fields (see ``add-auto-mode-investment-committee/design.md``'s "Committee
Decision (finalized)" for the authoritative specification):

  decision_id: str
      Deterministic identifier for this committee evaluation cycle
      (derived from the cycle's timestamp and the set of proposal_ids
      considered — not a random UUID, same rationale as
      StrategyProposal.proposal_id).
  evaluated_at: datetime
      When the committee ran this cycle.
  proposals_considered: List[str]
      Every proposal_id collected this cycle — the full candidate set,
      including ones later rejected.
  selected: List[SelectedAllocation]
      SelectedAllocation = {proposal_id, allocated_size, execution_priority}
      for each proposal chosen to execute (zero, one, or many).
  rejected: List[RejectedProposal]
      RejectedProposal = {proposal_id, rejection_step, rejection_reason}
      for every proposal considered and not selected. rejection_step names
      which Committee Process step rejected it (e.g. "expired",
      "edge_disqualified", "portfolio_risk", "ranked_below_selection") —
      never an unmeasurable judgement.
  trust_adjustments_applied: List[str]
      References to every TrustAdjustment record consulted this cycle,
      for audit.
  ranking_snapshot: List[str]
      The full ranked order of every proposal that survived rejection, by
      proposal_id — so a human (or a future learning system) can
      reconstruct exactly how selection followed from ranking.

CommitteeDecision is itself immutable once produced, for the same reason
every StrategyProposal is: a decision, once made, is a permanent audit
record. A new cycle produces a new CommitteeDecision, never an edit to a
prior one.

======================================================================
TrustAdjustment (future contract — specified, not implemented)
======================================================================

External information (news, social sentiment, Fear & Greed, macro,
funding, options, futures) belongs exclusively to Auto — never to
strategies. It is represented as independent TrustAdjustment records that
reference, but never modify, a StrategyProposal:

  proposal_id: str
      Which proposal this adjusts.
  source: str
      e.g. "fear_greed", "news_sentiment", "funding_rate", "exchange_health".
  adjustment: float
      A multiplier or delta applied ONLY within the committee's ranking of
      this proposal — never to decision_score or any other proposal field.
  generated_at: datetime

======================================================================
Why this file has no code
======================================================================

Auto Mode (including all committee/ranking/allocation logic) is
explicitly out of scope for both `add-strategy-decision-framework`'s
Phase 0 and this file's own change. Defining even an unused dataclass here
would be "implementing" in a way the task explicitly forbids — the value
of this file is exclusively that Phase 0-6 and a future Auto Mode
implementation change agree on these shapes in writing, now, without
either being obligated to keep an unused code artifact in sync with the
authoritative OpenSpec design docs. When Auto Mode is actually
implemented, these shapes belong in real, tested code (likely
``app/services/auto_committee/``, per
``add-auto-mode-investment-committee/tasks.md``) — not retrofitted from
this docstring.
"""
