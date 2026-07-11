## ADDED Requirements
### Requirement: Per-Regime Backtest Breakdown
The system SHALL report a strategy's backtest performance broken down by the
market regime active at each trade's entry.

#### Scenario: Regime breakdown for a completed backtest
- **WHEN** a backtest with a non-empty trade list completes
- **THEN** the system can report expectancy, win rate, and drawdown
  separately for each regime bucket represented among the strategy's trades

#### Scenario: Regime classification uses the canonical detector
- **WHEN** regime-conditioned reporting classifies a trade's entry
- **THEN** it uses the same regime detector `_strategy_auto` uses for live
  strategy selection, not a separate ad hoc classification
