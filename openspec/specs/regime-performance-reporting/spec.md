# regime-performance-reporting Specification

## Purpose
Answer *when* a strategy works, not just whether it does. A single expectancy
figure averages over market states a strategy behaves differently in — one that
earns in chop and bleeds in trends has a mean describing neither — so this
capability buckets a completed backtest's trades by the market regime in force
at each trade's entry and reports expectancy, win rate, and drawdown per bucket.

Classification uses the canonical regime detector `_strategy_auto` already uses
for live strategy selection, so a per-regime claim is checkable against the same
notion of "regime" the system actually trades on, rather than a second, ad hoc
heuristic describing a market nothing trades in. Bucketing is post-hoc over
results already produced: it re-runs nothing and influences no decision.
## Requirements
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

