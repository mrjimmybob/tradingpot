-- Migration 009: Add decision_explanation and edge_management_category
-- columns to the orders table (add-strategy-decision-framework, Phase 0.6).
-- Closes the Pillar 8 persistence gap found in every strategy's audit: the
-- full structured DecisionExplanation (including, once a strategy is
-- migrated, its Evidence Report) previously lived only in the in-memory,
-- current-state-only DiagnosticsStore - a historical trade could only be
-- explained by its terse Order.reason string. Both columns are nullable
-- and additive; existing rows are unaffected and no existing behavior
-- changes.
ALTER TABLE orders ADD COLUMN decision_explanation JSON;
ALTER TABLE orders ADD COLUMN edge_management_category VARCHAR(20);
