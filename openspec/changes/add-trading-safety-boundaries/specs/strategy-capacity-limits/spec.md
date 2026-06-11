## ADDED Requirements
### Requirement: Strategy Capacity Configuration
The system SHALL allow per-strategy capacity limits, including max allocation percent and max concurrent bots.

#### Scenario: Configure capacity limits for a strategy
- **WHEN** the operator sets a max allocation percent and max bot count for a strategy
- **THEN** the system stores and validates the limits for enforcement

### Requirement: Auto-Mode Capacity Filtering
The system SHALL exclude over-capacity strategies from auto-mode selection.

#### Scenario: Auto-mode skips over-capacity strategy
- **WHEN** auto-mode evaluates eligible strategies and a strategy exceeds its capacity
- **THEN** that strategy is excluded from the eligible list

### Requirement: Capacity-Aware Order Sizing
The system SHALL cap order sizing to remain within remaining strategy capacity.

#### Scenario: Order size reduced to fit capacity
- **WHEN** a trade would exceed the remaining strategy allocation capacity
- **THEN** the order size is reduced to the maximum allowed
