## ADDED Requirements

### Requirement: Dip Recovery lifecycle state machine
The system SHALL implement a `dip_recovery` strategy with an explicit, persisted
lifecycle: IDLE, TRACKING_DROP, WAITING_REVERSAL, LONG_OPEN, COOLDOWN, plus a transient
ENTRY_ARMED state reported only via the explanation system on the tick a BUY is emitted.
The strategy SHALL NOT buy while price is still declining or has not yet reversed off a
tracked low by the required adaptive margin.

#### Scenario: Continuous decline never buys
- **WHEN** price falls steadily tick after tick with no bounce
- **THEN** the strategy transitions to and remains in TRACKING_DROP and never emits a BUY

#### Scenario: Confirmed reversal after a decline buys
- **WHEN** a significant decline has been tracked to a low and price then recovers from
  that low by at least the adaptive recovery threshold, with all enabled confirmation
  filters passing
- **THEN** the strategy emits a BUY and transitions to LONG_OPEN, recording entry price,
  entry time, and the ATR value locked at entry

#### Scenario: An unresolved setup expires back to IDLE
- **WHEN** a bot has been in TRACKING_DROP or WAITING_REVERSAL for longer than
  `setup_expiry_minutes` without a confirmed entry
- **THEN** the strategy resets all tracking fields and returns to IDLE

### Requirement: Adaptive, volatility-relative thresholds
The system SHALL derive the decline-detection threshold and the recovery-confirmation
threshold from current volatility (an ATR-percent proxy computed from the bot's own tick
price history), each floored at a configured minimum, rather than using a fixed
percentage. No formula or default SHALL assume a specific trading pair.

#### Scenario: High volatility expands the drop threshold
- **WHEN** recent price history shows large tick-to-tick movement (high ATR%)
- **THEN** a decline that would be significant in a calm market (e.g. 2%) is NOT enough to
  arm TRACKING_DROP, because the ATR-scaled threshold is larger

#### Scenario: Low volatility contracts the drop threshold
- **WHEN** recent price history shows minimal tick-to-tick movement (low ATR%)
- **THEN** the drop threshold falls back to its configured floor, so a modest decline
  (e.g. 2%) is enough to arm TRACKING_DROP

### Requirement: ATR-based exit management
The system SHALL manage an open Dip Recovery position with a take-profit target, a
monotonically-tightening trailing stop, an independent emergency stop wider than the
trailing stop, and a maximum position duration - all derived from the ATR value locked at
entry. Every exit SHALL route through a cooldown before monitoring resumes, extended when
the exit was a loss.

#### Scenario: Trailing stop ratchets up and exits on pullback
- **WHEN** price rises after entry to a new high
- **THEN** the trailing stop moves up to track it (never down), and a subsequent pullback
  through the ratcheted stop (not the original one) closes the position

#### Scenario: A losing exit gets a longer cooldown
- **WHEN** a position is closed at a price below its entry price
- **THEN** the strategy enters COOLDOWN for `loss_cooldown_seconds` (>= the normal
  `cooldown_seconds`) before monitoring resumes

### Requirement: Auto Mode integration
The system SHALL make `dip_recovery` an eligible Auto Mode candidate with a regime-based
eligibility declaration and a numeric opportunity score (0-10) reflecting how well current
market conditions match its setup, computed via the same formula the strategy uses for its
own diagnostics.

#### Scenario: Auto Mode can select Dip Recovery
- **WHEN** the current regime is a downtrend and `dip_recovery` is not in cooldown, not
  blacklisted, and not at strategy capacity
- **THEN** `dip_recovery` appears in Auto Mode's eligible-strategy list and may be selected
  based on its opportunity score

#### Scenario: Score reflects setup quality
- **WHEN** a bar history shows a decline followed by an early bounce off the resulting low
- **THEN** the opportunity score for `dip_recovery` is higher than for a market that is
  either sideways or still actively falling with no bounce yet

### Requirement: Persisted runtime state
The system SHALL persist all Dip Recovery runtime state (lifecycle state, reference high,
tracked low, entry information, highest price since entry, exit levels, cooldown timer)
through the existing generic strategy-state persistence mechanism, so it survives a
service restart, deploy, or bot restart without resetting to empty state.

#### Scenario: Restart mid-TRACKING_DROP restores state
- **WHEN** a bot is checkpointed while in TRACKING_DROP and the process restarts
- **THEN** the restored state has the same reference high, tracked low, and tick counters
  as before the restart

#### Scenario: Restart mid-LONG_OPEN restores state
- **WHEN** a bot is checkpointed while in LONG_OPEN and the process restarts
- **THEN** the restored state has the same entry price, entry time, locked ATR, and
  trailing stop as before the restart, and the strategy continues managing the same
  position rather than resetting it

### Requirement: Structured decision explanation
The system SHALL report, for every evaluation, the current lifecycle state, the value
being monitored, the threshold required, and the distance to the next action, via the
existing `ExplanationBuilder` used by every other strategy.

#### Scenario: Explanation carries exact calculated values
- **WHEN** the strategy evaluates a tick in any lifecycle state
- **THEN** the recorded explanation's metrics contain the exact numeric values used in
  that tick's decision (e.g. current price, reference high, drawdown percent, drop
  threshold percent)
