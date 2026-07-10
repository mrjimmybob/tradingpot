## REMOVED Requirements

### Requirement: Funding Carry Strategy
**Reason**: This bot only trades spot; a real funding-carry trade requires a spot position plus an opposite
perpetual futures position collecting funding payments. Without the futures leg, `funding_carry` cannot
generate funding yield — it is an unmanaged directional long strategy mislabeled as a carry trade. It never
traded real capital in this environment (its only bot instance is dry-run test data).
**Migration**: Full deletion, by explicit operator decision — no compatibility is preserved. All
`funding_carry` bots, positions, orders, and trades are deleted from the database as part of this change
(they are disposable test data, not real trading history). No code path retains special-case handling for a
`funding_carry` value appearing in `strategy`/`strategy_used` fields.

The system previously allowed `funding_carry` to be selected as a bot strategy, configured via `GET
/strategies` and `POST /bots`, and chosen by Auto Mode.

#### Scenario: Strategy no longer selectable
- **WHEN** a client requests `GET /strategies`
- **THEN** the response does not include a `funding_carry` entry

#### Scenario: Bot creation rejects the strategy
- **WHEN** a client calls `POST /bots` with `"strategy": "funding_carry"`
- **THEN** the request is rejected with a validation error naming `funding_carry` as unknown

#### Scenario: Auto Mode never selects it
- **WHEN** Auto Mode scores eligible strategies for entry
- **THEN** `funding_carry` is not present in the strategy capability table and cannot be scored or selected

#### Scenario: No funding_carry data remains in the database
- **WHEN** the database is queried for bots, positions, orders, or trades referencing `funding_carry`
- **THEN** no rows are found; all such rows were deleted as part of this change
