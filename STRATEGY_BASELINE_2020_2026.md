# Strategy baseline: out-of-sample measurement, 2020–2026

The first objective, out-of-sample measurement of this project's strategies.
Produced by `add-strategy-validation-tooling` (Section 5). Every number here is
a **measurement of the shipped defaults**, not a recommendation, and **no
strategy parameter was changed as a result of producing it** — the tooling has
no code path that could.

- **Date run:** 2026-07-26 (re-run the same day with benchmark-relative measures,
  which closed the "three strategies unmeasurable" gap this document originally
  recorded as open — see *Benchmark-relative measurement* below)
- **Data:** `binance/BTCUSDT`, local CSV, 2020-01-01 → 2026-01-01
- **Windows:** 13 × 180 days, non-overlapping (step = window), each measured cold
- **Benchmarks:** buy-and-hold and periodic DCA (weekly cadence), both paying the
  same 0.1% per-side fee the strategies paid
- **Parameters:** none supplied, so every strategy fell back to its own internal
  defaults — the ones that had never been checked against data. Fingerprint
  `44136fa355b3678a` (the empty parameter dict) on every window of every run.
- **Execution model:** 0.1% per-side fee, no spread, no slippage
- **Reproduce:**

  ```bash
  cd backend
  python -m app.backtesting.validation.cli --exchange binance --symbol BTCUSDT \
      --timeframe 4h --strategy all --start 2020-01-01 --end 2026-01-01 \
      --window-days 180 --quiet
  ```

  Full raw reports (per-window tables, per-regime breakdowns, every limitation
  in full) are committed at
  `openspec/changes/archive/2026-07-26-add-strategy-validation-tooling/baseline-raw-4h.txt`
  and `…/baseline-raw-1d.txt`.

---

## What was measured

Two runs, at 4h and 1d resolution. Both are reported because the difference
between them is itself one of the findings.

### 4h resolution

| Strategy | Windows | Traded in | Closed trades | Consistency | Pooled expectancy |
|---|---|---|---|---|---|
| dca_accumulator | 13 | 0 | 0 | not assessable | — |
| adaptive_grid | 13 | 0 | 0 | not assessable | — |
| mean_reversion | 13 | 13 | 159 | **mixed** | +1.43 |
| trend_following | 13 | 12 | 364 | **mixed** | +24.43 |
| volatility_breakout | 13 | 13 | 71 | **mixed** | +6.44 |
| dip_recovery | 13 | 0 | 0 | not assessable | — |

Validated records produced (4h):

| Strategy | Expectancy | Win rate | Profit factor | Sample |
|---|---|---|---|---|
| mean_reversion | +1.43 | 45.3% | 1.08 | 159 closed trades |
| trend_following | +24.43 | 39.3% | 1.33 | 364 closed trades |
| volatility_breakout | +6.44 | 33.8% | 1.17 | 71 closed trades |

### 1d resolution

| Strategy | Windows | Traded in | Closed trades | Consistency | Pooled expectancy |
|---|---|---|---|---|---|
| dca_accumulator | 13 | 0 | 0 | not assessable | — |
| adaptive_grid | 13 | 0 | 0 | not assessable | — |
| mean_reversion | 13 | 12 | 24 | **mixed** | +71.30 |
| trend_following | 13 | 10 | 29 | **mixed** | −26.76 |
| volatility_breakout | 13 | 6 | 7 | **mixed** | −0.81 |
| dip_recovery | 13 | 0 | 0 | not assessable | — |

⚠️ `dip_recovery`'s zeros above are a **measurement artefact**, not a property of
the strategy — 4h and 1d are both too coarse for its 240-minute setup window. See
finding 5 for what it actually does at 1h (341 trades, −27.40 expectancy).

These tables are in declaration order and are **not** sorted by performance.
They compare what was measured; they do not rank the strategies, and the sample
sizes here would not support a ranking.

---

## Findings

### 1. No strategy's edge is stable across time

Every strategy that closed trades came back **mixed** at both resolutions:
per-window expectancy changes sign across the 13 out-of-sample windows. Not one
of them earned consistently across 2020–2026.

This is the finding the whole change was built to obtain. A single full-range
backtest reports one number and cannot distinguish an edge from a market phase.
Thirteen windows show that for all three trading strategies, the answer depends
on which window you ask. The pooled expectancy figures above are therefore
averages over states the strategies behave differently in, and describe none of
them individually.

### 2. The headline number depends heavily on the measurement timeframe

`trend_following` measures **−26.76 per trade at 1d and +24.43 at 4h** — it
changes sign. `mean_reversion` measures **+71.30 at 1d and +1.43 at 4h** — a
50× difference in magnitude, on the same data, same range, same parameters,
same windows.

Neither number is wrong; they measure different things, because the candle
timeframe determines when a strategy is allowed to make a decision. The 1d
figures rest on 24 and 29 trades respectively, which is below any reasonable
floor. The lesson for anyone reading a backtest of this codebase is that
"strategy X earns Y per trade" is not a property of the strategy alone, and any
such claim that omits its timeframe is unfalsifiable.

### 3. Three of six strategies cannot be measured by expectancy at all

`dca_accumulator`, `adaptive_grid`, and `dip_recovery` closed **zero round
trips** in all 13 windows at both resolutions, so expectancy per closed trade is
not a meaningful measure for them and the tooling refuses to produce a validated
record rather than printing a misleading zero.

Two separate reasons, which the closed-trade counter cannot tell apart but the
benchmark-relative measures below can: `dca_accumulator` never sells by design,
`adaptive_grid` scales in and out without ever fully flattening (the portfolio
records a closed trade only on a **full** close), and `dip_recovery` — as it
turns out — never opened a position at all.

**Update (same-day re-run):** these three are now measured on return and
drawdown against benchmarks. See the next section; this is no longer an open
gap.

### 4. Per-regime behaviour is legible, and mostly as designed

At 4h, bucketing by the canonical trend regime (`_strategy_auto`'s own
detector), expectancy per closed trade:

| Strategy | up (44.7% of candles) | flat (19.0%) | down (36.2%) |
|---|---|---|---|
| mean_reversion | +4.78 | +5.37 | **−3.82** |
| trend_following | +23.50 | +44.29 | **−6.20** |
| volatility_breakout | −0.04 | +4.72 | **+35.46** |

`mean_reversion` and `trend_following` both earn in rising and flat markets and
lose in falling ones — a long-biased profile, which is what these
implementations are. `volatility_breakout` inverts it: it is the only strategy
of the three that earned in downtrends, and it was flat-to-negative in uptrends.
Its 10 down-regime trades are far too few to call this an edge, but it is the
one place in this baseline where the per-regime split contradicts the aggregate
rather than merely decomposing it.

---

## Benchmark-relative measurement

Measured on the equity curve rather than on closed trades, so it is defined for
every strategy including the three that close none. `Expo` is realised exposure
(beta of equity returns to the asset's) — roughly the fraction of the time the
strategy was effectively deployed. `> B&H` / `> DCA` count the windows in which
the strategy's return exceeded that benchmark's.

### 4h resolution

| Strategy | Exposure | > buy-and-hold | > periodic DCA |
|---|---|---|---|
| dca_accumulator | 0.97 | 6/13 | 8/13 |
| adaptive_grid | 0.31 | 5/13 | 7/13 |
| mean_reversion | 0.01 | 4/13 | 6/13 |
| trend_following | 0.15 | 3/13 | 6/13 |
| volatility_breakout | 0.01 | 4/13 | 5/13 |
| dip_recovery | 0.00 | 4/13 | 5/13 |

### 1d resolution

| Strategy | Exposure | > buy-and-hold | > periodic DCA |
|---|---|---|---|
| dca_accumulator | 0.87 | 4/13 | 10/13 |
| adaptive_grid | 0.21 | 4/13 | 6/13 |
| mean_reversion | 0.01 | 4/13 | 5/13 |
| trend_following | 0.03 | 4/13 | 5/13 |
| volatility_breakout | 0.00 | 4/13 | 5/13 |
| dip_recovery | 0.00 | 4/13 | 5/13 |

### 5. `dip_recovery`'s zero was a measurement artefact — and it loses money

**Corrected.** An earlier version of this document reported that `dip_recovery`
"never opened a position" at 4h and 1d and flagged it as a possible defect in the
strategy. That conclusion was wrong, and the fault was in the measurement.

`dip_recovery` declares `setup_expiry_minutes: 240`: a decline arms a setup, and
the setup is abandoned if no reversal is confirmed within 240 minutes. A 4h
candle **is** 240 minutes, so every setup expired on the very next evaluation —
463 of 464 setups in a probe of 2022 died with "Setup expired after 240.0 min".
The strategy's time constants are written for the ~60s cadence the engine runs at
live; a backtest evaluates once per candle. Measured at 4h or 1d it could not
possibly trade.

Measured at **1h**, where the same setup window gets four evaluations instead of
one, the same code with the same parameters trades freely — and the real result
is worse than the artefact suggested:

| | 1h, 2020-2026, 13 × 180d windows |
|---|---|
| Closed trades | 341 (every window traded) |
| Expectancy per trade | **−27.40** |
| Win rate | 32.6% |
| Profit factor | **0.61** |
| Consistency | mixed — only **1 of 13** windows had positive expectancy |
| Beat buy-and-hold / periodic DCA | 2 of 13 / 4 of 13 |

So `dip_recovery` is not inert. With its shipped defaults it trades often and
loses persistently: 12 of 13 out-of-sample windows negative, and a profit factor
below 1 in almost all of them. That is a far more actionable finding than the
"never trades" artefact it replaced. **No parameter was changed** — acting on
this is a separate decision.

The tooling now refuses to report this class of zero silently: a **cadence check**
compares each strategy's declared duration parameters against the measurement's
inferred candle interval and prints a prominent warning before any results when a
mechanism cannot function at that resolution. The 4h and 1d figures elsewhere in
this document carry that warning for `dip_recovery`.

### 6. `dca_accumulator` behaves like buy-and-hold, not like an accumulator

Exposure 0.97–1.00 in every 4h window, and per-window returns within a few points
of buy-and-hold (−12.8% to +6.6%). Over a 180-day window it deploys essentially
fully and early, so it tracks the asset rather than averaging into it. It beats
the weekly-DCA benchmark in 8 of 13 windows (10 of 13 at 1d) largely *because* it
deploys faster than weekly instalments — an exposure difference, not evidence of
entry timing.

That is worth knowing about a strategy whose stated role is being the project's
clean, boring accumulation reference: over these window lengths it is closer to a
lump sum than to dollar-cost averaging.

### 7. `adaptive_grid` trades a large drawdown reduction for return

The one strategy whose profile the old instrument hid completely. Exposure
averages 0.31 and it draws down **far less than either benchmark in almost every
window** — −45.1, −35.1, −26.7, −24.8 percentage points of drawdown versus
buy-and-hold. It beats buy-and-hold on return in only 5 of 13 windows, but it
beats periodic DCA in 7, while being much less exposed than both.

Its worst window (2021-12 → 2022-06) still lost 56.4% with a 60.7% drawdown, so
this is risk *reduction*, not risk elimination.

### 8. Most strategies are barely in the market

Exposure is at or below 0.15 for `mean_reversion`, `trend_following`, and
`volatility_breakout` at both resolutions — `mean_reversion` sits at 0.01 despite
closing 159 trades. They hold for very short periods and sit in cash the rest of
the time.

This reframes finding 1. Their per-trade expectancies are not small because each
decision is poor; they are small in portfolio terms because the strategies are
almost never deployed. It also explains why all six strategies beat buy-and-hold
in roughly the same 4 of 13 windows: those are the down windows, where being in
cash wins regardless of what the strategy did.

---

## Limitations

These are stated in full in the raw reports; the material ones are:

- **13 windows is not many.** Six years of history at 180-day windows is what is
  available. Agreement or disagreement between 13 windows is suggestive, not
  statistically strong.
- **One symbol.** BTCUSDT only. Nothing here generalises to SOLUSDT or to any
  asset not measured.
- **Cold starts.** Each window includes its own indicator and regime warm-up, so
  the earliest bars of every window are effectively non-trading.
- **Modelled costs.** Fees are modelled at 0.1% per side; spread, slippage,
  funding and partial fills are not. Live results will differ.
- **Regime labels use the measured timeframe.** The live detector runs on
  60-second bars; labels here apply the same classifier at 4h/1d bar size.
- **Entry attribution.** A trade is attributed entirely to the regime at its
  entry; a trade held across a regime change is not split.
- **Small per-regime samples.** Every per-regime bucket above is under the
  30-trade floor except `trend_following`'s "up" (287) and "flat" (52).
- **Exposure is a proxy, not a measured split.** It is the beta of equity returns
  to the asset's. The portfolio records only total equity, so the true cash/asset
  split is not recoverable without changing the engine.
- **Excess return is not skill when exposure differs.** A strategy at 0.01
  exposure trailing buy-and-hold has told you almost nothing about its selection
  quality — mostly that it was not deployed.
- **The periodic-DCA benchmark has a cadence** (weekly here). A different cadence
  gives a different benchmark; it is disclosed rather than tuned.
- **A candle interval coarser than a strategy's time constants invalidates the
  measurement.** `dip_recovery` is the worked example (finding 5). The tooling now
  warns automatically, but the principle applies to any strategy measured at a
  timeframe near its own durations: these are ~60s-cadence strategies, and a
  backtest evaluates once per candle.
- **Closed-trade counts understate scale-out strategies.** The backtest portfolio
  records a closed trade only on a full close
  (`backend/app/backtesting/portfolio.py:65-73`), a documented simplification.
  Realised P&L from partial exits appears in the equity curve but not the trade
  count. Fixing it would change every existing measurement and is its own change.

---

## What was deliberately not done

No parameter was changed, tuned, searched, or written anywhere as a result of
this baseline. The measurement tooling has no code path that could do so, and a
guard test enforces that on every test run. Acting on these findings — including
the obvious temptation to adjust the strategies that lose in downtrends — is a
separate decision requiring its own change, and would need re-measuring
afterwards against windows these results did not inform.

The validated records above are also **not** wired into runtime. Every live
`StrategyProposal.expected_edge_estimate` remains `None`; connecting them is a
separate change.
