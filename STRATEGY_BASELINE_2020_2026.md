# Strategy baseline: out-of-sample measurement, 2020–2026

The first objective, out-of-sample measurement of this project's strategies.
Produced by `add-strategy-validation-tooling` (Section 5). Every number here is
a **measurement of the shipped defaults**, not a recommendation, and **no
strategy parameter was changed as a result of producing it** — the tooling has
no code path that could.

- **Date run:** 2026-07-26
- **Data:** `binance/BTCUSDT`, local CSV, 2020-01-01 → 2026-01-01
- **Windows:** 13 × 180 days, non-overlapping (step = window), each measured cold
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
  `openspec/changes/add-strategy-validation-tooling/baseline-raw-4h.txt` and
  `…/baseline-raw-1d.txt`.

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
trips** in all 13 windows at both resolutions. They were not inactive — their
per-window Return% and MaxDD% move substantially and diverge from buy-and-hold
(`adaptive_grid` returned −1.64% in a window where buy-and-hold returned
+26.89%). They scale in and out without ever fully flattening a position, so the
closed-round-trip counter never increments.

Expectancy per closed trade is simply not a meaningful measure for these
strategies. The tooling refuses to produce a validated record for them rather
than printing a misleading zero. Judging them requires return- and
drawdown-based measures against a stated benchmark — which this tooling does not
yet provide, and which is a gap worth closing.

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
