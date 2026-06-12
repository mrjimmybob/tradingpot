"""Funding-rate diagnostic — READ-ONLY analysis. NEVER places trades.

Purpose: validate whether a funding-based strategy is likely to have a real
edge on the target exchange, by analysing historical perpetual funding rates
net of realistic round-trip trading costs.

Design constraints (consistent with the rest of the codebase):
- Read-only: fetches public funding-rate history; places no orders.
- Reuses ExchangeService for data access and ExecutionCostModel for cost
  estimation (RISK-only modelling, no alpha).
- Pure statistics are separated from I/O so they can be unit-tested in
  isolation and reused by the funding_carry strategy.
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from app.services.execution_cost_model import ExecutionCostModel, get_cost_model

logger = logging.getLogger(__name__)


@dataclass
class FundingStats:
    """Pure statistics over a series of per-interval funding rates.

    Rates are per-interval fractions (e.g. 0.0001 = 0.01% per interval).
    Percentage fields are expressed in percent (0.01 == 0.01%).
    """
    count: int
    interval_hours: float
    periods_per_year: float

    mean_rate: float
    median_rate: float
    min_rate: float
    max_rate: float
    stdev_rate: float

    positive_pct: float          # share of windows with rate > 0
    negative_pct: float          # share of windows with rate < 0

    mean_per_period_pct: float   # mean rate, in percent
    annualized_mean_pct: float   # mean rate compounded-free * periods/year, percent


def compute_funding_stats(rates: List[float], interval_hours: float = 8.0) -> FundingStats:
    """Compute summary statistics for a series of funding rates.

    Args:
        rates: Per-interval funding rates as fractions (oldest-first or any order).
        interval_hours: Hours between funding settlements (default 8h).

    Returns:
        FundingStats. For an empty series, all metrics are 0.
    """
    interval_hours = interval_hours if interval_hours and interval_hours > 0 else 8.0
    periods_per_year = (24.0 / interval_hours) * 365.0

    n = len(rates)
    if n == 0:
        return FundingStats(
            count=0, interval_hours=interval_hours, periods_per_year=periods_per_year,
            mean_rate=0.0, median_rate=0.0, min_rate=0.0, max_rate=0.0, stdev_rate=0.0,
            positive_pct=0.0, negative_pct=0.0,
            mean_per_period_pct=0.0, annualized_mean_pct=0.0,
        )

    mean_rate = sum(rates) / n
    median_rate = statistics.median(rates)
    min_rate = min(rates)
    max_rate = max(rates)
    stdev_rate = statistics.pstdev(rates) if n > 1 else 0.0

    positives = sum(1 for r in rates if r > 0)
    negatives = sum(1 for r in rates if r < 0)

    return FundingStats(
        count=n,
        interval_hours=interval_hours,
        periods_per_year=periods_per_year,
        mean_rate=mean_rate,
        median_rate=median_rate,
        min_rate=min_rate,
        max_rate=max_rate,
        stdev_rate=stdev_rate,
        positive_pct=(positives / n) * 100.0,
        negative_pct=(negatives / n) * 100.0,
        mean_per_period_pct=mean_rate * 100.0,
        annualized_mean_pct=mean_rate * periods_per_year * 100.0,
    )


@dataclass
class FundingDiagnosticReport:
    """Result of analysing funding rates for one symbol, net of costs."""
    symbol: str
    swap_symbol: str
    stats: FundingStats

    roundtrip_cost_pct: float          # one-time entry+exit cost, percent of notional
    assumed_holding_periods: int       # how many funding windows a position is held
    breakeven_periods: float           # periods of mean funding to cover round-trip cost

    gross_funding_pct: float           # mean funding * holding_periods, percent
    net_funding_pct: float             # gross funding minus round-trip cost, percent
    profitable_window_pct: float       # share of windows beating amortised cost

    best_period_rate: float
    best_period_time: Optional[datetime]
    worst_period_rate: float
    worst_period_time: Optional[datetime]

    viable: bool
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        s = self.stats
        lines = [
            f"Funding diagnostic for {self.symbol} (swap {self.swap_symbol})",
            f"  Samples              : {s.count} funding windows "
            f"({s.interval_hours:g}h each)",
            f"  Average funding      : {s.mean_per_period_pct:.5f}% / period "
            f"({s.annualized_mean_pct:.2f}% annualized)",
            f"  Median / Min / Max   : {s.median_rate * 100:.5f}% / "
            f"{s.min_rate * 100:.5f}% / {s.max_rate * 100:.5f}%",
            f"  Std dev              : {s.stdev_rate * 100:.5f}%",
            f"  Distribution         : {s.positive_pct:.1f}% positive, "
            f"{s.negative_pct:.1f}% negative",
            f"  Round-trip cost      : {self.roundtrip_cost_pct:.5f}% of notional",
            f"  Breakeven            : {self.breakeven_periods:.1f} periods of mean funding",
            f"  Net over {self.assumed_holding_periods:>2} periods : "
            f"{self.net_funding_pct:.5f}% (gross {self.gross_funding_pct:.5f}%)",
            f"  Profitable windows   : {self.profitable_window_pct:.1f}% beat amortised cost",
            f"  Verdict              : {'LIKELY EDGE' if self.viable else 'NO CLEAR EDGE'}",
        ]
        for note in self.notes:
            lines.append(f"  Note                 : {note}")
        return "\n".join(lines)


class FundingRateDiagnostic:
    """Read-only analyser for perpetual funding rates.

    Reuses an ExchangeService for data access and an ExecutionCostModel for
    realistic cost estimation. Places no trades.
    """

    def __init__(
        self,
        exchange,
        cost_model: Optional[ExecutionCostModel] = None,
        exchange_fee_pct: float = 0.1,
        market_spread_pct: float = 0.0,
        slippage_pct: float = 0.0,
    ):
        """Initialise the diagnostic.

        Args:
            exchange: A connected ExchangeService (or SimulatedExchangeService).
            cost_model: Optional pre-built cost model. If omitted, one is built
                from the fee/spread/slippage arguments.
            exchange_fee_pct: Per-side exchange fee, percent (default 0.1).
            market_spread_pct: Typical spread, percent (default 0.0).
            slippage_pct: Expected slippage, percent (default 0.0).
        """
        self.exchange = exchange
        self.cost_model = cost_model or get_cost_model(
            exchange_fee_pct=exchange_fee_pct,
            market_spread_pct=market_spread_pct,
            slippage_pct=slippage_pct,
        )

    async def analyze(
        self,
        spot_symbol: str,
        limit: int = 500,
        assumed_holding_periods: int = 3,
        notional_usd: float = 1000.0,
    ) -> Optional[FundingDiagnosticReport]:
        """Analyse funding rates for a spot symbol's perpetual.

        Args:
            spot_symbol: Spot pair (e.g. 'BTC/USDT'); the matching swap is queried.
            limit: Number of funding windows to retrieve.
            assumed_holding_periods: Funding windows a position is assumed to be
                held, used to amortise the one-time round-trip cost.
            notional_usd: Notional used for cost estimation (cancels out as a %).

        Returns:
            FundingDiagnosticReport, or None if no funding data is available.
        """
        swap_symbol = self.exchange.to_swap_symbol(spot_symbol)
        history = await self.exchange.get_funding_rate_history(swap_symbol, limit=limit)

        if not history:
            logger.warning(
                f"No funding-rate history available for {swap_symbol}; "
                "the exchange may not list this perpetual or support funding history."
            )
            return None

        interval_hours = history[-1].interval_hours
        rates = [h.funding_rate for h in history]
        stats = compute_funding_stats(rates, interval_hours)

        # Round-trip cost as a percentage of notional (price cancels out).
        roundtrip_cost_usd = self.cost_model.estimate_roundtrip_cost(notional_usd, 1.0)
        roundtrip_cost_pct = (
            (roundtrip_cost_usd / notional_usd) * 100.0 if notional_usd > 0 else 0.0
        )

        holding = max(1, assumed_holding_periods)
        gross_funding_pct = stats.mean_per_period_pct * holding
        net_funding_pct = gross_funding_pct - roundtrip_cost_pct

        # Amortised per-window cost: a window "pays" if its funding magnitude
        # exceeds the share of round-trip cost attributed to it.
        amortised_cost_per_window = (
            (roundtrip_cost_pct / holding) / 100.0 if holding > 0 else 0.0
        )
        profitable = sum(1 for r in rates if abs(r) > amortised_cost_per_window)
        profitable_window_pct = (profitable / stats.count) * 100.0

        mean_abs = abs(stats.mean_rate)
        breakeven_periods = (
            (roundtrip_cost_pct / 100.0) / mean_abs if mean_abs > 0 else float("inf")
        )

        best = max(history, key=lambda h: h.funding_rate)
        worst = min(history, key=lambda h: h.funding_rate)

        notes: List[str] = []
        if stats.count < holding:
            notes.append(
                f"Only {stats.count} windows available (< {holding} assumed holding)."
            )
        if breakeven_periods == float("inf"):
            notes.append("Mean funding is ~0; no directional carry signal.")

        viable = net_funding_pct > 0 and stats.count >= holding

        return FundingDiagnosticReport(
            symbol=spot_symbol,
            swap_symbol=swap_symbol,
            stats=stats,
            roundtrip_cost_pct=roundtrip_cost_pct,
            assumed_holding_periods=holding,
            breakeven_periods=breakeven_periods,
            gross_funding_pct=gross_funding_pct,
            net_funding_pct=net_funding_pct,
            profitable_window_pct=profitable_window_pct,
            best_period_rate=best.funding_rate,
            best_period_time=best.timestamp,
            worst_period_rate=worst.funding_rate,
            worst_period_time=worst.timestamp,
            viable=viable,
            notes=notes,
        )
