"""Run the funding-rate diagnostic from the command line. READ-ONLY.

Uses the public exchange market-data API (no credentials required, same as
dry-run mode) to fetch historical perpetual funding rates and report whether a
funding-based edge plausibly exists net of trading costs. Places no trades.

Examples:
    python -m scripts.run_funding_diagnostic
    python -m scripts.run_funding_diagnostic --symbols BTC/USDT ETH/USDT
    python -m scripts.run_funding_diagnostic --limit 1000 --fee 0.1 --hold 3
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/run_funding_diagnostic.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.exchange import SimulatedExchangeService  # noqa: E402
from app.services.funding_diagnostic import FundingRateDiagnostic  # noqa: E402


async def _run(args: argparse.Namespace) -> int:
    exchange = SimulatedExchangeService()
    if not await exchange.connect():
        print("ERROR: could not connect to exchange public market-data API.")
        return 1

    if not exchange.supports_funding_rates():
        print(
            f"ERROR: exchange '{exchange.exchange_id}' does not expose funding-rate "
            "history; a funding strategy is not viable here."
        )
        await exchange.disconnect()
        return 2

    diagnostic = FundingRateDiagnostic(exchange, exchange_fee_pct=args.fee)

    try:
        for symbol in args.symbols:
            report = await diagnostic.analyze(
                symbol,
                limit=args.limit,
                assumed_holding_periods=args.hold,
            )
            print("=" * 72)
            if report is None:
                print(f"No funding-rate data available for {symbol}.")
            else:
                print(report.summary())
        print("=" * 72)
    finally:
        await exchange.disconnect()

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Funding-rate diagnostic (read-only).")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT", "ETH/USDT"],
        help="Spot symbols to analyse (the matching perpetual is queried).",
    )
    parser.add_argument(
        "--limit", type=int, default=500, help="Funding windows to retrieve per symbol."
    )
    parser.add_argument(
        "--hold",
        type=int,
        default=3,
        help="Assumed holding period in funding windows (for cost amortisation).",
    )
    parser.add_argument(
        "--fee", type=float, default=0.1, help="Per-side exchange fee in percent."
    )
    args = parser.parse_args()

    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
