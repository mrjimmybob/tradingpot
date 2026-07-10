"""Historical backtesting foundation.

Replays historical candle data through the exact strategy decision code used
in live trading (TradingEngine._get_strategy_executor) with a realistic,
no-lookahead execution model. See openspec/changes/add-historical-backtesting/
for the design.

This package does not change, tune, or validate any strategy's profitability -
it only makes profitability measurable.
"""
