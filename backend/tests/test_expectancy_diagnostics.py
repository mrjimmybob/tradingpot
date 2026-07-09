"""Tests for closed-trade expectancy diagnostics (Finding 4).

Covers two things:
1. ``compute_expectancy`` computes closed-trade count, win rate, average
   win, average loss, and expectancy correctly (formula:
   expectancy = win_rate*avg_win + loss_rate*avg_loss, avg_loss signed <= 0).
2. ``find_expectancy_red_flags`` detects structurally IMPOSSIBLE patterns
   (a take-profit exit losing money on average, sign errors, a formula
   mismatch) without ever asserting the strategy must be profitable —
   a clean, negative-expectancy-but-internally-consistent dataset must
   produce zero flags.
"""
from __future__ import annotations

import pytest

from app.services.expectancy import (
    ClosedTrade,
    ExpectancyReport,
    ReasonStats,
    compute_expectancy,
    find_expectancy_red_flags,
)


# ---------------------------------------------------------------------------
# compute_expectancy
# ---------------------------------------------------------------------------

class TestComputeExpectancy:
    def test_empty_trades(self):
        report = compute_expectancy([])
        assert report.n == 0
        assert report.win_rate == 0.0
        assert report.loss_rate == 0.0
        assert report.avg_win == 0.0
        assert report.avg_loss == 0.0
        assert report.expectancy == 0.0

    def test_matches_forensic_dataset_shape(self):
        """Mirrors the real closed-trade shape from the forensic audit:
        16% win rate, small wins, larger losses -> negative expectancy.
        """
        trades = (
            [ClosedTrade(pnl=0.0225, exit_reason="mean reached")] * 10
            + [ClosedTrade(pnl=-0.047, exit_reason="mean reached")] * 36
            + [ClosedTrade(pnl=-0.076, exit_reason="Hard stop")] * 6
        )
        report = compute_expectancy(trades)

        assert report.n == 52
        assert report.wins == 10
        assert report.losses == 42
        assert report.win_rate == pytest.approx(10 / 52)
        assert report.avg_win == pytest.approx(0.0225)
        # weighted avg of the two loss groups
        expected_avg_loss = (36 * -0.047 + 6 * -0.076) / 42
        assert report.avg_loss == pytest.approx(expected_avg_loss)
        assert report.expectancy == pytest.approx(
            report.win_rate * report.avg_win + report.loss_rate * report.avg_loss
        )
        assert report.expectancy < 0  # this dataset genuinely is negative EV

    def test_breakeven_trades_counted_but_not_averaged_in(self):
        trades = [
            ClosedTrade(pnl=1.0), ClosedTrade(pnl=-1.0), ClosedTrade(pnl=0.0),
        ]
        report = compute_expectancy(trades)
        assert report.n == 3
        assert report.breakeven == 1
        assert report.avg_win == pytest.approx(1.0)
        assert report.avg_loss == pytest.approx(-1.0)

    def test_by_reason_breakdown(self):
        trades = [
            ClosedTrade(pnl=5.0, exit_reason="take profit"),
            ClosedTrade(pnl=-2.0, exit_reason="take profit"),
            ClosedTrade(pnl=-3.0, exit_reason="stop loss"),
        ]
        report = compute_expectancy(trades)
        assert set(report.by_reason.keys()) == {"take profit", "stop loss"}
        tp = report.by_reason["take profit"]
        assert tp.n == 2 and tp.wins == 1 and tp.losses == 1
        assert tp.avg_pnl == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# find_expectancy_red_flags
# ---------------------------------------------------------------------------

class TestFindExpectancyRedFlags:
    def test_clean_negative_expectancy_dataset_raises_no_flags(self):
        """A strategy can legitimately lose money (low win rate, wins/losses
        both correctly signed, target-labeled exits genuinely profitable on
        average) without any structural bug. Markets vary — must not flag.
        """
        trades = (
            [ClosedTrade(pnl=0.5, exit_reason="take profit")] * 3
            + [ClosedTrade(pnl=-1.0, exit_reason="stop loss")] * 7
        )
        report = compute_expectancy(trades)
        assert report.expectancy < 0  # genuinely losing...
        flags = find_expectancy_red_flags(report)
        assert flags == []  # ...but nothing here is IMPOSSIBLE

    def test_flags_take_profit_exit_with_negative_average(self):
        """The exact bug shape from Finding 1: an exit reason containing
        'mean reached' (a take-profit label) that loses money on average.
        """
        trades = (
            [ClosedTrade(pnl=0.02, exit_reason="mean reached")] * 11
            + [ClosedTrade(pnl=-0.05, exit_reason="mean reached")] * 36
        )
        report = compute_expectancy(trades)
        flags = find_expectancy_red_flags(report)
        assert any("mean reached" in f and "negative average" in f for f in flags)

    def test_does_not_flag_non_take_profit_reasons_for_losing(self):
        """A stop-loss exit losing money on average is expected behavior,
        not a structural bug — must not be flagged."""
        trades = [ClosedTrade(pnl=-1.0, exit_reason="Hard stop")] * 5
        report = compute_expectancy(trades)
        flags = find_expectancy_red_flags(report)
        assert flags == []

    def test_flags_sign_corrupted_avg_win(self):
        report = ExpectancyReport(
            n=2, wins=1, losses=1, breakeven=0,
            win_rate=0.5, loss_rate=0.5,
            avg_win=-1.0, avg_loss=-2.0,  # corrupted: a "win" is negative
            expectancy=0.5 * -1.0 + 0.5 * -2.0,
            total_pnl=-3.0,
        )
        flags = find_expectancy_red_flags(report)
        assert any("avg_win is negative" in f for f in flags)

    def test_flags_sign_corrupted_avg_loss(self):
        report = ExpectancyReport(
            n=2, wins=1, losses=1, breakeven=0,
            win_rate=0.5, loss_rate=0.5,
            avg_win=1.0, avg_loss=2.0,  # corrupted: a "loss" is positive
            expectancy=0.5 * 1.0 + 0.5 * 2.0,
            total_pnl=3.0,
        )
        flags = find_expectancy_red_flags(report)
        assert any("avg_loss is positive" in f for f in flags)

    def test_flags_rate_sum_exceeding_one(self):
        report = ExpectancyReport(
            n=10, wins=7, losses=6, breakeven=0,
            win_rate=0.7, loss_rate=0.6,  # impossible: sums to 1.3
            avg_win=1.0, avg_loss=-1.0,
            expectancy=0.7 * 1.0 + 0.6 * -1.0,
            total_pnl=0.0,
        )
        flags = find_expectancy_red_flags(report)
        assert any("> 1.0" in f for f in flags)

    def test_flags_expectancy_formula_mismatch(self):
        report = ExpectancyReport(
            n=10, wins=5, losses=5, breakeven=0,
            win_rate=0.5, loss_rate=0.5,
            avg_win=1.0, avg_loss=-1.0,
            expectancy=999.0,  # does not match win_rate*avg_win + loss_rate*avg_loss (0.0)
            total_pnl=0.0,
        )
        flags = find_expectancy_red_flags(report)
        assert any("does not match its own formula" in f for f in flags)
