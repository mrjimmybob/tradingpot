"""Is the measurement's candle interval fine enough for the strategy?

The regression these tests exist for: `dip_recovery` declares
`setup_expiry_minutes: 240`, and measured on 4h candles - exactly 240 minutes -
every setup expired on the next evaluation, so it recorded zero trades across
six years. The same code with the same parameters opened 127 positions when
measured on 1h candles. The zero was an artefact of the timeframe and was
initially read as a defect in the strategy.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List

import pytest

from app.backtesting.candle import Candle
from app.backtesting.validation.cadence import (
    MS_PER_MINUTE,
    CadenceWarning,
    check_cadence,
    declared_parameters,
    duration_parameters,
    format_cadence_warning,
    infer_candle_interval_ms,
)

START_MS = 1704067200000
_TRADING_ENGINE = Path(__file__).resolve().parents[1] / "app" / "services" / "trading_engine.py"


def _candles(count: int, step_ms: int) -> List[Candle]:
    return [
        Candle(
            timestamp=START_MS + i * step_ms, datetime="d", symbol="TESTUSD",
            open=100.0, high=100.0, low=100.0, close=100.0,
            base_volume=1.0, quote_volume=100.0, trade_count=1,
        )
        for i in range(count)
    ]


class TestCandleIntervalInference:
    def test_a_uniform_series_reports_its_step(self):
        assert infer_candle_interval_ms(_candles(50, 4 * 3_600_000)) == 4 * 3_600_000

    def test_a_data_gap_does_not_become_the_interval(self):
        """Taking the first gap, or the mean, would let one gap misreport the
        interval and suppress a warning that should have fired."""
        candles = _candles(50, 3_600_000)
        tail = _candles(5, 3_600_000)
        shifted = [
            Candle(**{**c.__dict__, "timestamp": c.timestamp + 30 * 86_400_000})
            for c in tail
        ]
        assert infer_candle_interval_ms(candles + shifted) == 3_600_000

    def test_too_short_a_series_has_no_interval(self):
        assert infer_candle_interval_ms([]) is None
        assert infer_candle_interval_ms(_candles(1, 60_000)) is None


class TestDurationParameters:
    def test_duration_suffixes_are_recognised_and_converted(self):
        durations = duration_parameters("dip_recovery")
        assert durations["setup_expiry_minutes"][0] == 240 * MS_PER_MINUTE
        assert durations["cooldown_seconds"][0] == 300 * 1_000

    def test_non_duration_parameters_are_ignored(self):
        durations = duration_parameters("dip_recovery")
        assert "atr_period" not in durations
        assert "min_drop_percent" not in durations

    def test_operator_values_override_declared_defaults(self):
        """The check must apply to what was actually measured."""
        durations = duration_parameters("dip_recovery", {"setup_expiry_minutes": 30})
        assert durations["setup_expiry_minutes"][0] == 30 * MS_PER_MINUTE

    def test_absent_or_nonsensical_values_are_skipped(self):
        durations = duration_parameters("dip_recovery", {"setup_expiry_minutes": 0})
        assert "setup_expiry_minutes" not in durations
        durations = duration_parameters("dip_recovery", {"setup_expiry_minutes": None})
        assert "setup_expiry_minutes" not in durations

    def test_an_unknown_strategy_has_no_declared_parameters(self):
        assert declared_parameters("no_such_strategy") == {}
        assert duration_parameters("no_such_strategy") == {}


class TestCadenceCheck:
    def test_the_dip_recovery_regression_is_caught(self):
        """setup_expiry_minutes (240) at a 4h candle interval (240 min) - the
        exact configuration that produced six years of silent zeros."""
        warnings = check_cadence("dip_recovery", _candles(100, 4 * 3_600_000))
        by_name = {w.parameter: w for w in warnings}

        assert "setup_expiry_minutes" in by_name
        expiry = by_name["setup_expiry_minutes"]
        assert expiry.is_fatal
        assert expiry.evaluations_per_period == pytest.approx(1.0)

    def test_a_fine_enough_timeframe_produces_no_warning_for_that_parameter(self):
        """At 1m candles the same parameter gets 240 evaluations."""
        warnings = check_cadence("dip_recovery", _candles(100, 60_000))
        assert "setup_expiry_minutes" not in {w.parameter for w in warnings}

    def test_an_hourly_timeframe_no_longer_makes_the_setup_fatal(self):
        """1h is where dip_recovery actually trades - four evaluations per
        setup window rather than one."""
        warnings = check_cadence("dip_recovery", _candles(100, 3_600_000))
        expiry = next(
            (w for w in warnings if w.parameter == "setup_expiry_minutes"), None
        )
        assert expiry is None or not expiry.is_fatal

    def test_warnings_are_ordered_worst_first(self):
        warnings = check_cadence("dip_recovery", _candles(100, 4 * 3_600_000))
        ratios = [w.evaluations_per_period for w in warnings]
        assert ratios == sorted(ratios)

    def test_operator_parameters_are_honoured_by_the_check(self):
        """Widening the expiry should clear its warning - the check reflects
        what was measured, not the schema default."""
        warnings = check_cadence(
            "dip_recovery", _candles(100, 4 * 3_600_000),
            {"setup_expiry_minutes": 10_000},
        )
        assert "setup_expiry_minutes" not in {w.parameter for w in warnings}

    def test_a_series_with_no_inferable_interval_warns_about_nothing(self):
        assert check_cadence("dip_recovery", _candles(1, 60_000)) == ()

    def test_a_strategy_with_no_declared_durations_is_never_flagged(self):
        assert check_cadence("no_such_strategy", _candles(100, 86_400_000)) == ()


class TestWarningReport:
    def _warnings(self, interval_ms=4 * 3_600_000):
        return check_cadence("dip_recovery", _candles(100, interval_ms))

    def test_a_sound_measurement_prints_nothing_at_all(self):
        assert format_cadence_warning("dip_recovery", (), num_trades=5) == ""

    def test_zero_trades_plus_a_fatal_constant_says_do_not_read_the_zero(self):
        report = format_cadence_warning("dip_recovery", self._warnings(), num_trades=0)
        assert "CADENCE WARNING" in report
        assert "Do NOT read the zero as evidence" in report
        assert "re-measure on a finer timeframe" in report

    def test_a_fatal_constant_with_trades_still_qualifies_the_result(self):
        report = format_cadence_warning("dip_recovery", self._warnings(), num_trades=9)
        assert "CADENCE WARNING" in report
        assert "Do NOT read the zero" not in report
        assert "do not describe how the strategy would behave live" in report

    def test_the_offending_parameters_are_named_with_their_descriptions(self):
        report = format_cadence_warning("dip_recovery", self._warnings(), num_trades=0)
        assert "setup_expiry_minutes" in report
        assert "Abandon an unresolved decline/reversal setup" in report

    def test_a_merely_coarse_constant_reads_as_understated_not_broken(self):
        warning = CadenceWarning(
            parameter="max_position_duration_minutes", value_ms=720 * MS_PER_MINUTE,
            candle_interval_ms=240 * MS_PER_MINUTE, description="force exit",
        )
        assert not warning.is_fatal
        report = format_cadence_warning("dip_recovery", (warning,), num_trades=3)
        assert "understate" in report


class TestDeclaredDefaultsMatchTheCode:
    """The check is only as good as the schema it reads.

    `config.py` declares the parameter defaults; the strategies read them with
    `params.get(name, default)`. If those two drift, this check would warn about
    a constant the strategy does not actually use - or stay silent about one it
    does. This test compares them directly.
    """

    # Drift that exists today, is NOT caused by this change, and needs a
    # behaviour decision rather than a silent edit:
    #
    #   volatility_breakout.cooldown_hours — config.py declares 72 ("SPARSE:
    #   72 = 3 days"), trading_engine.py:4694 reads `params.get("cooldown_hours",
    #   24)` and its docstring says 24. The Strategy Decision Framework audit
    #   (archive/2026-07-25-add-strategy-decision-framework/audits/
    #   volatility_breakout.md:142) records the constant as 24.
    #
    # A bot created through the UI therefore gets 72 while a bot created with
    # empty parameters gets 24. Aligning either side changes live behaviour, so
    # it is recorded here rather than decided here. Remove this entry when the
    # two are reconciled.
    KNOWN_DRIFT = {"volatility_breakout.cooldown_hours"}

    def _code_defaults(self):
        source = _TRADING_ENGINE.read_text(encoding="utf-8")
        found = {}
        for name, default in re.findall(
            r'params\.get\(\s*"([a-z0-9_]+)"\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)', source
        ):
            found.setdefault(name, set()).add(float(default))
        return found

    def test_declared_duration_defaults_match_the_strategy_code(self):
        from app.routers.config import STRATEGIES

        code_defaults = self._code_defaults()
        mismatches = []
        for info in STRATEGIES:
            for name, spec in info.parameters.items():
                if not any(
                    name.endswith(s) for s in ("_seconds", "_minutes", "_hours")
                ):
                    continue
                declared = spec.get("default")
                if not isinstance(declared, (int, float)):
                    continue
                seen = code_defaults.get(name)
                if not seen:
                    continue  # not read with a literal default; nothing to compare
                if f"{info.name}.{name}" in self.KNOWN_DRIFT:
                    continue
                if float(declared) not in seen:
                    mismatches.append(
                        f"{info.name}.{name}: config.py={declared} code={sorted(seen)}"
                    )
        assert not mismatches, (
            "Declared duration defaults have drifted from the strategy code, so the "
            f"cadence check would describe parameters nobody uses: {mismatches}"
        )

    def test_the_known_drift_still_exists_so_the_exemption_is_not_stale(self):
        """An allowlist entry that no longer describes reality quietly weakens
        the guard. If this fails, the drift was fixed - delete the entry."""
        from app.routers.config import STRATEGIES

        declared = next(
            i for i in STRATEGIES if i.name == "volatility_breakout"
        ).parameters["cooldown_hours"]["default"]
        assert declared == 72
        assert 24.0 in self._code_defaults()["cooldown_hours"]

    def test_the_dip_recovery_constant_behind_the_regression_is_pinned(self):
        """240 minutes is what made a 4h measurement degenerate. If this ever
        changes, the regression test above needs revisiting with it."""
        assert declared_parameters("dip_recovery")["setup_expiry_minutes"]["default"] == 240
