"""P0 regression tests: a bot's CONFIGURED strategy is immutable.

Background (the data-integrity failure these tests lock down):
  The trading loop ran ``full_risk_check`` on every bot. On a 3-loss streak
  ``check_consecutive_losses`` returned ROTATE_STRATEGY, and the loop called
  ``RiskManagementService.rotate_strategy`` which OVERWROTE ``Bot.strategy`` for
  every non-auto_mode bot - cycling it through ``_ALPHA_STRATEGIES``. After
  ``max_strategy_rotations`` (default 3) the bot paused at "Max strategy
  rotations reached (3)". This silently corrupted fixed-strategy bots:
      dca_accumulator -> ... -> trend_following
      mean_reversion  -> ... -> funding_carry
      trend_following -> ... -> dca_accumulator
  (each = 3 hops through _ALPHA_STRATEGIES).

The fix:
  * check_consecutive_losses PAUSES fixed-strategy bots (never rotates them) and
    leaves auto_mode to CONTINUE (it self-manages in-memory).
  * rotate_strategy REJECTS every request and never writes Bot.strategy.
  * repair_corrupted_strategies restores the original strategy (earliest
    rotation's from_strategy) and clears the rotation history.

These tests drive the real service against an in-memory SQLite session.
"""
import pytest
from datetime import datetime, timedelta

from app.models import Bot, BotStatus, Order
from app.models.order import OrderType, OrderStatus
from app.models.strategy_rotation import StrategyRotation
from app.models.tax_lot import RealizedGain
from app.services.risk_management import RiskManagementService, RiskAction
from app.services.trading_engine import TradingEngine, _ALPHA_STRATEGIES


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _make_bot(session, strategy, name="bot", max_rotations=3):
    bot = Bot(
        name=name, trading_pair="BTC/USDT", strategy=strategy,
        strategy_params={}, budget=1000.0, current_balance=1000.0,
        max_strategy_rotations=max_rotations, is_dry_run=True,
        status=BotStatus.RUNNING,
    )
    session.add(bot)
    await session.commit()
    await session.refresh(bot)
    return bot


async def _add_losing_realized_gains(session, bot_id, count=3, gain_loss=-10.0):
    """Insert ``count`` RealizedGain rows with negative gain_loss so that
    _count_consecutive_losses reports ``count`` consecutive losses."""
    base = datetime.utcnow()
    for i in range(count):
        session.add(RealizedGain(
            owner_id=str(bot_id),
            asset="BTC",
            quantity=0.001,
            proceeds=9.99,
            cost_basis=10.02,
            gain_loss=gain_loss,
            holding_period_days=0,
            is_long_term=False,
            purchase_trade_id=i + 1,
            sell_trade_id=i + 1000,
            tax_lot_id=i + 1,
            purchase_date=base + timedelta(seconds=i),
            sell_date=base + timedelta(seconds=i + 1),
        ))
    await session.commit()


async def _add_profitable_realized_gain(session, bot_id, gain_loss=5.0):
    """Insert one RealizedGain row with positive gain_loss (a winning trade)."""
    base = datetime.utcnow()
    session.add(RealizedGain(
        owner_id=str(bot_id),
        asset="BTC",
        quantity=0.001,
        proceeds=10.05,
        cost_basis=10.00,
        gain_loss=gain_loss,
        holding_period_days=0,
        is_long_term=False,
        purchase_trade_id=9001,
        sell_trade_id=9002,
        tax_lot_id=9001,
        purchase_date=base,
        sell_date=base + timedelta(seconds=1),
    ))
    await session.commit()


async def _add_rotation_chain(session, bot_id, chain, reason="Consecutive losses"):
    """Record a from->to rotation chain, e.g. ['a','b','c'] => a->b, b->c."""
    base = datetime.utcnow()
    for i in range(len(chain) - 1):
        session.add(StrategyRotation(
            bot_id=bot_id, from_strategy=chain[i], to_strategy=chain[i + 1],
            reason=reason, created_at=base + timedelta(seconds=i),
        ))
    await session.commit()


async def _rotation_count(session, bot_id):
    from sqlalchemy import select, func
    res = await session.execute(
        select(func.count(StrategyRotation.id)).where(StrategyRotation.bot_id == bot_id)
    )
    return res.scalar() or 0


# --------------------------------------------------------------------------- #
# 1 & 2: Fixed-strategy bots never change strategy / never increment rotations
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_consecutive_losses_enters_recovery_without_corrupting_fixed_bot(test_db):
    """Three consecutive losses enter RECOVERY_MODE (not PAUSE) without
    corrupting Bot.strategy or creating rotation rows."""
    bot = await _make_bot(test_db, "dca_accumulator")
    await _add_losing_realized_gains(test_db, bot.id, count=3)

    svc = RiskManagementService(test_db)
    count, result = await svc.check_consecutive_losses(bot.id, threshold=3)

    assert count == 3
    assert result.action == RiskAction.ENTER_RECOVERY_MODE
    assert "recovery mode" in result.reason.lower()
    # Strategy untouched, and NO rotation row created.
    await test_db.refresh(bot)
    assert bot.strategy == "dca_accumulator"
    assert await _rotation_count(test_db, bot.id) == 0


@pytest.mark.asyncio
async def test_rotate_strategy_rejected_for_every_fixed_strategy(test_db):
    """Every fixed alpha strategy is immutable - rotate_strategy rejects each and
    writes nothing."""
    svc = RiskManagementService(test_db)
    for strat in _ALPHA_STRATEGIES:
        bot = await _make_bot(test_db, strat, name=f"bot-{strat}")
        ok, msg = await svc.rotate_strategy(bot.id, "mean_reversion", "forced")
        assert ok is False
        assert "immutable" in msg.lower()
        await test_db.refresh(bot)
        assert bot.strategy == strat
        assert await _rotation_count(test_db, bot.id) == 0


# --------------------------------------------------------------------------- #
# 3: Auto Mode rotates internally without modifying the configured strategy
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_auto_mode_internal_switch_does_not_touch_bot_strategy(test_db):
    bot = await _make_bot(test_db, "auto_mode")
    engine = TradingEngine()
    auto_state = {}
    eligible = [
        ("trend_following", 90, "strong trend"),
        ("mean_reversion", 10, "range"),
    ]
    selected, should_switch, _ = engine._select_strategy_from_eligible(
        eligible, current_strategy="mean_reversion",
        auto_state=auto_state, min_switch_interval=0,
    )
    # The auto selector picks a runtime strategy in-memory...
    assert should_switch is True
    assert selected == "trend_following"
    # ...but the persisted configured strategy is still auto_mode.
    await test_db.refresh(bot)
    assert bot.strategy == "auto_mode"


@pytest.mark.asyncio
async def test_auto_mode_losing_streak_continues_not_paused(test_db):
    bot = await _make_bot(test_db, "auto_mode")
    await _add_losing_realized_gains(test_db, bot.id, count=3)
    svc = RiskManagementService(test_db)
    count, result = await svc.check_consecutive_losses(bot.id, threshold=3)
    assert count == 3
    assert result.action == RiskAction.CONTINUE
    await test_db.refresh(bot)
    assert bot.strategy == "auto_mode"


# --------------------------------------------------------------------------- #
# 4: Auto Mode (or a risk event) cannot modify a bot's persisted strategy
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_rotate_strategy_blocked_for_auto_mode(test_db):
    bot = await _make_bot(test_db, "auto_mode")
    svc = RiskManagementService(test_db)
    ok, msg = await svc.rotate_strategy(bot.id, "dca_accumulator", "forced")
    assert ok is False
    assert "auto_mode" in msg
    await test_db.refresh(bot)
    assert bot.strategy == "auto_mode"


# --------------------------------------------------------------------------- #
# 5: No write path other than rotate_strategy can overwrite Bot.strategy
# (guards against a websocket/serialization/update path being added that mutates
# the configured strategy)
# --------------------------------------------------------------------------- #
def test_bot_strategy_has_single_write_site():
    """Static guard: ``bot.strategy =`` (assignment) must not reappear anywhere
    in app/ except inside the repair routine. rotate_strategy no longer assigns
    it at all. Websocket/serialization layers only ever READ bot.strategy."""
    import pathlib
    import re

    app_dir = pathlib.Path(__file__).resolve().parent.parent / "app"
    # Match `<something>.strategy =` but not `==`, `>=`, `!=`, strategy_params, etc.
    pattern = re.compile(r"\.strategy\s*=(?!=)")
    offenders = []
    for path in app_dir.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "strategy_params" in line or "strategy_state" in line:
                continue
            if "strategy_used" in line:
                continue
            if pattern.search(line):
                offenders.append(f"{path.relative_to(app_dir)}:{lineno}: {stripped}")

    # The ONLY permitted assignment is inside repair_corrupted_strategies.
    allowed = [o for o in offenders if "risk_management.py" in o]
    unexpected = [o for o in offenders if "risk_management.py" not in o]
    assert not unexpected, (
        "Unexpected write to bot.strategy (configured strategy must be "
        f"immutable):\n" + "\n".join(unexpected)
    )
    # And in risk_management it must only be the repair routine, never a rotation.
    assert len(allowed) <= 1, f"Too many strategy writes in risk_management: {allowed}"


# --------------------------------------------------------------------------- #
# 6: Corrupted bots can be repaired (restore original + reset rotation counter)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_repair_restores_the_three_observed_bots(test_db):
    """Reproduce the exact production corruption and prove repair restores each
    bot's original configured strategy and clears its rotation history."""
    # TestBot2-DCA: dca -> adaptive_grid -> mean_reversion -> trend_following
    b2 = await _make_bot(test_db, "trend_following", name="TestBot2-DCA")
    await _add_rotation_chain(test_db, b2.id, [
        "dca_accumulator", "adaptive_grid", "mean_reversion", "trend_following",
    ])
    # TestBot3-MR: mean_reversion -> trend_following -> volatility_breakout -> funding_carry
    b3 = await _make_bot(test_db, "funding_carry", name="TestBot3-MR")
    await _add_rotation_chain(test_db, b3.id, [
        "mean_reversion", "trend_following", "volatility_breakout", "funding_carry",
    ])
    # TestBot9-TF: trend_following -> volatility_breakout -> funding_carry -> dca_accumulator
    b9 = await _make_bot(test_db, "dca_accumulator", name="TestBot9-TF")
    await _add_rotation_chain(test_db, b9.id, [
        "trend_following", "volatility_breakout", "funding_carry", "dca_accumulator",
    ])

    svc = RiskManagementService(test_db)
    repairs = await svc.repair_corrupted_strategies()

    assert len(repairs) == 3
    await test_db.refresh(b2)
    await test_db.refresh(b3)
    await test_db.refresh(b9)
    assert b2.strategy == "dca_accumulator"
    assert b3.strategy == "mean_reversion"
    assert b9.strategy == "trend_following"
    # Rotation history cleared -> rotation count back to 0 for all.
    for b in (b2, b3, b9):
        assert await _rotation_count(test_db, b.id) == 0


@pytest.mark.asyncio
async def test_repair_is_idempotent_and_skips_clean_bots(test_db):
    corrupted = await _make_bot(test_db, "funding_carry", name="corrupt")
    await _add_rotation_chain(test_db, corrupted.id, ["mean_reversion", "funding_carry"])
    clean = await _make_bot(test_db, "dca_accumulator", name="clean")
    auto = await _make_bot(test_db, "auto_mode", name="auto")

    svc = RiskManagementService(test_db)
    first = await svc.repair_corrupted_strategies()
    assert len(first) == 1
    assert first[0]["bot_id"] == corrupted.id

    # Second run: nothing left to repair.
    second = await svc.repair_corrupted_strategies()
    assert second == []

    await test_db.refresh(corrupted)
    await test_db.refresh(clean)
    await test_db.refresh(auto)
    assert corrupted.strategy == "mean_reversion"
    assert clean.strategy == "dca_accumulator"  # untouched
    assert auto.strategy == "auto_mode"  # untouched


# --------------------------------------------------------------------------- #
# 7: Existing serialization still surfaces the (correct) configured strategy
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_bot_api_reports_configured_strategy(client, test_db):
    bot = await _make_bot(test_db, "mean_reversion", name="api-bot")
    resp = await client.get(f"/api/bots/{bot.id}")
    assert resp.status_code == 200
    assert resp.json()["strategy"] == "mean_reversion"


# --------------------------------------------------------------------------- #
# 8: Consecutive loss classification uses realized P&L, not balance snapshots
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_dca_buy_only_never_increments_consecutive_losses(test_db):
    """DCA with only BUY orders must never trigger the consecutive loss counter.

    Previously, each BUY deducted only the fee from running_balance_after, creating
    a monotonically decreasing sequence that the old algorithm classified as losses.
    The new algorithm reads realized_gains, which only contains SELL exits, so
    BUY-only strategies produce zero consecutive losses regardless of how many
    buys have executed."""
    bot = await _make_bot(test_db, "dca_accumulator")
    # No realized gains — DCA has accumulated BTC but never sold.
    svc = RiskManagementService(test_db)
    count, result = await svc.check_consecutive_losses(bot.id, threshold=3)
    assert count == 0
    assert result.action == RiskAction.CONTINUE


@pytest.mark.asyncio
async def test_profitable_round_trip_resets_consecutive_loss_counter(test_db):
    """A profitable exit (gain_loss >= 0) resets the streak to 0.

    Two prior losses followed by one profitable exit must leave the counter at 0
    so the bot is not paused after the winning trade."""
    bot = await _make_bot(test_db, "mean_reversion")
    # Insert two losses first (oldest), then one win (newest).
    base = datetime.utcnow()
    for i, gl in enumerate([-5.0, -3.0, +8.0]):
        test_db.add(RealizedGain(
            owner_id=str(bot.id), asset="BTC",
            quantity=0.001, proceeds=10.0 + gl, cost_basis=10.0, gain_loss=gl,
            holding_period_days=0, is_long_term=False,
            purchase_trade_id=i + 1, sell_trade_id=i + 100, tax_lot_id=i + 1,
            purchase_date=base + timedelta(seconds=i),
            sell_date=base + timedelta(seconds=i + 1),
        ))
    await test_db.commit()

    svc = RiskManagementService(test_db)
    count, result = await svc.check_consecutive_losses(bot.id, threshold=3)
    assert count == 0
    assert result.action == RiskAction.CONTINUE


@pytest.mark.asyncio
async def test_losing_round_trip_increments_consecutive_loss_counter(test_db):
    """A losing exit (gain_loss < 0) increments the consecutive loss counter."""
    bot = await _make_bot(test_db, "trend_following")
    await _add_losing_realized_gains(test_db, bot.id, count=2)

    svc = RiskManagementService(test_db)
    count, result = await svc.check_consecutive_losses(bot.id, threshold=3)
    assert count == 2
    assert result.action == RiskAction.CONTINUE  # below threshold


@pytest.mark.asyncio
async def test_profitable_sell_never_classified_as_loss(test_db):
    """A SELL with gain_loss > 0 must never be counted as a loss, regardless of
    fee magnitude or running_balance_after snapshot timing.

    This guards the specific failure mode found in production: the old algorithm
    captured running_balance_after before crediting realized P&L, making even
    profitable SELLs appear as balance decreases."""
    bot = await _make_bot(test_db, "mean_reversion")
    # A profitable realized gain — price moved enough to overcome buy fee.
    await _add_profitable_realized_gain(test_db, bot.id, gain_loss=0.50)

    svc = RiskManagementService(test_db)
    count, result = await svc.check_consecutive_losses(bot.id, threshold=3)
    assert count == 0
    assert result.action == RiskAction.CONTINUE


@pytest.mark.asyncio
async def test_three_genuine_consecutive_losses_enter_recovery_mode(test_db):
    """Three consecutive realized losses trigger RECOVERY_MODE (not PAUSE)."""
    bot = await _make_bot(test_db, "trend_following")
    await _add_losing_realized_gains(test_db, bot.id, count=3, gain_loss=-15.0)

    svc = RiskManagementService(test_db)
    count, result = await svc.check_consecutive_losses(bot.id, threshold=3)
    assert count == 3
    assert result.action == RiskAction.ENTER_RECOVERY_MODE
    assert "consecutive losses" in result.reason
    assert "recovery mode" in result.reason.lower()
