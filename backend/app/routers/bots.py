"""Bot management router."""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import get_session, Bot, BotStatus, Order, Position
from ..services import trading_engine

router = APIRouter()


# Pydantic schemas
class BotCreate(BaseModel):
    """Schema for creating a bot."""
    name: str = Field(..., min_length=1, max_length=255)
    trading_pair: str = Field(..., min_length=1, max_length=50)
    strategy: str = Field(..., min_length=1, max_length=50)
    strategy_params: dict = Field(default_factory=dict)
    budget: float = Field(..., gt=0)
    compound_enabled: bool = False
    running_time_hours: Optional[float] = None
    stop_loss_percent: Optional[float] = Field(default=None, ge=0, le=100)
    stop_loss_absolute: Optional[float] = Field(default=None, ge=0)
    drawdown_limit_percent: Optional[float] = Field(default=None, ge=0, le=100)
    drawdown_limit_absolute: Optional[float] = Field(default=None, ge=0)
    daily_loss_limit: Optional[float] = Field(default=None, ge=0)
    weekly_loss_limit: Optional[float] = Field(default=None, ge=0)
    max_strategy_rotations: int = Field(default=3, ge=0)
    is_dry_run: bool = False


class BotUpdate(BaseModel):
    """Schema for updating a bot."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    strategy: Optional[str] = Field(default=None, min_length=1, max_length=50)
    strategy_params: Optional[dict] = None
    budget: Optional[float] = Field(default=None, gt=0)
    compound_enabled: Optional[bool] = None
    running_time_hours: Optional[float] = None
    stop_loss_percent: Optional[float] = Field(default=None, ge=0, le=100)
    stop_loss_absolute: Optional[float] = Field(default=None, ge=0)
    drawdown_limit_percent: Optional[float] = Field(default=None, ge=0, le=100)
    drawdown_limit_absolute: Optional[float] = Field(default=None, ge=0)
    daily_loss_limit: Optional[float] = Field(default=None, ge=0)
    weekly_loss_limit: Optional[float] = Field(default=None, ge=0)
    max_strategy_rotations: Optional[int] = Field(default=None, ge=0)


class BotResponse(BaseModel):
    """Schema for bot response."""
    id: int
    name: str
    trading_pair: str
    strategy: str
    strategy_params: dict
    budget: float
    compound_enabled: bool
    current_balance: float
    running_time_hours: Optional[float]
    stop_loss_percent: Optional[float]
    stop_loss_absolute: Optional[float]
    drawdown_limit_percent: Optional[float]
    drawdown_limit_absolute: Optional[float]
    daily_loss_limit: Optional[float]
    weekly_loss_limit: Optional[float]
    max_strategy_rotations: int
    is_dry_run: bool
    status: str
    total_pnl: float
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime]
    paused_at: Optional[datetime]

    class Config:
        from_attributes = True


class OrderResponse(BaseModel):
    """Schema for order response."""
    id: int
    bot_id: int
    exchange_order_id: Optional[str]
    order_type: str
    trading_pair: str
    amount: float
    price: float
    fees: float
    status: str
    strategy_used: str
    running_balance_after: Optional[float]
    is_simulated: bool
    created_at: datetime
    filled_at: Optional[datetime]

    class Config:
        from_attributes = True


class PositionResponse(BaseModel):
    """Schema for position response."""
    id: int
    bot_id: int
    trading_pair: str
    side: str
    entry_price: float
    current_price: float
    amount: float
    unrealized_pnl: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


@router.get("", response_model=List[BotResponse])
async def list_bots(
    session: AsyncSession = Depends(get_session),
    status_filter: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
):
    """List all bots with optional filtering."""
    query = select(Bot)
    if status_filter:
        query = query.where(Bot.status == status_filter)
    query = query.offset(skip).limit(limit)

    result = await session.execute(query)
    bots = result.scalars().all()
    return bots


@router.post("", response_model=BotResponse, status_code=status.HTTP_201_CREATED)
async def create_bot(
    bot_data: BotCreate,
    session: AsyncSession = Depends(get_session),
):
    """Create a new bot."""
    bot = Bot(
        name=bot_data.name,
        trading_pair=bot_data.trading_pair,
        strategy=bot_data.strategy,
        strategy_params=bot_data.strategy_params,
        budget=bot_data.budget,
        current_balance=bot_data.budget,
        compound_enabled=bot_data.compound_enabled,
        running_time_hours=bot_data.running_time_hours,
        stop_loss_percent=bot_data.stop_loss_percent,
        stop_loss_absolute=bot_data.stop_loss_absolute,
        drawdown_limit_percent=bot_data.drawdown_limit_percent,
        drawdown_limit_absolute=bot_data.drawdown_limit_absolute,
        daily_loss_limit=bot_data.daily_loss_limit,
        weekly_loss_limit=bot_data.weekly_loss_limit,
        max_strategy_rotations=bot_data.max_strategy_rotations,
        is_dry_run=bot_data.is_dry_run,
    )

    session.add(bot)
    await session.commit()
    await session.refresh(bot)
    return bot


@router.get("/{bot_id}", response_model=BotResponse)
async def get_bot(
    bot_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get a specific bot by ID."""
    result = await session.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if not bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot with id {bot_id} not found"
        )
    return bot


@router.put("/{bot_id}", response_model=BotResponse)
async def update_bot(
    bot_id: int,
    bot_data: BotUpdate,
    session: AsyncSession = Depends(get_session),
):
    """Update a bot configuration. Bot must be paused or stopped."""
    result = await session.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if not bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot with id {bot_id} not found"
        )

    if bot.status == BotStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot update a running bot. Pause or stop it first."
        )

    # Update fields
    update_data = bot_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(bot, field, value)

    bot.updated_at = datetime.utcnow()
    await session.commit()
    await session.refresh(bot)
    return bot


@router.delete("/{bot_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_bot(
    bot_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Delete a bot. Bot must be stopped."""
    result = await session.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if not bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot with id {bot_id} not found"
        )

    if bot.status == BotStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a running bot. Stop it first."
        )

    await session.delete(bot)
    await session.commit()


@router.post("/{bot_id}/start", response_model=BotResponse)
async def start_bot(
    bot_id: int,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    """Start a bot."""
    result = await session.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if not bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot with id {bot_id} not found"
        )

    if bot.status == BotStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot is already running"
        )

    # Start the trading engine for this bot
    background_tasks.add_task(trading_engine.start_bot, bot_id)

    # Update status (will also be updated by trading engine)
    bot.status = BotStatus.RUNNING
    bot.started_at = datetime.utcnow()
    bot.paused_at = None
    bot.updated_at = datetime.utcnow()

    await session.commit()
    await session.refresh(bot)
    return bot


@router.post("/{bot_id}/pause", response_model=BotResponse)
async def pause_bot(
    bot_id: int,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    """Pause a running bot."""
    result = await session.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if not bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot with id {bot_id} not found"
        )

    if bot.status != BotStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only pause a running bot"
        )

    # Pause the trading engine for this bot
    background_tasks.add_task(trading_engine.pause_bot, bot_id)

    bot.status = BotStatus.PAUSED
    bot.paused_at = datetime.utcnow()
    bot.updated_at = datetime.utcnow()

    await session.commit()
    await session.refresh(bot)
    return bot


@router.post("/{bot_id}/stop", response_model=BotResponse)
async def stop_bot(
    bot_id: int,
    cancel_orders: bool = True,
    background_tasks: BackgroundTasks = None,
    session: AsyncSession = Depends(get_session),
):
    """Stop a bot."""
    result = await session.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if not bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot with id {bot_id} not found"
        )

    if bot.status == BotStatus.STOPPED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot is already stopped"
        )

    # Stop the trading engine for this bot
    if background_tasks:
        background_tasks.add_task(trading_engine.stop_bot, bot_id, cancel_orders)
    else:
        await trading_engine.stop_bot(bot_id, cancel_orders)

    bot.status = BotStatus.STOPPED
    bot.updated_at = datetime.utcnow()

    await session.commit()
    await session.refresh(bot)
    return bot


@router.post("/{bot_id}/kill", response_model=BotResponse)
async def kill_bot(
    bot_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Kill switch for a bot - cancels all pending orders but does not liquidate positions."""
    result = await session.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if not bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot with id {bot_id} not found"
        )

    # Kill the bot via trading engine
    await trading_engine.kill_bot(bot_id)

    bot.status = BotStatus.STOPPED
    bot.updated_at = datetime.utcnow()

    # Log alert
    from ..models import Alert
    alert = Alert(
        bot_id=bot_id,
        alert_type="kill_switch",
        message=f"Kill switch activated for bot '{bot.name}' (ID: {bot_id})",
        email_sent=False,
    )
    session.add(alert)

    await session.commit()
    await session.refresh(bot)
    return bot


@router.post("/{bot_id}/go-live", response_model=BotResponse)
async def promote_to_live(
    bot_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Promote a dry run bot to live trading."""
    result = await session.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if not bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot with id {bot_id} not found"
        )

    if not bot.is_dry_run:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot is already live"
        )

    if bot.status == BotStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stop the bot before promoting to live"
        )

    bot.is_dry_run = False
    bot.updated_at = datetime.utcnow()

    await session.commit()
    await session.refresh(bot)
    return bot


@router.post("/{bot_id}/copy", response_model=BotResponse)
async def copy_bot(
    bot_id: int,
    new_name: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    """Copy a bot's configuration to a new bot."""
    result = await session.execute(select(Bot).where(Bot.id == bot_id))
    source_bot = result.scalar_one_or_none()

    if not source_bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot with id {bot_id} not found"
        )

    new_bot = Bot(
        name=new_name or f"{source_bot.name} (Copy)",
        trading_pair=source_bot.trading_pair,
        strategy=source_bot.strategy,
        strategy_params=source_bot.strategy_params.copy() if source_bot.strategy_params else {},
        budget=source_bot.budget,
        current_balance=source_bot.budget,
        compound_enabled=source_bot.compound_enabled,
        running_time_hours=source_bot.running_time_hours,
        stop_loss_percent=source_bot.stop_loss_percent,
        stop_loss_absolute=source_bot.stop_loss_absolute,
        drawdown_limit_percent=source_bot.drawdown_limit_percent,
        drawdown_limit_absolute=source_bot.drawdown_limit_absolute,
        daily_loss_limit=source_bot.daily_loss_limit,
        weekly_loss_limit=source_bot.weekly_loss_limit,
        max_strategy_rotations=source_bot.max_strategy_rotations,
        is_dry_run=source_bot.is_dry_run,
    )

    session.add(new_bot)
    await session.commit()
    await session.refresh(new_bot)
    return new_bot


@router.get("/{bot_id}/orders", response_model=List[OrderResponse])
async def get_bot_orders(
    bot_id: int,
    session: AsyncSession = Depends(get_session),
    skip: int = 0,
    limit: int = 100,
):
    """Get orders for a specific bot."""
    # First check if bot exists
    result = await session.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if not bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot with id {bot_id} not found"
        )

    result = await session.execute(
        select(Order)
        .where(Order.bot_id == bot_id)
        .order_by(Order.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    orders = result.scalars().all()
    return orders


@router.get("/{bot_id}/positions", response_model=List[PositionResponse])
async def get_bot_positions(
    bot_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get open positions for a specific bot."""
    # First check if bot exists
    result = await session.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if not bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot with id {bot_id} not found"
        )

    result = await session.execute(
        select(Position).where(Position.bot_id == bot_id)
    )
    positions = result.scalars().all()
    return positions
