"""Portfolio risk management API endpoints.

TODO: This is a basic implementation. UI/UX work pending.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
from pydantic import BaseModel

from ..models import get_session, PortfolioRisk
from ..services.portfolio_risk import PortfolioRiskService

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


class PortfolioRiskConfig(BaseModel):
    """Portfolio risk configuration request."""
    owner_id: str
    daily_loss_cap_pct: Optional[float] = None
    weekly_loss_cap_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    max_total_exposure_pct: Optional[float] = None
    enabled: bool = False


@router.get("/risk/{owner_id}")
async def get_portfolio_risk(
    owner_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get portfolio risk configuration for an owner.

    Args:
        owner_id: Owner identifier

    Returns:
        Portfolio risk configuration
    """
    result = await session.execute(
        select(PortfolioRisk).where(PortfolioRisk.owner_id == owner_id)
    )
    risk_config = result.scalar_one_or_none()

    if not risk_config:
        return {
            "owner_id": owner_id,
            "daily_loss_cap_pct": None,
            "weekly_loss_cap_pct": None,
            "max_drawdown_pct": None,
            "max_total_exposure_pct": None,
            "enabled": False,
        }

    return risk_config.to_dict()


@router.post("/risk")
async def create_or_update_portfolio_risk(
    config: PortfolioRiskConfig,
    session: AsyncSession = Depends(get_session),
):
    """Create or update portfolio risk configuration.

    Args:
        config: Portfolio risk configuration

    Returns:
        Updated configuration
    """
    # Check if config already exists
    result = await session.execute(
        select(PortfolioRisk).where(PortfolioRisk.owner_id == config.owner_id)
    )
    risk_config = result.scalar_one_or_none()

    if risk_config:
        # Update existing
        risk_config.daily_loss_cap_pct = config.daily_loss_cap_pct
        risk_config.weekly_loss_cap_pct = config.weekly_loss_cap_pct
        risk_config.max_drawdown_pct = config.max_drawdown_pct
        risk_config.max_total_exposure_pct = config.max_total_exposure_pct
        risk_config.enabled = config.enabled
    else:
        # Create new
        risk_config = PortfolioRisk(
            owner_id=config.owner_id,
            daily_loss_cap_pct=config.daily_loss_cap_pct,
            weekly_loss_cap_pct=config.weekly_loss_cap_pct,
            max_drawdown_pct=config.max_drawdown_pct,
            max_total_exposure_pct=config.max_total_exposure_pct,
            enabled=config.enabled,
        )
        session.add(risk_config)

    await session.commit()
    await session.refresh(risk_config)

    return risk_config.to_dict()


@router.get("/metrics/{owner_id}")
async def get_portfolio_metrics(
    owner_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get current portfolio metrics for an owner.

    Args:
        owner_id: Owner identifier

    Returns:
        Portfolio metrics
    """
    portfolio_risk = PortfolioRiskService(session)
    metrics = await portfolio_risk.get_portfolio_metrics(owner_id)

    return metrics


@router.delete("/risk/{owner_id}")
async def delete_portfolio_risk(
    owner_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete portfolio risk configuration.

    Args:
        owner_id: Owner identifier

    Returns:
        Success message
    """
    result = await session.execute(
        select(PortfolioRisk).where(PortfolioRisk.owner_id == owner_id)
    )
    risk_config = result.scalar_one_or_none()

    if not risk_config:
        raise HTTPException(status_code=404, detail="Portfolio risk config not found")

    await session.delete(risk_config)
    await session.commit()

    return {"message": "Portfolio risk config deleted successfully"}
