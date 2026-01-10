"""Alerts router."""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import get_session, Alert

router = APIRouter()


class AlertResponse(BaseModel):
    """Schema for alert response."""
    id: int
    bot_id: Optional[int]
    alert_type: str
    message: str
    email_sent: bool
    created_at: datetime

    class Config:
        from_attributes = True


@router.get("", response_model=List[AlertResponse])
async def list_alerts(
    session: AsyncSession = Depends(get_session),
    bot_id: Optional[int] = None,
    alert_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
):
    """List all alerts with optional filtering."""
    query = select(Alert).order_by(Alert.created_at.desc())

    if bot_id is not None:
        query = query.where(Alert.bot_id == bot_id)
    if alert_type:
        query = query.where(Alert.alert_type == alert_type)

    query = query.offset(skip).limit(limit)

    result = await session.execute(query)
    alerts = result.scalars().all()
    return alerts


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get a specific alert by ID."""
    result = await session.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert with id {alert_id} not found"
        )
    return alert
