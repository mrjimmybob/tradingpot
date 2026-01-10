"""Health check router."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "tradingbot",
        "version": "1.0.0"
    }
