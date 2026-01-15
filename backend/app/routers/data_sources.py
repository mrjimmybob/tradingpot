"""Data sources API router.

Provides endpoints for managing external data sources:
- Get all source configurations and status
- Enable/disable individual sources
- Update source settings
- Fetch data from sources
- Get aggregated trading signals

Features: #68, #69, #70, #71, #72, #73, #180
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..services.external_data import (
    external_data_service,
    DataSourceType,
    NewsItem,
    FearGreedData,
    OnchainMetrics,
    SocialSentiment,
    MarketConditions,
    AggregatedSignals,
)

router = APIRouter()


class SourceUpdateRequest(BaseModel):
    """Request to update a data source configuration."""
    enabled: Optional[bool] = None
    api_key: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class BulkEnableRequest(BaseModel):
    """Request to enable/disable all sources."""
    enabled: bool


class SourceConfigResponse(BaseModel):
    """Response with source configuration and status."""
    enabled: bool
    has_api_key: bool
    refresh_interval_seconds: int
    cache_ttl_seconds: int
    settings: Dict[str, Any]
    healthy: bool
    last_fetch: Optional[str]
    last_error: Optional[str]
    data_age_seconds: Optional[int]


class NewsItemResponse(BaseModel):
    """News item response."""
    title: str
    source: str
    url: str
    published_at: str
    sentiment_score: float
    relevance_score: float
    summary: Optional[str]


class FearGreedResponse(BaseModel):
    """Fear & Greed Index response."""
    value: int
    classification: str
    timestamp: str
    previous_value: Optional[int]
    previous_classification: Optional[str]


class OnchainMetricsResponse(BaseModel):
    """On-chain metrics response."""
    active_addresses: int
    transaction_volume: float
    exchange_inflow: float
    exchange_outflow: float
    whale_transactions: int
    timestamp: str
    accumulation_signal: float
    network_health: float


class SocialSentimentResponse(BaseModel):
    """Social sentiment response."""
    mentions_24h: int
    sentiment_score: float
    trending_score: float
    top_keywords: List[str]
    timestamp: str
    source: str


class MarketConditionsResponse(BaseModel):
    """Market conditions response."""
    btc_dominance: float
    total_market_cap: float
    market_cap_change_24h: float
    volume_24h: float
    trending_coins: List[str]
    timestamp: str
    market_phase: str


class AggregatedSignalsResponse(BaseModel):
    """Aggregated signals response."""
    overall_sentiment: float
    confidence: float
    signal: str
    contributing_sources: List[str]
    timestamp: str
    details: Dict[str, Any]


@router.get("/sources", response_model=Dict[str, SourceConfigResponse])
async def get_all_sources():
    """Get all data source configurations and status."""
    return external_data_service.get_source_configs()


@router.put("/sources/{source_type}", response_model=SourceConfigResponse)
async def update_source(source_type: str, request: SourceUpdateRequest):
    """Update configuration for a specific data source."""
    try:
        source_enum = DataSourceType(source_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid source type: {source_type}. Valid types: {[s.value for s in DataSourceType]}"
        )

    return external_data_service.update_source_config(
        source_type=source_enum,
        enabled=request.enabled,
        api_key=request.api_key,
        settings=request.settings
    )


@router.post("/sources/bulk", response_model=Dict[str, SourceConfigResponse])
async def bulk_update_sources(request: BulkEnableRequest):
    """Enable or disable all data sources at once."""
    return external_data_service.set_all_sources_enabled(request.enabled)


@router.get("/news", response_model=Optional[List[NewsItemResponse]])
async def get_news():
    """Get crypto news if news source is enabled."""
    news = await external_data_service.get_news()
    if news is None:
        return None
    return [
        NewsItemResponse(
            title=n.title,
            source=n.source,
            url=n.url,
            published_at=n.published_at.isoformat(),
            sentiment_score=n.sentiment_score,
            relevance_score=n.relevance_score,
            summary=n.summary
        )
        for n in news
    ]


@router.get("/fear-greed", response_model=Optional[FearGreedResponse])
async def get_fear_greed():
    """Get Fear & Greed Index if enabled."""
    data = await external_data_service.get_fear_greed()
    if data is None:
        return None
    return FearGreedResponse(
        value=data.value,
        classification=data.classification,
        timestamp=data.timestamp.isoformat(),
        previous_value=data.previous_value,
        previous_classification=data.previous_classification
    )


@router.get("/onchain", response_model=Optional[OnchainMetricsResponse])
async def get_onchain_metrics():
    """Get on-chain metrics if enabled."""
    data = await external_data_service.get_onchain_metrics()
    if data is None:
        return None
    return OnchainMetricsResponse(
        active_addresses=data.active_addresses,
        transaction_volume=data.transaction_volume,
        exchange_inflow=data.exchange_inflow,
        exchange_outflow=data.exchange_outflow,
        whale_transactions=data.whale_transactions,
        timestamp=data.timestamp.isoformat(),
        accumulation_signal=data.accumulation_signal,
        network_health=data.network_health
    )


@router.get("/social", response_model=Optional[SocialSentimentResponse])
async def get_social_sentiment():
    """Get social sentiment if enabled."""
    data = await external_data_service.get_social_sentiment()
    if data is None:
        return None
    return SocialSentimentResponse(
        mentions_24h=data.mentions_24h,
        sentiment_score=data.sentiment_score,
        trending_score=data.trending_score,
        top_keywords=data.top_keywords,
        timestamp=data.timestamp.isoformat(),
        source=data.source
    )


@router.get("/market", response_model=Optional[MarketConditionsResponse])
async def get_market_conditions():
    """Get market conditions if enabled."""
    data = await external_data_service.get_market_conditions()
    if data is None:
        return None
    return MarketConditionsResponse(
        btc_dominance=data.btc_dominance,
        total_market_cap=data.total_market_cap,
        market_cap_change_24h=data.market_cap_change_24h,
        volume_24h=data.volume_24h,
        trending_coins=data.trending_coins,
        timestamp=data.timestamp.isoformat(),
        market_phase=data.market_phase
    )


@router.get("/signals", response_model=AggregatedSignalsResponse)
async def get_aggregated_signals():
    """Get aggregated trading signals from all enabled sources."""
    signals = await external_data_service.get_aggregated_signals()
    return AggregatedSignalsResponse(
        overall_sentiment=signals.overall_sentiment,
        confidence=signals.confidence,
        signal=signals.signal,
        contributing_sources=signals.contributing_sources,
        timestamp=signals.timestamp.isoformat(),
        details=signals.details
    )
