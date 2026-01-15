"""External data sources service.

Provides access to various external data sources for trading signals:
- News API: Crypto news headlines and sentiment
- Fear & Greed Index: Market sentiment indicator
- On-chain metrics: Blockchain data
- Social sentiment: Twitter/Reddit mentions
- Market conditions: Overall market health indicators

Each source can be individually enabled/disabled via configuration.

Features: #68, #69, #70, #71, #72, #73, #180
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import aiohttp
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DataSourceType(str, Enum):
    """Types of external data sources."""
    NEWS_API = "news_api"
    FEAR_GREED = "fear_greed"
    ONCHAIN_METRICS = "onchain_metrics"
    SOCIAL_SENTIMENT = "social_sentiment"
    MARKET_CONDITIONS = "market_conditions"


@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""
    enabled: bool = False
    api_key: Optional[str] = None
    refresh_interval_seconds: int = 300  # 5 minutes default
    cache_ttl_seconds: int = 300
    # Source-specific settings
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSourceStatus:
    """Status of a data source."""
    source_type: DataSourceType
    enabled: bool
    healthy: bool
    last_fetch: Optional[datetime] = None
    last_error: Optional[str] = None
    data_age_seconds: Optional[int] = None


@dataclass
class NewsItem:
    """A news article."""
    title: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float  # -1 to 1
    relevance_score: float  # 0 to 1
    summary: Optional[str] = None


@dataclass
class FearGreedData:
    """Fear and Greed Index data."""
    value: int  # 0-100
    classification: str  # Extreme Fear, Fear, Neutral, Greed, Extreme Greed
    timestamp: datetime
    previous_value: Optional[int] = None
    previous_classification: Optional[str] = None


@dataclass
class OnchainMetrics:
    """On-chain metrics data."""
    active_addresses: int
    transaction_volume: float
    exchange_inflow: float
    exchange_outflow: float
    whale_transactions: int
    timestamp: datetime
    # Derived signals
    accumulation_signal: float  # -1 to 1
    network_health: float  # 0 to 1


@dataclass
class SocialSentiment:
    """Social media sentiment data."""
    mentions_24h: int
    sentiment_score: float  # -1 to 1
    trending_score: float  # 0 to 1
    top_keywords: List[str]
    timestamp: datetime
    source: str  # twitter, reddit, etc.


@dataclass
class MarketConditions:
    """Overall market conditions."""
    btc_dominance: float
    total_market_cap: float
    market_cap_change_24h: float
    volume_24h: float
    trending_coins: List[str]
    timestamp: datetime
    market_phase: str  # accumulation, markup, distribution, markdown


@dataclass
class AggregatedSignals:
    """Aggregated signals from all enabled data sources."""
    overall_sentiment: float  # -1 to 1
    confidence: float  # 0 to 1
    signal: str  # bullish, bearish, neutral, avoid
    contributing_sources: List[str]
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


class BaseDataSource(ABC):
    """Base class for external data sources."""

    def __init__(self, source_type: DataSourceType, config: DataSourceConfig):
        self.source_type = source_type
        self.config = config
        self._cache: Optional[Any] = None
        self._cache_time: Optional[datetime] = None
        self._last_error: Optional[str] = None
        self._healthy = True

    @property
    def is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if self._cache is None or self._cache_time is None:
            return False
        age = (datetime.utcnow() - self._cache_time).total_seconds()
        return age < self.config.cache_ttl_seconds

    @abstractmethod
    async def fetch(self) -> Any:
        """Fetch data from the source."""
        pass

    async def get_data(self) -> Optional[Any]:
        """Get data, using cache if valid."""
        if not self.config.enabled:
            return None

        if self.is_cache_valid:
            return self._cache

        try:
            self._cache = await self.fetch()
            self._cache_time = datetime.utcnow()
            self._healthy = True
            self._last_error = None
            return self._cache
        except Exception as e:
            self._healthy = False
            self._last_error = str(e)
            logger.error(f"Error fetching from {self.source_type}: {e}")
            # Return stale cache if available
            return self._cache

    def get_status(self) -> DataSourceStatus:
        """Get the status of this data source."""
        data_age = None
        if self._cache_time:
            data_age = int((datetime.utcnow() - self._cache_time).total_seconds())

        return DataSourceStatus(
            source_type=self.source_type,
            enabled=self.config.enabled,
            healthy=self._healthy,
            last_fetch=self._cache_time,
            last_error=self._last_error,
            data_age_seconds=data_age
        )


class NewsAPISource(BaseDataSource):
    """News API data source for crypto news."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(DataSourceType.NEWS_API, config)
        self.base_url = "https://newsapi.org/v2"

    async def fetch(self) -> List[NewsItem]:
        """Fetch crypto news from News API."""
        if not self.config.api_key:
            # Return mock data when no API key configured
            return self._get_mock_news()

        async with aiohttp.ClientSession() as session:
            params = {
                "q": "cryptocurrency OR bitcoin OR ethereum",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "apiKey": self.config.api_key
            }
            async with session.get(f"{self.base_url}/everything", params=params) as resp:
                if resp.status != 200:
                    raise Exception(f"News API returned {resp.status}")
                data = await resp.json()

                return [
                    NewsItem(
                        title=article["title"],
                        source=article["source"]["name"],
                        url=article["url"],
                        published_at=datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00")),
                        sentiment_score=self._analyze_sentiment(article["title"]),
                        relevance_score=0.8,
                        summary=article.get("description")
                    )
                    for article in data.get("articles", [])
                ]

    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis based on keywords."""
        positive_words = ["bullish", "surge", "rally", "gains", "up", "high", "record", "adoption", "growth"]
        negative_words = ["bearish", "crash", "drop", "fall", "down", "low", "fear", "sell", "decline"]

        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count + neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count)

    def _get_mock_news(self) -> List[NewsItem]:
        """Return mock news data for demo/testing."""
        return [
            NewsItem(
                title="Bitcoin continues steady growth as institutional adoption increases",
                source="CryptoNews",
                url="https://example.com/news/1",
                published_at=datetime.utcnow() - timedelta(hours=2),
                sentiment_score=0.6,
                relevance_score=0.9,
                summary="Major financial institutions continue to add Bitcoin to their portfolios."
            ),
            NewsItem(
                title="Ethereum network upgrade scheduled for next month",
                source="BlockchainDaily",
                url="https://example.com/news/2",
                published_at=datetime.utcnow() - timedelta(hours=5),
                sentiment_score=0.4,
                relevance_score=0.8,
                summary="The upcoming upgrade promises improved scalability and lower gas fees."
            )
        ]


class FearGreedSource(BaseDataSource):
    """Fear and Greed Index data source."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(DataSourceType.FEAR_GREED, config)
        self.api_url = "https://api.alternative.me/fng/"

    async def fetch(self) -> FearGreedData:
        """Fetch Fear and Greed Index."""
        async with aiohttp.ClientSession() as session:
            params = {"limit": 2}  # Get current and previous
            async with session.get(self.api_url, params=params) as resp:
                if resp.status != 200:
                    raise Exception(f"Fear & Greed API returned {resp.status}")
                data = await resp.json()

                items = data.get("data", [])
                if not items:
                    raise Exception("No data returned from Fear & Greed API")

                current = items[0]
                previous = items[1] if len(items) > 1 else None

                return FearGreedData(
                    value=int(current["value"]),
                    classification=current["value_classification"],
                    timestamp=datetime.fromtimestamp(int(current["timestamp"])),
                    previous_value=int(previous["value"]) if previous else None,
                    previous_classification=previous["value_classification"] if previous else None
                )


class OnchainMetricsSource(BaseDataSource):
    """On-chain metrics data source."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(DataSourceType.ONCHAIN_METRICS, config)

    async def fetch(self) -> OnchainMetrics:
        """Fetch on-chain metrics."""
        # In production, this would connect to Glassnode, IntoTheBlock, etc.
        # For now, return simulated data
        return OnchainMetrics(
            active_addresses=950000 + int(asyncio.get_event_loop().time() % 100000),
            transaction_volume=15000000000.0,
            exchange_inflow=5000.0,
            exchange_outflow=7500.0,
            whale_transactions=45,
            timestamp=datetime.utcnow(),
            accumulation_signal=0.3,  # Slight accumulation
            network_health=0.75
        )


class SocialSentimentSource(BaseDataSource):
    """Social media sentiment data source."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(DataSourceType.SOCIAL_SENTIMENT, config)

    async def fetch(self) -> SocialSentiment:
        """Fetch social sentiment data."""
        # In production, this would connect to LunarCrush, Santiment, etc.
        # For now, return simulated data
        return SocialSentiment(
            mentions_24h=125000,
            sentiment_score=0.15,
            trending_score=0.6,
            top_keywords=["bitcoin", "btc", "crypto", "ethereum", "defi"],
            timestamp=datetime.utcnow(),
            source="aggregated"
        )


class MarketConditionsSource(BaseDataSource):
    """Market conditions data source."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(DataSourceType.MARKET_CONDITIONS, config)
        self.api_url = "https://api.coingecko.com/api/v3/global"

    async def fetch(self) -> MarketConditions:
        """Fetch market conditions."""
        async with aiohttp.ClientSession() as session:
            async with session.get(self.api_url) as resp:
                if resp.status != 200:
                    raise Exception(f"CoinGecko API returned {resp.status}")
                data = await resp.json()

                market_data = data.get("data", {})

                # Determine market phase based on metrics
                market_cap_change = market_data.get("market_cap_change_percentage_24h_usd", 0)
                if market_cap_change > 3:
                    phase = "markup"
                elif market_cap_change < -3:
                    phase = "markdown"
                elif market_cap_change > 0:
                    phase = "accumulation"
                else:
                    phase = "distribution"

                return MarketConditions(
                    btc_dominance=market_data.get("market_cap_percentage", {}).get("btc", 0),
                    total_market_cap=market_data.get("total_market_cap", {}).get("usd", 0),
                    market_cap_change_24h=market_cap_change,
                    volume_24h=market_data.get("total_volume", {}).get("usd", 0),
                    trending_coins=[],  # Would need separate API call
                    timestamp=datetime.utcnow(),
                    market_phase=phase
                )


class ExternalDataService:
    """Service for managing external data sources.

    Provides a unified interface to all external data sources with:
    - Individual enable/disable per source
    - Caching to reduce API calls
    - Health monitoring
    - Aggregated signal generation
    """

    CONFIG_FILE = "data_sources.yaml"

    def __init__(self):
        self._sources: Dict[DataSourceType, BaseDataSource] = {}
        self._configs: Dict[DataSourceType, DataSourceConfig] = {}
        self._load_config()
        self._init_sources()

    def _get_config_path(self) -> Path:
        """Get the path to the config file."""
        backend_dir = Path(__file__).parent.parent.parent
        return backend_dir / self.CONFIG_FILE

    def _load_config(self):
        """Load configuration from file."""
        config_path = self._get_config_path()

        # Default configs (all disabled by default for safety)
        default_configs = {
            DataSourceType.NEWS_API: DataSourceConfig(
                enabled=False,
                refresh_interval_seconds=600,
                cache_ttl_seconds=300,
                settings={"keywords": ["bitcoin", "ethereum", "crypto"]}
            ),
            DataSourceType.FEAR_GREED: DataSourceConfig(
                enabled=False,
                refresh_interval_seconds=3600,  # Updates once per day anyway
                cache_ttl_seconds=1800,
            ),
            DataSourceType.ONCHAIN_METRICS: DataSourceConfig(
                enabled=False,
                refresh_interval_seconds=900,
                cache_ttl_seconds=600,
            ),
            DataSourceType.SOCIAL_SENTIMENT: DataSourceConfig(
                enabled=False,
                refresh_interval_seconds=900,
                cache_ttl_seconds=600,
            ),
            DataSourceType.MARKET_CONDITIONS: DataSourceConfig(
                enabled=False,
                refresh_interval_seconds=300,
                cache_ttl_seconds=180,
            ),
        }

        self._configs = default_configs

        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}

                sources_config = file_config.get("sources", {})
                for source_type in DataSourceType:
                    source_name = source_type.value
                    if source_name in sources_config:
                        source_cfg = sources_config[source_name]
                        self._configs[source_type] = DataSourceConfig(
                            enabled=source_cfg.get("enabled", False),
                            api_key=source_cfg.get("api_key"),
                            refresh_interval_seconds=source_cfg.get("refresh_interval_seconds", 300),
                            cache_ttl_seconds=source_cfg.get("cache_ttl_seconds", 300),
                            settings=source_cfg.get("settings", {})
                        )
                logger.info(f"Loaded data sources config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load data sources config: {e}, using defaults")

    def _save_config(self):
        """Save configuration to file."""
        config_path = self._get_config_path()

        config_data = {
            "sources": {}
        }

        for source_type, config in self._configs.items():
            config_data["sources"][source_type.value] = {
                "enabled": config.enabled,
                "api_key": config.api_key or "",
                "refresh_interval_seconds": config.refresh_interval_seconds,
                "cache_ttl_seconds": config.cache_ttl_seconds,
                "settings": config.settings
            }

        try:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            logger.info(f"Saved data sources config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save data sources config: {e}")

    def _init_sources(self):
        """Initialize data source instances."""
        source_classes = {
            DataSourceType.NEWS_API: NewsAPISource,
            DataSourceType.FEAR_GREED: FearGreedSource,
            DataSourceType.ONCHAIN_METRICS: OnchainMetricsSource,
            DataSourceType.SOCIAL_SENTIMENT: SocialSentimentSource,
            DataSourceType.MARKET_CONDITIONS: MarketConditionsSource,
        }

        for source_type, source_class in source_classes.items():
            config = self._configs.get(source_type, DataSourceConfig())
            self._sources[source_type] = source_class(config)

    def get_source_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all source configurations (for API response)."""
        result = {}
        for source_type, config in self._configs.items():
            status = self._sources[source_type].get_status()
            result[source_type.value] = {
                "enabled": config.enabled,
                "has_api_key": bool(config.api_key),
                "refresh_interval_seconds": config.refresh_interval_seconds,
                "cache_ttl_seconds": config.cache_ttl_seconds,
                "settings": config.settings,
                "healthy": status.healthy,
                "last_fetch": status.last_fetch.isoformat() if status.last_fetch else None,
                "last_error": status.last_error,
                "data_age_seconds": status.data_age_seconds,
            }
        return result

    def update_source_config(
        self,
        source_type: DataSourceType,
        enabled: Optional[bool] = None,
        api_key: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update configuration for a specific source."""
        if source_type not in self._configs:
            raise ValueError(f"Unknown source type: {source_type}")

        config = self._configs[source_type]

        if enabled is not None:
            config.enabled = enabled
        if api_key is not None:
            config.api_key = api_key if api_key else None
        if settings is not None:
            config.settings.update(settings)

        # Reinitialize the source with new config
        self._init_sources()

        # Save updated config
        self._save_config()

        return self.get_source_configs()[source_type.value]

    def set_all_sources_enabled(self, enabled: bool) -> Dict[str, Dict[str, Any]]:
        """Enable or disable all sources at once."""
        for config in self._configs.values():
            config.enabled = enabled

        self._init_sources()
        self._save_config()

        return self.get_source_configs()

    async def get_news(self) -> Optional[List[NewsItem]]:
        """Get news data if enabled."""
        return await self._sources[DataSourceType.NEWS_API].get_data()

    async def get_fear_greed(self) -> Optional[FearGreedData]:
        """Get Fear & Greed Index if enabled."""
        return await self._sources[DataSourceType.FEAR_GREED].get_data()

    async def get_onchain_metrics(self) -> Optional[OnchainMetrics]:
        """Get on-chain metrics if enabled."""
        return await self._sources[DataSourceType.ONCHAIN_METRICS].get_data()

    async def get_social_sentiment(self) -> Optional[SocialSentiment]:
        """Get social sentiment if enabled."""
        return await self._sources[DataSourceType.SOCIAL_SENTIMENT].get_data()

    async def get_market_conditions(self) -> Optional[MarketConditions]:
        """Get market conditions if enabled."""
        return await self._sources[DataSourceType.MARKET_CONDITIONS].get_data()

    async def get_aggregated_signals(self) -> AggregatedSignals:
        """Get aggregated signals from all enabled sources."""
        signals = []
        details = {}
        contributing_sources = []

        # Collect signals from each enabled source
        fear_greed = await self.get_fear_greed()
        if fear_greed:
            # Convert 0-100 to -1 to 1
            fg_signal = (fear_greed.value - 50) / 50
            signals.append(fg_signal)
            contributing_sources.append("fear_greed")
            details["fear_greed"] = {
                "value": fear_greed.value,
                "classification": fear_greed.classification,
                "signal": fg_signal
            }

        news = await self.get_news()
        if news:
            avg_sentiment = sum(n.sentiment_score for n in news) / len(news) if news else 0
            signals.append(avg_sentiment)
            contributing_sources.append("news")
            details["news"] = {
                "count": len(news),
                "avg_sentiment": avg_sentiment
            }

        social = await self.get_social_sentiment()
        if social:
            signals.append(social.sentiment_score)
            contributing_sources.append("social_sentiment")
            details["social_sentiment"] = {
                "sentiment": social.sentiment_score,
                "mentions": social.mentions_24h
            }

        onchain = await self.get_onchain_metrics()
        if onchain:
            signals.append(onchain.accumulation_signal)
            contributing_sources.append("onchain_metrics")
            details["onchain_metrics"] = {
                "accumulation_signal": onchain.accumulation_signal,
                "network_health": onchain.network_health
            }

        market = await self.get_market_conditions()
        if market:
            # Convert market cap change to signal
            mkt_signal = max(-1, min(1, market.market_cap_change_24h / 10))
            signals.append(mkt_signal)
            contributing_sources.append("market_conditions")
            details["market_conditions"] = {
                "phase": market.market_phase,
                "change_24h": market.market_cap_change_24h
            }

        # Calculate overall sentiment
        if signals:
            overall_sentiment = sum(signals) / len(signals)
            confidence = len(signals) / 5  # 5 possible sources
        else:
            overall_sentiment = 0
            confidence = 0

        # Determine signal
        if confidence < 0.2:
            signal = "neutral"  # Not enough data
        elif overall_sentiment > 0.3:
            signal = "bullish"
        elif overall_sentiment < -0.3:
            signal = "bearish"
        elif abs(overall_sentiment) < 0.1:
            signal = "neutral"
        else:
            signal = "neutral"

        return AggregatedSignals(
            overall_sentiment=overall_sentiment,
            confidence=confidence,
            signal=signal,
            contributing_sources=contributing_sources,
            timestamp=datetime.utcnow(),
            details=details
        )

    def get_all_statuses(self) -> List[DataSourceStatus]:
        """Get status of all data sources."""
        return [source.get_status() for source in self._sources.values()]


# Global instance
external_data_service = ExternalDataService()
