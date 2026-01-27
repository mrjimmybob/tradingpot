"""API tests for portfolio endpoints."""

import pytest
from unittest.mock import AsyncMock, patch, Mock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.routers.portfolio import router


@pytest.fixture
def app():
    """Create FastAPI test app."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


def mock_get_session():
    """Override get_session dependency."""
    mock_session = AsyncMock()
    yield mock_session


def create_mock_risk_config(owner_id="owner1", enabled=True):
    """Helper to create mock portfolio risk config."""
    config = Mock()
    config.owner_id = owner_id
    config.daily_loss_cap_pct = 5.0
    config.weekly_loss_cap_pct = 10.0
    config.max_drawdown_pct = 15.0
    config.max_total_exposure_pct = 80.0
    config.enabled = enabled
    config.to_dict = Mock(return_value={
        "owner_id": owner_id,
        "daily_loss_cap_pct": 5.0,
        "weekly_loss_cap_pct": 10.0,
        "max_drawdown_pct": 15.0,
        "max_total_exposure_pct": 80.0,
        "enabled": enabled,
    })
    return config


class TestGetPortfolioRisk:
    """Tests for GET /portfolio/risk/{owner_id}."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.portfolio.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_get_risk_config_success(self, client):
        """Returns existing risk configuration."""
        mock_config = create_mock_risk_config()
        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=mock_config)

        with patch("app.routers.portfolio.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/portfolio/risk/owner1")

        assert response.status_code == 200
        data = response.json()
        assert data["owner_id"] == "owner1"
        assert data["daily_loss_cap_pct"] == 5.0
        assert data["enabled"] is True

    def test_get_risk_config_not_found_returns_default(self, client):
        """Returns default config when not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=None)

        with patch("app.routers.portfolio.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/portfolio/risk/owner2")

        assert response.status_code == 200
        data = response.json()
        assert data["owner_id"] == "owner2"
        assert data["daily_loss_cap_pct"] is None
        assert data["enabled"] is False

    def test_response_json_shape(self, client):
        """Response has expected JSON shape."""
        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=None)

        with patch("app.routers.portfolio.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/portfolio/risk/owner1")

        assert response.status_code == 200
        data = response.json()
        assert "owner_id" in data
        assert "daily_loss_cap_pct" in data
        assert "weekly_loss_cap_pct" in data
        assert "max_drawdown_pct" in data
        assert "max_total_exposure_pct" in data
        assert "enabled" in data


class TestCreateOrUpdatePortfolioRisk:
    """Tests for POST /portfolio/risk."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.portfolio.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_create_new_risk_config(self, client):
        """Creates new risk configuration."""
        from app.models import PortfolioRisk as RealPortfolioRisk
        
        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=None)

        mock_new_config = create_mock_risk_config()

        with patch("app.routers.portfolio.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            with patch("app.routers.portfolio.AsyncSession.commit", new_callable=AsyncMock):
                with patch("app.routers.portfolio.AsyncSession.refresh", new_callable=AsyncMock):
                    with patch("app.routers.portfolio.AsyncSession.add", new_callable=Mock):
                        with patch.object(RealPortfolioRisk, "__init__", return_value=None):
                            with patch.object(RealPortfolioRisk, "to_dict", return_value=mock_new_config.to_dict()):
                                response = client.post("/portfolio/risk", json={
                                    "owner_id": "owner1",
                                    "daily_loss_cap_pct": 5.0,
                                    "enabled": True
                                })

        assert response.status_code == 200
        data = response.json()
        assert data["owner_id"] == "owner1"

    def test_update_existing_risk_config(self, client):
        """Updates existing risk configuration."""
        mock_existing = create_mock_risk_config()
        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=mock_existing)

        with patch("app.routers.portfolio.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            with patch("app.routers.portfolio.AsyncSession.commit", new_callable=AsyncMock):
                with patch("app.routers.portfolio.AsyncSession.refresh", new_callable=AsyncMock):
                    response = client.post("/portfolio/risk", json={
                        "owner_id": "owner1",
                        "daily_loss_cap_pct": 8.0,
                        "enabled": False
                    })

        assert response.status_code == 200

    def test_create_with_all_fields(self, client):
        """Creates config with all fields specified."""
        from app.models import PortfolioRisk as RealPortfolioRisk
        
        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=None)

        mock_new_config = create_mock_risk_config()

        with patch("app.routers.portfolio.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            with patch("app.routers.portfolio.AsyncSession.commit", new_callable=AsyncMock):
                with patch("app.routers.portfolio.AsyncSession.refresh", new_callable=AsyncMock):
                    with patch("app.routers.portfolio.AsyncSession.add", new_callable=Mock):
                        with patch.object(RealPortfolioRisk, "__init__", return_value=None):
                            with patch.object(RealPortfolioRisk, "to_dict", return_value=mock_new_config.to_dict()):
                                response = client.post("/portfolio/risk", json={
                                    "owner_id": "owner1",
                                    "daily_loss_cap_pct": 5.0,
                                    "weekly_loss_cap_pct": 10.0,
                                    "max_drawdown_pct": 15.0,
                                    "max_total_exposure_pct": 80.0,
                                    "enabled": True
                                })

        assert response.status_code == 200

    def test_missing_owner_id_rejected(self, client):
        """Missing owner_id returns 422."""
        response = client.post("/portfolio/risk", json={
            "daily_loss_cap_pct": 5.0,
            "enabled": True
        })

        assert response.status_code == 422


class TestGetPortfolioMetrics:
    """Tests for GET /portfolio/metrics/{owner_id}."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.portfolio.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_get_metrics_success(self, client):
        """Returns portfolio metrics."""
        mock_metrics = {
            "owner_id": "owner1",
            "total_equity": 100000.0,
            "cash_balance": 50000.0,
            "unrealized_pnl": 5000.0,
            "realized_pnl": 3000.0,
            "total_exposure": 50000.0,
            "exposure_pct": 50.0,
        }

        with patch("app.routers.portfolio.PortfolioRiskService.get_portfolio_metrics", new_callable=AsyncMock, return_value=mock_metrics):
            response = client.get("/portfolio/metrics/owner1")

        assert response.status_code == 200
        data = response.json()
        assert data["owner_id"] == "owner1"
        assert data["total_equity"] == 100000.0

    def test_empty_portfolio_metrics(self, client):
        """Returns metrics for empty portfolio."""
        mock_metrics = {
            "owner_id": "owner2",
            "total_equity": 0.0,
            "cash_balance": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "total_exposure": 0.0,
            "exposure_pct": 0.0,
        }

        with patch("app.routers.portfolio.PortfolioRiskService.get_portfolio_metrics", new_callable=AsyncMock, return_value=mock_metrics):
            response = client.get("/portfolio/metrics/owner2")

        assert response.status_code == 200
        data = response.json()
        assert data["total_equity"] == 0.0

    def test_metrics_with_open_positions(self, client):
        """Returns metrics with unrealized P&L."""
        mock_metrics = {
            "owner_id": "owner1",
            "total_equity": 105000.0,
            "cash_balance": 50000.0,
            "unrealized_pnl": 5000.0,
            "realized_pnl": 0.0,
            "total_exposure": 50000.0,
            "exposure_pct": 47.6,
        }

        with patch("app.routers.portfolio.PortfolioRiskService.get_portfolio_metrics", new_callable=AsyncMock, return_value=mock_metrics):
            response = client.get("/portfolio/metrics/owner1")

        assert response.status_code == 200
        data = response.json()
        assert data["unrealized_pnl"] == 5000.0

    def test_metrics_response_json_shape(self, client):
        """Response has expected JSON shape."""
        mock_metrics = {
            "owner_id": "owner1",
            "total_equity": 100000.0,
            "cash_balance": 50000.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "total_exposure": 0.0,
            "exposure_pct": 0.0,
        }

        with patch("app.routers.portfolio.PortfolioRiskService.get_portfolio_metrics", new_callable=AsyncMock, return_value=mock_metrics):
            response = client.get("/portfolio/metrics/owner1")

        assert response.status_code == 200
        data = response.json()
        assert "owner_id" in data
        assert "total_equity" in data
        assert "cash_balance" in data
        assert "unrealized_pnl" in data


class TestDeletePortfolioRisk:
    """Tests for DELETE /portfolio/risk/{owner_id}."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.portfolio.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_delete_success(self, client):
        """Deletes existing risk configuration."""
        mock_config = create_mock_risk_config()
        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=mock_config)

        with patch("app.routers.portfolio.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            with patch("app.routers.portfolio.AsyncSession.delete", new_callable=AsyncMock):
                with patch("app.routers.portfolio.AsyncSession.commit", new_callable=AsyncMock):
                    response = client.delete("/portfolio/risk/owner1")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_delete_not_found(self, client):
        """Returns 404 when config not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=None)

        with patch("app.routers.portfolio.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.delete("/portfolio/risk/owner999")

        assert response.status_code == 404


class TestPortfolioEdgeCases:
    """Tests for edge cases and validation."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.portfolio.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_service_exception_propagates(self, client):
        """Service layer exception propagates correctly."""
        with patch("app.routers.portfolio.PortfolioRiskService.get_portfolio_metrics", new_callable=AsyncMock, side_effect=Exception("Service error")):
            with pytest.raises(Exception) as exc_info:
                client.get("/portfolio/metrics/owner1")
            assert "Service error" in str(exc_info.value)

    def test_negative_percentages_accepted(self, client):
        """Negative percentages are accepted (loss caps)."""
        from app.models import PortfolioRisk as RealPortfolioRisk
        
        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=None)

        mock_new_config = create_mock_risk_config()

        with patch("app.routers.portfolio.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            with patch("app.routers.portfolio.AsyncSession.commit", new_callable=AsyncMock):
                with patch("app.routers.portfolio.AsyncSession.refresh", new_callable=AsyncMock):
                    with patch("app.routers.portfolio.AsyncSession.add", new_callable=Mock):
                        with patch.object(RealPortfolioRisk, "__init__", return_value=None):
                            with patch.object(RealPortfolioRisk, "to_dict", return_value=mock_new_config.to_dict()):
                                response = client.post("/portfolio/risk", json={
                                    "owner_id": "owner1",
                                    "daily_loss_cap_pct": -5.0,
                                    "enabled": True
                                })

        assert response.status_code == 200

    def test_zero_percentages_accepted(self, client):
        """Zero percentages are accepted."""
        from app.models import PortfolioRisk as RealPortfolioRisk
        
        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=None)

        mock_new_config = create_mock_risk_config()

        with patch("app.routers.portfolio.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            with patch("app.routers.portfolio.AsyncSession.commit", new_callable=AsyncMock):
                with patch("app.routers.portfolio.AsyncSession.refresh", new_callable=AsyncMock):
                    with patch("app.routers.portfolio.AsyncSession.add", new_callable=Mock):
                        with patch.object(RealPortfolioRisk, "__init__", return_value=None):
                            with patch.object(RealPortfolioRisk, "to_dict", return_value=mock_new_config.to_dict()):
                                response = client.post("/portfolio/risk", json={
                                    "owner_id": "owner1",
                                    "daily_loss_cap_pct": 0.0,
                                    "enabled": True
                                })

        assert response.status_code == 200

    def test_disabled_config_still_stored(self, client):
        """Disabled config is still stored."""
        from app.models import PortfolioRisk as RealPortfolioRisk
        
        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=None)

        mock_new_config = create_mock_risk_config(enabled=False)

        with patch("app.routers.portfolio.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            with patch("app.routers.portfolio.AsyncSession.commit", new_callable=AsyncMock):
                with patch("app.routers.portfolio.AsyncSession.refresh", new_callable=AsyncMock):
                    with patch("app.routers.portfolio.AsyncSession.add", new_callable=Mock):
                        with patch.object(RealPortfolioRisk, "__init__", return_value=None):
                            with patch.object(RealPortfolioRisk, "to_dict", return_value=mock_new_config.to_dict()):
                                response = client.post("/portfolio/risk", json={
                                    "owner_id": "owner1",
                                    "enabled": False
                                })

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False
