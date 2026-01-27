"""API tests for reports endpoints (compliance-critical).

Tests focus on HTTP layer, status codes, parameter validation, and JSON response shapes.
Service layer is mocked to isolate API contract testing.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.routers.reports import router
from app.services.reporting_service import (
    TaxSummaryRecord,
    AuditLogRecord,
    RealizedGainRecord,
    EquityCurveRecord,
    EquityEvent,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def app():
    """Create FastAPI test app."""
    test_app = FastAPI()
    test_app.include_router(router, prefix="/reports")
    return test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_session():
    """Create mock database session."""
    session = AsyncMock()
    return session


def mock_get_session():
    """Override get_session dependency."""
    mock_session = AsyncMock()
    yield mock_session


# ============================================================================
# Tax Summary Endpoint Tests
# ============================================================================


class TestTaxSummaryEndpoint:
    """Tests for GET /tax-summary/{year}."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.reports.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_success_case(self, client):
        """Success case returns 200 with correct JSON shape."""
        mock_summary = TaxSummaryRecord(
            total_realized_gain=15000.50,
            short_term_gain=8000.25,
            long_term_gain=7000.25,
            lot_count=42,
            trade_count=100,
        )

        with patch("app.services.reporting_service.ReportingService.get_tax_summary", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = mock_summary

            response = client.get("/reports/tax-summary/2025?is_simulated=true")

            assert response.status_code == 200
            data = response.json()
            assert data["total_realized_gain"] == 15000.50
            assert data["short_term_gain"] == 8000.25
            assert data["long_term_gain"] == 7000.25
            assert data["lot_count"] == 42
            assert data["trade_count"] == 100

    def test_owner_filter(self, client):
        """Owner filter parameter is passed to service."""
        mock_summary = TaxSummaryRecord(
            total_realized_gain=0.0,
            short_term_gain=0.0,
            long_term_gain=0.0,
            lot_count=0,
            trade_count=0,
        )

        with patch("app.services.reporting_service.ReportingService.get_tax_summary", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = mock_summary

            response = client.get("/reports/tax-summary/2025?is_simulated=false&owner_id=user123")

            assert response.status_code == 200
            mock_method.assert_called_once()
            call_kwargs = mock_method.call_args.kwargs
            assert call_kwargs["owner_id"] == "user123"

    def test_year_parameter_handling(self, client):
        """Year parameter is extracted from path."""
        mock_summary = TaxSummaryRecord(
            total_realized_gain=0.0,
            short_term_gain=0.0,
            long_term_gain=0.0,
            lot_count=0,
            trade_count=0,
        )

        with patch("app.services.reporting_service.ReportingService.get_tax_summary", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = mock_summary

            response = client.get("/reports/tax-summary/2023?is_simulated=true")

            assert response.status_code == 200
            mock_method.assert_called_once()
            call_kwargs = mock_method.call_args.kwargs
            assert call_kwargs["year"] == 2023

    def test_simulated_vs_live_mode(self, client):
        """Simulated vs live mode parameter is passed to service."""
        mock_summary = TaxSummaryRecord(
            total_realized_gain=0.0,
            short_term_gain=0.0,
            long_term_gain=0.0,
            lot_count=0,
            trade_count=0,
        )

        with patch("app.services.reporting_service.ReportingService.get_tax_summary", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = mock_summary

            # Test simulated=true
            response = client.get("/reports/tax-summary/2025?is_simulated=true")
            assert response.status_code == 200
            call_kwargs = mock_method.call_args.kwargs
            assert call_kwargs["is_simulated"] is True

            # Test simulated=false
            response = client.get("/reports/tax-summary/2025?is_simulated=false")
            assert response.status_code == 200
            call_kwargs = mock_method.call_args.kwargs
            assert call_kwargs["is_simulated"] is False

    def test_negative_gains_allowed(self, client):
        """Negative gains are allowed in response."""
        mock_summary = TaxSummaryRecord(
            total_realized_gain=-5000.00,
            short_term_gain=-3000.00,
            long_term_gain=-2000.00,
            lot_count=10,
            trade_count=25,
        )

        with patch("app.services.reporting_service.ReportingService.get_tax_summary", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = mock_summary

            response = client.get("/reports/tax-summary/2025?is_simulated=true")

            assert response.status_code == 200
            data = response.json()
            assert data["total_realized_gain"] == -5000.00
            assert data["short_term_gain"] == -3000.00
            assert data["long_term_gain"] == -2000.00

    def test_missing_required_is_simulated_param(self, client):
        """Missing is_simulated parameter returns 422."""
        response = client.get("/reports/tax-summary/2025")

        assert response.status_code == 422

    def test_invalid_year_too_low(self, client):
        """Invalid year (< 2000) returns 400."""
        response = client.get("/reports/tax-summary/1999?is_simulated=true")

        assert response.status_code == 400
        assert "Invalid year" in response.json()["detail"]

    def test_invalid_year_too_high(self, client):
        """Invalid year (> 2100) returns 400."""
        response = client.get("/reports/tax-summary/2101?is_simulated=true")

        assert response.status_code == 400
        assert "Invalid year" in response.json()["detail"]

    def test_response_json_shape(self, client):
        """Response has expected JSON shape."""
        mock_summary = TaxSummaryRecord(
            total_realized_gain=1000.0,
            short_term_gain=600.0,
            long_term_gain=400.0,
            lot_count=5,
            trade_count=10,
        )

        with patch("app.services.reporting_service.ReportingService.get_tax_summary", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = mock_summary

            response = client.get("/reports/tax-summary/2025?is_simulated=true")

            assert response.status_code == 200
            data = response.json()
            assert "total_realized_gain" in data
            assert "short_term_gain" in data
            assert "long_term_gain" in data
            assert "lot_count" in data
            assert "trade_count" in data
            assert isinstance(data["total_realized_gain"], (int, float))
            assert isinstance(data["lot_count"], int)

    def test_service_exception_propagates(self, client):
        """Service layer exception propagates (re-raised in test mode)."""
        with patch("app.services.reporting_service.ReportingService.get_tax_summary", new_callable=AsyncMock) as mock_method:
            mock_method.side_effect = Exception("Database error")

            with pytest.raises(Exception, match="Database error"):
                client.get("/reports/tax-summary/2025?is_simulated=true")


# ============================================================================
# Audit Log Endpoint Tests
# ============================================================================


class TestAuditLogEndpoint:
    """Tests for GET /audit-log."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.reports.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_success_case(self, client):
        """Success case returns 200 with list of audit logs."""
        mock_logs = [
            AuditLogRecord(
                id=1,
                timestamp=datetime(2025, 1, 15, 10, 0, 0),
                severity="error",
                source="risk_management",
                bot_id=1,
                message="Stop loss triggered",
                details={"loss_pct": 5.5},
            ),
            AuditLogRecord(
                id=2,
                timestamp=datetime(2025, 1, 15, 11, 0, 0),
                severity="info",
                source="strategy_rotation",
                bot_id=1,
                message="Strategy switched",
                details={"from": "grid", "to": "dca"},
            ),
        ]

        with patch("app.services.reporting_service.ReportingService.get_audit_log", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = mock_logs

            response = client.get("/reports/audit-log?is_simulated=true")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["id"] == 1
            assert data[0]["severity"] == "error"
            assert data[0]["message"] == "Stop loss triggered"
            assert data[1]["id"] == 2

    def test_bot_filter(self, client):
        """Bot filter parameter is passed to service."""
        with patch("app.services.reporting_service.ReportingService.get_audit_log", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = []

            response = client.get("/reports/audit-log?is_simulated=true&bot_id=5")

            assert response.status_code == 200
            mock_method.assert_called_once()
            call_kwargs = mock_method.call_args.kwargs
            assert call_kwargs["bot_id"] == 5

    def test_severity_filter(self, client):
        """Severity filter parameter is passed to service."""
        with patch("app.services.reporting_service.ReportingService.get_audit_log", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = []

            response = client.get("/reports/audit-log?is_simulated=true&severity=error")

            assert response.status_code == 200
            mock_method.assert_called_once()
            call_kwargs = mock_method.call_args.kwargs
            assert call_kwargs["severity"] == "error"

    def test_date_range_filter(self, client):
        """Date range filter parameters are passed to service."""
        with patch("app.services.reporting_service.ReportingService.get_audit_log", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = []

            response = client.get(
                "/reports/audit-log?is_simulated=true"
                "&start_date=2025-01-01T00:00:00"
                "&end_date=2025-01-31T23:59:59"
            )

            assert response.status_code == 200
            mock_method.assert_called_once()
            call_kwargs = mock_method.call_args.kwargs
            assert call_kwargs["start_date"] is not None
            assert call_kwargs["end_date"] is not None

    def test_empty_result(self, client):
        """Empty result returns 200 with empty list."""
        with patch("app.services.reporting_service.ReportingService.get_audit_log", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = []

            response = client.get("/reports/audit-log?is_simulated=true")

            assert response.status_code == 200
            assert response.json() == []

    def test_invalid_severity(self, client):
        """Invalid severity value returns 400."""
        response = client.get("/reports/audit-log?is_simulated=true&severity=critical")

        assert response.status_code == 400
        assert "severity must be" in response.json()["detail"]

    def test_response_json_shape(self, client):
        """Response has expected JSON shape."""
        mock_logs = [
            AuditLogRecord(
                id=1,
                timestamp=datetime(2025, 1, 15, 10, 0, 0),
                severity="warning",
                source="exchange",
                bot_id=None,
                message="Rate limit approached",
                details=None,
            ),
        ]

        with patch("app.services.reporting_service.ReportingService.get_audit_log", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = mock_logs

            response = client.get("/reports/audit-log?is_simulated=true")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert "id" in data[0]
            assert "timestamp" in data[0]
            assert "severity" in data[0]
            assert "source" in data[0]
            assert "bot_id" in data[0]
            assert "message" in data[0]
            assert "details" in data[0]

    def test_missing_is_simulated_param(self, client):
        """Missing is_simulated parameter returns 422."""
        response = client.get("/reports/audit-log")

        assert response.status_code == 422

    def test_service_exception_propagates(self, client):
        """Service layer exception propagates (re-raised in test mode)."""
        with patch("app.services.reporting_service.ReportingService.get_audit_log", new_callable=AsyncMock) as mock_method:
            mock_method.side_effect = Exception("Database error")

            with pytest.raises(Exception, match="Database error"):
                client.get("/reports/audit-log?is_simulated=true")


# ============================================================================
# Realized Gains Endpoint Tests
# ============================================================================


class TestRealizedGainsEndpoint:
    """Tests for GET /realized-gains."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.reports.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_success_case(self, client):
        """Success case returns 200 with list of realized gains."""
        mock_gains = [
            RealizedGainRecord(
                asset="BTC",
                quantity=0.5,
                buy_date=datetime(2024, 6, 1, 0, 0, 0),
                sell_date=datetime(2025, 1, 15, 0, 0, 0),
                cost_basis=20000.0,
                proceeds=25000.0,
                gain_loss=5000.0,
                holding_period_days=228,
                bot_id=1,
                is_simulated=True,
            ),
        ]

        with patch("app.services.reporting_service.ReportingService.get_realized_gains", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = mock_gains

            response = client.get("/reports/realized-gains?is_simulated=true")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["asset"] == "BTC"
            assert data[0]["gain_loss"] == 5000.0
            assert data[0]["quantity"] == 0.5

    def test_asset_filter(self, client):
        """Asset filter parameter is passed to service."""
        with patch("app.services.reporting_service.ReportingService.get_realized_gains", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = []

            response = client.get("/reports/realized-gains?is_simulated=true&asset=ETH")

            assert response.status_code == 200
            mock_method.assert_called_once()
            call_kwargs = mock_method.call_args.kwargs
            assert call_kwargs["asset"] == "ETH"

    def test_bot_filter(self, client):
        """Bot filter parameter is passed to service."""
        with patch("app.services.reporting_service.ReportingService.get_realized_gains", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = []

            response = client.get("/reports/realized-gains?is_simulated=true&bot_id=3")

            assert response.status_code == 200
            mock_method.assert_called_once()
            call_kwargs = mock_method.call_args.kwargs
            assert call_kwargs["bot_id"] == 3

    def test_loss_scenario(self, client):
        """Loss scenario (negative gain_loss) is allowed."""
        mock_gains = [
            RealizedGainRecord(
                asset="ETH",
                quantity=1.0,
                buy_date=datetime(2024, 12, 1, 0, 0, 0),
                sell_date=datetime(2025, 1, 15, 0, 0, 0),
                cost_basis=3000.0,
                proceeds=2500.0,
                gain_loss=-500.0,
                holding_period_days=45,
                bot_id=2,
                is_simulated=True,
            ),
        ]

        with patch("app.services.reporting_service.ReportingService.get_realized_gains", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = mock_gains

            response = client.get("/reports/realized-gains?is_simulated=true")

            assert response.status_code == 200
            data = response.json()
            assert data[0]["gain_loss"] == -500.0

    def test_missing_is_simulated_param(self, client):
        """Missing is_simulated parameter returns 422."""
        response = client.get("/reports/realized-gains")

        assert response.status_code == 422

    def test_response_json_shape(self, client):
        """Response has expected JSON shape."""
        mock_gains = [
            RealizedGainRecord(
                asset="BTC",
                quantity=0.1,
                buy_date=datetime(2024, 1, 1, 0, 0, 0),
                sell_date=datetime(2025, 1, 1, 0, 0, 0),
                cost_basis=4000.0,
                proceeds=5000.0,
                gain_loss=1000.0,
                holding_period_days=366,
                bot_id=1,
                is_simulated=False,
            ),
        ]

        with patch("app.services.reporting_service.ReportingService.get_realized_gains", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = mock_gains

            response = client.get("/reports/realized-gains?is_simulated=false")

            assert response.status_code == 200
            data = response.json()
            assert "asset" in data[0]
            assert "quantity" in data[0]
            assert "buy_date" in data[0]
            assert "sell_date" in data[0]
            assert "cost_basis" in data[0]
            assert "proceeds" in data[0]
            assert "gain_loss" in data[0]
            assert "holding_period_days" in data[0]
            assert "bot_id" in data[0]
            assert "is_simulated" in data[0]

    def test_empty_result(self, client):
        """Empty result returns 200 with empty list."""
        with patch("app.services.reporting_service.ReportingService.get_realized_gains", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = []

            response = client.get("/reports/realized-gains?is_simulated=true")

            assert response.status_code == 200
            assert response.json() == []

    def test_service_exception_propagates(self, client):
        """Service layer exception propagates (re-raised in test mode)."""
        with patch("app.services.reporting_service.ReportingService.get_realized_gains", new_callable=AsyncMock) as mock_method:
            mock_method.side_effect = Exception("Database error")

            with pytest.raises(Exception, match="Database error"):
                client.get("/reports/realized-gains?is_simulated=true")


# ============================================================================
# Equity Curve Endpoint Tests
# ============================================================================


class TestEquityCurveEndpoint:
    """Tests for GET /equity-curve."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.reports.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_success_case(self, client):
        """Success case returns 200 with curve and events."""
        mock_curve = [
            EquityCurveRecord(
                timestamp=datetime(2025, 1, 1, 0, 0, 0),
                equity=10000.0,
            ),
            EquityCurveRecord(
                timestamp=datetime(2025, 1, 2, 0, 0, 0),
                equity=10500.0,
            ),
        ]
        mock_events = [
            EquityEvent(
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                event_type="strategy_switch",
                description="Switched from grid to dca",
                bot_id=1,
            ),
        ]

        with patch("app.services.reporting_service.ReportingService.get_equity_curve", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = (mock_curve, mock_events)

            response = client.get("/reports/equity-curve?is_simulated=true")

            assert response.status_code == 200
            data = response.json()
            assert "curve" in data
            assert "events" in data
            assert len(data["curve"]) == 2
            assert len(data["events"]) == 1
            assert data["curve"][0]["equity"] == 10000.0
            assert data["events"][0]["event_type"] == "strategy_switch"

    def test_owner_filter(self, client):
        """Owner filter parameter is passed to service."""
        with patch("app.services.reporting_service.ReportingService.get_equity_curve", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = ([], [])

            response = client.get("/reports/equity-curve?is_simulated=true&owner_id=user456")

            assert response.status_code == 200
            mock_method.assert_called_once()
            call_kwargs = mock_method.call_args.kwargs
            assert call_kwargs["owner_id"] == "user456"

    def test_asset_parameter(self, client):
        """Asset parameter is passed to service."""
        with patch("app.services.reporting_service.ReportingService.get_equity_curve", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = ([], [])

            response = client.get("/reports/equity-curve?is_simulated=true&asset=BTC")

            assert response.status_code == 200
            mock_method.assert_called_once()
            call_kwargs = mock_method.call_args.kwargs
            assert call_kwargs["asset"] == "BTC"

    def test_empty_events(self, client):
        """Empty events list is handled correctly."""
        mock_curve = [
            EquityCurveRecord(
                timestamp=datetime(2025, 1, 1, 0, 0, 0),
                equity=10000.0,
            ),
        ]

        with patch("app.services.reporting_service.ReportingService.get_equity_curve", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = (mock_curve, [])

            response = client.get("/reports/equity-curve?is_simulated=true")

            assert response.status_code == 200
            data = response.json()
            assert len(data["curve"]) == 1
            assert data["events"] == []

    def test_multiple_events(self, client):
        """Multiple events are returned correctly."""
        mock_curve = [
            EquityCurveRecord(
                timestamp=datetime(2025, 1, 1, 0, 0, 0),
                equity=10000.0,
            ),
        ]
        mock_events = [
            EquityEvent(
                timestamp=datetime(2025, 1, 1, 10, 0, 0),
                event_type="kill_switch",
                description="Kill switch activated",
                bot_id=1,
            ),
            EquityEvent(
                timestamp=datetime(2025, 1, 1, 14, 0, 0),
                event_type="regime_change",
                description="Regime changed to bearish",
                bot_id=None,
            ),
        ]

        with patch("app.services.reporting_service.ReportingService.get_equity_curve", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = (mock_curve, mock_events)

            response = client.get("/reports/equity-curve?is_simulated=true")

            assert response.status_code == 200
            data = response.json()
            assert len(data["events"]) == 2
            assert data["events"][0]["event_type"] == "kill_switch"
            assert data["events"][1]["event_type"] == "regime_change"

    def test_missing_is_simulated_param(self, client):
        """Missing is_simulated parameter returns 422."""
        response = client.get("/reports/equity-curve")

        assert response.status_code == 422

    def test_response_json_shape(self, client):
        """Response has expected JSON shape."""
        mock_curve = [
            EquityCurveRecord(
                timestamp=datetime(2025, 1, 1, 0, 0, 0),
                equity=10000.0,
            ),
        ]
        mock_events = [
            EquityEvent(
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                event_type="large_loss",
                description="Loss exceeded threshold",
                bot_id=2,
            ),
        ]

        with patch("app.services.reporting_service.ReportingService.get_equity_curve", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = (mock_curve, mock_events)

            response = client.get("/reports/equity-curve?is_simulated=true")

            assert response.status_code == 200
            data = response.json()
            assert "curve" in data
            assert "events" in data
            assert "timestamp" in data["curve"][0]
            assert "equity" in data["curve"][0]
            assert "timestamp" in data["events"][0]
            assert "event_type" in data["events"][0]
            assert "description" in data["events"][0]
            assert "bot_id" in data["events"][0]

    def test_service_exception_propagates(self, client):
        """Service layer exception propagates (re-raised in test mode)."""
        with patch("app.services.reporting_service.ReportingService.get_equity_curve", new_callable=AsyncMock) as mock_method:
            mock_method.side_effect = Exception("Database error")

            with pytest.raises(Exception, match="Database error"):
                client.get("/reports/equity-curve?is_simulated=true")


# ============================================================================
# Additional Edge Cases
# ============================================================================


class TestReportsAPIEdgeCases:
    """Additional edge cases and error handling tests."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.reports.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_invalid_query_parameter_type(self, client):
        """Invalid query parameter type returns 422."""
        response = client.get("/reports/tax-summary/2025?is_simulated=notaboolean")

        assert response.status_code == 422

    def test_tax_summary_year_as_string(self, client):
        """Year parameter must be numeric."""
        response = client.get("/reports/tax-summary/year2025?is_simulated=true")

        assert response.status_code == 422

    def test_audit_log_invalid_datetime_format(self, client):
        """Invalid datetime format returns 422."""
        response = client.get("/reports/audit-log?is_simulated=true&start_date=invalid-date")

        assert response.status_code == 422

    def test_realized_gains_date_range(self, client):
        """Date range parameters are passed to service."""
        with patch("app.services.reporting_service.ReportingService.get_realized_gains", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = []

            response = client.get(
                "/reports/realized-gains?is_simulated=true"
                "&start_date=2025-01-01T00:00:00"
                "&end_date=2025-01-31T23:59:59"
            )

            assert response.status_code == 200
            mock_method.assert_called_once()
            call_kwargs = mock_method.call_args.kwargs
            assert call_kwargs["start_date"] is not None
            assert call_kwargs["end_date"] is not None

    def test_equity_curve_default_asset_parameter(self, client):
        """Asset parameter defaults to USDT."""
        with patch("app.services.reporting_service.ReportingService.get_equity_curve", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = ([], [])

            response = client.get("/reports/equity-curve?is_simulated=true")

            assert response.status_code == 200
            mock_method.assert_called_once()
            call_kwargs = mock_method.call_args.kwargs
            assert call_kwargs["asset"] == "USDT"

    def test_audit_log_all_severities(self, client):
        """All valid severity values are accepted."""
        with patch("app.services.reporting_service.ReportingService.get_audit_log", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = []

            for severity in ["info", "warning", "error"]:
                response = client.get(f"/reports/audit-log?is_simulated=true&severity={severity}")
                assert response.status_code == 200

    def test_tax_summary_zero_values(self, client):
        """Zero values in tax summary are handled correctly."""
        mock_summary = TaxSummaryRecord(
            total_realized_gain=0.0,
            short_term_gain=0.0,
            long_term_gain=0.0,
            lot_count=0,
            trade_count=0,
        )

        with patch("app.services.reporting_service.ReportingService.get_tax_summary", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = mock_summary

            response = client.get("/reports/tax-summary/2025?is_simulated=true")

            assert response.status_code == 200
            data = response.json()
            assert data["total_realized_gain"] == 0.0
            assert data["lot_count"] == 0
