"""
Compliance Mode Contract Test (Backend ↔ Frontend)

This test validates that backend compliance endpoints return exactly what
the frontend expects. Ensures report schemas cannot silently drift and
break tax/audit UI.

Contract tests between backend /reports/* endpoints and frontend expectations.
Uses FastAPI TestClient with real routers and mocked service layer.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.services.reporting_service import (
    TaxSummaryRecord,
    AuditLogRecord,
    RealizedGainRecord,
    EquityCurveRecord,
    EquityEvent,
)


@pytest.fixture
def mock_reporting_service():
    """Mock ReportingService with deterministic responses."""
    with patch('app.routers.reports.ReportingService') as mock:
        yield mock


@pytest.fixture
def mock_get_session():
    """Mock database session."""
    with patch('app.routers.reports.get_session') as mock:
        mock.return_value = AsyncMock()
        yield mock


@pytest.fixture
def client():
    """FastAPI test client with real routers."""
    return TestClient(app)


class TestTaxSummaryContract:
    """Contract tests for GET /reports/tax-summary/{year}"""

    def test_http_correctness(self, client, mock_reporting_service, mock_get_session):
        """Verify HTTP 200, content-type application/json"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_tax_summary = AsyncMock(
            return_value=TaxSummaryRecord(
                total_realized_gain=5000.0,
                short_term_gain=3000.0,
                long_term_gain=2000.0,
                lot_count=10,
                trade_count=25,
            )
        )

        response = client.get('/api/reports/tax-summary/2025?is_simulated=true')

        assert response.status_code == 200
        assert response.headers['content-type'] == 'application/json'

    def test_required_query_params_enforced(self, client, mock_get_session):
        """Verify is_simulated parameter is required"""
        response = client.get('/api/reports/tax-summary/2025')

        assert response.status_code == 422
        assert 'detail' in response.json()

    def test_json_schema_shape(self, client, mock_reporting_service, mock_get_session):
        """Verify JSON schema matches frontend expectations"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_tax_summary = AsyncMock(
            return_value=TaxSummaryRecord(
                total_realized_gain=7000.0,
                short_term_gain=5000.0,
                long_term_gain=2000.0,
                lot_count=15,
                trade_count=30,
            )
        )

        response = client.get('/api/reports/tax-summary/2025?is_simulated=true')
        data = response.json()

        # Assert presence and types
        assert 'total_realized_gain' in data
        assert isinstance(data['total_realized_gain'], (int, float))

        assert 'short_term_gain' in data
        assert isinstance(data['short_term_gain'], (int, float))

        assert 'long_term_gain' in data
        assert isinstance(data['long_term_gain'], (int, float))

        assert 'lot_count' in data
        assert isinstance(data['lot_count'], int)

        assert 'trade_count' in data
        assert isinstance(data['trade_count'], int)

    def test_field_values_match_expected(self, client, mock_reporting_service, mock_get_session):
        """Verify field values are correctly returned"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_tax_summary = AsyncMock(
            return_value=TaxSummaryRecord(
                total_realized_gain=7000.0,
                short_term_gain=5000.0,
                long_term_gain=2000.0,
                lot_count=15,
                trade_count=30,
            )
        )

        response = client.get('/api/reports/tax-summary/2025?is_simulated=true')
        data = response.json()

        assert data['total_realized_gain'] == 7000.0
        assert data['short_term_gain'] == 5000.0
        assert data['long_term_gain'] == 2000.0
        assert data['lot_count'] == 15
        assert data['trade_count'] == 30

    def test_invalid_year_returns_400(self, client, mock_get_session):
        """Verify invalid year returns 400 Bad Request"""
        response = client.get('/api/reports/tax-summary/1999?is_simulated=true')

        assert response.status_code == 400
        assert 'detail' in response.json()
        assert 'Invalid year' in response.json()['detail']

    def test_year_too_high_returns_400(self, client, mock_get_session):
        """Verify year > 2100 returns 400"""
        response = client.get('/api/reports/tax-summary/2101?is_simulated=true')

        assert response.status_code == 400
        assert 'detail' in response.json()

    def test_negative_gains_allowed(self, client, mock_reporting_service, mock_get_session):
        """Verify negative gains are valid (losses)"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_tax_summary = AsyncMock(
            return_value=TaxSummaryRecord(
                total_realized_gain=-1500.0,
                short_term_gain=-1000.0,
                long_term_gain=-500.0,
                lot_count=8,
                trade_count=20,
            )
        )

        response = client.get('/api/reports/tax-summary/2025?is_simulated=true')
        data = response.json()

        assert response.status_code == 200
        assert data['total_realized_gain'] == -1500.0
        assert data['short_term_gain'] == -1000.0
        assert data['long_term_gain'] == -500.0

    def test_zero_values_allowed(self, client, mock_reporting_service, mock_get_session):
        """Verify zero values are valid"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_tax_summary = AsyncMock(
            return_value=TaxSummaryRecord(
                total_realized_gain=0.0,
                short_term_gain=0.0,
                long_term_gain=0.0,
                lot_count=0,
                trade_count=0,
            )
        )

        response = client.get('/api/reports/tax-summary/2025?is_simulated=true')
        data = response.json()

        assert response.status_code == 200
        assert data['total_realized_gain'] == 0.0
        assert data['lot_count'] == 0


class TestAuditLogContract:
    """Contract tests for GET /reports/audit-log"""

    def test_http_correctness(self, client, mock_reporting_service, mock_get_session):
        """Verify HTTP 200, content-type application/json"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_audit_log = AsyncMock(
            return_value=[
                AuditLogRecord(
                    id=1,
                    timestamp=datetime(2025, 1, 26, 10, 0, 0),
                    severity='info',
                    source='trading_engine',
                    bot_id=1,
                    message='Trade executed',
                    details=None,
                )
            ]
        )

        response = client.get('/api/reports/audit-log?is_simulated=true')

        assert response.status_code == 200
        assert response.headers['content-type'] == 'application/json'

    def test_required_query_params_enforced(self, client, mock_get_session):
        """Verify is_simulated parameter is required"""
        response = client.get('/api/reports/audit-log')

        assert response.status_code == 422
        assert 'detail' in response.json()

    def test_json_schema_shape(self, client, mock_reporting_service, mock_get_session):
        """Verify JSON schema matches frontend expectations"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_audit_log = AsyncMock(
            return_value=[
                AuditLogRecord(
                    id=1,
                    timestamp=datetime(2025, 1, 26, 10, 0, 0),
                    severity='warning',
                    source='risk_management',
                    bot_id=2,
                    message='Stop loss triggered',
                    details={'loss_pct': 5.0},
                )
            ]
        )

        response = client.get('/api/reports/audit-log?is_simulated=true')
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 1

        entry = data[0]
        assert 'id' in entry
        assert isinstance(entry['id'], int)

        assert 'timestamp' in entry
        assert isinstance(entry['timestamp'], str)  # ISO8601 string

        assert 'severity' in entry
        assert isinstance(entry['severity'], str)

        assert 'source' in entry
        assert isinstance(entry['source'], str)

        assert 'bot_id' in entry
        # bot_id can be int or null

        assert 'message' in entry
        assert isinstance(entry['message'], str)

        assert 'details' in entry
        # details can be dict or null

    def test_array_structure(self, client, mock_reporting_service, mock_get_session):
        """Verify response is array, empty lists allowed"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_audit_log = AsyncMock(return_value=[])

        response = client.get('/api/reports/audit-log?is_simulated=true')
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 0

    def test_multiple_entries(self, client, mock_reporting_service, mock_get_session):
        """Verify multiple entries returned correctly"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_audit_log = AsyncMock(
            return_value=[
                AuditLogRecord(
                    id=1,
                    timestamp=datetime(2025, 1, 26, 10, 0, 0),
                    severity='info',
                    source='system',
                    bot_id=None,
                    message='System started',
                    details=None,
                ),
                AuditLogRecord(
                    id=2,
                    timestamp=datetime(2025, 1, 26, 11, 0, 0),
                    severity='error',
                    source='trading_engine',
                    bot_id=1,
                    message='Order failed',
                    details={'reason': 'Insufficient balance'},
                ),
            ]
        )

        response = client.get('/api/reports/audit-log?is_simulated=true')
        data = response.json()

        assert len(data) == 2
        assert data[0]['id'] == 1
        assert data[1]['id'] == 2

    def test_null_bot_id_allowed(self, client, mock_reporting_service, mock_get_session):
        """Verify bot_id can be null (system-level events)"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_audit_log = AsyncMock(
            return_value=[
                AuditLogRecord(
                    id=1,
                    timestamp=datetime(2025, 1, 26, 10, 0, 0),
                    severity='info',
                    source='system',
                    bot_id=None,
                    message='System event',
                    details=None,
                )
            ]
        )

        response = client.get('/api/reports/audit-log?is_simulated=true')
        data = response.json()

        assert data[0]['bot_id'] is None

    def test_null_details_allowed(self, client, mock_reporting_service, mock_get_session):
        """Verify details can be null"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_audit_log = AsyncMock(
            return_value=[
                AuditLogRecord(
                    id=1,
                    timestamp=datetime(2025, 1, 26, 10, 0, 0),
                    severity='info',
                    source='system',
                    bot_id=1,
                    message='Simple message',
                    details=None,
                )
            ]
        )

        response = client.get('/api/reports/audit-log?is_simulated=true')
        data = response.json()

        assert data[0]['details'] is None

    def test_invalid_severity_returns_400(self, client, mock_get_session):
        """Verify invalid severity returns 400"""
        response = client.get('/api/reports/audit-log?is_simulated=true&severity=invalid')

        assert response.status_code == 400
        assert 'detail' in response.json()
        assert 'severity' in response.json()['detail'].lower()

    def test_severity_filter_accepted(self, client, mock_reporting_service, mock_get_session):
        """Verify valid severity values accepted"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_audit_log = AsyncMock(return_value=[])

        for severity in ['info', 'warning', 'error']:
            response = client.get(f'/api/reports/audit-log?is_simulated=true&severity={severity}')
            assert response.status_code == 200


class TestRealizedGainsContract:
    """Contract tests for GET /reports/realized-gains"""

    def test_http_correctness(self, client, mock_reporting_service, mock_get_session):
        """Verify HTTP 200, content-type application/json"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_realized_gains = AsyncMock(
            return_value=[
                RealizedGainRecord(
                    asset='BTC',
                    quantity=0.5,
                    buy_date=datetime(2025, 1, 1, 10, 0, 0),
                    sell_date=datetime(2025, 1, 15, 10, 0, 0),
                    cost_basis=45000.0,
                    proceeds=48000.0,
                    gain_loss=3000.0,
                    holding_period_days=14,
                    bot_id=1,
                    is_simulated=True,
                )
            ]
        )

        response = client.get('/api/reports/realized-gains?is_simulated=true')

        assert response.status_code == 200
        assert response.headers['content-type'] == 'application/json'

    def test_required_query_params_enforced(self, client, mock_get_session):
        """Verify is_simulated parameter is required"""
        response = client.get('/api/reports/realized-gains')

        assert response.status_code == 422
        assert 'detail' in response.json()

    def test_json_schema_shape(self, client, mock_reporting_service, mock_get_session):
        """Verify JSON schema matches frontend expectations"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_realized_gains = AsyncMock(
            return_value=[
                RealizedGainRecord(
                    asset='ETH',
                    quantity=2.0,
                    buy_date=datetime(2025, 1, 1, 10, 0, 0),
                    sell_date=datetime(2025, 1, 20, 10, 0, 0),
                    cost_basis=6000.0,
                    proceeds=6500.0,
                    gain_loss=500.0,
                    holding_period_days=19,
                    bot_id=2,
                    is_simulated=True,
                )
            ]
        )

        response = client.get('/api/reports/realized-gains?is_simulated=true')
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 1

        entry = data[0]
        assert 'asset' in entry
        assert isinstance(entry['asset'], str)

        assert 'quantity' in entry
        assert isinstance(entry['quantity'], (int, float))

        assert 'buy_date' in entry
        assert isinstance(entry['buy_date'], str)

        assert 'sell_date' in entry
        assert isinstance(entry['sell_date'], str)

        assert 'cost_basis' in entry
        assert isinstance(entry['cost_basis'], (int, float))

        assert 'proceeds' in entry
        assert isinstance(entry['proceeds'], (int, float))

        assert 'gain_loss' in entry
        assert isinstance(entry['gain_loss'], (int, float))

        assert 'holding_period_days' in entry
        assert isinstance(entry['holding_period_days'], int)

        assert 'bot_id' in entry
        # bot_id can be int or null

        assert 'is_simulated' in entry
        assert isinstance(entry['is_simulated'], bool)

    def test_array_structure(self, client, mock_reporting_service, mock_get_session):
        """Verify response is array, empty lists allowed"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_realized_gains = AsyncMock(return_value=[])

        response = client.get('/api/reports/realized-gains?is_simulated=true')
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 0

    def test_negative_gain_loss_allowed(self, client, mock_reporting_service, mock_get_session):
        """Verify negative gain_loss (losses) are valid"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_realized_gains = AsyncMock(
            return_value=[
                RealizedGainRecord(
                    asset='BTC',
                    quantity=0.5,
                    buy_date=datetime(2025, 1, 1, 10, 0, 0),
                    sell_date=datetime(2025, 1, 15, 10, 0, 0),
                    cost_basis=50000.0,
                    proceeds=45000.0,
                    gain_loss=-5000.0,
                    holding_period_days=14,
                    bot_id=1,
                    is_simulated=True,
                )
            ]
        )

        response = client.get('/api/reports/realized-gains?is_simulated=true')
        data = response.json()

        assert response.status_code == 200
        assert data[0]['gain_loss'] == -5000.0

    def test_multiple_gains(self, client, mock_reporting_service, mock_get_session):
        """Verify multiple gain records returned correctly"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_realized_gains = AsyncMock(
            return_value=[
                RealizedGainRecord(
                    asset='BTC',
                    quantity=0.5,
                    buy_date=datetime(2025, 1, 1, 10, 0, 0),
                    sell_date=datetime(2025, 1, 15, 10, 0, 0),
                    cost_basis=45000.0,
                    proceeds=48000.0,
                    gain_loss=3000.0,
                    holding_period_days=14,
                    bot_id=1,
                    is_simulated=True,
                ),
                RealizedGainRecord(
                    asset='ETH',
                    quantity=2.0,
                    buy_date=datetime(2025, 1, 5, 10, 0, 0),
                    sell_date=datetime(2025, 1, 20, 10, 0, 0),
                    cost_basis=6000.0,
                    proceeds=6500.0,
                    gain_loss=500.0,
                    holding_period_days=15,
                    bot_id=1,
                    is_simulated=True,
                ),
            ]
        )

        response = client.get('/api/reports/realized-gains?is_simulated=true')
        data = response.json()

        assert len(data) == 2
        assert data[0]['asset'] == 'BTC'
        assert data[1]['asset'] == 'ETH'

    def test_timestamp_format_is_iso8601(self, client, mock_reporting_service, mock_get_session):
        """Verify timestamps are ISO8601 strings"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_realized_gains = AsyncMock(
            return_value=[
                RealizedGainRecord(
                    asset='BTC',
                    quantity=0.5,
                    buy_date=datetime(2025, 1, 1, 10, 0, 0),
                    sell_date=datetime(2025, 1, 15, 10, 0, 0),
                    cost_basis=45000.0,
                    proceeds=48000.0,
                    gain_loss=3000.0,
                    holding_period_days=14,
                    bot_id=1,
                    is_simulated=True,
                )
            ]
        )

        response = client.get('/api/reports/realized-gains?is_simulated=true')
        data = response.json()

        # ISO8601 format check (simplified)
        assert 'T' in data[0]['buy_date']
        assert 'T' in data[0]['sell_date']


class TestEquityCurveContract:
    """Contract tests for GET /reports/equity-curve"""

    def test_http_correctness(self, client, mock_reporting_service, mock_get_session):
        """Verify HTTP 200, content-type application/json"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_equity_curve = AsyncMock(
            return_value=(
                [
                    EquityCurveRecord(
                        timestamp=datetime(2025, 1, 1, 10, 0, 0),
                        equity=10000.0,
                    )
                ],
                []
            )
        )

        response = client.get('/api/reports/equity-curve?is_simulated=true')

        assert response.status_code == 200
        assert response.headers['content-type'] == 'application/json'

    def test_required_query_params_enforced(self, client, mock_get_session):
        """Verify is_simulated parameter is required"""
        response = client.get('/api/reports/equity-curve')

        assert response.status_code == 422
        assert 'detail' in response.json()

    def test_json_schema_shape(self, client, mock_reporting_service, mock_get_session):
        """Verify JSON schema matches frontend expectations"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_equity_curve = AsyncMock(
            return_value=(
                [
                    EquityCurveRecord(
                        timestamp=datetime(2025, 1, 1, 10, 0, 0),
                        equity=10000.0,
                    ),
                    EquityCurveRecord(
                        timestamp=datetime(2025, 1, 15, 10, 0, 0),
                        equity=11234.56,
                    ),
                ],
                [
                    EquityEvent(
                        timestamp=datetime(2025, 1, 10, 10, 0, 0),
                        event_type='strategy_switch',
                        description='Switched to momentum',
                        bot_id=1,
                    )
                ]
            )
        )

        response = client.get('/api/reports/equity-curve?is_simulated=true')
        data = response.json()

        # Top-level structure
        assert 'curve' in data
        assert isinstance(data['curve'], list)

        assert 'events' in data
        assert isinstance(data['events'], list)

        # Curve point structure
        if len(data['curve']) > 0:
            point = data['curve'][0]
            assert 'timestamp' in point
            assert isinstance(point['timestamp'], str)

            assert 'equity' in point
            assert isinstance(point['equity'], (int, float))

        # Event structure
        if len(data['events']) > 0:
            event = data['events'][0]
            assert 'timestamp' in event
            assert isinstance(event['timestamp'], str)

            assert 'event_type' in event
            assert isinstance(event['event_type'], str)

            assert 'description' in event
            assert isinstance(event['description'], str)

            assert 'bot_id' in event
            # bot_id can be int or null

    def test_empty_curve_allowed(self, client, mock_reporting_service, mock_get_session):
        """Verify empty curve is valid"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_equity_curve = AsyncMock(
            return_value=([], [])
        )

        response = client.get('/api/reports/equity-curve?is_simulated=true')
        data = response.json()

        assert response.status_code == 200
        assert len(data['curve']) == 0
        assert len(data['events']) == 0

    def test_curve_without_events(self, client, mock_reporting_service, mock_get_session):
        """Verify curve can exist without events"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_equity_curve = AsyncMock(
            return_value=(
                [
                    EquityCurveRecord(
                        timestamp=datetime(2025, 1, 1, 10, 0, 0),
                        equity=10000.0,
                    )
                ],
                []
            )
        )

        response = client.get('/api/reports/equity-curve?is_simulated=true')
        data = response.json()

        assert len(data['curve']) == 1
        assert len(data['events']) == 0

    def test_multiple_equity_points(self, client, mock_reporting_service, mock_get_session):
        """Verify multiple equity curve points"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_equity_curve = AsyncMock(
            return_value=(
                [
                    EquityCurveRecord(
                        timestamp=datetime(2025, 1, 1, 10, 0, 0),
                        equity=10000.0,
                    ),
                    EquityCurveRecord(
                        timestamp=datetime(2025, 1, 15, 10, 0, 0),
                        equity=11000.0,
                    ),
                    EquityCurveRecord(
                        timestamp=datetime(2025, 1, 26, 10, 0, 0),
                        equity=12000.0,
                    ),
                ],
                []
            )
        )

        response = client.get('/api/reports/equity-curve?is_simulated=true')
        data = response.json()

        assert len(data['curve']) == 3
        assert data['curve'][0]['equity'] == 10000.0
        assert data['curve'][1]['equity'] == 11000.0
        assert data['curve'][2]['equity'] == 12000.0

    def test_timestamp_ordering(self, client, mock_reporting_service, mock_get_session):
        """Verify timestamps are in ascending order"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_equity_curve = AsyncMock(
            return_value=(
                [
                    EquityCurveRecord(
                        timestamp=datetime(2025, 1, 1, 10, 0, 0),
                        equity=10000.0,
                    ),
                    EquityCurveRecord(
                        timestamp=datetime(2025, 1, 15, 10, 0, 0),
                        equity=11000.0,
                    ),
                ],
                []
            )
        )

        response = client.get('/api/reports/equity-curve?is_simulated=true')
        data = response.json()

        timestamps = [point['timestamp'] for point in data['curve']]
        assert timestamps == sorted(timestamps)


class TestContractStability:
    """Tests to ensure contract stability and backward compatibility"""

    def test_all_endpoints_return_json(self, client, mock_reporting_service, mock_get_session):
        """Verify all compliance endpoints return JSON"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_tax_summary = AsyncMock(
            return_value=TaxSummaryRecord(
                total_realized_gain=0.0,
                short_term_gain=0.0,
                long_term_gain=0.0,
                lot_count=0,
                trade_count=0,
            )
        )
        mock_service_instance.get_audit_log = AsyncMock(return_value=[])
        mock_service_instance.get_realized_gains = AsyncMock(return_value=[])
        mock_service_instance.get_equity_curve = AsyncMock(return_value=([], []))

        endpoints = [
            '/api/reports/tax-summary/2025?is_simulated=true',
            '/api/reports/audit-log?is_simulated=true',
            '/api/reports/realized-gains?is_simulated=true',
            '/api/reports/equity-curve?is_simulated=true',
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            assert 'application/json' in response.headers['content-type']

    def test_no_extra_fields_in_response(self, client, mock_reporting_service, mock_get_session):
        """Verify responses don't contain unexpected extra fields"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_tax_summary = AsyncMock(
            return_value=TaxSummaryRecord(
                total_realized_gain=5000.0,
                short_term_gain=3000.0,
                long_term_gain=2000.0,
                lot_count=10,
                trade_count=25,
            )
        )

        response = client.get('/api/reports/tax-summary/2025?is_simulated=true')
        data = response.json()

        expected_fields = {
            'total_realized_gain',
            'short_term_gain',
            'long_term_gain',
            'lot_count',
            'trade_count',
        }

        assert set(data.keys()) == expected_fields

    def test_response_size_greater_than_zero_when_data_present(
        self, client, mock_reporting_service, mock_get_session
    ):
        """Verify response size > 0 when data is present"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_realized_gains = AsyncMock(
            return_value=[
                RealizedGainRecord(
                    asset='BTC',
                    quantity=0.5,
                    buy_date=datetime(2025, 1, 1, 10, 0, 0),
                    sell_date=datetime(2025, 1, 15, 10, 0, 0),
                    cost_basis=45000.0,
                    proceeds=48000.0,
                    gain_loss=3000.0,
                    holding_period_days=14,
                    bot_id=1,
                    is_simulated=True,
                )
            ]
        )

        response = client.get('/api/reports/realized-gains?is_simulated=true')
        data = response.json()

        assert len(data) > 0

    def test_deterministic_response_on_repeated_calls(
        self, client, mock_reporting_service, mock_get_session
    ):
        """Verify stable order on repeated calls"""
        mock_service_instance = mock_reporting_service.return_value
        mock_service_instance.get_tax_summary = AsyncMock(
            return_value=TaxSummaryRecord(
                total_realized_gain=5000.0,
                short_term_gain=3000.0,
                long_term_gain=2000.0,
                lot_count=10,
                trade_count=25,
            )
        )

        response1 = client.get('/api/reports/tax-summary/2025?is_simulated=true')
        data1 = response1.json()

        response2 = client.get('/api/reports/tax-summary/2025?is_simulated=true')
        data2 = response2.json()

        assert data1 == data2
