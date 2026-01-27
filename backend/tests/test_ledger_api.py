"""API tests for ledger endpoints (compliance-critical).

Tests focus on HTTP layer, filtering, error handling, and data separation.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, Mock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.routers.ledger import router


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


def create_mock_ledger_entry(entry_id=1, owner_id="owner1", bot_id=1, asset="USDT", amount=1000.0, is_simulated=True):
    """Helper to create mock ledger entries."""
    entry = Mock()
    entry.id = entry_id
    entry.owner_id = owner_id
    entry.bot_id = bot_id
    entry.asset = asset
    entry.delta_amount = amount
    entry.balance_after = 10000.0
    entry.reason = "trade"
    entry.is_simulated = is_simulated
    entry.created_at = datetime(2025, 1, 15, 10, 0, 0)
    entry.to_dict = Mock(return_value={
        "id": entry_id,
        "owner_id": owner_id,
        "bot_id": bot_id,
        "asset": asset,
        "delta_amount": amount,
        "balance_after": 10000.0,
        "reason": "trade",
        "is_simulated": is_simulated,
        "created_at": "2025-01-15T10:00:00",
    })
    return entry


class TestLedgerList:
    """Tests for GET /ledger/entries."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.ledger.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_returns_all_ledger_entries(self, client):
        """Returns all ledger entries without filters."""
        mock_entries = [
            create_mock_ledger_entry(entry_id=1, asset="USDT"),
            create_mock_ledger_entry(entry_id=2, asset="BTC"),
        ]

        mock_result = Mock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=mock_entries)))

        with patch("app.routers.ledger.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/ledger/entries")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == 1
        assert data[1]["id"] == 2

    def test_filter_by_bot_id(self, client):
        """Filters entries by bot_id."""
        mock_entries = [create_mock_ledger_entry(entry_id=1, bot_id=5)]

        mock_result = Mock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=mock_entries)))

        with patch("app.routers.ledger.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/ledger/entries?bot_id=5")

        assert response.status_code == 200
        data = response.json()
        assert data[0]["bot_id"] == 5

    def test_response_json_shape(self, client):
        """Response has expected JSON shape."""
        mock_entries = [create_mock_ledger_entry()]

        mock_result = Mock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=mock_entries)))

        with patch("app.routers.ledger.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/ledger/entries")

        assert response.status_code == 200
        data = response.json()
        assert "id" in data[0]
        assert "bot_id" in data[0]
        assert "asset" in data[0]
        assert "is_simulated" in data[0]


class TestLedgerBalance:
    """Tests for GET /ledger/balance/{owner_id}/{asset}."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.ledger.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_get_balance_success(self, client):
        """Returns current balance."""
        with patch("app.routers.ledger.LedgerWriterService.get_balance", new_callable=AsyncMock, return_value=5000.0):
            response = client.get("/ledger/balance/owner1/USDT")

        assert response.status_code == 200
        data = response.json()
        assert data["owner_id"] == "owner1"
        assert data["asset"] == "USDT"
        assert data["balance"] == 5000.0

    def test_get_balance_with_bot_id(self, client):
        """Returns balance filtered by bot."""
        with patch("app.routers.ledger.LedgerWriterService.get_balance", new_callable=AsyncMock, return_value=2500.0):
            response = client.get("/ledger/balance/owner1/USDT?bot_id=1")

        assert response.status_code == 200
        data = response.json()
        assert data["bot_id"] == 1
        assert data["balance"] == 2500.0


class TestLedgerTrades:
    """Tests for GET /ledger/trades endpoints."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.ledger.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_get_trades_success(self, client):
        """Returns all trades."""
        mock_trade = Mock()
        mock_trade.to_dict = Mock(return_value={
            "id": 1,
            "bot_id": 1,
            "side": "buy",
            "base_asset": "BTC",
            "quantity": 0.1,
            "price": 50000.0,
        })

        mock_result = Mock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[mock_trade])))

        with patch("app.routers.ledger.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/ledger/trades")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == 1

    def test_get_trade_by_id_success(self, client):
        """Returns specific trade."""
        mock_trade = Mock()
        mock_trade.to_dict = Mock(return_value={"id": 5, "side": "sell"})

        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=mock_trade)

        with patch("app.routers.ledger.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/ledger/trades/5")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 5

    def test_get_trade_not_found(self, client):
        """Returns 404 for missing trade."""
        mock_result = Mock()
        mock_result.scalar_one_or_none = Mock(return_value=None)

        with patch("app.routers.ledger.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/ledger/trades/999")

        assert response.status_code == 404


class TestLedgerTaxLots:
    """Tests for GET /ledger/tax-lots endpoints."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.ledger.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_get_tax_lots_success(self, client):
        """Returns tax lots."""
        mock_lot = Mock()
        mock_lot.to_dict = Mock(return_value={
            "id": 1,
            "asset": "BTC",
            "quantity_remaining": 0.5,
            "unit_cost": 50000.0,
        })

        mock_result = Mock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[mock_lot])))

        with patch("app.routers.ledger.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/ledger/tax-lots")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["asset"] == "BTC"

    def test_get_tax_lot_summary(self, client):
        """Returns tax lot summary."""
        mock_row = Mock()
        mock_row.asset = "BTC"
        mock_row.total_quantity = 1.0
        mock_row.total_cost = 50000.0
        mock_row.avg_cost = 50000.0
        mock_row.lot_count = 2

        mock_result = [mock_row]

        with patch("app.routers.ledger.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/ledger/tax-lots/summary/owner1")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["asset"] == "BTC"
        assert data[0]["lot_count"] == 2


class TestLedgerRealizedGains:
    """Tests for GET /ledger/realized-gains endpoints."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.ledger.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_get_realized_gains_success(self, client):
        """Returns realized gains."""
        mock_gain = Mock()
        mock_gain.to_dict = Mock(return_value={
            "id": 1,
            "asset": "BTC",
            "gain_loss": 5000.0,
            "is_long_term": False,
        })

        mock_result = Mock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[mock_gain])))

        with patch("app.routers.ledger.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/ledger/realized-gains")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["gain_loss"] == 5000.0

    def test_filter_by_year(self, client):
        """Filters realized gains by year."""
        mock_result = Mock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[])))

        with patch("app.routers.ledger.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/ledger/realized-gains?year=2024")

        assert response.status_code == 200

    def test_filter_by_long_term(self, client):
        """Filters by long-term status."""
        mock_result = Mock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[])))

        with patch("app.routers.ledger.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/ledger/realized-gains?is_long_term=true")

        assert response.status_code == 200


class TestLedgerCompliance:
    """Tests for compliance-critical data separation."""

    def setup_method(self):
        """Patch get_session dependency."""
        self.mock_get_session_patcher = patch("app.routers.ledger.get_session", mock_get_session)
        self.mock_get_session_patcher.start()

    def teardown_method(self):
        """Stop patching."""
        self.mock_get_session_patcher.stop()

    def test_simulated_entries_in_response(self, client):
        """Response includes is_simulated field."""
        mock_entries = [
            create_mock_ledger_entry(entry_id=1, is_simulated=True),
            create_mock_ledger_entry(entry_id=2, is_simulated=False),
        ]

        mock_result = Mock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=mock_entries)))

        with patch("app.routers.ledger.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/ledger/entries")

        assert response.status_code == 200
        data = response.json()
        assert data[0]["is_simulated"] is True
        assert data[1]["is_simulated"] is False

    def test_empty_result_set(self, client):
        """Handles empty result gracefully."""
        mock_result = Mock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[])))

        with patch("app.routers.ledger.AsyncSession.execute", new_callable=AsyncMock, return_value=mock_result):
            response = client.get("/ledger/entries")

        assert response.status_code == 200
        assert response.json() == []
