# TradingBot

A comprehensive crypto trading bot system with multiple trading strategies, server-client architecture for managing multiple bot instances, web UI for monitoring and control, and full logging for tax compliance.

> [!Warning]
>This is a AI spec driven project for testing purposes only, it is neither complete nor secure. This has completely and 100% been writen by AI.
>                    *** DO NOT USE WITH YOUR MONEY ***

## Features

- **9 Trading Strategies**: DCA Accumulator, Adaptive Grid, Mean Reversion, Trend Following, Cross-Sectional Momentum, Volatility Breakout, TWAP, VWAP, and Auto Mode
- **Multi-Bot Management**: Run multiple bots simultaneously with different strategies and trading pairs
- **Risk Management**: Virtual wallets, stop losses, drawdown limits, consecutive loss detection, and kill switches
- **Real-time Monitoring**: WebSocket-based live updates for P&L, positions, and bot status
- **Tax Compliance**: Complete order logging with fiscal export functionality
- **Dark Theme UI**: Professional monitoring dashboard

## Technology Stack

### Backend
- Python 3.11+
- FastAPI (async web framework)
- SQLAlchemy with SQLite
- ccxt (unified exchange API)
- asyncio for concurrent bot management

### Frontend
- React 18
- Tailwind CSS
- React Query for state management
- Recharts for P&L visualization
- WebSocket for real-time updates

## Prerequisites

- Python 3.11 or higher
- Node.js 18+
- MEXC exchange account with API key/secret
- (Optional) CoinMarketCap API key
- (Optional) CoinGecko API key

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd tradingbot
chmod +x init.sh
./init.sh
```

### 2. Configure API Keys

Edit `config/exchanges.yaml` with your MEXC API credentials:

```yaml
mexc:
  api_key: "YOUR_API_KEY"
  api_secret: "YOUR_API_SECRET"
```

### 3. Start the Backend

```bash
source venv/bin/activate  # or venv/Scripts/activate on Windows
cd backend
uvicorn app.main:app --reload
```

### 4. Start the Frontend

```bash
cd frontend
npm run dev
```

### 5. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Project Structure

```
tradingbot/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── models/           # SQLAlchemy models
│   │   ├── routers/          # API endpoints
│   │   ├── services/         # Business logic
│   │   ├── strategies/       # Trading strategies
│   │   └── exchange/         # Exchange integration
│   ├── requirements.txt
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Page components
│   │   ├── hooks/            # Custom hooks
│   │   ├── services/         # API services
│   │   └── App.tsx
│   ├── package.json
│   └── tailwind.config.js
├── config/
│   ├── exchanges.yaml        # Exchange API keys
│   ├── email.yaml            # SMTP configuration
│   ├── data_sources.yaml     # External data sources
│   └── defaults.yaml         # Default parameters
├── logs/                     # Bot log files
├── init.sh                   # Setup script
└── README.md
```

## Configuration

### Exchange Configuration (`config/exchanges.yaml`)

```yaml
mexc:
  api_key: "YOUR_API_KEY"
  api_secret: "YOUR_API_SECRET"
  sandbox: false
```

### Email Alerts (`config/email.yaml`)

```yaml
smtp:
  host: "smtp.gmail.com"
  port: 587
  username: "your-email@gmail.com"
  password: "your-app-password"
  use_tls: true

notifications:
  enabled: true
  recipient: "your-email@gmail.com"
```

### Data Sources (`config/data_sources.yaml`)

```yaml
coinmarketcap:
  api_key: "YOUR_CMC_API_KEY"
  priority: 1
  enabled: false

coingecko:
  priority: 2
  enabled: true

poll_frequency: 300  # 5 minutes
```

## Trading Strategies

| Strategy | Description |
|----------|-------------|
| DCA Accumulator | Dollar-cost averaging with configurable intervals |
| Adaptive Grid | Dynamic grid trading with auto-rebalancing |
| Mean Reversion | Trade reversions to mean using Bollinger Bands |
| Trend Following | Conservative long-only momentum with EMA crossover and ATR-based stops |
| Cross-Sectional Momentum | Relative strength strategy that ranks assets and holds top performers |
| Volatility Breakout | Enters on price breakouts following low-volatility compression |
| TWAP | Time-weighted average price execution |
| VWAP | Volume-weighted average price targeting |
| Auto Mode | Regime-based automatic strategy selection policy (detects market regimes and selects optimal strategy) |

## Safety Features

- **Virtual Wallet**: Budget cap separate from actual exchange balance
- **Stop Losses**: Per-trade and per-bot drawdown limits
- **Kill Switch**: Per-bot and global emergency stop
- **Consecutive Loss Protection**: Automatic strategy rotation and pause
- **Time-Based Limits**: Daily and weekly loss limits
- **Dry Run Mode**: Test strategies with real market data, no real orders

## API Endpoints

### Bots
- `GET /api/bots` - List all bots
- `POST /api/bots` - Create new bot
- `GET /api/bots/{id}` - Get bot details
- `PUT /api/bots/{id}` - Update bot config
- `DELETE /api/bots/{id}` - Delete bot
- `POST /api/bots/{id}/start` - Start bot
- `POST /api/bots/{id}/pause` - Pause bot
- `POST /api/bots/{id}/stop` - Stop bot
- `POST /api/bots/{id}/kill` - Kill switch for bot

### Global
- `POST /api/kill-all` - Global kill switch
- `GET /api/stats` - Global statistics
- `GET /api/pnl` - P&L data for chart

### Reports
- `GET /api/reports/pnl` - P&L report
- `GET /api/reports/tax` - Tax export
- `GET /api/reports/fees` - Fee totals

### WebSocket
- `WS /ws` - Real-time updates

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Database

The application uses SQLite for simplicity. The database file is created at `backend/tradingbot.db`.

To reset the database:
```bash
rm backend/tradingbot.db
# Restart the backend server
```

## Security Notes

- Keep your `config/exchanges.yaml` file secure - it contains API keys
- Never commit API keys to version control
- Use API keys with minimal permissions (trade-only, no withdrawal)
- Start with dry run mode to test strategies
- Use small budgets initially when going live

## License

Personal use only.

## Support

For issues and feature requests, please use the project's issue tracker.
