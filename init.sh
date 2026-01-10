#!/bin/bash

# TradingBot - Development Environment Setup Script
# This script sets up and runs the development environment for the crypto trading bot

set -e

echo "=========================================="
echo "  TradingBot - Development Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
        echo -e "${GREEN}Python $PYTHON_VERSION found - OK${NC}"
    else
        echo -e "${RED}Python 3.11+ required, found $PYTHON_VERSION${NC}"
        exit 1
    fi
else
    echo -e "${RED}Python 3 not found. Please install Python 3.11+${NC}"
    exit 1
fi

# Check Node.js version
echo -e "\n${YELLOW}Checking Node.js version...${NC}"
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -ge 18 ]; then
        echo -e "${GREEN}Node.js $(node -v) found - OK${NC}"
    else
        echo -e "${RED}Node.js 18+ required, found $(node -v)${NC}"
        exit 1
    fi
else
    echo -e "${RED}Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi

# Create Python virtual environment if it doesn't exist
echo -e "\n${YELLOW}Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "${GREEN}Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install Python dependencies
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
if [ -f "backend/requirements.txt" ]; then
    pip install -r backend/requirements.txt
    echo -e "${GREEN}Python dependencies installed${NC}"
else
    echo -e "${YELLOW}No requirements.txt found - will be created during development${NC}"
fi

# Install Node.js dependencies
echo -e "\n${YELLOW}Installing Node.js dependencies...${NC}"
if [ -f "frontend/package.json" ]; then
    cd frontend
    npm install
    cd ..
    echo -e "${GREEN}Node.js dependencies installed${NC}"
else
    echo -e "${YELLOW}No package.json found - will be created during development${NC}"
fi

# Create necessary directories
echo -e "\n${YELLOW}Creating directory structure...${NC}"
mkdir -p config
mkdir -p logs
mkdir -p backend/app
mkdir -p frontend/src

# Create sample config files if they don't exist
if [ ! -f "config/exchanges.yaml" ]; then
    cat > config/exchanges.yaml << 'EOF'
# Exchange API Configuration
# Add your API keys here (keep this file secure!)

mexc:
  api_key: "YOUR_MEXC_API_KEY"
  api_secret: "YOUR_MEXC_API_SECRET"
  sandbox: false  # Set to true for testing

# Additional exchanges can be added here
# binance:
#   api_key: "YOUR_BINANCE_API_KEY"
#   api_secret: "YOUR_BINANCE_API_SECRET"
EOF
    echo -e "${GREEN}Created config/exchanges.yaml template${NC}"
fi

if [ ! -f "config/email.yaml" ]; then
    cat > config/email.yaml << 'EOF'
# SMTP Email Configuration for Alerts

smtp:
  host: "smtp.gmail.com"
  port: 587
  username: "your-email@gmail.com"
  password: "your-app-password"
  use_tls: true

notifications:
  enabled: false  # Set to true after configuring SMTP
  recipient: "your-email@gmail.com"
EOF
    echo -e "${GREEN}Created config/email.yaml template${NC}"
fi

if [ ! -f "config/data_sources.yaml" ]; then
    cat > config/data_sources.yaml << 'EOF'
# External Data Sources Configuration

coinmarketcap:
  api_key: "YOUR_CMC_API_KEY"
  priority: 1
  enabled: false

coingecko:
  api_key: "YOUR_COINGECKO_API_KEY"  # Optional for free tier
  priority: 2
  enabled: true

# Poll frequency in seconds
poll_frequency: 300  # 5 minutes

# Extreme market condition thresholds
extreme_conditions:
  crash_threshold_percent: -15  # Pause bots if market drops this much in 24h
  enabled: true
EOF
    echo -e "${GREEN}Created config/data_sources.yaml template${NC}"
fi

if [ ! -f "config/defaults.yaml" ]; then
    cat > config/defaults.yaml << 'EOF'
# Default Strategy Parameters

strategies:
  dca_accumulator:
    interval_minutes: 60
    amount_percent: 10  # Percent of budget per buy

  adaptive_grid:
    grid_count: 10
    grid_spacing_percent: 1.0
    range_percent: 10

  mean_reversion:
    bollinger_period: 20
    bollinger_std: 2.0

  twap:
    execution_period_minutes: 60
    slice_count: 10

  vwap:
    lookback_period_minutes: 30

  breakdown_momentum:
    breakout_threshold_percent: 2.0
    volume_threshold_multiplier: 1.5

  scalping:
    take_profit_percent: 0.5
    max_position_time_seconds: 300

  arbitrage:
    min_spread_percent: 0.3

  event_filler:
    event_sources: []

  auto_mode:
    factor_precedence:
      - trend
      - volume
      - volatility
    disabled_factors: []
    switch_threshold: 0.7

# Default risk parameters
risk_defaults:
  stop_loss_percent: 5.0
  drawdown_limit_percent: 10.0
  daily_loss_limit: null  # No limit by default
  weekly_loss_limit: null
  max_strategy_rotations: 3
  consecutive_loss_threshold: 3
EOF
    echo -e "${GREEN}Created config/defaults.yaml template${NC}"
fi

echo -e "\n${GREEN}=========================================="
echo -e "  Setup Complete!"
echo -e "==========================================${NC}"
echo -e "\n${YELLOW}Next Steps:${NC}"
echo "1. Configure your API keys in config/exchanges.yaml"
echo "2. (Optional) Configure email alerts in config/email.yaml"
echo "3. (Optional) Configure data sources in config/data_sources.yaml"
echo ""
echo -e "${YELLOW}To start the backend server:${NC}"
echo "  source venv/bin/activate  # or venv/Scripts/activate on Windows"
echo "  cd backend && uvicorn app.main:app --reload"
echo ""
echo -e "${YELLOW}To start the frontend development server:${NC}"
echo "  cd frontend && npm run dev"
echo ""
echo -e "${YELLOW}Access the application:${NC}"
echo "  Frontend: http://localhost:5173"
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
