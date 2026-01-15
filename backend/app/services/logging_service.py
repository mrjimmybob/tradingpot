"""Logging service for per-bot file logging and fiscal records."""

import os
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Base logs directory
LOGS_BASE_DIR = Path(__file__).parent.parent.parent / "logs"


@dataclass
class TradeLogEntry:
    """Represents a trade log entry."""
    timestamp: datetime
    bot_id: int
    bot_name: str
    order_id: int
    order_type: str
    trading_pair: str
    amount: float
    price: float
    fees: float
    status: str
    strategy: str
    running_balance: Optional[float]
    is_simulated: bool
    pnl: Optional[float] = None


@dataclass
class FiscalLogEntry:
    """Represents a fiscal/tax log entry for realized gains/losses."""
    date: datetime
    trading_pair: str
    token: str
    buy_date: Optional[datetime]
    buy_price: float
    sale_price: float
    amount: float
    proceeds: float
    cost_basis: float
    gain_loss: float
    holding_period_days: Optional[int]
    is_simulated: bool


class BotLoggingService:
    """Service for managing per-bot log files."""

    def __init__(self, bot_id: int, bot_name: str = "", is_dry_run: bool = False):
        """Initialize logging service for a bot.

        Args:
            bot_id: The bot ID
            bot_name: The bot name for log entries
            is_dry_run: Whether this is a dry run bot
        """
        self.bot_id = bot_id
        self.bot_name = bot_name
        self.is_dry_run = is_dry_run
        self.bot_log_dir = LOGS_BASE_DIR / str(bot_id)

        # Ensure log directory exists
        self._ensure_log_directory()

    def _ensure_log_directory(self) -> None:
        """Create the bot's log directory if it doesn't exist."""
        try:
            self.bot_log_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Bot {self.bot_id}: Log directory ensured at {self.bot_log_dir}")
        except Exception as e:
            logger.error(f"Bot {self.bot_id}: Failed to create log directory: {e}")

    def get_log_directory(self) -> Path:
        """Get the bot's log directory path."""
        return self.bot_log_dir

    def log_trade(self, entry: TradeLogEntry) -> None:
        """Log a trade to the bot's trade log file.

        Args:
            entry: Trade log entry to write
        """
        # Determine log file name based on dry run status
        if entry.is_simulated:
            log_file = self.bot_log_dir / "trades_simulated.csv"
        else:
            log_file = self.bot_log_dir / "trades.csv"

        # Check if file exists to determine if we need headers
        write_header = not log_file.exists()

        try:
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                if write_header:
                    writer.writerow([
                        'timestamp', 'bot_id', 'bot_name', 'order_id', 'order_type',
                        'trading_pair', 'amount', 'price', 'fees', 'status',
                        'strategy', 'running_balance', 'is_simulated', 'pnl'
                    ])

                writer.writerow([
                    entry.timestamp.isoformat(),
                    entry.bot_id,
                    entry.bot_name,
                    entry.order_id,
                    entry.order_type,
                    entry.trading_pair,
                    f"{entry.amount:.8f}",
                    f"{entry.price:.8f}",
                    f"{entry.fees:.8f}",
                    entry.status,
                    entry.strategy,
                    f"{entry.running_balance:.2f}" if entry.running_balance else "",
                    entry.is_simulated,
                    f"{entry.pnl:.2f}" if entry.pnl else ""
                ])

            logger.debug(f"Bot {self.bot_id}: Logged trade {entry.order_id} to {log_file.name}")

        except Exception as e:
            logger.error(f"Bot {self.bot_id}: Failed to log trade: {e}")

    def log_fiscal_entry(self, entry: FiscalLogEntry) -> None:
        """Log a fiscal/tax entry for realized gains/losses.

        Args:
            entry: Fiscal log entry to write
        """
        # Determine fiscal file based on year and dry run status
        year = entry.date.year
        if entry.is_simulated:
            fiscal_file = self.bot_log_dir / f"fiscal_{year}_simulated.csv"
        else:
            fiscal_file = self.bot_log_dir / f"fiscal_{year}.csv"

        write_header = not fiscal_file.exists()

        try:
            with open(fiscal_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                if write_header:
                    writer.writerow([
                        'sale_date', 'trading_pair', 'token', 'buy_date',
                        'buy_price', 'sale_price', 'amount', 'proceeds',
                        'cost_basis', 'gain_loss', 'holding_period_days',
                        'short_term', 'is_simulated'
                    ])

                # Determine if short-term (< 1 year) or long-term
                short_term = True
                if entry.holding_period_days is not None:
                    short_term = entry.holding_period_days < 365

                writer.writerow([
                    entry.date.strftime('%Y-%m-%d'),
                    entry.trading_pair,
                    entry.token,
                    entry.buy_date.strftime('%Y-%m-%d') if entry.buy_date else "",
                    f"{entry.buy_price:.8f}",
                    f"{entry.sale_price:.8f}",
                    f"{entry.amount:.8f}",
                    f"{entry.proceeds:.2f}",
                    f"{entry.cost_basis:.2f}",
                    f"{entry.gain_loss:.2f}",
                    entry.holding_period_days if entry.holding_period_days else "",
                    short_term,
                    entry.is_simulated
                ])

            logger.debug(f"Bot {self.bot_id}: Logged fiscal entry to {fiscal_file.name}")

        except Exception as e:
            logger.error(f"Bot {self.bot_id}: Failed to log fiscal entry: {e}")

    def log_activity(self, message: str, level: str = "INFO") -> None:
        """Log general bot activity to activity log.

        Args:
            message: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR)
        """
        activity_file = self.bot_log_dir / "activity.log"

        try:
            with open(activity_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.utcnow().isoformat()
                prefix = "[DRY RUN] " if self.is_dry_run else ""
                f.write(f"{timestamp} [{level}] {prefix}{message}\n")

        except Exception as e:
            logger.error(f"Bot {self.bot_id}: Failed to log activity: {e}")


def ensure_bot_log_directory(bot_id: int) -> Path:
    """Ensure a bot's log directory exists and return its path.

    Args:
        bot_id: The bot ID

    Returns:
        Path to the bot's log directory
    """
    bot_log_dir = LOGS_BASE_DIR / str(bot_id)
    bot_log_dir.mkdir(parents=True, exist_ok=True)
    return bot_log_dir


def get_bot_logging_service(bot_id: int, bot_name: str = "", is_dry_run: bool = False) -> BotLoggingService:
    """Get or create a logging service for a bot.

    Args:
        bot_id: The bot ID
        bot_name: The bot name
        is_dry_run: Whether this is a dry run bot

    Returns:
        BotLoggingService instance
    """
    return BotLoggingService(bot_id, bot_name, is_dry_run)
