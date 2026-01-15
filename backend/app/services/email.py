"""
Email service for sending SMTP notifications.
"""
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


@dataclass
class EmailConfig:
    enabled: bool
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    from_address: str
    to_addresses: List[str]
    use_tls: bool = True


def load_email_config() -> Optional[EmailConfig]:
    """Load email configuration from config/email.yaml"""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "email.yaml"

    if not config_path.exists():
        logger.warning(f"Email config not found at {config_path}. Email notifications disabled.")
        return None

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        email_config = config.get("email", {})

        return EmailConfig(
            enabled=email_config.get("enabled", False),
            smtp_host=email_config.get("smtp_host", ""),
            smtp_port=email_config.get("smtp_port", 587),
            smtp_user=email_config.get("smtp_user", ""),
            smtp_password=email_config.get("smtp_password", ""),
            from_address=email_config.get("from_address", ""),
            to_addresses=email_config.get("to_addresses", []),
            use_tls=email_config.get("use_tls", True),
        )
    except Exception as e:
        logger.error(f"Failed to load email config: {e}")
        return None


class EmailService:
    """Service for sending email notifications."""

    def __init__(self):
        self.config = load_email_config()
        self._is_enabled = self.config is not None and self.config.enabled

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    def send_email(self, subject: str, body_html: str, body_text: Optional[str] = None) -> bool:
        """
        Send an email notification.

        Args:
            subject: Email subject
            body_html: HTML body content
            body_text: Plain text body (optional, derived from HTML if not provided)

        Returns:
            True if email was sent successfully, False otherwise
        """
        if not self._is_enabled or not self.config:
            logger.info(f"Email not enabled. Would send: {subject}")
            logger.debug(f"Email body: {body_text or body_html}")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.config.from_address
            msg["To"] = ", ".join(self.config.to_addresses)

            # Plain text fallback
            if body_text:
                msg.attach(MIMEText(body_text, "plain"))

            # HTML body
            msg.attach(MIMEText(body_html, "html"))

            # Send email
            if self.config.use_tls:
                with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                    server.starttls()
                    server.login(self.config.smtp_user, self.config.smtp_password)
                    server.sendmail(
                        self.config.from_address,
                        self.config.to_addresses,
                        msg.as_string()
                    )
            else:
                with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                    if self.config.smtp_user:
                        server.login(self.config.smtp_user, self.config.smtp_password)
                    server.sendmail(
                        self.config.from_address,
                        self.config.to_addresses,
                        msg.as_string()
                    )

            logger.info(f"Email sent successfully: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_bot_paused_alert(
        self,
        bot_id: int,
        bot_name: str,
        reason: str,
        pnl: float,
        trading_pair: str
    ) -> bool:
        """
        Send alert when a bot is paused due to risk limits.

        Args:
            bot_id: Bot ID
            bot_name: Bot name
            reason: Reason for pause (e.g., "Drawdown limit reached")
            pnl: Current P&L
            trading_pair: Trading pair

        Returns:
            True if email was sent successfully
        """
        subject = f"[TradingBot Alert] Bot Paused: {bot_name}"

        pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
        pnl_sign = "+" if pnl >= 0 else ""

        body_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #1f2937; color: #f3f4f6; padding: 20px; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: #374151; border-radius: 8px; padding: 20px; }}
                .header {{ font-size: 24px; font-weight: bold; color: #f59e0b; margin-bottom: 20px; }}
                .alert-box {{ background-color: #7f1d1d; border-left: 4px solid #ef4444; padding: 15px; margin: 15px 0; border-radius: 4px; }}
                .info-row {{ display: flex; margin: 10px 0; }}
                .label {{ color: #9ca3af; width: 120px; }}
                .value {{ color: #f3f4f6; font-weight: bold; }}
                .pnl {{ color: {pnl_color}; font-family: monospace; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">Bot Paused Alert</div>

                <div class="alert-box">
                    <strong>Reason:</strong> {reason}
                </div>

                <div class="info-row">
                    <span class="label">Bot Name:</span>
                    <span class="value">{bot_name}</span>
                </div>

                <div class="info-row">
                    <span class="label">Bot ID:</span>
                    <span class="value">{bot_id}</span>
                </div>

                <div class="info-row">
                    <span class="label">Trading Pair:</span>
                    <span class="value">{trading_pair}</span>
                </div>

                <div class="info-row">
                    <span class="label">Current P&L:</span>
                    <span class="value pnl">{pnl_sign}${pnl:.2f}</span>
                </div>

                <p style="margin-top: 20px; color: #9ca3af; font-size: 12px;">
                    This is an automated alert from TradingBot. Review your bot settings and take action if necessary.
                </p>
            </div>
        </body>
        </html>
        """

        body_text = f"""
Bot Paused Alert

Reason: {reason}

Bot Name: {bot_name}
Bot ID: {bot_id}
Trading Pair: {trading_pair}
Current P&L: {pnl_sign}${pnl:.2f}

This is an automated alert from TradingBot. Review your bot settings and take action if necessary.
        """

        return self.send_email(subject, body_html, body_text)

    def send_kill_switch_alert(
        self,
        affected_bots: List[dict],
        total_pnl: float
    ) -> bool:
        """
        Send alert when global kill switch is triggered.

        Args:
            affected_bots: List of bot dicts with id, name, pnl, status
            total_pnl: Total P&L across all affected bots

        Returns:
            True if email was sent successfully
        """
        subject = f"[TradingBot Alert] Global Kill Switch Activated"

        pnl_color = "#22c55e" if total_pnl >= 0 else "#ef4444"
        pnl_sign = "+" if total_pnl >= 0 else ""

        bot_rows = ""
        for bot in affected_bots:
            bot_pnl = bot.get("pnl", 0)
            bot_pnl_color = "#22c55e" if bot_pnl >= 0 else "#ef4444"
            bot_pnl_sign = "+" if bot_pnl >= 0 else ""
            bot_rows += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #4b5563;">{bot.get('name', 'Unknown')}</td>
                <td style="padding: 8px; border-bottom: 1px solid #4b5563;">{bot.get('trading_pair', 'N/A')}</td>
                <td style="padding: 8px; border-bottom: 1px solid #4b5563; color: {bot_pnl_color}; font-family: monospace;">{bot_pnl_sign}${bot_pnl:.2f}</td>
            </tr>
            """

        body_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #1f2937; color: #f3f4f6; padding: 20px; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: #374151; border-radius: 8px; padding: 20px; }}
                .header {{ font-size: 24px; font-weight: bold; color: #ef4444; margin-bottom: 20px; }}
                .alert-box {{ background-color: #7f1d1d; border-left: 4px solid #ef4444; padding: 15px; margin: 15px 0; border-radius: 4px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th {{ text-align: left; padding: 8px; background-color: #1f2937; color: #9ca3af; }}
                .summary {{ background-color: #1f2937; padding: 15px; border-radius: 4px; margin-top: 15px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">Global Kill Switch Activated</div>

                <div class="alert-box">
                    <strong>All running bots have been stopped!</strong>
                </div>

                <h3 style="color: #f3f4f6;">Affected Bots ({len(affected_bots)})</h3>

                <table>
                    <tr>
                        <th>Bot Name</th>
                        <th>Trading Pair</th>
                        <th>P&L</th>
                    </tr>
                    {bot_rows}
                </table>

                <div class="summary">
                    <span style="color: #9ca3af;">Total P&L:</span>
                    <span style="color: {pnl_color}; font-family: monospace; font-weight: bold; font-size: 18px;">
                        {pnl_sign}${total_pnl:.2f}
                    </span>
                </div>

                <p style="margin-top: 20px; color: #9ca3af; font-size: 12px;">
                    This is an automated alert from TradingBot. The global kill switch was triggered, stopping all active bots.
                </p>
            </div>
        </body>
        </html>
        """

        # Plain text version
        bot_text_lines = "\n".join([
            f"  - {bot.get('name', 'Unknown')} ({bot.get('trading_pair', 'N/A')}): ${bot.get('pnl', 0):.2f}"
            for bot in affected_bots
        ])

        body_text = f"""
Global Kill Switch Activated

All running bots have been stopped!

Affected Bots ({len(affected_bots)}):
{bot_text_lines}

Total P&L: {pnl_sign}${total_pnl:.2f}

This is an automated alert from TradingBot. The global kill switch was triggered, stopping all active bots.
        """

        return self.send_email(subject, body_html, body_text)


# Singleton instance
email_service = EmailService()
