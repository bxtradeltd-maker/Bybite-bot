"""
Configuration for the SMC Bybit Auto-Trader.
Set these as environment variables (recommended) or edit the defaults directly.
"""

import os

# ---- BYBIT API ----
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "true").lower() == "true"  # start on testnet!

# ---- WEBHOOK / DASHBOARD SERVER ----
SERVER_HOST = "0.0.0.0"
# Railway (and most cloud hosts) assign the port dynamically via $PORT.
# Falls back to 5000 for local testing.
SERVER_PORT = int(os.getenv("PORT", "5000"))
# Shared secret — must match the "secret" field in your TradingView alert JSON.
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-this-to-something-long-and-random")
# Password to view/use the HTML dashboard itself.
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "change-this-too")

# ---- RISK MANAGEMENT ----
RISK_PER_TRADE_PERCENT = 1.0
MAX_DAILY_LOSS_PERCENT = 5.0
MAX_OPEN_TRADES = 3
DEFAULT_SL_PERCENT = 1.0   # used if signal doesn't include an explicit SL price
DEFAULT_TP_PERCENT = 2.0
LEVERAGE = 5

# ---- SYMBOLS ----
# TradingView ticker -> Bybit symbol (usually identical for USDT perpetuals)
SYMBOL_MAP = {
    "BTCUSDT": "BTCUSDT",
    "ETHUSDT": "ETHUSDT",
}
