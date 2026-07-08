"""
Server that:
  1. Receives TradingView webhook alerts -> auto-executes trades on Bybit
  2. Serves a small JSON API that the HTML dashboard (dashboard.html) calls
     to show balance, open positions, and recent signals.

Run with:
    python server.py
Then open dashboard.html in a browser (update API_BASE inside it to point
here, e.g. http://YOUR_SERVER_IP:5000).
"""

from flask import Flask, request, jsonify
from collections import deque
from datetime import datetime
import config
import bybit_trader
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("server")

app = Flask(__name__)

# Keep the last 50 signals/trades in memory to show on the dashboard
recent_activity = deque(maxlen=50)


def _log_activity(entry):
    entry["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    recent_activity.appendleft(entry)


def _check_dashboard_auth(data):
    return data.get("password") == config.DASHBOARD_PASSWORD


# ---------- TradingView webhook (auto-trading) ----------

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"status": "error", "reason": "invalid_json"}), 400

    if data.get("secret") != config.WEBHOOK_SECRET:
        log.warning("Rejected webhook: bad secret")
        return jsonify({"status": "error", "reason": "unauthorized"}), 401

    symbol = data.get("symbol")
    action = data.get("action", "").lower()
    sl = data.get("sl")
    tp = data.get("tp")

    if not symbol or action not in ("buy", "sell"):
        return jsonify({"status": "error", "reason": "missing_or_invalid_fields"}), 400

    log.info(f"Signal received: {action.upper()} {symbol} SL={sl} TP={tp}")
    result = bybit_trader.place_trade(symbol, action, sl_price=sl, tp_price=tp)
    _log_activity({"source": "tradingview", "symbol": symbol, "action": action, "result": result})
    return jsonify(result)


# ---------- Dashboard API ----------

@app.route("/api/status", methods=["POST"])
def api_status():
    data = request.get_json(silent=True) or {}
    if not _check_dashboard_auth(data):
        return jsonify({"status": "error", "reason": "unauthorized"}), 401
    return jsonify(bybit_trader.get_account_summary())


@app.route("/api/positions", methods=["POST"])
def api_positions():
    data = request.get_json(silent=True) or {}
    if not _check_dashboard_auth(data):
        return jsonify({"status": "error", "reason": "unauthorized"}), 401
    return jsonify({"positions": bybit_trader.get_open_positions()})


@app.route("/api/activity", methods=["POST"])
def api_activity():
    data = request.get_json(silent=True) or {}
    if not _check_dashboard_auth(data):
        return jsonify({"status": "error", "reason": "unauthorized"}), 401
    return jsonify({"activity": list(recent_activity)})


@app.route("/api/manual_trade", methods=["POST"])
def api_manual_trade():
    """Lets the dashboard fire a trade manually (e.g. a 'Confirm' button)."""
    data = request.get_json(silent=True) or {}
    if not _check_dashboard_auth(data):
        return jsonify({"status": "error", "reason": "unauthorized"}), 401

    symbol = data.get("symbol")
    action = data.get("action", "").lower()
    if not symbol or action not in ("buy", "sell"):
        return jsonify({"status": "error", "reason": "missing_or_invalid_fields"}), 400

    result = bybit_trader.place_trade(symbol, action, sl_price=data.get("sl"), tp_price=data.get("tp"))
    _log_activity({"source": "dashboard_manual", "symbol": symbol, "action": action, "result": result})
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "testnet": config.BYBIT_TESTNET})


if __name__ == "__main__":
    log.info(f"Starting server on {config.SERVER_HOST}:{config.SERVER_PORT} (testnet={config.BYBIT_TESTNET})")
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT)
