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
import backtest
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


@app.route("/api/backtest", methods=["GET"])
def api_backtest():
    """
    Runs a backtest and returns the results as JSON. No auth - open on
    purpose per your request, so it's a plain GET you can hit from a
    phone browser. Just don't share the URL publicly since it does pull
    live data from Bybit each call and can take a little while to run.

    Example:
      /api/backtest?symbol=ETHUSDT&days=30&min_score=3
    """
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"status": "error", "reason": "missing_symbol_param"}), 400

    days = request.args.get("days", default=30, type=int)
    min_score = request.args.get("min_score", default=backtest.strategy.MIN_CONFLUENCE_SCORE, type=int)
    window = request.args.get("window", default=backtest.strategy.CANDLE_LIMIT, type=int)

    try:
        result = backtest.run_backtest(symbol, days, min_score, window, verbose=False)
    except Exception as e:
        log.error(f"Backtest failed for {symbol}: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500

    return jsonify(result)


@app.route("/api/sweep", methods=["GET"])
def api_sweep():
    """
    Tests multiple min_score thresholds across multiple day windows in one
    call, and reports which threshold is STABLE (consistent win rate
    across different time periods) rather than just good on one sample.

    Example:
      /api/sweep?symbol=ETHUSDT&days=30,60,90&scores=0,1,2,3
    """
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"status": "error", "reason": "missing_symbol_param"}), 400

    days_param = request.args.get("days", default="30,60,90")
    scores_param = request.args.get("scores", default="0,1,2,3")
    try:
        day_windows = [int(d.strip()) for d in days_param.split(",")]
        score_values = [int(s.strip()) for s in scores_param.split(",")]
    except ValueError:
        return jsonify({"status": "error", "reason": "days and scores must be comma-separated integers"}), 400

    window = request.args.get("window", default=backtest.strategy.CANDLE_LIMIT, type=int)

    try:
        result = backtest.sweep_scores(symbol, day_windows, score_values, window)
    except Exception as e:
        log.error(f"Sweep failed for {symbol}: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500

    return jsonify(result)


@app.route("/api/sweep_risk", methods=["GET"])
def api_sweep_risk():
    """
    Tests min_score thresholds TOGETHER with SL/TP risk-reward pairs, so
    you're not just tuning the confluence filter in isolation - a
    strategy can look bad on win rate alone but be fine (or better) once
    the risk/reward ratio actually matches what it can realistically hit.

    Example:
      /api/sweep_risk?symbol=ETHUSDT&days=30,60,90&scores=0,1,2&sl_tp=1:2,1:3,0.5:2,1.5:3
    sl_tp pairs are "SL:TP" in percent, comma-separated.
    """
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"status": "error", "reason": "missing_symbol_param"}), 400

    days_param = request.args.get("days", default="30,60,90")
    scores_param = request.args.get("scores", default="0,1,2,3")
    sl_tp_param = request.args.get(
        "sl_tp",
        default=f"{backtest.config.DEFAULT_SL_PERCENT}:{backtest.config.DEFAULT_TP_PERCENT},1:2,1:3,0.5:2"
    )
    try:
        day_windows = [int(d.strip()) for d in days_param.split(",")]
        score_values = [int(s.strip()) for s in scores_param.split(",")]
        sl_tp_pairs = []
        for pair in sl_tp_param.split(","):
            sl_str, tp_str = pair.split(":")
            sl_tp_pairs.append((float(sl_str), float(tp_str)))
    except ValueError:
        return jsonify({
            "status": "error",
            "reason": "days/scores must be comma-separated integers, sl_tp must be comma-separated SL:TP pairs like 1:2,1:3"
        }), 400

    window = request.args.get("window", default=backtest.strategy.CANDLE_LIMIT, type=int)

    try:
        result = backtest.sweep_risk_reward(symbol, day_windows, score_values, sl_tp_pairs, window)
    except Exception as e:
        log.error(f"Risk sweep failed for {symbol}: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500

    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "testnet": config.BYBIT_TESTNET})


if __name__ == "__main__":
    log.info(f"Starting server on {config.SERVER_HOST}:{config.SERVER_PORT} (testnet={config.BYBIT_TESTNET})")
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT)
