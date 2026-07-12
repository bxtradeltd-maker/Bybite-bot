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
from flask_cors import CORS
from collections import deque
from datetime import datetime
import config
import bybit_trader
import backtest
import backtest_grid
import trend_strategy
import selective_trend
import grid_ea
import liquidity_sweep
import backtest_sweep
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("server")

app = Flask(__name__)
CORS(app)

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
    Runs a backtest and returns the results as JSON.
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


@app.route("/api/grid_backtest", methods=["GET"])
def api_grid_backtest():
    """
    Runs the grid/basket EA backtest and returns JSON results.

    Example:
      /api/grid_backtest?symbol=BTCUSDT&days=90&direction=buy&repeat=20&balance=20
    """
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"status": "error", "reason": "missing_symbol_param"}), 400

    days = request.args.get("days", default=90, type=int)
    direction = request.args.get("direction", default="buy")
    repeat = request.args.get("repeat", default=20, type=int)
    balance = request.args.get("balance", default=20.0, type=float)

    if direction not in ("buy", "sell"):
        return jsonify({"status": "error", "reason": "direction must be buy or sell"}), 400

    initial_qty_usd = balance * grid_ea.INITIAL_QTY_PCT
    below_minimum = initial_qty_usd < grid_ea.MIN_NOTIONAL_USD

    try:
        candles = backtest.fetch_history(symbol, "5", days)
        if len(candles) < 50:
            return jsonify({"status": "error", "reason": "not_enough_candles", "candles_fetched": len(candles)}), 200

        results = []
        idx = 0
        for _ in range(repeat):
            if idx >= len(candles) - 1:
                break
            r = backtest_grid.simulate_one_basket(
                candles, idx, direction,
                initial_qty_usd, grid_ea.LOT_MULTIPLIER, grid_ea.GRID_STEP_PCT,
                grid_ea.MAX_GRID_LEVELS, grid_ea.BASKET_TP_PCT, grid_ea.EQUITY_STOP_PCT,
            )
            results.append(r)
            idx = r["exit_idx"] + 1

        tp_count = sum(1 for r in results if r["outcome"] == "take_profit")
        stop_count = sum(1 for r in results if r["outcome"] == "equity_stop")
        unresolved = sum(1 for r in results if r["outcome"] == "unresolved_end_of_data")
        total_notional = sum(r["notional"] for r in results) or 1
        weighted_pnl = sum(r["final_pnl_pct"] * r["notional"] for r in results) / total_notional
        worst_drawdown = min((r["final_pnl_pct"] for r in results), default=0)
        worst_dollar_loss = min((r["final_pnl_pct"] / 100 * r["notional"] for r in results), default=0)
        max_levels_seen = max((r["levels_used"] for r in results), default=0)

        return jsonify({
            "status": "ok",
            "symbol": symbol,
            "direction": direction,
            "starting_balance_usd": balance,
            "level_1_size_usd": round(initial_qty_usd, 2),
            "below_exchange_minimum_warning": below_minimum,
            "baskets_run": len(results),
            "take_profit_closes": tp_count,
            "equity_stop_closes": stop_count,
            "unresolved_at_end_of_data": unresolved,
            "max_grid_levels_hit": max_levels_seen,
            "grid_level_cap": grid_ea.MAX_GRID_LEVELS,
            "worst_single_basket_pnl_pct": round(worst_drawdown, 3),
            "worst_single_basket_dollar_loss": round(worst_dollar_loss, 2),
            "notional_weighted_avg_pnl_pct": round(weighted_pnl, 3),
            "results": results,
        })
    except Exception as e:
        log.error(f"Grid backtest failed for {symbol}: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route("/api/liquidity_sweep_backtest", methods=["GET"])
def api_liquidity_sweep_backtest():
    """
    Example:
      /api/liquidity_sweep_backtest?symbol=BTCUSDT&days=90
    """
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"status": "error", "reason": "missing_symbol_param"}), 400
    days = request.args.get("days", default=90, type=int)
    balance = request.args.get("balance", default=20.0, type=float)

    try:
        ltf_candles = backtest.fetch_history(symbol, "15", days)
        htf_candles = backtest.fetch_history(symbol, "240", days)
        if len(ltf_candles) < 100 or len(htf_candles) < 20:
            return jsonify({"status": "error", "reason": "not_enough_candles",
                             "ltf_fetched": len(ltf_candles), "htf_fetched": len(htf_candles)}), 200

        ratio = 240 // 15
        trades = backtest_sweep.simulate(ltf_candles, htf_candles, ratio)

        wins = sum(1 for t in trades if t["result"] == "win")
        losses = sum(1 for t in trades if t["result"] == "loss")
        unresolved = sum(1 for t in trades if t["result"] == "unresolved")
        total_r = sum(t["r_multiple"] for t in trades)
        win_rate = wins / (wins + losses) * 100 if (wins + losses) else 0
        breakeven_win_rate = 1 / (1 + liquidity_sweep.TP2_R_MULTIPLE) * 100

        from collections import Counter
        breakdown_counts = Counter(t["detailed"]["result"] for t in trades)
        multi_tp_breakdown = {
            label: {"count": count, "pct": round(count / len(trades) * 100, 1) if trades else 0}
            for label, count in breakdown_counts.items()
        }
        detailed_total_r = sum(t["detailed"]["r_multiple"] for t in trades)

        # position sizing preview at the given balance - shows real dollar
        # risk per trade and flags if trades would be too small for the exchange
        avg_risk_distance_pct = None
        sizing_preview = None
        if trades:
            risk_usd = balance * liquidity_sweep.RISK_PCT_PER_TRADE
            sizing_preview = {
                "risk_usd_per_trade": round(risk_usd, 2),
                "risk_pct_used": liquidity_sweep.RISK_PCT_PER_TRADE,
                "note": "This is how much you'd lose in dollars on a losing trade at this "
                        "balance and risk %. Position size (qty) scales automatically so "
                        "this dollar risk stays constant regardless of SL distance.",
            }

        return jsonify({
            "status": "ok",
            "symbol": symbol,
            "days": days,
            "starting_balance_usd": balance,
            "sizing_preview": sizing_preview,
            "total_signals": len(trades),
            "wins": wins,
            "losses": losses,
            "unresolved": unresolved,
            "win_rate_pct": round(win_rate, 2),
            "breakeven_win_rate_pct": round(breakeven_win_rate, 2),
            "edge_pct_points": round(win_rate - breakeven_win_rate, 2),
            "total_r": round(total_r, 2),
            "avg_r_per_trade": round(total_r / len(trades), 3) if trades else None,
            "multi_tp_breakdown": multi_tp_breakdown,
            "detailed_total_r_with_partials": round(detailed_total_r, 2),
        })
    except Exception as e:
        log.error(f"Liquidity sweep backtest failed for {symbol}: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route("/api/trend_backtest", methods=["GET"])
def api_trend_backtest():
    """
    Trend-following strategy with ATR-normalized (volatility-adaptive)
    SL/TP, designed to generalize across symbols rather than being tuned
    to one asset's typical move size.

    Example:
      /api/trend_backtest?symbol=SOLUSDT&days=150
    """
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"status": "error", "reason": "missing_symbol_param"}), 400
    days = request.args.get("days", default=150, type=int)
    interval = request.args.get("interval", default="15")

    try:
        result = trend_strategy.run_backtest(symbol, days, interval, verbose=False)
        return jsonify(result)
    except Exception as e:
        log.error(f"Trend backtest failed for {symbol}: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route("/api/selective_trend_backtest", methods=["GET"])
def api_selective_trend_backtest():
    """
    Selective trend strategy: ADX trend filter + RSI pullback entries,
    wide 1:4 reward:risk, built specifically to survive fees by trading
    less often with a bigger per-trade edge.

    Example:
      /api/selective_trend_backtest?symbol=BTCUSDT&days=365
    """
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"status": "error", "reason": "missing_symbol_param"}), 400
    days = request.args.get("days", default=365, type=int)
    interval = request.args.get("interval", default="15")

    try:
        result = selective_trend.run_backtest(symbol, days, interval, verbose=False)
        return jsonify(result)
    except Exception as e:
        log.error(f"Selective trend backtest failed for {symbol}: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "testnet": config.BYBIT_TESTNET})


if __name__ == "__main__":
    log.info(f"Starting server on {config.SERVER_HOST}:{config.SERVER_PORT} (testnet={config.BYBIT_TESTNET})")
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT)
