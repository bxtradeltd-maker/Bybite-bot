"""
Handles all Bybit API connection and order execution logic.
Uses the official `pybit` library (Bybit's unified v5 API).
"""

from pybit.unified_trading import HTTP
import config
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bybit_trader")

session = HTTP(
    testnet=config.BYBIT_TESTNET,
    api_key=config.BYBIT_API_KEY,
    api_secret=config.BYBIT_API_SECRET,
)

_daily_loss_locked = False
_daily_target_locked = False
_daily_lock_date = None
_start_of_day_equity = None


def _get_equity():
    resp = session.get_wallet_balance(accountType="UNIFIED")
    try:
        return float(resp["result"]["list"][0]["totalEquity"])
    except (KeyError, IndexError, TypeError):
        log.error(f"Could not read equity from balance response: {resp}")
        return None


def _refresh_daily_baseline():
    """Resets both locks and captures start-of-day equity when the date rolls over."""
    global _daily_loss_locked, _daily_target_locked, _daily_lock_date, _start_of_day_equity
    today = datetime.now().date()
    if _daily_lock_date != today:
        _daily_loss_locked = False
        _daily_target_locked = False
        _daily_lock_date = today
        _start_of_day_equity = _get_equity()


def _check_daily_loss_lock():
    """Drawdown stop - halts trading for the day if losses hit MAX_DAILY_LOSS_PERCENT."""
    global _daily_loss_locked
    _refresh_daily_baseline()

    if _start_of_day_equity is None or _start_of_day_equity <= 0:
        return _daily_loss_locked

    current_equity = _get_equity()
    if current_equity is None:
        return _daily_loss_locked

    loss_percent = (_start_of_day_equity - current_equity) / _start_of_day_equity * 100
    if loss_percent >= config.MAX_DAILY_LOSS_PERCENT:
        if not _daily_loss_locked:
            log.warning(f"Daily drawdown limit hit ({loss_percent:.2f}%). Trading locked for today.")
        _daily_loss_locked = True

    return _daily_loss_locked


def _check_daily_target_lock():
    """Profit-target stop - halts NEW trades once DAILY_TARGET_PERCENT is reached.
    Does NOT increase position sizing to try to reach this target faster."""
    global _daily_target_locked
    _refresh_daily_baseline()

    if _start_of_day_equity is None or _start_of_day_equity <= 0:
        return _daily_target_locked

    current_equity = _get_equity()
    if current_equity is None:
        return _daily_target_locked

    gain_percent = (current_equity - _start_of_day_equity) / _start_of_day_equity * 100
    if gain_percent >= config.DAILY_TARGET_PERCENT:
        if not _daily_target_locked:
            log.info(f"Daily target reached (+{gain_percent:.2f}%). New trades paused for today.")
        _daily_target_locked = True

    return _daily_target_locked


def count_open_positions():
    resp = session.get_positions(category="linear", settleCoin="USDT")
    positions = [p for p in resp.get("result", {}).get("list", []) if float(p.get("size", 0)) > 0]
    return len(positions)


def _get_instrument_filters(symbol):
    resp = session.get_instruments_info(category="linear", symbol=symbol)
    info = resp["result"]["list"][0]
    lot_filter = info["lotSizeFilter"]
    return {
        "qty_step": float(lot_filter["qtyStep"]),
        "min_qty": float(lot_filter["minOrderQty"]),
    }


def _get_available_usdt():
    """Actual spendable USDT margin - excludes BTC/other coin holdings that
    aren't usable as margin for USDT-linear contracts."""
    resp = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
    try:
        coins = resp["result"]["list"][0]["coin"]
        for c in coins:
            if c["coin"] == "USDT":
                return float(c["availableToWithdraw"] or c["walletBalance"] or 0)
    except (KeyError, IndexError, TypeError):
        log.error(f"Could not read available USDT from balance response: {resp}")
    return 0.0


def _calc_qty_from_amount(symbol, entry_price, amount_usd):
    """Position size based on a specific dollar amount (e.g. from a manual
    trade), still capped by actual available USDT balance."""
    available_usdt = _get_available_usdt()
    max_notional = available_usdt * config.LEVERAGE * 0.95
    notional = min(amount_usd * config.LEVERAGE, max_notional)

    qty = notional / entry_price if entry_price > 0 else 0
    filters = _get_instrument_filters(symbol)
    qty = max(filters["min_qty"], qty)
    step = filters["qty_step"]
    qty = round(qty / step) * step
    return round(qty, 6)


def get_live_prices():
    """Fetches last price + 24h change for every symbol in SYMBOL_MAP."""
    prices = {}
    try:
        resp = session.get_tickers(category="linear")
        rows = resp.get("result", {}).get("list", [])
        wanted = set(config.SYMBOL_MAP.values())
        for r in rows:
            if r.get("symbol") in wanted:
                prices[r["symbol"]] = {
                    "last_price": float(r.get("lastPrice", 0)),
                    "change_24h_percent": round(float(r.get("price24hPcnt", 0)) * 100, 2),
                }
    except Exception as e:
        log.error(f"Failed to fetch live prices: {e}")
    return prices


def _calc_qty(symbol, entry_price, sl_price, equity):
    """Position size based on % risk of equity and SL distance, capped so the
    required margin never exceeds actual available USDT balance."""
    risk_amount = equity * (config.RISK_PER_TRADE_PERCENT / 100)
    sl_distance = abs(entry_price - sl_price)
    if sl_distance <= 0:
        sl_distance = entry_price * (config.DEFAULT_SL_PERCENT / 100)

    qty = risk_amount / sl_distance

    # Cap by actual available USDT margin (not total equity, which may include
    # non-USDT holdings like BTC that aren't usable as margin here).
    available_usdt = _get_available_usdt()
    max_notional = available_usdt * config.LEVERAGE * 0.95  # 5% safety buffer
    max_qty_by_balance = max_notional / entry_price if entry_price > 0 else 0
    qty = min(qty, max_qty_by_balance)

    filters = _get_instrument_filters(symbol)
    qty = max(filters["min_qty"], qty)
    step = filters["qty_step"]
    qty = round(qty / step) * step
    return round(qty, 6)


def place_trade(tv_symbol: str, action: str, sl_price: float = None, tp_price: float = None, amount_usd: float = None):
    """
    action: 'buy' or 'sell'
    sl_price / tp_price: optional exact prices from your Pine Script signal.
    Falls back to DEFAULT_SL_PERCENT / DEFAULT_TP_PERCENT if omitted.
    amount_usd: optional - if provided (e.g. from a manual trade), the position
    size is based on this dollar amount instead of the automatic risk-% calc.
    Still capped by available balance and leverage.
    """
    if _check_daily_loss_lock():
        return {"status": "blocked", "reason": "daily_drawdown_limit_hit"}

    if _check_daily_target_lock():
        return {"status": "blocked", "reason": "daily_target_reached"}

    if count_open_positions() >= config.MAX_OPEN_TRADES:
        return {"status": "blocked", "reason": "max_open_trades_reached"}

    symbol = config.SYMBOL_MAP.get(tv_symbol.upper())
    if symbol is None:
        return {"status": "error", "reason": "symbol_not_in_map"}

    ticker_resp = session.get_tickers(category="linear", symbol=symbol)
    try:
        last_price = float(ticker_resp["result"]["list"][0]["lastPrice"])
    except (KeyError, IndexError, TypeError):
        return {"status": "error", "reason": "could_not_fetch_price"}

    side = "Buy" if action.lower() == "buy" else "Sell"

    if sl_price is None:
        sl_price = last_price * (1 - config.DEFAULT_SL_PERCENT / 100) if side == "Buy" \
            else last_price * (1 + config.DEFAULT_SL_PERCENT / 100)
    if tp_price is None:
        tp_price = last_price * (1 + config.DEFAULT_TP_PERCENT / 100) if side == "Buy" \
            else last_price * (1 - config.DEFAULT_TP_PERCENT / 100)

    equity = _get_equity()
    if equity is None:
        return {"status": "error", "reason": "could_not_fetch_equity"}

    if amount_usd is not None and amount_usd > 0:
        qty = _calc_qty_from_amount(symbol, last_price, amount_usd)
    else:
        qty = _calc_qty(symbol, last_price, sl_price, equity)

    try:
        session.set_leverage(category="linear", symbol=symbol,
                              buyLeverage=str(config.LEVERAGE), sellLeverage=str(config.LEVERAGE))
    except Exception as e:
        log.warning(f"Leverage set skipped/failed (may already be set): {e}")

    try:
        result = session.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            stopLoss=str(round(sl_price, 4)),
            takeProfit=str(round(tp_price, 4)),
        )
    except Exception as e:
        log.error(f"Order failed: {e}")
        return {"status": "error", "reason": str(e)}

    if result.get("retCode") != 0:
        log.error(f"Order rejected: {result}")
        return {"status": "error", "reason": result.get("retMsg"), "raw": result}

    log.info(f"Order placed: {side} {qty} {symbol} @ ~{last_price} SL={sl_price} TP={tp_price}")
    return {
        "status": "ok",
        "symbol": symbol,
        "action": action,
        "qty": qty,
        "price": last_price,
        "sl": round(sl_price, 4),
        "tp": round(tp_price, 4),
        "order_id": result["result"].get("orderId"),
    }


def get_open_positions():
    resp = session.get_positions(category="linear", settleCoin="USDT")
    positions = [p for p in resp.get("result", {}).get("list", []) if float(p.get("size", 0)) > 0]
    return [
        {
            "symbol": p["symbol"],
            "side": p["side"],
            "size": p["size"],
            "entry_price": p["avgPrice"],
            "unrealized_pnl": p["unrealisedPnl"],
            "leverage": p["leverage"],
        }
        for p in positions
    ]


def get_account_summary():
    equity = _get_equity()
    import money_plan
    plan_status = money_plan.get_status(equity)
    return {
        "equity": equity,
        "open_positions": count_open_positions(),
        "daily_loss_locked": _daily_loss_locked,
        "daily_target_locked": _daily_target_locked,
        "money_plan": plan_status,
    }


def get_trade_history(limit=50):
    """Fetches recent closed trades from Bybit and computes summary stats."""
    try:
        resp = session.get_closed_pnl(category="linear", limit=limit)
        rows = resp.get("result", {}).get("list", [])
    except Exception as e:
        log.error(f"Failed to fetch closed PnL: {e}")
        return {"trades": [], "summary": {}}

    trades = []
    wins = 0
    losses = 0
    total_pnl = 0.0

    for r in rows:
        pnl = float(r.get("closedPnl", 0))
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
        trades.append({
            "symbol": r.get("symbol"),
            "side": r.get("side"),
            "qty": r.get("qty"),
            "entry_price": r.get("avgEntryPrice"),
            "exit_price": r.get("avgExitPrice"),
            "pnl": round(pnl, 4),
            "closed_time": r.get("updatedTime"),
        })

    summary = {
        "total_closed": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round((wins / len(trades) * 100), 1) if trades else 0,
        "total_pnl": round(total_pnl, 4),
        "open_positions": count_open_positions(),
    }

    return {"trades": trades, "summary": summary}
