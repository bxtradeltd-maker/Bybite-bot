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
_daily_lock_date = None
_start_of_day_equity = None


def _get_equity():
    resp = session.get_wallet_balance(accountType="UNIFIED")
    try:
        return float(resp["result"]["list"][0]["totalEquity"])
    except (KeyError, IndexError, TypeError):
        log.error(f"Could not read equity from balance response: {resp}")
        return None


def _check_daily_loss_lock():
    global _daily_loss_locked, _daily_lock_date, _start_of_day_equity
    today = datetime.now().date()

    if _daily_lock_date != today:
        _daily_loss_locked = False
        _daily_lock_date = today
        _start_of_day_equity = _get_equity()

    if _start_of_day_equity is None or _start_of_day_equity <= 0:
        return _daily_loss_locked

    current_equity = _get_equity()
    if current_equity is None:
        return _daily_loss_locked

    loss_percent = (_start_of_day_equity - current_equity) / _start_of_day_equity * 100
    if loss_percent >= config.MAX_DAILY_LOSS_PERCENT:
        if not _daily_loss_locked:
            log.warning(f"Daily loss limit hit ({loss_percent:.2f}%). Trading locked for today.")
        _daily_loss_locked = True

    return _daily_loss_locked


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


def _calc_qty(symbol, entry_price, sl_price, equity):
    """Position size based on % risk of equity and SL distance."""
    risk_amount = equity * (config.RISK_PER_TRADE_PERCENT / 100)
    sl_distance = abs(entry_price - sl_price)
    if sl_distance <= 0:
        sl_distance = entry_price * (config.DEFAULT_SL_PERCENT / 100)

    qty = risk_amount / sl_distance
    filters = _get_instrument_filters(symbol)
    qty = max(filters["min_qty"], qty)
    step = filters["qty_step"]
    qty = round(qty / step) * step
    return round(qty, 6)


def place_trade(tv_symbol: str, action: str, sl_price: float = None, tp_price: float = None):
    """
    action: 'buy' or 'sell'
    sl_price / tp_price: optional exact prices from your Pine Script signal.
    Falls back to DEFAULT_SL_PERCENT / DEFAULT_TP_PERCENT if omitted.
    """
    if _check_daily_loss_lock():
        return {"status": "blocked", "reason": "daily_loss_limit_hit"}

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
    return {
        "equity": equity,
        "open_positions": count_open_positions(),
        "daily_loss_locked": _daily_loss_locked,
    }
