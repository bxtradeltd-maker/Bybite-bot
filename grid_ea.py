"""
Grid/basket EA - an original implementation of the same core idea as
Waka Waka style EAs: open an initial position, add to it at fixed
intervals against you with a lot multiplier, close the whole basket
when combined floating profit hits a target.

This is NOT a copy of any commercial EA's code - it's the standard,
publicly-documented grid/martingale mechanic, written from scratch to
plug into your existing bybit_trader.py / config.py.

HARD SAFETY RAILS (not optional, read before changing):
  - MAX_GRID_LEVELS caps how many times it can add to a losing basket.
    Uncapped martingale is how these accounts blow up. Don't remove this.
  - EQUITY_STOP_PCT force-closes the whole basket at a max floating loss
    regardless of grid level, so a single trend can't nuke the account.
  - One basket per symbol at a time. No overlapping grids on the same pair.

You need to wire in three functions from your bybit_trader.py that
aren't shown here since bybit_trader.py wasn't uploaded:
    bybit_trader.place_trade(symbol, action)          -> opens a market order, returns order info incl. avg fill price
    bybit_trader.get_open_positions(symbol=None)       -> list of open positions w/ symbol, side, size, avg_price, unrealized_pnl
    bybit_trader.close_position(symbol)                -> market-closes all open exposure on that symbol
Adjust the three calls in _open_level() / _close_basket() / basket_floating_pnl_pct()
below if your actual function signatures differ.
"""

import time
import logging

import config
import bybit_trader

log = logging.getLogger("grid_ea")

# ---------------- Tunable parameters ----------------

INITIAL_QTY_USD = 50          # notional size of the first grid level
LOT_MULTIPLIER = 1.5          # each add-on level is this much bigger than the last
GRID_STEP_PCT = 0.6           # price must move this % against the basket to add a level
MAX_GRID_LEVELS = 5           # HARD CAP - do not add more than this many levels
BASKET_TP_PCT = 1.0           # close whole basket when floating profit hits this % of total notional
EQUITY_STOP_PCT = -8.0        # force-close whole basket at this % floating loss, no matter the level
POLL_SECONDS = 15             # how often to check price / manage the basket

# in-memory basket state, keyed by symbol
_baskets = {}


def _new_basket(symbol, direction):
    return {
        "symbol": symbol,
        "direction": direction,   # "buy" or "sell"
        "levels": [],             # list of {"qty_usd":, "entry_price":}
        "opened_at": time.time(),
    }


def _next_level_qty(basket):
    if not basket["levels"]:
        return INITIAL_QTY_USD
    return basket["levels"][-1]["qty_usd"] * LOT_MULTIPLIER


def _avg_entry_price(basket):
    total_qty = sum(l["qty_usd"] for l in basket["levels"])
    weighted = sum(l["qty_usd"] * l["entry_price"] for l in basket["levels"])
    return weighted / total_qty if total_qty else 0.0


def _basket_notional(basket):
    return sum(l["qty_usd"] for l in basket["levels"])


def basket_floating_pnl_pct(basket, current_price):
    """% floating PnL of the whole basket relative to its total notional."""
    if not basket["levels"]:
        return 0.0
    avg_entry = _avg_entry_price(basket)
    if basket["direction"] == "buy":
        raw_pct = (current_price - avg_entry) / avg_entry * 100
    else:
        raw_pct = (avg_entry - current_price) / avg_entry * 100
    return raw_pct


def _open_level(basket, price):
    qty_usd = _next_level_qty(basket)
    action = basket["direction"]
    log.info(f"[{basket['symbol']}] opening grid level {len(basket['levels']) + 1} "
              f"({action}, ${qty_usd:.2f} notional) @ {price}")
    # NOTE: adjust this call to match your actual bybit_trader.place_trade signature.
    # If place_trade requires qty in contracts rather than USD notional, convert first.
    result = bybit_trader.place_trade(basket["symbol"], action)
    basket["levels"].append({"qty_usd": qty_usd, "entry_price": price, "order_result": result})
    return result


def _close_basket(basket, reason):
    log.info(f"[{basket['symbol']}] closing basket ({reason}), "
              f"{len(basket['levels'])} levels, notional ${_basket_notional(basket):.2f}")
    bybit_trader.close_position(basket["symbol"])
    _baskets.pop(basket["symbol"], None)


def _get_current_price(symbol):
    positions = bybit_trader.get_open_positions(symbol=symbol)
    if positions:
        # most bybit_trader implementations return mark/last price on the position dict
        return positions[0].get("mark_price") or positions[0].get("avg_price")
    return None


def start_basket(symbol, direction):
    """Call this to open a new grid basket. Refuses if one's already running on this symbol."""
    if symbol in _baskets:
        log.warning(f"[{symbol}] basket already active, ignoring start request")
        return {"status": "error", "reason": "basket_already_active"}

    basket = _new_basket(symbol, direction)
    price = _get_current_price(symbol)
    if price is None:
        return {"status": "error", "reason": "no_price_available"}

    _open_level(basket, price)
    _baskets[symbol] = basket
    return {"status": "ok", "symbol": symbol, "direction": direction}


def manage_basket(symbol):
    """Call this on a timer (every POLL_SECONDS) for each active symbol.
    Adds a grid level if price has moved against the basket by GRID_STEP_PCT
    (and MAX_GRID_LEVELS hasn't been hit), closes on TP or equity stop."""
    basket = _baskets.get(symbol)
    if not basket:
        return {"status": "no_active_basket"}

    price = _get_current_price(symbol)
    if price is None:
        return {"status": "error", "reason": "no_price_available"}

    pnl_pct = basket_floating_pnl_pct(basket, price)

    # 1. Equity stop - overrides everything else, always checked first
    if pnl_pct <= EQUITY_STOP_PCT:
        _close_basket(basket, f"equity_stop hit ({pnl_pct:.2f}%)")
        return {"status": "closed", "reason": "equity_stop", "pnl_pct": pnl_pct}

    # 2. Take profit
    if pnl_pct >= BASKET_TP_PCT:
        _close_basket(basket, f"basket_tp hit ({pnl_pct:.2f}%)")
        return {"status": "closed", "reason": "take_profit", "pnl_pct": pnl_pct}

    # 3. Grid add-on
    avg_entry = _avg_entry_price(basket)
    if basket["direction"] == "buy":
        moved_against_pct = (avg_entry - price) / avg_entry * 100
    else:
        moved_against_pct = (price - avg_entry) / avg_entry * 100

    if moved_against_pct >= GRID_STEP_PCT and len(basket["levels"]) < MAX_GRID_LEVELS:
        _open_level(basket, price)
        return {"status": "level_added", "level": len(basket["levels"]), "pnl_pct": pnl_pct}

    if moved_against_pct >= GRID_STEP_PCT and len(basket["levels"]) >= MAX_GRID_LEVELS:
        log.warning(f"[{symbol}] MAX_GRID_LEVELS reached, basket riding it out until "
                     f"TP or equity_stop - no more adds")

    return {"status": "holding", "pnl_pct": pnl_pct, "levels": len(basket["levels"])}


def run_forever(symbol):
    """Simple polling loop. In server.py you'd more likely call manage_basket()
    from an APScheduler job per active symbol instead of blocking like this."""
    while symbol in _baskets:
        manage_basket(symbol)
        time.sleep(POLL_SECONDS)
