"""
Liquidity Sweep strategy - standalone module. Original implementation of
the public H4-swing / M15-sweep mechanic (find swing highs/lows on a
higher timeframe, wait for price to wick beyond one on a lower timeframe
then close back inside, trade the reversal).

Matches your existing candle format: list of dicts with
timestamp/open/high/low/close/volume, oldest-first.

This is intentionally standalone - not wired into strategy.py yet. Test
it on its own first (see backtest_sweep.py), then decide whether it's
worth adding as a confluence factor.
"""

# ---------------- Tunable parameters ----------------

SWING_LEFT_RIGHT = 3        # bars on each side that must be lower/higher for a swing point (H4)
MAX_ACTIVE_SWINGS = 6       # keep at most this many recent unswept swing highs/lows per side
SWEEP_MAX_WICK_ATR = 2.0    # sweep wick beyond the swing must be within this * ATR (avoid giant runaway moves)

SL_BUFFER_ATR = 0.2         # base stop loss buffer beyond the sweep extreme
SL_LOOKBACK_CANDLES = 5     # dynamic SL also considers the extreme of the last N candles
SL_MIN_ATR = 0.4            # SL distance can never be tighter than this * ATR (avoid getting stopped by noise)
SL_MAX_ATR = 3.0            # SL distance can never be wider than this * ATR (caps worst-case loss per trade)

TP1_R_MULTIPLE = 1.0        # first partial exit at 1x the SL distance
TP2_R_MULTIPLE = 2.0        # second partial exit at 2x the SL distance
TP3_R_MULTIPLE = 3.5        # final target at 3.5x the SL distance

BREAKEVEN_TRIGGER_R = 1.0   # move SL to entry once price reaches this multiple of SL distance in profit
TRAILING_ATR_MULT = 1.0     # once TP1 is hit, trail SL this * ATR behind price
ALLOW_OPPOSITE_SIGNAL_FLIP = True  # if a fresh opposite-direction signal fires while a position is
                                     # open, close it and open the new direction instead of ignoring it

# --- signal quality filter ---
VOLUME_CONFIRMATION_MULT = 1.3   # sweep candle volume must exceed this * the average volume of
                                   # the lookback window - real liquidity grabs tend to show a
                                   # volume spike; filters out low-conviction fake sweeps
VOLUME_LOOKBACK = 20              # candles used to compute the average volume baseline

# --- position sizing (risk-based, works for any account size incl. small capital) ---
RISK_PCT_PER_TRADE = 0.02         # risk this % of account balance per trade (0.02 = 2%)
MIN_NOTIONAL_USD = 5.0            # Bybit's typical minimum order notional - position sizing
                                    # below this refuses to open rather than failing at the exchange


def _atr_series(candles, length=14):
    trs = [0.0] * len(candles)
    for i in range(1, len(candles)):
        h, l, prev_close = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        trs[i] = max(h - l, abs(h - prev_close), abs(l - prev_close))
    atr = [0.0] * len(candles)
    if len(candles) > length:
        atr[length] = sum(trs[1:length + 1]) / length
        for i in range(length + 1, len(candles)):
            atr[i] = (atr[i - 1] * (length - 1) + trs[i]) / length
    return atr


def find_swing_points(htf_candles):
    """
    Scans H4 candles for confirmed swing highs/lows using a simple
    left/right pivot check. Returns two lists of dicts:
    swing_highs, swing_lows - each {price, idx, swept}
    """
    highs, lows = [], []
    n = len(htf_candles)
    lr = SWING_LEFT_RIGHT

    for i in range(lr, n - lr):
        window = htf_candles[i - lr:i + lr + 1]
        h = htf_candles[i]["high"]
        l = htf_candles[i]["low"]

        if all(h >= c["high"] for c in window):
            highs.append({"price": h, "idx": i, "swept": False})
        if all(l <= c["low"] for c in window):
            lows.append({"price": l, "idx": i, "swept": False})

    return highs[-MAX_ACTIVE_SWINGS:], lows[-MAX_ACTIVE_SWINGS:]


def _has_volume_confirmation(candles, i):
    """Real liquidity sweeps tend to show a volume spike as stops get
    triggered. Requires the sweep candle's volume to clear the average
    of the preceding VOLUME_LOOKBACK candles by VOLUME_CONFIRMATION_MULT."""
    if i < VOLUME_LOOKBACK:
        return True  # not enough history yet - don't block early signals
    lookback = candles[i - VOLUME_LOOKBACK:i]
    avg_volume = sum(c["volume"] for c in lookback) / len(lookback)
    if avg_volume <= 0:
        return True
    return candles[i]["volume"] >= avg_volume * VOLUME_CONFIRMATION_MULT


def check_sweep_signal(ltf_candles, swing_highs, swing_lows):
    """
    Checks the most recent closed LTF (e.g. M15) candle for a sweep of
    any unswept swing high/low, confirmed by a close back inside AND
    a volume spike (see VOLUME_CONFIRMATION_MULT).
    Returns a signal dict or None.
    """
    if len(ltf_candles) < 2:
        return None

    atr = _atr_series(ltf_candles)
    i = len(ltf_candles) - 1
    a = atr[i]
    if a <= 0:
        return None

    if not _has_volume_confirmation(ltf_candles, i):
        return None

    candle = ltf_candles[i]

    # bearish setup: wick above a swing high, close back below it -> sell
    for swing in swing_highs:
        if swing["swept"]:
            continue
        if candle["high"] > swing["price"] and candle["close"] < swing["price"]:
            wick_size = candle["high"] - swing["price"]
            if wick_size <= SWEEP_MAX_WICK_ATR * a:
                swing["swept"] = True
                return _build_signal("sell", candle, swing["price"], candle["high"], a, ltf_candles[:i + 1])

    # bullish setup: wick below a swing low, close back above it -> buy
    for swing in swing_lows:
        if swing["swept"]:
            continue
        if candle["low"] < swing["price"] and candle["close"] > swing["price"]:
            wick_size = swing["price"] - candle["low"]
            if wick_size <= SWEEP_MAX_WICK_ATR * a:
                swing["swept"] = True
                return _build_signal("buy", candle, swing["price"], candle["low"], a, ltf_candles[:i + 1])

    return None


def compute_position_size(balance, entry_price, sl_price, risk_pct=None):
    """
    Risk-based sizing: risk a fixed % of account balance on this trade,
    sized by how far away the stop loss is. Works correctly at any
    account size, including small capital like $20 - the position
    naturally scales down instead of using a fixed dollar amount that
    might be way too large (or too small) for the account.

    Returns dict: {qty_base_asset, notional_usd, risk_usd, below_minimum}
    Wire qty_base_asset into your bybit_trader.place_trade() call.
    """
    risk_pct = risk_pct if risk_pct is not None else RISK_PCT_PER_TRADE
    risk_usd = balance * risk_pct
    price_risk_distance = abs(entry_price - sl_price)

    if price_risk_distance <= 0:
        return {"qty_base_asset": 0, "notional_usd": 0, "risk_usd": risk_usd, "below_minimum": True}

    qty_base_asset = risk_usd / price_risk_distance
    notional_usd = qty_base_asset * entry_price
    below_minimum = notional_usd < MIN_NOTIONAL_USD

    return {
        "qty_base_asset": qty_base_asset,
        "notional_usd": notional_usd,
        "risk_usd": risk_usd,
        "below_minimum": below_minimum,
    }


def _build_signal(direction, candle, swing_price, sweep_extreme, atr_now, recent_candles=None):
    entry = candle["close"]

    # base SL: beyond the sweep extreme + buffer
    if direction == "buy":
        sl = sweep_extreme - SL_BUFFER_ATR * atr_now
    else:
        sl = sweep_extreme + SL_BUFFER_ATR * atr_now

    # dynamic adjustment: also respect the extreme of the last N candles
    if recent_candles:
        lookback = recent_candles[-SL_LOOKBACK_CANDLES:]
        if direction == "buy":
            extreme_low = min(c["low"] for c in lookback)
            sl = min(sl, extreme_low - 0.1 * atr_now)
        else:
            extreme_high = max(c["high"] for c in lookback)
            sl = max(sl, extreme_high + 0.1 * atr_now)

    # clamp SL distance to a sane min/max range (never absurdly tight or wide)
    risk = abs(entry - sl)
    min_risk, max_risk = SL_MIN_ATR * atr_now, SL_MAX_ATR * atr_now
    if risk < min_risk:
        risk = min_risk
    elif risk > max_risk:
        risk = max_risk
    sl = entry - risk if direction == "buy" else entry + risk

    if direction == "buy":
        tp1, tp2, tp3 = entry + risk * TP1_R_MULTIPLE, entry + risk * TP2_R_MULTIPLE, entry + risk * TP3_R_MULTIPLE
        breakeven_trigger = entry + risk * BREAKEVEN_TRIGGER_R
    else:
        tp1, tp2, tp3 = entry - risk * TP1_R_MULTIPLE, entry - risk * TP2_R_MULTIPLE, entry - risk * TP3_R_MULTIPLE
        breakeven_trigger = entry - risk * BREAKEVEN_TRIGGER_R

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "risk_distance": risk,
        "swept_level": swing_price,
        "breakeven_trigger_price": breakeven_trigger,
        "breakeven_done": False,
        "tp1_hit": False,
        "tp2_hit": False,
    }


def manage_open_position(position, current_price, atr_now):
    """
    Call each new candle on an open position. Handles break-even move
    and trailing stop after TP1. Returns the updated position dict
    (mutates sl in place, marks tp1_hit/tp2_hit as levels are crossed
    by the caller's own order-fill tracking).
    """
    direction = position["direction"]

    if not position["breakeven_done"]:
        reached = (current_price >= position["breakeven_trigger_price"] if direction == "buy"
                   else current_price <= position["breakeven_trigger_price"])
        if reached:
            position["sl"] = position["entry"]
            position["breakeven_done"] = True

    if position["tp1_hit"]:
        trail_distance = TRAILING_ATR_MULT * atr_now
        if direction == "buy":
            new_sl = current_price - trail_distance
            if new_sl > position["sl"]:
                position["sl"] = new_sl
        else:
            new_sl = current_price + trail_distance
            if new_sl < position["sl"]:
                position["sl"] = new_sl

    return position


def check_opposite_signal_flip(position, new_signal):
    """
    If ALLOW_OPPOSITE_SIGNAL_FLIP is on and a fresh signal fires in the
    opposite direction of an open position, this returns True to signal
    "close current position and open the new one instead." Wire into
    your execution loop: if True, call close_position() then open
    new_signal as normal.
    """
    if not ALLOW_OPPOSITE_SIGNAL_FLIP or not position or not new_signal:
        return False
    return new_signal["direction"] != position["direction"]


def check_symbol_sweep(htf_candles, ltf_candles):
    """
    Entry point matching your existing check_symbol() convention.
    htf_candles: H4 candles for swing detection
    ltf_candles: M15 (or your chosen entry timeframe) candles for the sweep+entry
    """
    swing_highs, swing_lows = find_swing_points(htf_candles)
    return check_sweep_signal(ltf_candles, swing_highs, swing_lows)
