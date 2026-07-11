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
SL_BUFFER_ATR = 0.2         # stop loss placed this * ATR beyond the sweep extreme
TP_R_MULTIPLE = 2.0         # single take profit at this multiple of the SL distance (start simple - one clean R:R)


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


def check_sweep_signal(ltf_candles, swing_highs, swing_lows):
    """
    Checks the most recent closed LTF (e.g. M15) candle for a sweep of
    any unswept swing high/low, confirmed by a close back inside.
    Returns a signal dict or None.
    """
    if len(ltf_candles) < 2:
        return None

    atr = _atr_series(ltf_candles)
    i = len(ltf_candles) - 1
    a = atr[i]
    if a <= 0:
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
                return _build_signal("sell", candle, swing["price"], candle["high"], a)

    # bullish setup: wick below a swing low, close back above it -> buy
    for swing in swing_lows:
        if swing["swept"]:
            continue
        if candle["low"] < swing["price"] and candle["close"] > swing["price"]:
            wick_size = swing["price"] - candle["low"]
            if wick_size <= SWEEP_MAX_WICK_ATR * a:
                swing["swept"] = True
                return _build_signal("buy", candle, swing["price"], candle["low"], a)

    return None


def _build_signal(direction, candle, swing_price, sweep_extreme, atr_now):
    entry = candle["close"]
    if direction == "buy":
        sl = sweep_extreme - SL_BUFFER_ATR * atr_now
    else:
        sl = sweep_extreme + SL_BUFFER_ATR * atr_now

    risk = abs(entry - sl)
    tp = entry + risk * TP_R_MULTIPLE if direction == "buy" else entry - risk * TP_R_MULTIPLE

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk_distance": risk,
        "swept_level": swing_price,
    }


def check_symbol_sweep(htf_candles, ltf_candles):
    """
    Entry point matching your existing check_symbol() convention.
    htf_candles: H4 candles for swing detection
    ltf_candles: M15 (or your chosen entry timeframe) candles for the sweep+entry
    """
    swing_highs, swing_lows = find_swing_points(htf_candles)
    return check_sweep_signal(ltf_candles, swing_highs, swing_lows)
