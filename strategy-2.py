"""
Server-side port of the "Volume-Trend Order Block Engine" Pine Script,
extended with a 7-indicator confluence filter on top of the base OB retest
trigger.

Runs directly against Bybit candle data on Railway - no TradingView needed.

Base trigger (unchanged): bar-by-bar Supertrend + pivot-based order block
detection + retest signal, matching Pine's barstate.isconfirmed behavior.

Confluence layer (new): once the base trigger fires a candidate direction
(buy or sell), that single direction is scored against 7 indicators split
into two groups:

  Gates (must both pass, direction-agnostic, don't add to the score):
    - ATR ok        -> volatility above a minimum floor
    - RSI veto       -> not extreme in the direction being traded

  Directional confluence (must AGREE with the candidate direction,
  each agreeing indicator adds 1 to the score, out of 5):
    - VWAP position  -> buy: price below VWAP / sell: price above VWAP
    - MKR trend       -> buy: MKR trend == 1 / sell: MKR trend == -1  [placeholder]
    - HTF structure  -> buy: 4H trend up / sell: 4H trend down
    - OTE zone       -> price sitting in the 0.618-0.786 retracement of
                        the most recent swing leg, on the trade side
    - FVG confluence -> unfilled fair value gap in the trade direction  [placeholder]

Only the side that actually fired the base trigger is scored - never both
sides on the same bar. The base trigger already establishes the candidate
direction, so scoring the opposite side too would let strong confluence on
a direction with no base signal invent a trade the core logic never asked
for.

Every base-trigger firing is logged to the activity feed via
set_activity_logger(), including REJECTED signals (fired but score too low
or a gate failed) with their full score breakdown. That's the data needed
to tune MIN_CONFLUENCE_SCORE from real numbers instead of guessing.
"""

import logging
import config
import bybit_trader

log = logging.getLogger("strategy")

# ---- Base trigger settings (mirrors the Pine Script inputs) ----
ST_LEN = 50             # Volatility SMA Length
ST_MULT = 3.5           # Supertrend multiplier
PIVOT_LEN = 7           # Pivot strength (left/right bars)
BULL_VOL_PCT = 0.50     # Min buy % required for a bullish retest
BEAR_VOL_PCT = 0.50     # Min sell % required for a bearish retest
DELETE_ON_BREAK = True

TIMEFRAME = "5"         # Bybit kline interval string: 5-minute
CANDLE_LIMIT = 300      # how many historical candles to pull each check

# ---- Confluence settings ----
HTF_TIMEFRAME = "240"       # 4-hour candles for HTF structure
HTF_CANDLE_LIMIT = 200
HTF_EMA_LEN = 50            # HTF trend = close vs EMA(50) on the 4H

RSI_LEN = 14
RSI_OVERBOUGHT = 70         # veto buys when RSI >= this
RSI_OVERSOLD = 30           # veto sells when RSI <= this

ATR_MIN_PERCENT = 0.05      # ATR must be >= 0.05% of price to count as "sufficient volatility"

OTE_LOW = 0.618
OTE_HIGH = 0.786
OTE_LOOKBACK = 50           # bars searched back for the most recent swing leg

MIN_CONFLUENCE_SCORE = 3    # out of 5 directional indicators (tune from logged rejects)

# NOTE: MKR trend and FVG confluence are placeholders below (always return
# False) until those indicators are ported. That means the max achievable
# score right now is 3 (VWAP + HTF + OTE) - keep MIN_CONFLUENCE_SCORE <= 3
# until MKR/FVG land, or every signal will be rejected.

# Activity logging hook, set by server.py at startup via set_activity_logger()
_activity_logger = None


def set_activity_logger(fn):
    """Called once from server.py so strategy.py can log rejected/executed
    signals without importing server.py (would cause a circular import)."""
    global _activity_logger
    _activity_logger = fn


def _log_activity(entry):
    if _activity_logger is not None:
        try:
            _activity_logger(entry)
        except Exception as e:
            log.error(f"activity logger failed: {e}")


# ---------------------------------------------------------------------
# Candle fetching
# ---------------------------------------------------------------------

def fetch_candles(symbol: str, interval: str = TIMEFRAME, limit: int = CANDLE_LIMIT):
    """Fetch recent closed candles from Bybit. Returns list of dicts, oldest first."""
    resp = bybit_trader.session.get_kline(
        category="linear", symbol=symbol, interval=interval, limit=limit
    )
    rows = resp.get("result", {}).get("list", [])
    # Bybit returns newest-first; reverse to oldest-first
    rows = list(reversed(rows))
    candles = []
    for r in rows:
        candles.append({
            "timestamp": int(r[0]),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
        })
    # Drop the last candle if it's still forming (Bybit sometimes includes the live bar)
    return candles[:-1] if len(candles) > 1 else candles


def fetch_htf_candles(symbol: str):
    return fetch_candles(symbol, interval=HTF_TIMEFRAME, limit=HTF_CANDLE_LIMIT)


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _sma(values, length, i):
    if i + 1 < length:
        return None
    window = values[i + 1 - length: i + 1]
    return sum(window) / length


def _ema_series(values, length):
    """Full EMA series (None until enough bars have accumulated)."""
    ema = [None] * len(values)
    k = 2 / (length + 1)
    seed = None
    for i, v in enumerate(values):
        if i + 1 < length:
            continue
        if seed is None:
            seed = sum(values[i + 1 - length: i + 1]) / length
            ema[i] = seed
        else:
            ema[i] = v * k + ema[i - 1] * (1 - k)
    return ema


def _rsi_series(closes, length):
    """Wilder's RSI. Returns a list aligned with closes (None until warmed up)."""
    n = len(closes)
    rsi = [None] * n
    if n < length + 1:
        return rsi
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        change = closes[i] - closes[i - 1]
        gains[i] = max(change, 0.0)
        losses[i] = max(-change, 0.0)

    avg_gain = sum(gains[1:length + 1]) / length
    avg_loss = sum(losses[1:length + 1]) / length
    rsi[length] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))

    for i in range(length + 1, n):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        rsi[i] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))

    return rsi


def _session_vwap_series(candles):
    """VWAP that resets each UTC day, aligned with candles."""
    vwap = [None] * len(candles)
    day = None
    cum_pv = 0.0
    cum_vol = 0.0
    for i, c in enumerate(candles):
        bar_day = c["timestamp"] // 86_400_000  # ms -> day bucket
        if bar_day != day:
            day = bar_day
            cum_pv = 0.0
            cum_vol = 0.0
        typical = (c["high"] + c["low"] + c["close"]) / 3
        cum_pv += typical * c["volume"]
        cum_vol += c["volume"]
        vwap[i] = cum_pv / cum_vol if cum_vol > 0 else typical
    return vwap


def _is_pivot_low(lows, p, left, right):
    if p - left < 0 or p + right >= len(lows):
        return False
    window = lows[p - left: p + right + 1]
    return lows[p] == min(window)


def _is_pivot_high(highs, p, left, right):
    if p - left < 0 or p + right >= len(highs):
        return False
    window = highs[p - left: p + right + 1]
    return highs[p] == max(window)


# ---------------------------------------------------------------------
# Placeholders - wire these up once the indicators are ported
# ---------------------------------------------------------------------

def compute_mkr_trend(candles):
    """TODO: port the MKR trend indicator. Returning 0 (neutral) means this
    never agrees with either direction, so it never adds to the score."""
    return 0


def has_fvg_confluence(candles, direction):
    """TODO: port fair value gap detection. Returning False means this
    never adds to the score until implemented."""
    return False


# ---------------------------------------------------------------------
# HTF structure + OTE zone
# ---------------------------------------------------------------------

def compute_htf_trend(htf_candles):
    """Simple HTF structure proxy: close vs EMA(HTF_EMA_LEN) on the 4H.
    Returns 1 (up), -1 (down), or 0 (not enough data)."""
    closes = [c["close"] for c in htf_candles]
    if len(closes) < HTF_EMA_LEN + 1:
        return 0
    ema = _ema_series(closes, HTF_EMA_LEN)
    last_ema = ema[-1]
    if last_ema is None:
        return 0
    return 1 if closes[-1] > last_ema else -1


def _find_last_swing_leg(highs, lows, i, pivot_len, lookback):
    """Walk back from bar i looking for the most recent confirmed pivot low
    and pivot high within `lookback` bars, and return them as a leg
    (older_extreme, newer_extreme, leg_type) where leg_type is 'up' if the
    leg runs low->high (most recent pivot is the high) or 'down' otherwise.
    Returns None if no usable leg is found."""
    last_low = None   # (index, price)
    last_high = None  # (index, price)
    start = max(pivot_len, i - lookback)
    for p in range(i - pivot_len, start - 1, -1):
        if p < 0:
            break
        if last_low is None and _is_pivot_low(lows, p, pivot_len, pivot_len):
            last_low = (p, lows[p])
        if last_high is None and _is_pivot_high(highs, p, pivot_len, pivot_len):
            last_high = (p, highs[p])
        if last_low and last_high:
            break

    if not last_low or not last_high:
        return None

    if last_high[0] > last_low[0]:
        return last_low[1], last_high[1], "up"     # leg ran low -> high
    else:
        return last_high[1], last_low[1], "down"   # leg ran high -> low


def in_ote_zone(highs, lows, i, price, direction, pivot_len=PIVOT_LEN, lookback=OTE_LOOKBACK):
    """True if `price` sits in the 0.618-0.786 retracement of the most
    recent swing leg, on the side relevant to `direction`."""
    leg = _find_last_swing_leg(highs, lows, i, pivot_len, lookback)
    if leg is None:
        return False
    leg_start, leg_end, leg_type = leg
    span = leg_end - leg_start
    if span == 0:
        return False

    if direction == "buy" and leg_type == "up":
        # Retracement down from the high, buy zone is 0.618-0.786 back toward the low
        low_bound = leg_end - OTE_HIGH * span
        high_bound = leg_end - OTE_LOW * span
        return low_bound <= price <= high_bound

    if direction == "sell" and leg_type == "down":
        low_bound = leg_end + OTE_LOW * span
        high_bound = leg_end + OTE_HIGH * span
        return low_bound <= price <= high_bound

    return False


# ---------------------------------------------------------------------
# Base trigger (order block retest) - unchanged logic, now also returns
# the diagnostics the confluence layer needs for the last bar.
# ---------------------------------------------------------------------

def compute_signal(candles):
    """
    Runs the full bar-by-bar state machine over the candle history and
    returns the signal state for the LAST bar only, plus diagnostics
    needed for confluence scoring.
    """
    n = len(candles)
    if n < (2 * PIVOT_LEN + ST_LEN + 5):
        return {"buy": False, "sell": False, "reason": "not_enough_candles"}

    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]

    tr = [highs[i] - lows[i] for i in range(n)]
    rsi_series = _rsi_series(closes, RSI_LEN)
    vwap_series = _session_vwap_series(candles)

    market_trend = 1
    trend_stop = None

    active_top = None
    active_bot = None
    active_buy_ratio = None
    active_ob_trend = 0

    buy_retest = False
    sell_retest = False
    last_atr = None

    for i in range(n):
        atr = _sma(tr, ST_LEN, i)
        if atr is None:
            continue
        last_atr = atr

        hl2 = (highs[i] + lows[i]) / 2
        upper_band = hl2 + ST_MULT * atr
        lower_band = hl2 - ST_MULT * atr

        prev_stop = trend_stop if trend_stop is not None else (lower_band if market_trend == 1 else upper_band)
        if market_trend == 1:
            new_stop = max(lower_band, prev_stop)
        else:
            new_stop = min(upper_band, prev_stop)

        market_trend_changed = False
        if i > 0 and closes[i] > prev_stop and closes[i - 1] <= prev_stop:
            market_trend = 1
            new_stop = lower_band
            market_trend_changed = True
        elif i > 0 and closes[i] < prev_stop and closes[i - 1] >= prev_stop:
            market_trend = -1
            new_stop = upper_band
            market_trend_changed = True

        trend_stop = new_stop

        # Pivot detection: pivot confirmed pivot_len bars back from current bar i
        p_idx = i - PIVOT_LEN
        pivot_low_here = p_idx >= 0 and _is_pivot_low(lows, p_idx, PIVOT_LEN, PIVOT_LEN)
        pivot_high_here = p_idx >= 0 and _is_pivot_high(highs, p_idx, PIVOT_LEN, PIVOT_LEN)

        def window_buy_ratio(end_idx, length):
            buy_vol = 0.0
            sell_vol = 0.0
            for k in range(end_idx, end_idx - length - 1, -1):
                if k < 0:
                    break
                if closes[k] >= opens[k]:
                    buy_vol += volumes[k]
                else:
                    sell_vol += volumes[k]
            total = buy_vol + sell_vol
            return buy_vol / total if total > 0 else 0.5

        def overlapping(new_top, new_bot):
            if active_top is None or active_bot is None:
                return False
            return not (new_bot > active_top or new_top < active_bot)

        if market_trend == 1 and pivot_low_here and p_idx < len(opens):
            ob_top = min(opens[p_idx], closes[p_idx])
            ob_bot = ob_top - atr
            if not overlapping(ob_top, ob_bot):
                active_top = ob_top
                active_bot = ob_bot
                active_ob_trend = 1
                active_buy_ratio = window_buy_ratio(p_idx, PIVOT_LEN)

        if market_trend == -1 and pivot_high_here and p_idx < len(opens):
            ob_bot = max(opens[p_idx], closes[p_idx])
            ob_top = ob_bot + atr
            if not overlapping(ob_top, ob_bot):
                active_top = ob_top
                active_bot = ob_bot
                active_ob_trend = -1
                active_buy_ratio = window_buy_ratio(p_idx, PIVOT_LEN)

        # Invalidation
        if DELETE_ON_BREAK and active_top is not None and active_bot is not None:
            is_broken = (active_ob_trend == 1 and highs[i] < active_bot) or \
                        (active_ob_trend == -1 and lows[i] > active_top)
            if is_broken:
                active_top = None
                active_bot = None
                active_ob_trend = 0
                active_buy_ratio = None

        # Retest signals (only meaningful on the most recent bar, but compute every bar
        # so crossover/crossunder comparisons against bar i-1 are correct)
        buy_retest = False
        sell_retest = False
        if i > 0 and active_top is not None and active_buy_ratio is not None:
            crossover = lows[i] > active_top and lows[i - 1] <= active_top
            if crossover and not pivot_low_here and \
               active_buy_ratio >= BULL_VOL_PCT and not market_trend_changed:
                buy_retest = True

        if i > 0 and active_bot is not None and active_buy_ratio is not None:
            active_sell_ratio = 1.0 - active_buy_ratio
            crossunder = highs[i] < active_bot and highs[i - 1] >= active_bot
            if crossunder and not pivot_high_here and \
               active_sell_ratio >= BEAR_VOL_PCT and not market_trend_changed:
                sell_retest = True

    return {
        "buy": buy_retest,
        "sell": sell_retest,
        "trend": market_trend,
        # diagnostics for the confluence layer, all as of the last bar
        "last_close": closes[-1],
        "last_high": highs[-1],
        "last_low": lows[-1],
        "atr": last_atr,
        "rsi": rsi_series[-1] if rsi_series else None,
        "vwap": vwap_series[-1] if vwap_series else None,
        "highs": highs,
        "lows": lows,
    }


# ---------------------------------------------------------------------
# Confluence scoring
# ---------------------------------------------------------------------

def evaluate_confluence(direction, signal, htf_trend, candles):
    """
    Scores the fired direction ('buy' or 'sell') against the 7 indicators.
    Returns (gates_ok, score, breakdown) where breakdown lists every
    indicator's individual result for logging.
    """
    price = signal["last_close"]
    atr = signal["atr"]
    rsi = signal["rsi"]
    vwap = signal["vwap"]
    highs = signal["highs"]
    lows = signal["lows"]
    last_idx = len(highs) - 1

    # ---- Gates (direction-agnostic, must both pass) ----
    atr_ok = atr is not None and price > 0 and (atr / price) * 100 >= ATR_MIN_PERCENT

    if rsi is None:
        rsi_ok = False
    elif direction == "buy":
        rsi_ok = rsi < RSI_OVERBOUGHT
    else:
        rsi_ok = rsi > RSI_OVERSOLD

    gates_ok = atr_ok and rsi_ok

    # ---- Directional confluence (must agree with `direction`) ----
    vwap_agree = (direction == "buy" and vwap is not None and price < vwap) or \
                 (direction == "sell" and vwap is not None and price > vwap)

    mkr_trend = compute_mkr_trend(candles)
    mkr_agree = (direction == "buy" and mkr_trend == 1) or \
                (direction == "sell" and mkr_trend == -1)

    htf_agree = (direction == "buy" and htf_trend == 1) or \
                (direction == "sell" and htf_trend == -1)

    ote_agree = in_ote_zone(highs, lows, last_idx, price, direction)

    fvg_agree = has_fvg_confluence(candles, direction)

    breakdown = {
        "gates": {"atr_ok": atr_ok, "rsi_ok": rsi_ok, "rsi_value": rsi},
        "directional": {
            "vwap": vwap_agree,
            "mkr_trend": mkr_agree,
            "htf_structure": htf_agree,
            "ote_zone": ote_agree,
            "fvg": fvg_agree,
        },
    }
    score = sum([vwap_agree, mkr_agree, htf_agree, ote_agree, fvg_agree])

    return gates_ok, score, breakdown


# ---------------------------------------------------------------------
# Execution flow
# ---------------------------------------------------------------------

def check_symbol(tv_symbol: str):
    """Fetches candles for a symbol, computes the base signal, scores it
    against the confluence filter, and places a trade only if the fired
    direction clears both gates and MIN_CONFLUENCE_SCORE. Every fired
    signal (executed or rejected) is logged."""
    bybit_symbol = config.SYMBOL_MAP.get(tv_symbol.upper())
    if not bybit_symbol:
        log.warning(f"No symbol mapping for {tv_symbol}")
        return None

    try:
        candles = fetch_candles(bybit_symbol)
    except Exception as e:
        log.error(f"Failed to fetch candles for {bybit_symbol}: {e}")
        return None

    signal = compute_signal(candles)
    log.info(f"{bybit_symbol} base signal: buy={signal.get('buy')} sell={signal.get('sell')}")

    direction = "buy" if signal.get("buy") else "sell" if signal.get("sell") else None
    if direction is None:
        return None

    try:
        htf_candles = fetch_htf_candles(bybit_symbol)
        htf_trend = compute_htf_trend(htf_candles)
    except Exception as e:
        log.error(f"Failed to fetch HTF candles for {bybit_symbol}: {e}")
        htf_trend = 0

    gates_ok, score, breakdown = evaluate_confluence(direction, signal, htf_trend, candles)
    passed = gates_ok and score >= MIN_CONFLUENCE_SCORE

    entry = {
        "source": "strategy",
        "symbol": bybit_symbol,
        "direction": direction,
        "score": score,
        "min_required": MIN_CONFLUENCE_SCORE,
        "gates_ok": gates_ok,
        "breakdown": breakdown,
        "executed": passed,
    }

    if not passed:
        log.info(f"{bybit_symbol} {direction} REJECTED - gates_ok={gates_ok} score={score}/{MIN_CONFLUENCE_SCORE}")
        _log_activity(entry)
        return None

    log.info(f"{bybit_symbol} {direction.upper()} signal confirmed - score={score}/5")
    result = bybit_trader.place_trade(tv_symbol, direction)
    entry["result"] = result
    _log_activity(entry)
    return result


def run_all_symbols():
    results = {}
    for tv_symbol in config.SYMBOL_MAP.keys():
        results[tv_symbol] = check_symbol(tv_symbol)
    return results
