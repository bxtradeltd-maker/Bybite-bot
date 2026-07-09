"""
Server-side port of the "Volume-Trend Order Block Engine" Pine Script.
Runs directly against Bybit candle data on Railway - no TradingView needed.

This replicates, bar by bar:
  - the custom Supertrend (trend direction + trailing stop)
  - pivot-based order block detection (top/bottom + buy/sell volume ratio)
  - order block invalidation when price fully breaks through
  - buy/sell "retest" signals (price crossing back into a still-valid block)

Runs on a fixed timeframe (default 5 minute) and is checked on a schedule
(see scheduler.py). Only the most recently CLOSED candle is used to decide
signals, matching Pine's barstate.isconfirmed behavior.
"""

import logging
import config
import bybit_trader

log = logging.getLogger("strategy")

# ---- Settings (mirrors the Pine Script inputs) ----
ST_LEN = 50            # Volatility SMA Length
ST_MULT = 3.5          # Supertrend multiplier
PIVOT_LEN = 7           # Pivot strength (left/right bars)
BULL_VOL_PCT = 0.50     # Min buy % required for a bullish retest
BEAR_VOL_PCT = 0.50     # Min sell % required for a bearish retest
DELETE_ON_BREAK = True

TIMEFRAME = "5"         # Bybit kline interval string: 5-minute
CANDLE_LIMIT = 300      # how many historical candles to pull each check


def fetch_candles(symbol: str):
    """Fetch recent closed candles from Bybit. Returns list of dicts, oldest first."""
    resp = bybit_trader.session.get_kline(
        category="linear", symbol=symbol, interval=TIMEFRAME, limit=CANDLE_LIMIT
    )
    rows = resp.get("result", {}).get("list", [])
    # Bybit returns newest-first; reverse to oldest-first
    rows = list(reversed(rows))
    candles = []
    for r in rows:
        candles.append({
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
        })
    # Drop the last candle if it's still forming (Bybit sometimes includes the live bar)
    return candles[:-1] if len(candles) > 1 else candles


def _sma(values, length, i):
    if i + 1 < length:
        return None
    window = values[i + 1 - length: i + 1]
    return sum(window) / length


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


def compute_signal(candles, collect_all=False):
    """
    Runs the full bar-by-bar state machine over the candle history.
    By default returns only the LAST bar's signal: {'buy': bool, 'sell': bool}.
    If collect_all=True, also returns every historical signal found, as
    {'all_signals': [{'index': i, 'type': 'buy'/'sell', 'price': close[i]}, ...]}
    - used for backtesting accuracy.
    """
    n = len(candles)
    if n < (2 * PIVOT_LEN + ST_LEN + 5):
        return {"buy": False, "sell": False, "reason": "not_enough_candles", "all_signals": []}

    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]

    tr = [highs[i] - lows[i] for i in range(n)]

    market_trend = 1
    trend_stop = None

    active_top = None
    active_bot = None
    active_buy_ratio = None
    active_ob_trend = 0

    buy_retest = False
    sell_retest = False
    all_signals = []

    for i in range(n):
        atr = _sma(tr, ST_LEN, i)
        if atr is None:
            continue

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

        if DELETE_ON_BREAK and active_top is not None and active_bot is not None:
            is_broken = (active_ob_trend == 1 and highs[i] < active_bot) or \
                        (active_ob_trend == -1 and lows[i] > active_top)
            if is_broken:
                active_top = None
                active_bot = None
                active_ob_trend = 0
                active_buy_ratio = None

        buy_retest = False
        sell_retest = False
        if i > 0 and active_top is not None and active_buy_ratio is not None:
            crossover = lows[i] > active_top and lows[i - 1] <= active_top
            if crossover and not pivot_low_here and \
               active_buy_ratio >= BULL_VOL_PCT and not market_trend_changed:
                buy_retest = True
                if collect_all:
                    all_signals.append({"index": i, "type": "buy", "price": closes[i]})

        if i > 0 and active_bot is not None and active_buy_ratio is not None:
            active_sell_ratio = 1.0 - active_buy_ratio
            crossunder = highs[i] < active_bot and highs[i - 1] >= active_bot
            if crossunder and not pivot_high_here and \
               active_sell_ratio >= BEAR_VOL_PCT and not market_trend_changed:
                sell_retest = True
                if collect_all:
                    all_signals.append({"index": i, "type": "sell", "price": closes[i]})

    return {"buy": buy_retest, "sell": sell_retest, "trend": market_trend, "all_signals": all_signals}


def backtest_symbol(tv_symbol: str, lookback_candles: int = 1000, max_forward_bars: int = 100):
    """
    Runs the strategy over historical candles and checks, for every signal
    it would have fired, whether price hit the take-profit or stop-loss
    level FIRST (using the same DEFAULT_SL_PERCENT / DEFAULT_TP_PERCENT the
    live bot uses). Reports an actual measured win rate - not a guess.
    """
    bybit_symbol = config.SYMBOL_MAP.get(tv_symbol.upper())
    if not bybit_symbol:
        return {"error": "symbol_not_in_map"}

    try:
        resp = bybit_trader.session.get_kline(
            category="linear", symbol=bybit_symbol, interval=TIMEFRAME, limit=min(lookback_candles, 1000)
        )
        rows = list(reversed(resp.get("result", {}).get("list", [])))
        candles = [{
            "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
            "close": float(r[4]), "volume": float(r[5]),
        } for r in rows]
    except Exception as e:
        log.error(f"Backtest fetch failed for {bybit_symbol}: {e}")
        return {"error": str(e)}

    result = compute_signal(candles, collect_all=True)
    signals = result.get("all_signals", [])

    wins = 0
    losses = 0
    no_result = 0

    for sig in signals:
        idx = sig["index"]
        entry = sig["price"]
        is_buy = sig["type"] == "buy"

        if is_buy:
            tp = entry * (1 + config.DEFAULT_TP_PERCENT / 100)
            sl = entry * (1 - config.DEFAULT_SL_PERCENT / 100)
        else:
            tp = entry * (1 - config.DEFAULT_TP_PERCENT / 100)
            sl = entry * (1 + config.DEFAULT_SL_PERCENT / 100)

        outcome = None
        for j in range(idx + 1, min(idx + 1 + max_forward_bars, len(candles))):
            bar = candles[j]
            if is_buy:
                if bar["low"] <= sl:
                    outcome = "loss"
                    break
                if bar["high"] >= tp:
                    outcome = "win"
                    break
            else:
                if bar["high"] >= sl:
                    outcome = "loss"
                    break
                if bar["low"] <= tp:
                    outcome = "win"
                    break

        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            no_result += 1

    total_decided = wins + losses
    win_rate = round((wins / total_decided * 100), 1) if total_decided > 0 else None

    return {
        "symbol": bybit_symbol,
        "candles_tested": len(candles),
        "total_signals": len(signals),
        "wins": wins,
        "losses": losses,
        "no_result_yet": no_result,
        "win_rate_percent": win_rate,
        "tp_percent_used": config.DEFAULT_TP_PERCENT,
        "sl_percent_used": config.DEFAULT_SL_PERCENT,
    }


def check_symbol(tv_symbol: str):
    """Fetches candles for a symbol, computes the signal, and places a trade if triggered."""
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
    log.info(f"{bybit_symbol} signal check: {signal}")

    if signal.get("buy"):
        log.info(f"BUY signal on {bybit_symbol}")
        return bybit_trader.place_trade(tv_symbol, "buy")
    elif signal.get("sell"):
        log.info(f"SELL signal on {bybit_symbol}")
        return bybit_trader.place_trade(tv_symbol, "sell")

    return None


def run_all_symbols():
    results = {}
    for tv_symbol in config.SYMBOL_MAP.keys():
        results[tv_symbol] = check_symbol(tv_symbol)
    return results
