"""
Trend-following strategy with volatility-normalized (ATR-based) stop-loss
and take-profit levels, designed to generalize across symbols rather than
being tuned to one asset's typical move size.

Why this design (vs. the fixed-R-multiple liquidity-sweep strategy):
  The liquidity-sweep backtest showed a strong edge on BTCUSDT (avg R 0.443)
  but a flat-to-losing edge on ETHUSDT/BNBUSDT/SOLUSDT, because its SL/TP
  distances were expressed as fixed multiples of a fixed risk distance.
  Different coins have very different typical volatility, so a fixed
  distance that's "3.5x risk" on BTC might be an enormous ask on a calmer
  coin, or trivially easy on a wilder one.

  This strategy instead measures each symbol's own recent ATR (Average
  True Range) at signal time and sizes SL/TP relative to *that*, so the
  same strategy logic should behave consistently across BTC, ETH, SOL,
  BNB, etc. without per-symbol hand-tuning.

Entry logic:
  - Trend filter: EMA_FAST > EMA_SLOW for longs (below for shorts), plus
    price must be trading on the correct side of EMA_SLOW.
  - Momentum trigger: a pullback to EMA_FAST followed by a strong-bodied
    candle back in the trend direction (basic continuation entry).
  - ATR filter: skip signals when ATR is unusually low (dead/illiquid
    conditions) or unusually high (news-spike conditions), since both
    tend to produce unreliable fills.

Risk management:
  - Stop-loss = ATR_SL_MULT * ATR at entry
  - Take-profit = ATR_TP_MULT * ATR at entry (single target, no partials,
    to keep this first version simple and easy to reason about)

Usage:
    python trend_strategy.py BTCUSDT --days 150
    python trend_strategy.py ETHUSDT --days 150
    python trend_strategy.py SOLUSDT --days 150
    python trend_strategy.py BNBUSDT --days 150

Caveats (read before trusting the numbers):
  - PnL is expressed in R multiples, not modeling fees, slippage, or your
    actual position sizing.
  - This is a v1 template meant for comparison against the existing
    liquidity-sweep strategy, not a finished, live-ready system. Validate
    on a longer history (1yr+) before considering it for real capital.
  - Not financial advice - this is a statistical/engineering tool to help
    you evaluate whether an approach has a plausible edge.
"""

import argparse
import statistics

import bybit_trader  # reuses the mainnet-only market_data_session


# ---- Strategy parameters (kept symbol-agnostic on purpose) ----
EMA_FAST = 21
EMA_SLOW = 55
ATR_LEN = 14
ATR_SL_MULT = 1.5   # stop-loss distance = 1.5x ATR
ATR_TP_MULT = 3.0   # take-profit distance = 3x ATR (2:1 reward:risk)
ATR_MIN_PCTL = 20    # skip signals when ATR is below this percentile (too quiet)
ATR_MAX_PCTL = 90    # skip signals when ATR is above this percentile (too wild)
MIN_BODY_RATIO = 0.5  # trigger candle body must be at least this % of its range


def fetch_history(symbol: str, interval: str, days: int):
    """Same pagination approach as backtest.py, but explicitly routed
    through the mainnet market_data_session (public data, no geo-block)."""
    interval_minutes = int(interval)
    bars_per_day = (24 * 60) / interval_minutes
    target_bars = int(bars_per_day * days) + 50

    all_rows = []
    end_time = None
    while len(all_rows) < target_bars:
        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["end"] = end_time
        resp = bybit_trader.market_data_session.get_kline(**params)
        rows = resp.get("result", {}).get("list", [])
        if not rows:
            break
        all_rows.extend(rows)
        oldest_ts = int(rows[-1][0])
        end_time = oldest_ts - 1
        if len(rows) < 1000:
            break

    seen = {}
    for r in all_rows:
        seen[int(r[0])] = r
    ordered = [seen[t] for t in sorted(seen.keys())]

    candles = []
    for r in ordered:
        candles.append({
            "timestamp": int(r[0]),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
        })
    return candles


def _ema(values, period):
    """Returns a list of EMA values, same length as input (None for warmup)."""
    k = 2 / (period + 1)
    ema_vals = [None] * len(values)
    for i, v in enumerate(values):
        if i < period - 1:
            continue
        if ema_vals[i - 1] is None:
            seed = sum(values[i - period + 1:i + 1]) / period
            ema_vals[i] = seed
        else:
            ema_vals[i] = v * k + ema_vals[i - 1] * (1 - k)
    return ema_vals


def _atr(candles, period):
    """Returns a list of ATR values (Wilder's smoothing), same length,
    None for warmup."""
    trs = [None] * len(candles)
    for i in range(1, len(candles)):
        h, l, prev_close = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        trs[i] = max(h - l, abs(h - prev_close), abs(l - prev_close))

    atr_vals = [None] * len(candles)
    for i in range(period, len(candles)):
        if atr_vals[i - 1] is None:
            window = [t for t in trs[i - period + 1:i + 1] if t is not None]
            atr_vals[i] = sum(window) / len(window)
        else:
            atr_vals[i] = (atr_vals[i - 1] * (period - 1) + trs[i]) / period
    return atr_vals


def _percentile_rank(value, population):
    """% of population values <= value."""
    if not population:
        return 50
    below = sum(1 for p in population if p <= value)
    return 100 * below / len(population)


def detect_signals(candles):
    """Walks the candles and returns a list of signal dicts:
    {index, direction, entry_price, sl_price, tp_price}"""
    closes = [c["close"] for c in candles]
    ema_fast = _ema(closes, EMA_FAST)
    ema_slow = _ema(closes, EMA_SLOW)
    atr = _atr(candles, ATR_LEN)

    signals = []
    warmup = max(EMA_SLOW, ATR_LEN) + 5
    # rolling window of recent ATR values for percentile filtering
    atr_window_size = 200

    for i in range(warmup, len(candles) - 1):
        if ema_fast[i] is None or ema_slow[i] is None or atr[i] is None:
            continue

        recent_atrs = [a for a in atr[max(0, i - atr_window_size):i] if a is not None]
        atr_pctl = _percentile_rank(atr[i], recent_atrs)
        if atr_pctl < ATR_MIN_PCTL or atr_pctl > ATR_MAX_PCTL:
            continue

        c = candles[i]
        body = abs(c["close"] - c["open"])
        rng = c["high"] - c["low"]
        if rng == 0 or body / rng < MIN_BODY_RATIO:
            continue

        uptrend = ema_fast[i] > ema_slow[i] and c["close"] > ema_slow[i]
        downtrend = ema_fast[i] < ema_slow[i] and c["close"] < ema_slow[i]

        # pullback-and-continuation trigger: prior candle touched ema_fast,
        # this candle closes strongly back in the trend direction
        prev = candles[i - 1]
        touched_fast = prev["low"] <= ema_fast[i - 1] <= prev["high"] if ema_fast[i - 1] else False

        direction = None
        if uptrend and touched_fast and c["close"] > c["open"]:
            direction = "buy"
        elif downtrend and touched_fast and c["close"] < c["open"]:
            direction = "sell"

        if direction is None:
            continue

        entry_price = candles[i + 1]["open"]
        risk = atr[i] * ATR_SL_MULT
        reward = atr[i] * ATR_TP_MULT
        if direction == "buy":
            sl_price = entry_price - risk
            tp_price = entry_price + reward
        else:
            sl_price = entry_price + risk
            tp_price = entry_price - reward

        signals.append({
            "index": i,
            "direction": direction,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
        })

    return signals


def simulate_signal(candles, signal, max_bars_forward=300):
    """Walks forward from the signal's entry bar until SL or TP hit.
    Returns ('win'|'loss'|None, r_multiple)."""
    idx = signal["index"] + 1
    if idx >= len(candles):
        return None, 0.0

    direction = signal["direction"]
    entry = signal["entry_price"]
    sl, tp = signal["sl_price"], signal["tp_price"]

    for j in range(idx, min(idx + max_bars_forward, len(candles))):
        bar = candles[j]
        if direction == "buy":
            hit_sl = bar["low"] <= sl
            hit_tp = bar["high"] >= tp
        else:
            hit_sl = bar["high"] >= sl
            hit_tp = bar["low"] <= tp

        if hit_sl and hit_tp:
            return "loss", -1.0  # worst-case tie-break, same convention as existing backtest
        if hit_sl:
            return "loss", -1.0
        if hit_tp:
            r = abs(tp - entry) / abs(entry - sl)
            return "win", r

    return None, 0.0  # never resolved within window


def run_backtest(symbol: str, days: int, interval: str = "15", verbose: bool = True):
    candles = fetch_history(symbol, interval, days)
    if len(candles) < 100:
        return {"status": "error", "reason": "not_enough_candles", "symbol": symbol}

    signals = detect_signals(candles)
    results = [simulate_signal(candles, s) for s in signals]
    resolved = [(o, r) for o, r in results if o is not None]

    wins = [r for o, r in resolved if o == "win"]
    losses = [r for o, r in resolved if o == "loss"]
    total_r = sum(r for _, r in resolved)
    win_rate = 100 * len(wins) / len(resolved) if resolved else 0
    avg_r = total_r / len(resolved) if resolved else 0
    avg_win_r = statistics.mean(wins) if wins else 0
    # breakeven win rate given fixed R:R of ATR_TP_MULT/ATR_SL_MULT
    rr = ATR_TP_MULT / ATR_SL_MULT
    breakeven_wr = 100 / (1 + rr)

    out = {
        "status": "ok",
        "symbol": symbol,
        "days": days,
        "interval_minutes": interval,
        "total_signals": len(signals),
        "resolved": len(resolved),
        "unresolved": len(signals) - len(resolved),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate, 2),
        "breakeven_win_rate_pct": round(breakeven_wr, 2),
        "edge_pct_points": round(win_rate - breakeven_wr, 2),
        "avg_r_per_trade": round(avg_r, 3),
        "avg_win_r": round(avg_win_r, 3),
        "total_r": round(total_r, 2),
        "reward_risk_ratio": round(rr, 2),
        "atr_sl_mult": ATR_SL_MULT,
        "atr_tp_mult": ATR_TP_MULT,
    }

    if verbose:
        print(f"\n=== Trend Strategy Backtest: {symbol} ({days}d) ===")
        for k, v in out.items():
            print(f"  {k}: {v}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("--days", type=int, default=150)
    parser.add_argument("--interval", default="15")
    args = parser.parse_args()
    run_backtest(args.symbol, args.days, args.interval)
