"""
Selective trend strategy: ADX trend-strength filter + RSI pullback entry,
with a wide 1:4 reward:risk ratio.

Why this design (lessons from everything tested so far):
  1. liquidity_sweep and trend_strategy (EMA/ATR v1) both looked profitable
     on a 150-day window and both flipped negative or lost their edge once
     tested on 365 days - short windows can't be trusted.
  2. trend_strategy (v1) had a real, consistent gross edge across 4 symbols,
     but traded so often (500-1500+ signals/year) that fees/slippage (which
     cost roughly 0.25-0.46 R per trade) completely erased it, since its
     gross edge was only 0.04-0.1 R per trade.

  The fix attempted here: trade far less often, but demand a much bigger
  payoff per trade, so fixed per-trade costs become a small fraction of
  the edge instead of several multiples larger than it.

  - ADX > ADX_MIN only trades when the market is actually trending
    (skips chop, historically the source of most losing signals).
  - RSI pullback: in an uptrend, wait for RSI to dip into an oversold-ish
    pullback zone then turn back up (not just "any candle").
  - TP is set far out (4x the SL distance) so a single win comfortably
    covers several losing trades' worth of fees.

Usage:
    python selective_trend.py BTCUSDT --days 365

This is still an unverified, first-pass system - test with fees included
(default) across multiple symbols and the full year before trusting it.
Not financial advice.
"""

import argparse

import bybit_trader


# ---- Strategy parameters ----
EMA_TREND = 100      # long-term trend filter
RSI_LEN = 14
RSI_PULLBACK_LOW = 40    # uptrend pullback zone (buy dips into here)
RSI_PULLBACK_HIGH = 60   # downtrend pullback zone (sell rallies into here)
ADX_LEN = 14
ADX_MIN = 25          # only trade when ADX confirms a real trend
ATR_LEN = 14
ATR_SL_MULT = 1.5
ATR_TP_MULT = 6.0     # 1:4 reward:risk (6.0 / 1.5)

TAKER_FEE_PCT = 0.06
SLIPPAGE_PCT = 0.03
ROUND_TRIP_COST_PCT = 2 * (TAKER_FEE_PCT + SLIPPAGE_PCT)


def fetch_history(symbol: str, interval: str, days: int):
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
            "timestamp": int(r[0]), "open": float(r[1]), "high": float(r[2]),
            "low": float(r[3]), "close": float(r[4]), "volume": float(r[5]),
        })
    return candles


def _ema(values, period):
    k = 2 / (period + 1)
    out = [None] * len(values)
    for i, v in enumerate(values):
        if i < period - 1:
            continue
        if out[i - 1] is None:
            out[i] = sum(values[i - period + 1:i + 1]) / period
        else:
            out[i] = v * k + out[i - 1] * (1 - k)
    return out


def _rsi(closes, period):
    out = [None] * len(closes)
    gains, losses = [0.0] * len(closes), [0.0] * len(closes)
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains[i] = max(change, 0)
        losses[i] = max(-change, 0)

    avg_gain = avg_loss = None
    for i in range(period, len(closes)):
        if avg_gain is None:
            avg_gain = sum(gains[i - period + 1:i + 1]) / period
            avg_loss = sum(losses[i - period + 1:i + 1]) / period
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i] = 100
        else:
            rs = avg_gain / avg_loss
            out[i] = 100 - (100 / (1 + rs))
    return out


def _atr(candles, period):
    trs = [None] * len(candles)
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        trs[i] = max(h - l, abs(h - pc), abs(l - pc))
    out = [None] * len(candles)
    for i in range(period, len(candles)):
        if out[i - 1] is None:
            window = [t for t in trs[i - period + 1:i + 1] if t is not None]
            out[i] = sum(window) / len(window)
        else:
            out[i] = (out[i - 1] * (period - 1) + trs[i]) / period
    return out


def _adx(candles, period):
    """Standard Wilder ADX."""
    n = len(candles)
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        up_move = candles[i]["high"] - candles[i - 1]["high"]
        down_move = candles[i - 1]["low"] - candles[i]["low"]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))

    def wilder_smooth(values, period):
        out = [None] * len(values)
        for i in range(period, len(values)):
            if out[i - 1] is None:
                out[i] = sum(values[i - period + 1:i + 1])
            else:
                out[i] = out[i - 1] - (out[i - 1] / period) + values[i]
        return out

    tr_smooth = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)

    adx = [None] * n
    dx_vals = [None] * n
    for i in range(period, n):
        if tr_smooth[i] in (None, 0):
            continue
        plus_di = 100 * plus_dm_smooth[i] / tr_smooth[i]
        minus_di = 100 * minus_dm_smooth[i] / tr_smooth[i]
        denom = plus_di + minus_di
        dx_vals[i] = 100 * abs(plus_di - minus_di) / denom if denom else 0

    for i in range(2 * period, n):
        window = [d for d in dx_vals[i - period + 1:i + 1] if d is not None]
        if window:
            adx[i] = sum(window) / len(window)
    return adx


def detect_signals(candles):
    closes = [c["close"] for c in candles]
    ema = _ema(closes, EMA_TREND)
    rsi = _rsi(closes, RSI_LEN)
    atr = _atr(candles, ATR_LEN)
    adx = _adx(candles, ADX_LEN)

    signals = []
    warmup = max(EMA_TREND, ADX_LEN * 2, ATR_LEN) + 5

    for i in range(warmup, len(candles) - 1):
        if None in (ema[i], rsi[i], rsi[i - 1], atr[i], adx[i]):
            continue
        if adx[i] < ADX_MIN:
            continue

        c = candles[i]
        uptrend = c["close"] > ema[i]
        downtrend = c["close"] < ema[i]

        direction = None
        # pullback into the RSI zone, then turning back the trend's way
        if uptrend and rsi[i - 1] < RSI_PULLBACK_LOW and rsi[i] > rsi[i - 1] and c["close"] > c["open"]:
            direction = "buy"
        elif downtrend and rsi[i - 1] > RSI_PULLBACK_HIGH and rsi[i] < rsi[i - 1] and c["close"] < c["open"]:
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
            "index": i, "direction": direction, "entry_price": entry_price,
            "sl_price": sl_price, "tp_price": tp_price,
        })

    return signals


def simulate_signal(candles, signal, max_bars_forward=500, apply_costs=True):
    idx = signal["index"] + 1
    if idx >= len(candles):
        return None, 0.0

    direction = signal["direction"]
    entry = signal["entry_price"]
    sl, tp = signal["sl_price"], signal["tp_price"]
    risk_distance = abs(entry - sl)
    cost_r = 0.0
    if apply_costs and risk_distance > 0:
        cost_r = (ROUND_TRIP_COST_PCT / 100 * entry) / risk_distance

    for j in range(idx, min(idx + max_bars_forward, len(candles))):
        bar = candles[j]
        if direction == "buy":
            hit_sl = bar["low"] <= sl
            hit_tp = bar["high"] >= tp
        else:
            hit_sl = bar["high"] >= sl
            hit_tp = bar["low"] <= tp

        if hit_sl and hit_tp:
            return "loss", -1.0 - cost_r
        if hit_sl:
            return "loss", -1.0 - cost_r
        if hit_tp:
            r = abs(tp - entry) / abs(entry - sl)
            return "win", r - cost_r

    return None, 0.0


def run_backtest(symbol: str, days: int, interval: str = "15", verbose: bool = True):
    candles = fetch_history(symbol, interval, days)
    if len(candles) < 250:
        return {"status": "error", "reason": "not_enough_candles", "symbol": symbol}

    signals = detect_signals(candles)

    results_net = [simulate_signal(candles, s, apply_costs=True) for s in signals]
    resolved_net = [(o, r) for o, r in results_net if o is not None]
    wins_net = [r for o, r in resolved_net if o == "win"]
    losses_net = [r for o, r in resolved_net if o == "loss"]
    total_r_net = sum(r for _, r in resolved_net)
    win_rate = 100 * len(wins_net) / len(resolved_net) if resolved_net else 0
    avg_r_net = total_r_net / len(resolved_net) if resolved_net else 0

    results_gross = [simulate_signal(candles, s, apply_costs=False) for s in signals]
    resolved_gross = [(o, r) for o, r in results_gross if o is not None]
    total_r_gross = sum(r for _, r in resolved_gross)
    avg_r_gross = total_r_gross / len(resolved_gross) if resolved_gross else 0

    rr = ATR_TP_MULT / ATR_SL_MULT
    breakeven_wr = 100 / (1 + rr)

    out = {
        "status": "ok",
        "symbol": symbol,
        "days": days,
        "interval_minutes": interval,
        "total_signals": len(signals),
        "resolved": len(resolved_net),
        "unresolved": len(signals) - len(resolved_net),
        "wins": len(wins_net),
        "losses": len(losses_net),
        "win_rate_pct": round(win_rate, 2),
        "breakeven_win_rate_pct": round(breakeven_wr, 2),
        "edge_pct_points": round(win_rate - breakeven_wr, 2),
        "avg_r_per_trade_net_of_costs": round(avg_r_net, 3),
        "avg_r_per_trade_gross_no_costs": round(avg_r_gross, 3),
        "cost_drag_per_trade_r": round(avg_r_gross - avg_r_net, 3),
        "total_r_net_of_costs": round(total_r_net, 2),
        "total_r_gross_no_costs": round(total_r_gross, 2),
        "round_trip_cost_pct_assumed": round(ROUND_TRIP_COST_PCT, 3),
        "reward_risk_ratio": round(rr, 2),
        "adx_min": ADX_MIN,
        "atr_sl_mult": ATR_SL_MULT,
        "atr_tp_mult": ATR_TP_MULT,
    }

    if verbose:
        print(f"\n=== Selective Trend Backtest: {symbol} ({days}d) ===")
        for k, v in out.items():
            print(f"  {k}: {v}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--interval", default="15")
    args = parser.parse_args()
    run_backtest(args.symbol, args.days, args.interval)
