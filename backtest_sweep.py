"""
Backtest for liquidity_sweep.py - walks M15 candles bar-by-bar (no
lookahead), re-detecting H4 swings as they would have been known at
each point in time, and checks each sweep signal through to its SL/TP
outcome.

Run on a machine with Bybit access:
    python backtest_sweep.py BTCUSDT --days 90
"""

import argparse

import liquidity_sweep
from backtest import fetch_history


def check_sweep_signal_at(ltf_candles, atr_series, i, swing_highs, swing_lows):
    """
    Same logic as liquidity_sweep.check_sweep_signal but takes a
    precomputed ATR series and target index, so the backtest doesn't
    have to recompute ATR from scratch on every single bar (that was
    the main cause of timeouts on longer day windows).
    """
    a = atr_series[i]
    if a <= 0:
        return None

    if not liquidity_sweep._has_volume_confirmation(ltf_candles, i):
        return None

    candle = ltf_candles[i]

    for swing in swing_highs:
        if swing["swept"]:
            continue
        if candle["high"] > swing["price"] and candle["close"] < swing["price"]:
            wick_size = candle["high"] - swing["price"]
            if wick_size <= liquidity_sweep.SWEEP_MAX_WICK_ATR * a:
                swing["swept"] = True
                return liquidity_sweep._build_signal("sell", candle, swing["price"], candle["high"], a, ltf_candles[:i + 1])

    for swing in swing_lows:
        if swing["swept"]:
            continue
        if candle["low"] < swing["price"] and candle["close"] > swing["price"]:
            wick_size = swing["price"] - candle["low"]
            if wick_size <= liquidity_sweep.SWEEP_MAX_WICK_ATR * a:
                swing["swept"] = True
                return liquidity_sweep._build_signal("buy", candle, swing["price"], candle["low"], a, ltf_candles[:i + 1])

    return None


def simulate(ltf_candles, htf_candles, htf_interval_ratio):
    """
    htf_interval_ratio: how many LTF candles = 1 HTF candle (e.g. H4/M15 = 16)
    Re-derives swing points using only HTF candles that would have closed
    by the current LTF bar, so this doesn't cheat with future H4 data.

    Performance notes (this is what made 150+ day backtests time out):
      1. Swing points only actually change once a new H4 candle becomes
         visible (every htf_interval_ratio LTF bars) - cached and only
         recomputed on those boundaries instead of every LTF candle.
      2. ATR is computed ONCE upfront over the full candle list, instead
         of being recalculated from scratch on the growing slice every
         iteration (that was silently O(n^2) and the main slowdown).
    """
    trades = []
    swept_high_prices = set()
    swept_low_prices = set()

    atr_series = liquidity_sweep._atr_series(ltf_candles)

    swing_highs, swing_lows = [], []
    last_htf_visible_count = -1

    for i in range(50, len(ltf_candles)):
        htf_visible_count = min(len(htf_candles), (i // htf_interval_ratio) + 1)
        if htf_visible_count < liquidity_sweep.SWING_LEFT_RIGHT * 2 + 1:
            continue

        if htf_visible_count != last_htf_visible_count:
            htf_slice = htf_candles[:htf_visible_count]
            swing_highs, swing_lows = liquidity_sweep.find_swing_points(htf_slice)
            for s in swing_highs:
                if s["price"] in swept_high_prices:
                    s["swept"] = True
            for s in swing_lows:
                if s["price"] in swept_low_prices:
                    s["swept"] = True
            last_htf_visible_count = htf_visible_count

        signal = check_sweep_signal_at(ltf_candles, atr_series, i, swing_highs, swing_lows)
        if not signal:
            continue

        if signal["direction"] == "sell":
            swept_high_prices.add(signal["swept_level"])
        else:
            swept_low_prices.add(signal["swept_level"])

        outcome = _walk_to_outcome(ltf_candles, i + 1, signal)
        outcome["detailed"] = simulate_full_management(ltf_candles, atr_series, i + 1, signal)
        trades.append(outcome)

    return trades


def _walk_to_outcome(candles, start_idx, signal):
    """
    Validates against TP2 (2:1 R:R) - the same target that produced the
    proven +13.2pt edge in earlier tests. Kept simple/binary on purpose
    so these numbers stay comparable to what's already been validated.
    See simulate_full_management() for the detailed TP1/TP2/TP3 breakdown.
    """
    direction = signal["direction"]
    for j in range(start_idx, len(candles)):
        bar = candles[j]
        check_order = (bar["low"], bar["high"]) if direction == "buy" else (bar["high"], bar["low"])
        for price in check_order:
            if direction == "buy":
                if price <= signal["sl"]:
                    return {"result": "loss", "r_multiple": -1.0, "bars_held": j - start_idx}
                if price >= signal["tp2"]:
                    return {"result": "win", "r_multiple": liquidity_sweep.TP2_R_MULTIPLE, "bars_held": j - start_idx}
            else:
                if price >= signal["sl"]:
                    return {"result": "loss", "r_multiple": -1.0, "bars_held": j - start_idx}
                if price <= signal["tp2"]:
                    return {"result": "win", "r_multiple": liquidity_sweep.TP2_R_MULTIPLE, "bars_held": j - start_idx}
    return {"result": "unresolved", "r_multiple": 0.0, "bars_held": len(candles) - start_idx}


def simulate_full_management(candles, atr_series, start_idx, signal):
    """
    Walks the trade through TP1/TP2/TP3 with breakeven and trailing
    applied exactly as manage_open_position() would live. Assumes an
    even 1/3 position split closed at each TP level (the standard way
    to run a 3-target system) and reports which level was the final
    outcome plus the blended R-multiple across the three thirds.
    """
    direction = signal["direction"]
    sl = signal["sl"]
    tp1_hit = tp2_hit = tp3_hit = False
    r_realized = 0.0  # accumulates as each third closes

    for j in range(start_idx, len(candles)):
        bar = candles[j]
        a = atr_series[j] if j < len(atr_series) else atr_series[-1]
        check_order = (bar["low"], bar["high"]) if direction == "buy" else (bar["high"], bar["low"])

        for price in check_order:
            hit_sl = (price <= sl) if direction == "buy" else (price >= sl)
            if hit_sl:
                if not tp1_hit:
                    # full loss on all three thirds
                    return {"result": "sl_before_tp1", "r_multiple": -1.0, "bars_held": j - start_idx}
                else:
                    # stopped at breakeven or trailing after partial profit already banked
                    remaining_thirds = (1 if tp2_hit else 2) if not tp3_hit else 0
                    remaining_r = 0.0 if sl == signal["entry"] else (
                        (sl - signal["entry"]) / signal["risk_distance"] if direction == "buy"
                        else (signal["entry"] - sl) / signal["risk_distance"]
                    )
                    r_realized += remaining_r * remaining_thirds
                    label = "tp2_then_stopped" if tp2_hit else "tp1_then_stopped"
                    return {"result": label, "r_multiple": round(r_realized, 3), "bars_held": j - start_idx}

            if not tp1_hit:
                reached_tp1 = (price >= signal["tp1"]) if direction == "buy" else (price <= signal["tp1"])
                if reached_tp1:
                    tp1_hit = True
                    r_realized += liquidity_sweep.TP1_R_MULTIPLE / 3
                    sl = signal["entry"]  # breakeven move
                    continue

            if tp1_hit and not tp2_hit:
                reached_tp2 = (price >= signal["tp2"]) if direction == "buy" else (price <= signal["tp2"])
                if reached_tp2:
                    tp2_hit = True
                    r_realized += liquidity_sweep.TP2_R_MULTIPLE / 3
                    trail = liquidity_sweep.TRAILING_ATR_MULT * a
                    sl = (price - trail) if direction == "buy" else (price + trail)
                    continue

            if tp2_hit and not tp3_hit:
                reached_tp3 = (price >= signal["tp3"]) if direction == "buy" else (price <= signal["tp3"])
                if reached_tp3:
                    tp3_hit = True
                    r_realized += liquidity_sweep.TP3_R_MULTIPLE / 3
                    return {"result": "tp1_tp2_tp3", "r_multiple": round(r_realized, 3), "bars_held": j - start_idx}

            # trailing stop update once TP1 has been hit and price is progressing
            if tp1_hit and not tp3_hit:
                trail = liquidity_sweep.TRAILING_ATR_MULT * a
                trailing_sl = (price - trail) if direction == "buy" else (price + trail)
                if direction == "buy":
                    sl = max(sl, trailing_sl)
                else:
                    sl = min(sl, trailing_sl)

    # ran out of data
    if not tp1_hit:
        return {"result": "unresolved_no_tp1", "r_multiple": 0.0, "bars_held": len(candles) - start_idx}
    label = "unresolved_after_tp2" if tp2_hit else "unresolved_after_tp1"
    return {"result": label, "r_multiple": round(r_realized, 3), "bars_held": len(candles) - start_idx}


def run(symbol, days):
    ltf_candles = fetch_history(symbol, "15", days)
    htf_candles = fetch_history(symbol, "240", days)

    if len(ltf_candles) < 100 or len(htf_candles) < 20:
        print(f"Not enough candles (LTF: {len(ltf_candles)}, HTF: {len(htf_candles)})")
        return

    ratio = 240 // 15  # 16 M15 candles per H4 candle
    trades = simulate(ltf_candles, htf_candles, ratio)

    wins = sum(1 for t in trades if t["result"] == "win")
    losses = sum(1 for t in trades if t["result"] == "loss")
    unresolved = sum(1 for t in trades if t["result"] == "unresolved")
    total_r = sum(t["r_multiple"] for t in trades)
    win_rate = wins / (wins + losses) * 100 if (wins + losses) else 0
    breakeven_win_rate = 1 / (1 + liquidity_sweep.TP2_R_MULTIPLE) * 100

    print(f"\nSymbol: {symbol} | Days: {days} | Total signals: {len(trades)}")
    print(f"  Wins: {wins} | Losses: {losses} | Unresolved: {unresolved}")
    print(f"  Win rate: {win_rate:.1f}%  (breakeven at this R:R is {breakeven_win_rate:.1f}%)")
    print(f"  Edge: {win_rate - breakeven_win_rate:+.1f} percentage points")
    print(f"  Total R: {total_r:+.2f}  (sum of all trade R-multiples)")
    print(f"  Avg R per trade: {total_r / len(trades):+.3f}" if trades else "  No trades to average")

    print("\n  --- Multi-TP breakdown (partial exits, 1/3 per level) ---")
    from collections import Counter
    labels = Counter(t["detailed"]["result"] for t in trades)
    for label, count in labels.most_common():
        pct = count / len(trades) * 100 if trades else 0
        print(f"  {label}: {count} ({pct:.1f}%)")
    detailed_total_r = sum(t["detailed"]["r_multiple"] for t in trades)
    print(f"  Detailed total R (with partial exits): {detailed_total_r:+.2f}")

    print("\nRead this as: win_rate needs to beat breakeven_win_rate for this R:R to be "
          "profitable. If edge is negative or total signals is very low (<20-30), this "
          "isn't a usable strategy yet - either the logic needs work or this symbol/period "
          "doesn't suit it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    run(args.symbol, args.days)
