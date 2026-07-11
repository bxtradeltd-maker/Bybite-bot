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


def simulate(ltf_candles, htf_candles, htf_interval_ratio):
    """
    htf_interval_ratio: how many LTF candles = 1 HTF candle (e.g. H4/M15 = 16)
    Re-derives swing points using only HTF candles that would have closed
    by the current LTF bar, so this doesn't cheat with future H4 data.
    """
    trades = []
    swept_high_prices = set()
    swept_low_prices = set()

    for i in range(50, len(ltf_candles)):
        htf_visible_count = min(len(htf_candles), (i // htf_interval_ratio) + 1)
        if htf_visible_count < liquidity_sweep.SWING_LEFT_RIGHT * 2 + 1:
            continue

        htf_slice = htf_candles[:htf_visible_count]
        swing_highs, swing_lows = liquidity_sweep.find_swing_points(htf_slice)

        for s in swing_highs:
            if s["price"] in swept_high_prices:
                s["swept"] = True
        for s in swing_lows:
            if s["price"] in swept_low_prices:
                s["swept"] = True

        ltf_slice = ltf_candles[:i + 1]
        signal = liquidity_sweep.check_sweep_signal(ltf_slice, swing_highs, swing_lows)
        if not signal:
            continue

        if signal["direction"] == "sell":
            swept_high_prices.add(signal["swept_level"])
        else:
            swept_low_prices.add(signal["swept_level"])

        outcome = _walk_to_outcome(ltf_candles, i + 1, signal)
        trades.append(outcome)

    return trades


def _walk_to_outcome(candles, start_idx, signal):
    direction = signal["direction"]
    for j in range(start_idx, len(candles)):
        bar = candles[j]
        check_order = (bar["low"], bar["high"]) if direction == "buy" else (bar["high"], bar["low"])
        for price in check_order:
            if direction == "buy":
                if price <= signal["sl"]:
                    return {"result": "loss", "r_multiple": -1.0, "bars_held": j - start_idx}
                if price >= signal["tp"]:
                    return {"result": "win", "r_multiple": liquidity_sweep.TP_R_MULTIPLE, "bars_held": j - start_idx}
            else:
                if price >= signal["sl"]:
                    return {"result": "loss", "r_multiple": -1.0, "bars_held": j - start_idx}
                if price <= signal["tp"]:
                    return {"result": "win", "r_multiple": liquidity_sweep.TP_R_MULTIPLE, "bars_held": j - start_idx}
    return {"result": "unresolved", "r_multiple": 0.0, "bars_held": len(candles) - start_idx}


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
    breakeven_win_rate = 1 / (1 + liquidity_sweep.TP_R_MULTIPLE) * 100

    print(f"\nSymbol: {symbol} | Days: {days} | Total signals: {len(trades)}")
    print(f"  Wins: {wins} | Losses: {losses} | Unresolved: {unresolved}")
    print(f"  Win rate: {win_rate:.1f}%  (breakeven at this R:R is {breakeven_win_rate:.1f}%)")
    print(f"  Edge: {win_rate - breakeven_win_rate:+.1f} percentage points")
    print(f"  Total R: {total_r:+.2f}  (sum of all trade R-multiples)")
    print(f"  Avg R per trade: {total_r / len(trades):+.3f}" if trades else "  No trades to average")
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
