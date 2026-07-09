"""
Backtest for the confluence-filtered OB retest strategy in strategy.py.

Run this on a machine with internet access to Bybit (e.g. your Railway
server, or locally with the same env vars as config.py) - it can't run in
a sandboxed environment with no network access.

Usage:
    python backtest.py ETHUSDT --days 30
    python backtest.py ADAUSDT --days 90 --min-score 3

What it does:
  1. Pulls historical 5-min candles (and matching 4H candles) from Bybit,
     paginating past the 1000-candle-per-call limit.
  2. Slides a rolling window (same size the live bot uses, CANDLE_LIMIT)
     across history, exactly mirroring what check_symbol() sees on a real
     run - so the backtest can't "cheat" by looking at future candles.
  3. On every bar where the base OB-retest trigger fires, scores it with
     the same evaluate_confluence() used live. If it passes, simulates
     the trade forward using config's SL/TP percentages until one is hit.
  4. Prints win rate, total PnL (in %, since real position size isn't
     modeled), and a breakdown of how many signals were rejected and why.

Important caveats before trusting the numbers this prints:
  - MKR trend and FVG confluence are still placeholders (see strategy.py).
    This backtest is testing 5 of the 7 indicators, not all 7.
  - PnL here is in %, applied to a flat notional per trade - it does NOT
    model your actual position sizing, leverage, fees, or slippage. Treat
    the win rate as the primary number, PnL as directional only.
  - Exits are simplified: whichever of SL/TP is touched first within the
    same bar is assumed to be the outcome (checked worst-case: SL wins
    ties). Real fills may differ intra-bar.
"""

import argparse
import time

import config
import bybit_trader
import strategy


def fetch_history(symbol: str, interval: str, days: int):
    """Paginate backwards past Bybit's 1000-candle-per-call limit until we
    have `days` worth of history (or run out of data). Returns oldest-first."""
    interval_minutes = int(interval)
    bars_per_day = (24 * 60) / interval_minutes
    target_bars = int(bars_per_day * days) + 50  # small buffer

    all_rows = []
    end_time = None
    while len(all_rows) < target_bars:
        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["end"] = end_time
        resp = bybit_trader.session.get_kline(**params)
        rows = resp.get("result", {}).get("list", [])
        if not rows:
            break
        # Bybit returns newest-first within a page
        all_rows.extend(rows)
        oldest_ts = int(rows[-1][0])
        end_time = oldest_ts - 1
        if len(rows) < 1000:
            break
        time.sleep(0.15)  # be polite to the rate limit

    # dedupe + sort oldest-first
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


def simulate_trade(candles, entry_idx, direction, sl_pct, tp_pct, max_bars_forward=500):
    """Walks forward from entry_idx (enters at the OPEN of the next bar)
    until SL or TP is hit, or we run out of forward data / max_bars_forward.
    Returns (outcome, pnl_pct) where outcome is 'win', 'loss', or None if
    the trade never resolved within the available data."""
    if entry_idx + 1 >= len(candles):
        return None, 0.0

    entry_price = candles[entry_idx + 1]["open"]
    if direction == "buy":
        sl_price = entry_price * (1 - sl_pct / 100)
        tp_price = entry_price * (1 + tp_pct / 100)
    else:
        sl_price = entry_price * (1 + sl_pct / 100)
        tp_price = entry_price * (1 - tp_pct / 100)

    end = min(len(candles), entry_idx + 2 + max_bars_forward)
    for i in range(entry_idx + 1, end):
        bar = candles[i]
        if direction == "buy":
            hit_sl = bar["low"] <= sl_price
            hit_tp = bar["high"] >= tp_price
        else:
            hit_sl = bar["high"] >= sl_price
            hit_tp = bar["low"] <= tp_price

        if hit_sl and hit_tp:
            return "loss", -sl_pct   # worst-case tie-break: assume SL hit first
        if hit_sl:
            return "loss", -sl_pct
        if hit_tp:
            return "win", tp_pct

    return None, 0.0  # never resolved within window


def run_backtest(tv_symbol: str, days: int, min_score: int, window: int, verbose: bool = True):
    """Runs the backtest and returns a results dict. Prints a report too
    when verbose=True (used by the CLI); server.py's route calls this with
    verbose=False and just returns the dict as JSON."""
    def out(msg):
        if verbose:
            print(msg)

    bybit_symbol = config.SYMBOL_MAP.get(tv_symbol.upper())
    if not bybit_symbol:
        out(f"No symbol mapping for {tv_symbol} in config.SYMBOL_MAP")
        return {"status": "error", "reason": "unknown_symbol", "symbol": tv_symbol}

    out(f"Fetching ~{days}d of {strategy.TIMEFRAME}-min candles for {bybit_symbol}...")
    candles = fetch_history(bybit_symbol, strategy.TIMEFRAME, days)
    out(f"  got {len(candles)} candles")

    out(f"Fetching {strategy.HTF_TIMEFRAME}-min (HTF) candles...")
    htf_candles = fetch_history(bybit_symbol, strategy.HTF_TIMEFRAME, days)
    out(f"  got {len(htf_candles)} HTF candles")

    if len(candles) < window + 10:
        out("Not enough candle history for the configured window size.")
        return {"status": "error", "reason": "not_enough_candles", "symbol": bybit_symbol, "candles_fetched": len(candles)}

    strategy.MIN_CONFLUENCE_SCORE = min_score

    results = []
    rejected_gate = 0
    rejected_score = 0
    in_trade_until = -1  # bar index; block overlapping trades (matches MAX_OPEN_TRADES=1-per-symbol behavior)

    htf_timestamps = [c["timestamp"] for c in htf_candles]

    for end in range(window, len(candles)):
        window_slice = candles[end - window:end]
        signal = strategy.compute_signal(window_slice)
        direction = "buy" if signal.get("buy") else "sell" if signal.get("sell") else None
        if direction is None:
            continue
        if end <= in_trade_until:
            continue  # skip signals while a simulated trade is still open

        # align HTF candles to this point in time (only candles closed before the current bar)
        cur_ts = candles[end - 1]["timestamp"]
        htf_slice = [c for c in htf_candles if c["timestamp"] <= cur_ts]
        htf_trend = strategy.compute_htf_trend(htf_slice) if htf_slice else 0

        gates_ok, score, breakdown = strategy.evaluate_confluence(
            direction, signal, htf_trend, window_slice
        )

        if not gates_ok:
            rejected_gate += 1
            continue
        if score < min_score:
            rejected_score += 1
            continue

        outcome, pnl_pct = simulate_trade(
            candles, end - 1, direction,
            config.DEFAULT_SL_PERCENT, config.DEFAULT_TP_PERCENT
        )
        if outcome is None:
            continue

        results.append({"idx": end, "direction": direction, "score": score, "outcome": outcome, "pnl_pct": pnl_pct})
        # rough overlap guard: block new entries until this one would have resolved
        in_trade_until = end + 1

    wins = sum(1 for r in results if r["outcome"] == "win")
    losses = sum(1 for r in results if r["outcome"] == "loss")
    total_pnl = sum(r["pnl_pct"] for r in results)
    total_trades = wins + losses
    total_signals = total_trades + rejected_gate + rejected_score

    summary = {
        "status": "ok",
        "symbol": bybit_symbol,
        "days": days,
        "window": window,
        "min_score": min_score,
        "signals_fired_total": total_signals,
        "rejected_gate": rejected_gate,
        "rejected_score": rejected_score,
        "trades_simulated": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(100 * wins / total_trades, 1) if total_trades else None,
        "total_pnl_pct": round(total_pnl, 2),
        "avg_pnl_pct": round(total_pnl / total_trades, 3) if total_trades else None,
        "note": "MKR trend and FVG confluence are placeholders in strategy.py - "
                "this reflects 5 of 7 indicators, not the finished strategy.",
    }

    out("\n" + "=" * 50)
    out(f"BACKTEST RESULTS - {bybit_symbol}  ({days}d, window={window}, min_score={min_score})")
    out("=" * 50)
    out(f"Base-trigger signals fired total: {total_signals}")
    out(f"  Rejected (gate failed - ATR/RSI): {rejected_gate}")
    out(f"  Rejected (score < {min_score}):        {rejected_score}")
    out(f"  Passed & simulated:              {total_trades}")
    out("-" * 50)
    if total_trades == 0:
        out("No trades cleared the filter - nothing to report. Try a lower --min-score or more --days.")
        return summary
    out(f"Wins: {wins}   Losses: {losses}   Win rate: {summary['win_rate_pct']}%")
    out(f"Total PnL: {total_pnl:+.2f}%  (flat-size, no fees/slippage/leverage)")
    out(f"Avg PnL per trade: {summary['avg_pnl_pct']:+.3f}%")
    out("=" * 50)
    out("\nReminder: MKR trend and FVG confluence are placeholders in strategy.py -")
    out("this reflects 5 of 7 indicators, not the finished strategy.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the confluence-filtered strategy")
    parser.add_argument("symbol", help="TradingView-style symbol, e.g. ETHUSDT")
    parser.add_argument("--days", type=int, default=30, help="How many days of history to test")
    parser.add_argument("--min-score", type=int, default=strategy.MIN_CONFLUENCE_SCORE,
                         help="Override MIN_CONFLUENCE_SCORE for this run")
    parser.add_argument("--window", type=int, default=strategy.CANDLE_LIMIT,
                         help="Rolling window size, should match live CANDLE_LIMIT")
    args = parser.parse_args()

    run_backtest(args.symbol, args.days, args.min_score, args.window)
