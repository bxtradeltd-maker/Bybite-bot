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


def _evaluate_all_signals(candles, htf_candles):
    """
    The core speed fix: walks the FULL candle history exactly once to find
    every fired base-trigger signal, scores each one exactly once, and
    simulates its outcome exactly once. A day-window or min_score change
    afterwards is just filtering this list - no recomputation needed.

    Returns a list of dicts, one per fired signal:
      {idx, timestamp, direction, gates_ok, score, outcome, pnl_pct}
    """
    series = strategy.compute_signal_series(candles)
    if not series["signals"]:
        return []

    htf_trends = strategy.htf_trend_series(candles, htf_candles) if htf_candles else [0] * len(candles)

    evaluated = []
    for sig in series["signals"]:
        idx = sig["idx"]
        direction = sig["direction"]
        htf_trend = htf_trends[idx] if idx < len(htf_trends) else 0

        gates_ok, score, _breakdown = strategy.evaluate_confluence_at(idx, direction, series, htf_trend, candles)

        outcome, pnl_pct = simulate_trade(
            candles, idx, direction,
            config.DEFAULT_SL_PERCENT, config.DEFAULT_TP_PERCENT
        )
        if outcome is None:
            continue  # trade never resolved within available forward data

        evaluated.append({
            "idx": idx,
            "timestamp": candles[idx]["timestamp"],
            "direction": direction,
            "gates_ok": gates_ok,
            "score": score,
            "outcome": outcome,
            "pnl_pct": pnl_pct,
        })

    return evaluated


def _aggregate(evaluated_signals, min_score, days, window):
    """Filters a precomputed signal list down to what would have actually
    traded at a given min_score (respecting the same light overlap-guard
    as before: skip a signal if it fires before the prior counted trade's
    entry bar has passed), and totals up the results."""
    rejected_gate = 0
    rejected_score = 0
    trades = []
    last_entry_idx = -1

    for sig in evaluated_signals:
        if not sig["gates_ok"]:
            rejected_gate += 1
            continue
        if sig["score"] < min_score:
            rejected_score += 1
            continue
        if sig["idx"] <= last_entry_idx:
            continue  # overlap guard
        trades.append(sig)
        last_entry_idx = sig["idx"] + 1

    wins = sum(1 for t in trades if t["outcome"] == "win")
    losses = sum(1 for t in trades if t["outcome"] == "loss")
    total_pnl = sum(t["pnl_pct"] for t in trades)
    total_trades = wins + losses

    return {
        "days": days,
        "window": window,
        "min_score": min_score,
        "signals_fired_total": total_trades + rejected_gate + rejected_score,
        "rejected_gate": rejected_gate,
        "rejected_score": rejected_score,
        "trades_simulated": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(100 * wins / total_trades, 1) if total_trades else None,
        "total_pnl_pct": round(total_pnl, 2),
        "avg_pnl_pct": round(total_pnl / total_trades, 3) if total_trades else None,
    }


def run_backtest(tv_symbol: str, days: int, min_score: int, window: int, verbose: bool = True):
    """Runs a single backtest and returns a results dict. Prints a report
    too when verbose=True (CLI use); server.py's route calls this with
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

    min_bars = 2 * strategy.PIVOT_LEN + strategy.ST_LEN + 5
    if len(candles) < min_bars:
        out("Not enough candle history to run the strategy at all.")
        return {"status": "error", "reason": "not_enough_candles", "symbol": bybit_symbol, "candles_fetched": len(candles)}

    evaluated = _evaluate_all_signals(candles, htf_candles)
    summary = _aggregate(evaluated, min_score, days, window)
    summary["status"] = "ok"
    summary["symbol"] = bybit_symbol
    summary["note"] = ("MKR trend and FVG confluence are placeholders in strategy.py - "
                        "this reflects 5 of 7 indicators, not the finished strategy.")

    out("\n" + "=" * 50)
    out(f"BACKTEST RESULTS - {bybit_symbol}  ({days}d, min_score={min_score})")
    out("=" * 50)
    out(f"Base-trigger signals fired total: {summary['signals_fired_total']}")
    out(f"  Rejected (gate failed - ATR/RSI): {summary['rejected_gate']}")
    out(f"  Rejected (score < {min_score}):        {summary['rejected_score']}")
    out(f"  Passed & simulated:              {summary['trades_simulated']}")
    out("-" * 50)
    if summary["trades_simulated"] == 0:
        out("No trades cleared the filter - nothing to report. Try a lower --min-score or more --days.")
        return summary
    out(f"Wins: {summary['wins']}   Losses: {summary['losses']}   Win rate: {summary['win_rate_pct']}%")
    out(f"Total PnL: {summary['total_pnl_pct']:+.2f}%  (flat-size, no fees/slippage/leverage)")
    out(f"Avg PnL per trade: {summary['avg_pnl_pct']:+.3f}%")
    out("=" * 50)
    return summary


def sweep_scores(tv_symbol: str, day_windows, score_values, window: int = None):
    """
    Tests every (days, min_score) combination and returns a comparison
    table plus a stability summary - the tool for picking
    MIN_CONFLUENCE_SCORE off more than one lucky sample.

    Fetches candle history ONCE (for the longest day window - shorter
    windows are just a slice of it), scores every fired signal ONCE, then
    each (days, min_score) combination is just filtering that
    already-computed list. No repeated simulation, so this stays fast
    even with many combinations.
    """
    if window is None:
        window = strategy.CANDLE_LIMIT

    bybit_symbol = config.SYMBOL_MAP.get(tv_symbol.upper())
    if not bybit_symbol:
        return {"status": "error", "reason": "unknown_symbol", "symbol": tv_symbol}

    max_days = max(day_windows)
    full_candles = fetch_history(bybit_symbol, strategy.TIMEFRAME, max_days)
    full_htf = fetch_history(bybit_symbol, strategy.HTF_TIMEFRAME, max_days)

    min_bars = 2 * strategy.PIVOT_LEN + strategy.ST_LEN + 5
    if len(full_candles) < min_bars:
        return {"status": "error", "reason": "not_enough_candles", "symbol": bybit_symbol,
                "candles_fetched": len(full_candles)}

    interval_minutes = int(strategy.TIMEFRAME)
    htf_interval_minutes = int(strategy.HTF_TIMEFRAME)

    rows = []
    for days in sorted(day_windows):
        bars_needed = int((24 * 60 / interval_minutes) * days) + 50
        candles = full_candles[-bars_needed:] if len(full_candles) > bars_needed else full_candles
        htf_bars_needed = int((24 * 60 / htf_interval_minutes) * days) + 50
        htf_candles = full_htf[-htf_bars_needed:] if len(full_htf) > htf_bars_needed else full_htf

        if len(candles) < min_bars:
            for min_score in sorted(score_values):
                rows.append({"days": days, "window": window, "min_score": min_score,
                              "trades_simulated": 0, "win_rate_pct": None,
                              "status": "not_enough_candles"})
            continue

        evaluated = _evaluate_all_signals(candles, htf_candles)  # computed ONCE per day window
        for min_score in sorted(score_values):
            rows.append(_aggregate(evaluated, min_score, days, window))

    by_score = {}
    for r in rows:
        by_score.setdefault(r["min_score"], []).append(r)

    stability = []
    for score, group in sorted(by_score.items()):
        win_rates = [g["win_rate_pct"] for g in group if g.get("win_rate_pct") is not None]
        trade_counts = [g.get("trades_simulated", 0) for g in group]
        if len(win_rates) >= 2:
            mean_wr = sum(win_rates) / len(win_rates)
            variance = sum((w - mean_wr) ** 2 for w in win_rates) / len(win_rates)
            std_dev = variance ** 0.5
        else:
            mean_wr = win_rates[0] if win_rates else None
            std_dev = None
        stability.append({
            "min_score": score,
            "avg_win_rate_pct": round(mean_wr, 1) if mean_wr is not None else None,
            "win_rate_std_dev": round(std_dev, 1) if std_dev is not None else None,
            "total_trades_across_windows": sum(trade_counts),
            "day_windows_with_zero_trades": sum(1 for g in group if g.get("trades_simulated", 0) == 0),
        })

    return {
        "status": "ok",
        "symbol": bybit_symbol,
        "day_windows_tested": sorted(day_windows),
        "min_scores_tested": sorted(score_values),
        "results": rows,
        "stability_summary": stability,
        "how_to_read": (
            "Look at stability_summary: a good min_score has a low win_rate_std_dev "
            "(consistent across time windows) AND enough total_trades_across_windows "
            "to trust the number (more than ~20-30). Avoid a score with a high avg "
            "win rate built on very few trades - that's noise, not edge."
        ),
        "note": "MKR trend and FVG confluence are still placeholders in strategy.py.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the confluence-filtered strategy")
    parser.add_argument("symbol", help="TradingView-style symbol, e.g. ETHUSDT")
    parser.add_argument("--days", type=int, default=30, help="How many days of history to test")
    parser.add_argument("--min-score", type=int, default=strategy.MIN_CONFLUENCE_SCORE,
                         help="Override MIN_CONFLUENCE_SCORE for this run")
    parser.add_argument("--window", type=int, default=strategy.CANDLE_LIMIT,
                         help="Rolling window size, kept for reporting only in the fast path")
    args = parser.parse_args()

    run_backtest(args.symbol, args.days, args.min_score, args.window)

