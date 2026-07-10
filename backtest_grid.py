"""
Backtest for grid_ea.py's basket mechanic.

Run on a machine with Bybit access:
    python backtest_grid.py BTCUSDT --days 90
    python backtest_grid.py BTCUSDT --days 90 --direction sell

What it does:
  Walks 5-min candles bar-by-bar (no lookahead). Opens a basket at the
  start of the window, applies the exact same grid/TP/equity-stop logic
  as grid_ea.py, and reports how the basket would have played out -
  including the WORST case: does EQUITY_STOP_PCT actually get hit before
  the basket blows past what you can afford to lose?

This only tests ONE basket per run, opened at the start of the window and
run to conclusion (TP, equity stop, or max_levels+ride-out to end of data).
Use --repeat to chain N consecutive baskets back-to-back across the window
to see how the strategy behaves over many cycles, not just one.
"""

import argparse

import grid_ea
from backtest import fetch_history


def simulate_one_basket(candles, start_idx, direction,
                          initial_qty_usd, lot_multiplier, grid_step_pct,
                          max_levels, tp_pct, equity_stop_pct):
    """Pure simulation - no bybit_trader calls. Mirrors grid_ea's math exactly."""
    levels = []  # list of (qty_usd, entry_price)

    def avg_entry():
        total = sum(q for q, _ in levels)
        return sum(q * p for q, p in levels) / total if total else 0.0

    def floating_pnl_pct(price):
        if not levels:
            return 0.0
        avg = avg_entry()
        if direction == "buy":
            return (price - avg) / avg * 100
        return (avg - price) / avg * 100

    # open level 1 at the open of start_idx
    entry_price = candles[start_idx]["open"]
    levels.append((initial_qty_usd, entry_price))

    for i in range(start_idx, len(candles)):
        bar = candles[i]
        # check both high and low within the bar (worst-case ordering: stop before TP)
        for price in (bar["low"], bar["high"]) if direction == "buy" else (bar["high"], bar["low"]):
            pnl_pct = floating_pnl_pct(price)

            if pnl_pct <= equity_stop_pct:
                return {
                    "outcome": "equity_stop", "exit_idx": i, "levels_used": len(levels),
                    "final_pnl_pct": pnl_pct, "notional": sum(q for q, _ in levels),
                    "bars_held": i - start_idx,
                }
            if pnl_pct >= tp_pct:
                return {
                    "outcome": "take_profit", "exit_idx": i, "levels_used": len(levels),
                    "final_pnl_pct": pnl_pct, "notional": sum(q for q, _ in levels),
                    "bars_held": i - start_idx,
                }

            avg = avg_entry()
            moved_against = ((avg - price) / avg * 100 if direction == "buy"
                              else (price - avg) / avg * 100)
            if moved_against >= grid_step_pct and len(levels) < max_levels:
                next_qty = levels[-1][0] * lot_multiplier
                levels.append((next_qty, price))

    # ran out of data still holding
    final_price = candles[-1]["close"]
    return {
        "outcome": "unresolved_end_of_data", "exit_idx": len(candles) - 1,
        "levels_used": len(levels), "final_pnl_pct": floating_pnl_pct(final_price),
        "notional": sum(q for q, _ in levels), "bars_held": len(candles) - 1 - start_idx,
    }


def run(symbol, days, direction, repeat):
    candles = fetch_history(symbol, "5", days)
    if len(candles) < 50:
        print(f"Not enough candles fetched ({len(candles)}).")
        return

    results = []
    idx = 0
    for _ in range(repeat):
        if idx >= len(candles) - 1:
            break
        r = simulate_one_basket(
            candles, idx, direction,
            grid_ea.INITIAL_QTY_USD, grid_ea.LOT_MULTIPLIER, grid_ea.GRID_STEP_PCT,
            grid_ea.MAX_GRID_LEVELS, grid_ea.BASKET_TP_PCT, grid_ea.EQUITY_STOP_PCT,
        )
        results.append(r)
        idx = r["exit_idx"] + 1  # next basket starts right after this one resolves

    tp_count = sum(1 for r in results if r["outcome"] == "take_profit")
    stop_count = sum(1 for r in results if r["outcome"] == "equity_stop")
    unresolved = sum(1 for r in results if r["outcome"] == "unresolved_end_of_data")
    total_pnl_pct_of_notional = sum(r["final_pnl_pct"] * r["notional"] for r in results)
    total_notional = sum(r["notional"] for r in results) or 1
    max_levels_seen = max((r["levels_used"] for r in results), default=0)
    worst_drawdown = min((r["final_pnl_pct"] for r in results), default=0)

    print(f"\nSymbol: {symbol} | Direction: {direction} | Baskets run: {len(results)}")
    print(f"  Take-profit closes:  {tp_count}")
    print(f"  Equity-stop closes:  {stop_count}  <-- these are the ones that would've hurt")
    print(f"  Unresolved (still open at end of data): {unresolved}")
    print(f"  Max grid levels hit in any single basket: {max_levels_seen} (cap is {grid_ea.MAX_GRID_LEVELS})")
    print(f"  Worst single-basket floating PnL%: {worst_drawdown:.2f}%")
    print(f"  Notional-weighted avg PnL% per basket: {total_pnl_pct_of_notional / total_notional:.3f}%")
    print("\nRead this as: if equity_stop closes are common or the worst drawdown is "
          "deep, the grid step/multiplier/level-cap combo is too aggressive for this "
          "pair's volatility. If unresolved count is high, baskets are getting stuck "
          "open past your test window - check what happens if you extend --days.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--direction", choices=["buy", "sell"], default="buy")
    parser.add_argument("--repeat", type=int, default=20,
                         help="how many consecutive baskets to chain through the window")
    args = parser.parse_args()
    run(args.symbol, args.days, args.direction, args.repeat)
