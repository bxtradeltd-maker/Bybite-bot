"""
Money Management Plan tracker.

Mirrors the "Day / Start Capital / Expected Profit / Total P&L" table style
(starting capital compounding at a fixed daily % target), and compares it
against your ACTUAL account equity.

Important: this module only tracks and reports progress. It never changes
position sizing or forces extra trades to "catch up" to a missed daily
target - that pattern (increasing risk to chase a schedule) is how accounts
blow up. Position sizing stays governed by RISK_PER_TRADE_PERCENT in
config.py regardless of how this plan is doing.

The plan's start date is read from config.MONEY_PLAN_START_DATE (a Railway
env var you set once, format YYYY-MM-DD). This makes "Day X" survive server
restarts. If that variable is left blank, it falls back to today's date in
memory - which WILL reset to Day 1 on the next restart.
"""

from datetime import date, datetime
import logging
import config

log = logging.getLogger("money_plan")

if config.MONEY_PLAN_START_DATE:
    try:
        _START_DATE = datetime.strptime(config.MONEY_PLAN_START_DATE, "%Y-%m-%d").date()
    except ValueError:
        log.error(f"Invalid MONEY_PLAN_START_DATE '{config.MONEY_PLAN_START_DATE}', expected YYYY-MM-DD. Falling back to today.")
        _START_DATE = date.today()
else:
    log.warning("MONEY_PLAN_START_DATE not set - Day count will reset to Day 1 on every restart. "
                "Set it in Railway's Variables tab to make it permanent.")
    _START_DATE = date.today()


def _expected_capital(day: int) -> float:
    """Compounded expected capital at the END of the given day (1-indexed)."""
    rate = config.DAILY_TARGET_PERCENT / 100
    return config.STARTING_CAPITAL * ((1 + rate) ** day)


def current_day() -> int:
    elapsed = (date.today() - _START_DATE).days + 1
    return min(elapsed, config.PLAN_TOTAL_DAYS)


def build_table():
    """Returns the full expected-progress table, Day 1..PLAN_TOTAL_DAYS."""
    table = []
    start = config.STARTING_CAPITAL
    for day in range(1, config.PLAN_TOTAL_DAYS + 1):
        expected_profit = start * (config.DAILY_TARGET_PERCENT / 100)
        end_capital = start + expected_profit
        table.append({
            "day": day,
            "start_capital": round(start, 2),
            "expected_profit": round(expected_profit, 2),
            "expected_end_capital": round(end_capital, 2),
        })
        start = end_capital
    return table


def get_status(current_equity: float):
    day = current_day()
    expected_today = _expected_capital(day)
    expected_yesterday = _expected_capital(day - 1) if day > 1 else config.STARTING_CAPITAL

    achieved = current_equity is not None and current_equity >= expected_today

    return {
        "day": day,
        "total_days": config.PLAN_TOTAL_DAYS,
        "starting_capital": config.STARTING_CAPITAL,
        "daily_target_percent": config.DAILY_TARGET_PERCENT,
        "expected_capital_today": round(expected_today, 2),
        "expected_capital_start_of_day": round(expected_yesterday, 2),
        "actual_equity": current_equity,
        "target_achieved": achieved,
        "plan_complete": day >= config.PLAN_TOTAL_DAYS,
    }
