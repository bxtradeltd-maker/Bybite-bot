"""
Defender Bot (Python port) — Deriv Options Trading Bot
--------------------------------------------------------
Ported from the original Deriv Bot Builder (DBot) XML strategy, with a
compounding daily money management plan layered on top (per the
"Money Management Plan Progress Report" — BinaryMonarch style: start
capital, 10%/day compounding target, 30 days).

STRATEGY (signal logic unchanged from the original XML):
  Market: Volatility 10 Index (R_10)
  Indicators: RSI(period=2) and SMA(period=3), both on 1-minute candle closes
  Entry rules (evaluated every tick):
    - If price > SMA and 60 < RSI < 75         -> BUY CALL ("Higher")
    - Elif RSI > 97                            -> BUY PUT  ("Lower")  [extreme reversal]
    - Elif price < SMA and 25 < RSI < 40        -> BUY PUT  ("Lower")
    - Elif RSI < 3                             -> BUY CALL ("Higher") [extreme reversal]
  Contract: 1 tick duration, CALL/PUT (Rise/Fall)

MONEY MANAGEMENT (new — replaces the flat stake/profit-target/max-loss
from the original XML):
  - Each "day" has a starting capital and a daily profit target
    (default 10% of that day's starting capital).
  - Stake per trade = a % of the day's starting capital (default 3.5%,
    roughly matching the original bot's $0.35 stake on a $10 account).
  - On a LOSS, stake still increases via the martingale multiplier,
    same as the original bot — but the day stops trading once losses
    reach the daily max-loss cap (default 20% of day's starting capital),
    so a losing streak can't wipe far past the day's risk budget.
  - Once the day's profit target is hit (or the daily loss cap is hit),
    the bot pauses new trades and waits for the next calendar day (UTC)
    to begin, then rolls the actual achieved capital into the new day's
    starting capital and continues.
  - Progress persists to a local JSON file so the day count and capital
    survive a script *restart*. NOTE: if Railway does a full *redeploy*
    (new build), this file resets since Railway's filesystem is rebuilt.
    For guaranteed persistence across redeploys, this state should later
    move to a small database — flagging this now so it isn't a surprise.

⚠️ WARNING: This strategy still uses martingale-style staking after a
loss. The daily loss cap limits the damage per day, but a string of
losing days can still erode capital over time. Test on DEMO first.

CONFIG (all overridable via Railway environment variables):
  DERIV_APP_ID              (required)
  DERIV_API_TOKEN           (required)
  DEFENDER_SYMBOL            default: R_10
  DEFENDER_USE_DEMO          default: true    (set "false" to trade the real account)
  DEFENDER_MARTINGALE        default: 1.2

  MM_STARTING_CAPITAL        default: 10      (only used the very first run)
  MM_DAILY_TARGET_PCT        default: 0.10    (10% daily profit target)
  MM_DAILY_MAX_LOSS_PCT      default: 0.20    (stop the day at -20% of day's start capital)
  MM_STAKE_PERCENT           default: 0.035   (stake = 3.5% of day's start capital)
  MM_STATE_FILE              default: money_management_state.json
"""

import asyncio
import json
import os
import sys
from collections import deque
from datetime import datetime, timezone

import requests
import websockets

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_ID = os.environ.get("DERIV_APP_ID")
API_TOKEN = os.environ.get("DERIV_API_TOKEN")

SYMBOL = os.environ.get("DEFENDER_SYMBOL", "R_10")
USE_DEMO = os.environ.get("DEFENDER_USE_DEMO", "true").lower() != "false"
MARTINGALE_LEVEL = float(os.environ.get("DEFENDER_MARTINGALE", "1.2"))

MM_STARTING_CAPITAL = float(os.environ.get("MM_STARTING_CAPITAL", "10"))
MM_DAILY_TARGET_PCT = float(os.environ.get("MM_DAILY_TARGET_PCT", "0.10"))
MM_DAILY_MAX_LOSS_PCT = float(os.environ.get("MM_DAILY_MAX_LOSS_PCT", "0.20"))
MM_STAKE_PERCENT = float(os.environ.get("MM_STAKE_PERCENT", "0.035"))
MM_STATE_FILE = os.environ.get("MM_STATE_FILE", "money_management_state.json")

RSI_PERIOD = 2
SMA_PERIOD = 3
CANDLE_GRANULARITY = 60  # seconds (1-minute candles, matches original bot)

REST_BASE = "https://api.derivws.com/trading/v1/options"


# ---------------------------------------------------------------------------
# REST: accounts + OTP
# ---------------------------------------------------------------------------
def get_accounts():
    url = f"{REST_BASE}/accounts"
    headers = {"Deriv-App-ID": APP_ID, "Authorization": f"Bearer {API_TOKEN}"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json().get("data", [])


def get_otp_url(account_id):
    url = f"{REST_BASE}/accounts/{account_id}/otp"
    headers = {"Deriv-App-ID": APP_ID, "Authorization": f"Bearer {API_TOKEN}"}
    resp = requests.post(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json().get("data", {}).get("url")


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def calculate_sma(closes, period):
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def calculate_rsi(closes, period):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(-period, 0):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def decide_signal(current_price, closes):
    sma = calculate_sma(list(closes), SMA_PERIOD)
    rsi = calculate_rsi(list(closes), RSI_PERIOD)
    if sma is None or rsi is None:
        return None, rsi, sma

    if current_price > sma:
        if 60 < rsi < 75:
            return "CALL", rsi, sma
    elif rsi > 97:
        return "PUT", rsi, sma

    if current_price < sma:
        if 25 < rsi < 40:
            return "PUT", rsi, sma
    elif rsi < 3:
        return "CALL", rsi, sma

    return None, rsi, sma


# ---------------------------------------------------------------------------
# Money management plan (compounding daily target)
# ---------------------------------------------------------------------------
class MoneyManagementPlan:
    def __init__(self, state_file):
        self.state_file = state_file
        self.state = self._load_or_init()

    def _today_str(self):
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load_or_init(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                state = json.load(f)
            print(f"Loaded money management state: day {state['day']}, "
                  f"day_start_capital={state['day_start_capital']:.2f}, "
                  f"total_capital={state['total_capital']:.2f}")
            return state

        state = {
            "day": 1,
            "day_start_capital": MM_STARTING_CAPITAL,
            "total_capital": MM_STARTING_CAPITAL,
            "day_profit": 0.0,
            "day_date": self._today_str(),
            "day_complete": False,
        }
        self._save(state)
        print(f"Initialized new money management plan: starting capital "
              f"${MM_STARTING_CAPITAL:.2f}, daily target {MM_DAILY_TARGET_PCT*100:.1f}%")
        return state

    def _save(self, state=None):
        with open(self.state_file, "w") as f:
            json.dump(state or self.state, f, indent=2)

    def daily_target(self):
        return self.state["day_start_capital"] * MM_DAILY_TARGET_PCT

    def daily_max_loss(self):
        return self.state["day_start_capital"] * MM_DAILY_MAX_LOSS_PCT

    def stake_size(self):
        return round(self.state["day_start_capital"] * MM_STAKE_PERCENT, 2)

    def record_trade(self, profit):
        self.state["day_profit"] += profit
        self.state["total_capital"] += profit
        self._save()

    def day_status(self):
        """Returns 'target_hit', 'loss_limit_hit', or 'in_progress'."""
        if self.state["day_profit"] >= self.daily_target():
            return "target_hit"
        if self.state["day_profit"] <= -self.daily_max_loss():
            return "loss_limit_hit"
        return "in_progress"

    def mark_day_complete(self, reason):
        self.state["day_complete"] = True
        self._save()
        print(f"\n=== Day {self.state['day']} complete ({reason}) ===")
        print(f"Start capital: ${self.state['day_start_capital']:.2f} | "
              f"Target: ${self.daily_target():.2f} | "
              f"Actual profit: ${self.state['day_profit']:.2f} | "
              f"Achieved: {'YES' if reason == 'target_hit' else 'NO'}")
        print(f"Total capital now: ${self.state['total_capital']:.2f}\n")

    def try_advance_day(self):
        """If a new UTC calendar day has started, roll over to the next day."""
        today = self._today_str()
        if today == self.state["day_date"]:
            return False

        self.state["day"] += 1
        self.state["day_start_capital"] = self.state["total_capital"]
        self.state["day_profit"] = 0.0
        self.state["day_date"] = today
        self.state["day_complete"] = False
        self._save()
        print(f"=== Starting Day {self.state['day']} | "
              f"start capital ${self.state['day_start_capital']:.2f} | "
              f"target ${self.daily_target():.2f} ===")
        return True

    def is_day_complete(self):
        return self.state["day_complete"]


# ---------------------------------------------------------------------------
# WebSocket helpers
# ---------------------------------------------------------------------------
async def send_and_wait(ws, request, expect_msg_type):
    await ws.send(json.dumps(request))
    while True:
        raw = await ws.recv()
        data = json.loads(raw)
        if data.get("msg_type") == expect_msg_type:
            return data
        if "error" in data:
            raise RuntimeError(f"Deriv API error: {data['error']}")


async def refresh_candles(ws, closes_holder):
    request = {
        "ticks_history": SYMBOL,
        "count": 30,
        "end": "latest",
        "style": "candles",
        "granularity": CANDLE_GRANULARITY,
    }
    data = await send_and_wait(ws, request, "candles")
    candles = data.get("candles", [])
    closes_holder["closes"] = deque([c["close"] for c in candles], maxlen=50)
    if candles:
        closes_holder["last_epoch"] = candles[-1]["epoch"]


async def place_contract(ws, contract_type, stake):
    proposal_req = {
        "proposal": 1,
        "amount": stake,
        "basis": "stake",
        "contract_type": contract_type,
        "currency": "USD",
        "duration": 1,
        "duration_unit": "t",
        "symbol": SYMBOL,
    }
    proposal_resp = await send_and_wait(ws, proposal_req, "proposal")
    proposal_id = proposal_resp["proposal"]["id"]
    ask_price = proposal_resp["proposal"]["ask_price"]

    buy_req = {"buy": proposal_id, "price": ask_price}
    buy_resp = await send_and_wait(ws, buy_req, "buy")
    return buy_resp["buy"]["contract_id"]


async def wait_for_contract_result(ws, contract_id):
    request = {"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1}
    await ws.send(json.dumps(request))
    while True:
        raw = await ws.recv()
        data = json.loads(raw)
        if data.get("msg_type") != "proposal_open_contract":
            continue
        poc = data["proposal_open_contract"]
        if poc.get("is_sold"):
            return float(poc.get("profit", 0))


# ---------------------------------------------------------------------------
# Main trading loop
# ---------------------------------------------------------------------------
async def run_bot():
    if not APP_ID or not API_TOKEN:
        print("ERROR: DERIV_APP_ID and/or DERIV_API_TOKEN environment variables are missing.")
        sys.exit(1)

    print("Fetching accounts...")
    accounts = get_accounts()
    if not accounts:
        print("No accounts found.")
        sys.exit(1)

    target_type = "demo" if USE_DEMO else "real"
    account = next((a for a in accounts if a.get("account_type") == target_type), accounts[0])
    account_id = account["account_id"]
    print(f"Using account {account_id} ({account.get('account_type')}) "
          f"balance: {account.get('balance')} {account.get('currency')}")

    mm = MoneyManagementPlan(MM_STATE_FILE)
    current_stake = mm.stake_size()
    in_trade = False
    closes_holder = {"closes": deque(maxlen=50), "last_epoch": None}

    ws_url = get_otp_url(account_id)
    print("Connecting WebSocket...")
    print(f"Day {mm.state['day']} | start capital ${mm.state['day_start_capital']:.2f} | "
          f"target ${mm.daily_target():.2f} | daily loss cap ${mm.daily_max_loss():.2f} | "
          f"stake ${current_stake:.2f}")

    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
        await refresh_candles(ws, closes_holder)
        await ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))

        while True:
            raw = await ws.recv()
            data = json.loads(raw)

            if data.get("msg_type") != "tick":
                continue

            # If the day is complete, just wait for the next UTC day to roll over
            if mm.is_day_complete():
                if mm.try_advance_day():
                    current_stake = mm.stake_size()
                continue

            current_price = float(data["tick"]["quote"])
            tick_epoch = data["tick"]["epoch"]

            if closes_holder["last_epoch"] is None or \
                    tick_epoch - closes_holder["last_epoch"] >= CANDLE_GRANULARITY:
                await refresh_candles(ws, closes_holder)

            if in_trade:
                continue

            signal, rsi, sma = decide_signal(current_price, closes_holder["closes"])
            if signal is None:
                continue

            print(f"Signal: {signal} | price={current_price} rsi={rsi:.2f} sma={sma:.5f} "
                  f"stake=${current_stake:.2f} day_profit=${mm.state['day_profit']:.2f} "
                  f"/ target ${mm.daily_target():.2f}")

            in_trade = True
            try:
                contract_id = await place_contract(ws, signal, current_stake)
                print(f"Bought contract {contract_id} ({signal}) at stake ${current_stake:.2f}")
                profit = await wait_for_contract_result(ws, contract_id)
            except Exception as e:
                print(f"Trade error: {e}")
                in_trade = False
                continue

            mm.record_trade(profit)

            if profit > 0:
                print(f"WIN  profit=${profit:.2f} | day_profit=${mm.state['day_profit']:.2f}")
                current_stake = mm.stake_size()  # reset to base stake for the day
            else:
                print(f"LOSS profit=${profit:.2f} | day_profit=${mm.state['day_profit']:.2f}")
                current_stake = round(current_stake + abs(profit) * MARTINGALE_LEVEL, 2)

            status = mm.day_status()
            if status == "target_hit":
                mm.mark_day_complete("target_hit")
            elif status == "loss_limit_hit":
                mm.mark_day_complete("loss_limit_hit")

            in_trade = False


if __name__ == "__main__":
    asyncio.run(run_bot())
