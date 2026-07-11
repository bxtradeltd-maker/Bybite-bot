"""
Deriv API connection test script (updated for Deriv's current API, 2026).

Deriv retired the old "connect directly with app_id" method. The new flow is:
  1. List your accounts (REST) to find your account_id
  2. Request an OTP (one-time password) for that account (REST)
  3. Connect to WebSocket using the URL returned, which has the OTP baked in

Run this as a ONE-OFF script on Railway (not your main web process).
Requires DERIV_APP_ID and DERIV_API_TOKEN as environment variables.

Expected output:
1. A list of your account(s) with balance
2. A generated WebSocket URL with OTP
3. Confirmation the WebSocket connection opened successfully
"""

import asyncio
import json
import os
import sys

import requests
import websockets

APP_ID = os.environ.get("DERIV_APP_ID")
API_TOKEN = os.environ.get("DERIV_API_TOKEN")

REST_BASE = "https://api.derivws.com/trading/v1/options"


def get_accounts():
    """Step 1: list accounts tied to this token."""
    url = f"{REST_BASE}/accounts"
    headers = {
        "Deriv-App-ID": APP_ID,
        "Authorization": f"Bearer {API_TOKEN}",
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_otp_url(account_id):
    """Step 2: request a one-time-password WebSocket URL for this account."""
    url = f"{REST_BASE}/accounts/{account_id}/otp"
    headers = {
        "Deriv-App-ID": APP_ID,
        "Authorization": f"Bearer {API_TOKEN}",
    }
    resp = requests.post(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


async def test_websocket(ws_url):
    """Step 3: connect to the WebSocket using the OTP-embedded URL."""
    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
        print("\n--- WEBSOCKET CONNECTION OPENED SUCCESSFULLY ---")

        # Simple ping to confirm the session is live
        await ws.send(json.dumps({"ping": 1}))
        response = json.loads(await ws.recv())
        print("Ping response:", response)


def main():
    if not APP_ID or not API_TOKEN:
        print("ERROR: DERIV_APP_ID and/or DERIV_API_TOKEN environment variables are missing.")
        print("Set them in Railway's Variables tab before running this script.")
        sys.exit(1)

    print("Step 1: Fetching accounts...")
    try:
        accounts_data = get_accounts()
    except requests.HTTPError as e:
        print(f"ACCOUNTS REQUEST FAILED: {e}")
        print(f"Response body: {e.response.text}")
        sys.exit(1)

    accounts = accounts_data.get("data", [])
    if not accounts:
        print("No accounts found for this token.")
        sys.exit(1)

    print(f"Found {len(accounts)} account(s):")
    for acc in accounts:
        print(f"  - {acc.get('account_id')} | {acc.get('account_type')} | "
              f"balance: {acc.get('balance')} {acc.get('currency')}")

    account_id = accounts[0]["account_id"]
    print(f"\nStep 2: Requesting OTP for account {account_id}...")

    try:
        otp_data = get_otp_url(account_id)
    except requests.HTTPError as e:
        print(f"OTP REQUEST FAILED: {e}")
        print(f"Response body: {e.response.text}")
        sys.exit(1)

    ws_url = otp_data.get("data", {}).get("url")
    if not ws_url:
        print("No WebSocket URL returned in OTP response.")
        print(otp_data)
        sys.exit(1)

    print(f"Got WebSocket URL (OTP embedded).")
    print("\nStep 3: Connecting to WebSocket...")

    try:
        asyncio.run(test_websocket(ws_url))
        print("\nConnection test PASSED. Your app_id and token are working correctly.")
    except Exception as e:
        print(f"\nWEBSOCKET CONNECTION ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
