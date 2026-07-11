"""
Deriv API connection test script.

Purpose: confirm your DERIV_APP_ID and DERIV_API_TOKEN work correctly
before wiring Deriv into the main trading bot.

Run this as a ONE-OFF script on Railway (not your main web process).
On Railway: add DERIV_APP_ID and DERIV_API_TOKEN as environment variables,
then run this file manually via Railway's "Run Command" or a temporary
service pointed at this script.

Expected output:
1. An "authorize" response showing your account info (confirms token works)
2. A "candles" response showing recent price data for Volatility 100 Index

If either step fails, the printed error will tell you what's wrong
(bad token, bad app_id, wrong scopes, etc).
"""

import asyncio
import json
import os
import sys

import websockets

APP_ID = os.environ.get("DERIV_APP_ID")
API_TOKEN = os.environ.get("DERIV_API_TOKEN")


async def test_connection():
    if not APP_ID or not API_TOKEN:
        print("ERROR: DERIV_APP_ID and/or DERIV_API_TOKEN environment variables are missing.")
        print("Set them in Railway's Variables tab before running this script.")
        sys.exit(1)

    uri = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
    print(f"Connecting to {uri} ...")

    try:
        async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
            # Step 1: Authorize
            await ws.send(json.dumps({"authorize": API_TOKEN}))
            auth_response = json.loads(await ws.recv())

            if "error" in auth_response:
                print("AUTHORIZATION FAILED:")
                print(json.dumps(auth_response["error"], indent=2))
                sys.exit(1)

            account_info = auth_response.get("authorize", {})
            print("\n--- AUTHORIZATION SUCCESSFUL ---")
            print(f"Account: {account_info.get('loginid')}")
            print(f"Currency: {account_info.get('currency')}")
            print(f"Balance: {account_info.get('balance')}")
            print(f"Is Virtual (demo): {account_info.get('is_virtual')}")

            # Step 2: Request sample candle data
            await ws.send(json.dumps({
                "ticks_history": "R_100",
                "count": 10,
                "end": "latest",
                "style": "candles",
                "granularity": 900  # 15-minute candles
            }))
            candle_response = json.loads(await ws.recv())

            if "error" in candle_response:
                print("\nCANDLE REQUEST FAILED:")
                print(json.dumps(candle_response["error"], indent=2))
                sys.exit(1)

            candles = candle_response.get("candles", [])
            print(f"\n--- RECEIVED {len(candles)} CANDLES for R_100 ---")
            for c in candles[-3:]:
                print(c)

            print("\nConnection test PASSED. Your app_id and token are working.")

    except Exception as e:
        print(f"\nCONNECTION ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_connection())
